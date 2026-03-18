"""
Hamilton DAG node functions for the Chalkline pipeline.

Each function is a node whose parameter names declare dependencies
that Hamilton resolves automatically. The geometry track
(vectorization, PCA, clustering) and co-occurrence track (PMI
network) fork after vectorization and merge at the pathway graph,
enabling parallel execution of independent branches.
"""

import numpy  as np
import pandas as pd

from collections                 import Counter, defaultdict
from datetime                    import datetime, timezone
from hamilton.function_modifiers import extract_fields
from loguru                      import logger
from pydantic                    import TypeAdapter
from sklearn.pipeline            import Pipeline

from chalkline                          import SkillMap
from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.clustering.hierarchical  import HierarchicalClusterer
from chalkline.clustering.schemas       import ClusterLabel
from chalkline.collection.storage       import CorpusStorage
from chalkline.extraction.lexicons      import LexiconRegistry
from chalkline.extraction.loaders       import LexiconLoader
from chalkline.extraction.occupations   import OccupationIndex
from chalkline.extraction.skills        import SkillExtractor
from chalkline.extraction.vectorize     import SkillVectorizer
from chalkline.matching.matcher         import ResumeMatcher
from chalkline.pathways.graph           import CareerPathwayGraph
from chalkline.pathways.routing         import CareerRouter
from chalkline.pipeline.schemas         import ApprenticeshipContext, ClusterProfile
from chalkline.pipeline.schemas         import PipelineConfig, PipelineManifest
from chalkline.pipeline.schemas         import ProgramRecommendation
from chalkline.pipeline.trades          import TradeIndex
from chalkline.reduction.pca            import PcaReducer



def cluster_labels(
    clusterer     : HierarchicalClusterer,
    feature_names : list[str],
    tfidf_matrix  : object
) -> list[ClusterLabel]:
    """
    Generate TF-IDF centroid labels for each cluster.
    """
    return clusterer.labels(
        feature_names = feature_names,
        tfidf_matrix  = tfidf_matrix,
        top_n         = 50
    )


def clusterer(
    coordinates  : np.ndarray,
    document_ids : list[str]
) -> HierarchicalClusterer:
    """
    Fit Ward-linkage HAC on PCA coordinates.
    """
    return HierarchicalClusterer(
        coordinates  = coordinates,
        document_ids = document_ids
    )


def config(pipeline_config: PipelineConfig) -> PipelineConfig:
    """
    Pass pipeline configuration through as a DAG node.
    """
    return pipeline_config


def corpus(pipeline_config: PipelineConfig) -> dict[str, str]:
    """
    Load posting texts from the corpus directory.

    Raises:
        FileNotFoundError: When no JSON postings exist in the
            configured directory.
    """
    if not (result := {
        p.id: p.description
        for p in CorpusStorage(pipeline_config.postings_dir).load()
    }):
        raise FileNotFoundError(
            f"No postings found in {pipeline_config.postings_dir}"
        )
    return result


def density(
    binary_matrix    : object,
    extracted_skills : SkillMap
) -> dict:
    """
    Evaluate feature density of the lexicon-matching extraction
    paradigm.

    Logs vocabulary size, mean skills per posting, and the
    fraction of posting pairs with zero skill overlap to assess
    whether the feature space supports meaningful clustering
    and gap ranking.
    """
    n_docs     = len(extracted_skills)
    vocab_size = len({s for skills in extracted_skills.values() for s in skills})
    mean_skills = np.mean([len(s) for s in extracted_skills.values()])

    overlap = binary_matrix @ binary_matrix.T
    overlap.setdiag(0)
    n_pairs      = n_docs * (n_docs - 1) // 2
    zero_overlap = 1.0 - (overlap.nnz // 2) / n_pairs

    logger.info(
        f"Feature density: {vocab_size} vocabulary, "
        f"{mean_skills:.1f} mean skills/posting, "
        f"{zero_overlap:.1%} zero-overlap pairs"
    )

    return {
        "mean_skills_per_posting" : float(mean_skills),
        "vocabulary_size"         : vocab_size,
        "zero_overlap_fraction"   : float(zero_overlap)
    }


def extracted_skills(
    corpus    : dict[str, str],
    extractor : SkillExtractor
) -> SkillMap:
    """
    Extract canonical skills from each posting via
    Aho-Corasick pattern matching.
    """
    return extractor.extract(corpus)


def extractor(registry: LexiconRegistry) -> SkillExtractor:
    """
    Build an Aho-Corasick skill extractor from the registry.
    """
    return SkillExtractor(registry)


def geometry_pipeline(
    reducer_pipeline    : Pipeline,
    vectorizer_pipeline : Pipeline
) -> Pipeline:
    """
    Chain fitted vectorization and reduction into a single
    sklearn `Pipeline` for resume projection.

    Extracts the named steps from the vectorizer pipeline
    (DictVectorizer, TfidfTransformer, Normalizer) and the
    reducer pipeline (TruncatedSVD, StandardScaler) into one
    five-step `Pipeline` whose `transform([skill_dict])`
    projects a resume directly into PCA-scaled coordinates.
    """
    pca_steps = reducer_pipeline.named_steps
    vec_steps = vectorizer_pipeline.named_steps
    return Pipeline([
        ("vec",    vec_steps["vec"]),
        ("tfidf",  vec_steps["tfidf"]),
        ("norm",   vec_steps["norm"]),
        ("svd",    pca_steps["svd"]),
        ("scaler", pca_steps["scaler"])
    ])


def graph(
    network         : CooccurrenceNetwork,
    pipeline_config : PipelineConfig,
    profiles        : dict[int, ClusterProfile]
) -> CareerPathwayGraph:
    """
    Build the career pathway DAG from cluster profiles and
    PMI co-occurrence edges.
    """
    return CareerPathwayGraph(
        max_density = pipeline_config.max_graph_density,
        network     = network,
        profiles    = profiles
    )


def lexicons(pipeline_config: PipelineConfig) -> LexiconLoader:
    """
    Load all lexicon files from the configured directory.
    """
    return LexiconLoader(pipeline_config.lexicon_dir)


def manifest(
    extracted_skills  : SkillMap,
    geometry_pipeline : Pipeline
) -> PipelineManifest:
    """
    Build provenance metadata from fitted artifacts.
    """
    return PipelineManifest(
        corpus_size     = len(extracted_skills),
        geometry_params = geometry_pipeline.get_params(deep=True),
        posting_ids     = sorted(extracted_skills),
        timestamp       = datetime.now(timezone.utc).isoformat()
    )


def matcher(
    cluster_labels    : list[ClusterLabel],
    clusterer         : HierarchicalClusterer,
    extracted_skills  : SkillMap,
    geometry_pipeline : Pipeline,
    pipeline_config   : PipelineConfig,
    ppmi_df           : pd.DataFrame,
    profiles          : dict[int, ClusterProfile],
    trades            : TradeIndex
) -> ResumeMatcher:
    """
    Fit nearest-neighbor models on cluster centroids for
    single-resume inference.
    """
    return ResumeMatcher(
        cluster_labels    = cluster_labels,
        clusterer         = clusterer,
        extracted_skills  = extracted_skills,
        geometry_pipeline = geometry_pipeline,
        metric            = pipeline_config.distance_metric,
        ppmi_df           = ppmi_df,
        profiles          = profiles,
        top_k_gaps        = pipeline_config.top_k_gaps,
        trades            = trades
    )


def network(
    binary_matrix   : object,
    feature_names   : list[str],
    pipeline_config : PipelineConfig
) -> CooccurrenceNetwork:
    """
    Build the PMI co-occurrence network from the binary
    skill-presence matrix.
    """
    return CooccurrenceNetwork(
        binary_matrix    = binary_matrix,
        feature_names    = feature_names,
        min_cooccurrence = pipeline_config.min_cooccurrence,
        random_seed      = pipeline_config.random_seed
    )


def occupation_index(lexicons: LexiconLoader) -> OccupationIndex:
    """
    Build an O*NET occupation index from loaded lexicon data.
    """
    return OccupationIndex(lexicons.occupations)


def ppmi_df(network: CooccurrenceNetwork) -> pd.DataFrame:
    """
    Materialize the positive PMI DataFrame for gap ranking.
    """
    return network.ppmi_dataframe()


def profiles(
    cluster_labels   : list[ClusterLabel],
    clusterer        : HierarchicalClusterer,
    extracted_skills : SkillMap,
    occupation_index : OccupationIndex,
    sector_labels    : list[str],
    trades           : TradeIndex
) -> dict[int, ClusterProfile]:
    """
    Enrich HAC clusters with sector, Job Zone, apprenticeship,
    and program annotations.

    Aggregates the union skill set per cluster from extracted
    skills, resolves sector via majority vote on SOC-matched
    labels, assigns Job Zone via overlap coefficient against
    concrete O*NET profiles, and links apprenticeships and
    programs via the shared `TradeIndex` prefix lookups.
    """
    cluster_sectors = defaultdict(list)
    cluster_skills  = defaultdict(set)
    for doc, soc, cid in zip(
        clusterer.document_ids, sector_labels, clusterer.assignments
    ):
        cluster_sectors[cid].append(occupation_index.get(soc).sector)
        cluster_skills[cid].update(extracted_skills.get(doc, []))

    majority_sector = {
        cid: Counter(sectors).most_common(1)[0][0]
        for cid, sectors in cluster_sectors.items()
    }

    return {
        label.cluster_id: ClusterProfile(
            apprenticeship = next(iter(apps), None),
            cluster_id     = label.cluster_id,
            job_zone       = occupation_index.job_zone_for_skills(skills),
            programs       = progs,
            sector         = majority_sector[label.cluster_id],
            size           = label.size,
            skills         = skills,
            terms          = label.terms
        )
        for label       in cluster_labels
        for skills      in [cluster_skills[label.cluster_id]]
        for apps, progs in [trades.lookup([*label.terms, *skills])]
    }


@extract_fields({
    "coordinates"      : np.ndarray,
    "reducer_pipeline" : Pipeline
})
def reducer(
    pipeline_config : PipelineConfig,
    tfidf_matrix    : object
) -> dict:
    """
    Fit PCA via TruncatedSVD and expose coordinates and the
    fitted sklearn pipeline.

    The `coordinates` feed the clusterer while the
    `reducer_pipeline` feeds geometry pipeline composition.
    """
    r = PcaReducer(
        max_components     = pipeline_config.max_components,
        random_seed        = pipeline_config.random_seed,
        tfidf_matrix       = tfidf_matrix,
        variance_threshold = pipeline_config.variance_threshold
    )
    return {
        "coordinates"      : r.coordinates,
        "reducer_pipeline" : r.pipeline
    }


def registry(lexicons: LexiconLoader) -> LexiconRegistry:
    """
    Build a `LexiconRegistry` from loaded lexicon data.
    """
    return LexiconRegistry(
        certifications   = lexicons.certifications,
        occupations      = lexicons.occupations,
        osha_terms       = lexicons.osha_terms,
        supplement_terms = lexicons.supplement_terms
    )


def router(
    graph    : CareerPathwayGraph,
    profiles : dict[int, ClusterProfile],
    trades   : TradeIndex
) -> CareerRouter:
    """
    Build the career router with centrality and edge
    enrichment.
    """
    return CareerRouter(
        graph    = graph.graph,
        profiles = profiles,
        trades   = trades
    )


def sector_labels(
    document_ids     : list[str],
    extracted_skills : SkillMap,
    occupation_index : OccupationIndex
) -> list[str]:
    """
    Map postings to SOC codes via Jaccard-nearest occupation.

    For each document in row order, finds the O*NET occupation
    whose skill profile has maximum Jaccard overlap with the
    posting's canonical skill set, then returns that
    occupation's SOC code.
    """
    return [
        occupation_index.get(
            occupation_index.nearest(set(extracted_skills[doc]))
        ).soc_code
        for doc in document_ids
    ]


def trades(pipeline_config: PipelineConfig) -> TradeIndex:
    """
    Load apprenticeship and program reference data into a
    `TradeIndex`.
    """
    d = pipeline_config.lexicon_dir
    return TradeIndex(
        apprenticeships = TypeAdapter(
            list[ApprenticeshipContext]
        ).validate_json((d / "apprenticeships.json").read_bytes()),
        programs = TypeAdapter(
            list[ProgramRecommendation]
        ).validate_json((d / "programs.json").read_bytes())
    )


@extract_fields({
    "binary_matrix"       : object,
    "document_ids"        : list,
    "feature_names"       : list,
    "tfidf_matrix"        : object,
    "vectorizer_pipeline" : Pipeline
})
def vectorizer(
    extracted_skills: SkillMap
) -> dict:
    """
    Fit TF-IDF vectorization and expose intermediate matrices.

    Returns fields consumed independently by downstream nodes:
    `tfidf_matrix` for PCA and cluster labeling,
    `binary_matrix` for co-occurrence, `document_ids` and
    `feature_names` for alignment, and `vectorizer_pipeline`
    for geometry composition.
    """
    v = SkillVectorizer(extracted_skills)
    return {
        "binary_matrix"       : v.binary_matrix,
        "document_ids"        : v.document_ids,
        "feature_names"       : v.feature_names,
        "tfidf_matrix"        : v.tfidf_matrix,
        "vectorizer_pipeline" : v.pipeline
    }
