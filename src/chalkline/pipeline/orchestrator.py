"""
End-to-end pipeline orchestrator for Chalkline career mapping.

Coordinates the geometry track (extraction, vectorization, PCA,
clustering) and co-occurrence track (PMI network, Louvain communities)
into a fitted `Chalkline` instance constructed via `fit()` or `load()`.
All fitted artifacts are guaranteed present by construction, and
`match()` projects resumes without re-fitting.
"""

import pandas as pd

from collections               import Counter, defaultdict
from dataclasses               import dataclass
from datetime                  import datetime, timezone
from functools                 import cached_property
from joblib                    import dump, load
from json                      import dumps, loads
from logging                   import getLogger
from pathlib                   import Path
from pydantic                  import TypeAdapter
from sklearn.pipeline          import Pipeline
from sklearn.utils.validation  import check_is_fitted
from typing                    import Self

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
from chalkline.matching.schemas         import MatchResult
from chalkline.pathways.graph           import CareerPathwayGraph
from chalkline.pathways.routing         import CareerRouter
from chalkline.pipeline.schemas         import ApprenticeshipContext, ClusterProfile
from chalkline.pipeline.schemas         import PipelineConfig, PipelineManifest
from chalkline.pipeline.schemas         import ProgramRecommendation
from chalkline.pipeline.trades          import TradeIndex
from chalkline.reduction.pca            import PcaReducer

logger = getLogger(__name__)


def build_profiles(
    cluster_labels   : list[ClusterLabel],
    clusterer        : HierarchicalClusterer,
    extracted_skills : dict[str, list[str]],
    occupation_index : OccupationIndex,
    sector_labels    : list[str],
    trades           : TradeIndex
) -> dict[int, ClusterProfile]:
    """
    Enrich HAC clusters with sector, Job Zone, apprenticeship, and
    program annotations.

    Aggregates the union skill set per cluster from extracted skills,
    resolves sector via majority vote on SOC-matched labels, assigns
    Job Zone via overlap coefficient against concrete O*NET profiles,
    and links apprenticeships and programs via the shared
    `TradeIndex` prefix lookups.

    Args:
        cluster_labels   : TF-IDF centroid labels per cluster.
        clusterer        : Fitted hierarchical clusterer.
        extracted_skills : Skills per document identifier.
        occupation_index : O*NET occupation index for Job Zone.
        sector_labels    : SOC codes aligned with document order.
        trades           : Precomputed prefix lookup index.

    Returns:
        Enriched `ClusterProfile` per cluster ID.
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
        for apps, progs in [trades.match([*label.terms, *skills])]
    }


def compose_geometry(
    reducer    : PcaReducer,
    vectorizer : SkillVectorizer
) -> Pipeline:
    """
    Chain fitted vectorization and reduction steps into a single
    sklearn `Pipeline` for resume projection.

    Extracts the named steps from `SkillVectorizer.pipeline`
    (DictVectorizer, TfidfTransformer, Normalizer) and
    `PcaReducer.pipeline` (TruncatedSVD, StandardScaler) into one
    `Pipeline` whose `transform([skill_dict])` projects a resume
    directly into PCA-scaled coordinates.

    Args:
        reducer    : Fitted PCA reducer.
        vectorizer : Fitted skill vectorizer.

    Returns:
        Five-step sklearn `Pipeline` ready for `transform()`.
    """
    vec_steps = vectorizer.pipeline.named_steps
    pca_steps = reducer.pipeline.named_steps
    return Pipeline([
        ("vec",    vec_steps["vec"]),
        ("tfidf",  vec_steps["tfidf"]),
        ("norm",   vec_steps["norm"]),
        ("svd",    pca_steps["svd"]),
        ("scaler", pca_steps["scaler"])
    ])


def compute_sector_labels(
    document_ids     : list[str],
    extracted_skills : dict[str, list[str]],
    occupation_index : OccupationIndex
) -> list[str]:
    """
    Map postings to SOC codes via Jaccard-nearest occupation.

    For each document in row order, finds the O*NET occupation
    whose skill profile has maximum Jaccard overlap with the
    posting's canonical skill set, then returns that occupation's
    SOC code. This produces 21 distinct ground-truth classes for
    ARI evaluation rather than 3 broad sectors.

    Args:
        document_ids     : Posting identifiers in row order.
        extracted_skills : Skills per document identifier.
        occupation_index : O*NET occupation index.

    Returns:
        SOC code strings aligned with `document_ids` row order.
    """
    return [
        occupation_index.get(
            occupation_index.nearest(set(extracted_skills[doc]))
        ).soc_code
        for doc in document_ids
    ]


def _build_registry(lexicons: LexiconLoader) -> LexiconRegistry:
    """
    Build a `LexiconRegistry` from loaded lexicon data.
    """
    return LexiconRegistry(
        certifications   = lexicons.certifications,
        occupations      = lexicons.occupations,
        osha_terms       = lexicons.osha_terms,
        supplement_terms = lexicons.supplement_terms
    )


def _load_trades(config: PipelineConfig) -> TradeIndex:
    """
    Load apprenticeship and program reference data into a
    `TradeIndex`.
    """
    d = config.lexicon_dir
    return TradeIndex(
        apprenticeships = TypeAdapter(list[ApprenticeshipContext]).validate_json(
            (d / "apprenticeships.json").read_bytes()
        ),
        programs = TypeAdapter(list[ProgramRecommendation]).validate_json(
            (d / "programs.json").read_bytes()
        )
    )


def _validate_data(config: PipelineConfig):
    """
    Verify that all required data files exist before processing.

    Raises:
        FileNotFoundError: When any required file or directory
            is missing.
    """
    d = config.lexicon_dir
    missing = [
        str(d / name)
        for name in (
            "apprenticeships.json",
            "certifications.json",
            "onet.json",
            "osha.json",
            "programs.json",
            "supplement.json"
        )
        if not (d / name).exists()
    ]

    if not any(config.postings_dir.glob("*.json")):
        missing.append(f"{config.postings_dir}/*.json")

    if missing:
        raise FileNotFoundError(
            f"Missing required data files: {', '.join(missing)}"
        )


@dataclass(kw_only=True)
class Chalkline:
    """
    Fitted career mapping pipeline.

    Coordinates the geometry track (extraction, vectorization, PCA,
    clustering) and co-occurrence track (PMI network, Louvain
    communities) into a fitted landscape. Construct via `fit()` or
    `load()`, then access fitted artifacts directly or call `match()`
    for single-resume inference.
    """

    cluster_labels    : list[ClusterLabel]
    clusterer         : HierarchicalClusterer
    config            : PipelineConfig
    extracted_skills  : dict[str, list[str]]
    extractor         : SkillExtractor
    geometry_pipeline : Pipeline
    graph             : CareerPathwayGraph
    ppmi_df           : pd.DataFrame
    profiles          : dict[int, ClusterProfile]
    router            : CareerRouter
    trades            : TradeIndex

    @cached_property
    def matcher(self) -> ResumeMatcher:
        """
        Lazily construct the resume matcher from fitted artifacts.

        Deferred because `ResumeMatcher` fits nearest-neighbor
        models and computes centroids, and notebook users
        inspecting profiles or graphs should not pay that cost.
        """
        return ResumeMatcher(
            cluster_labels    = self.cluster_labels,
            clusterer         = self.clusterer,
            extracted_skills  = self.extracted_skills,
            geometry_pipeline = self.geometry_pipeline,
            metric            = self.config.distance_metric,
            ppmi_df           = self.ppmi_df,
            top_k_gaps        = self.config.top_k_gaps,
            trades            = self.trades
        )

    @classmethod
    def fit(cls, config: PipelineConfig) -> Self:
        """
        Execute all pipeline steps and return a fitted pipeline.

        Runs the geometry track (extraction, vectorization, PCA,
        clustering) and co-occurrence track (PMI network) in
        sequence, enriches clusters with stakeholder reference data,
        and builds the career pathway graph and router.

        Args:
            config: End-to-end pipeline configuration.

        Returns:
            A fully fitted `Chalkline` instance.
        """
        _validate_data(config)

        lexicons         = LexiconLoader(config.lexicon_dir)
        extractor        = SkillExtractor(_build_registry(lexicons))
        trades           = _load_trades(config)
        corpus = {
            p.id: p.description
            for p in CorpusStorage(config.postings_dir).load()
        }
        occupation_index = OccupationIndex(lexicons.occupations)

        extracted_skills = extractor.extract(corpus)
        vectorizer       = SkillVectorizer(extracted_skills)

        reducer = PcaReducer(
            max_components     = config.max_components,
            random_seed        = config.random_seed,
            tfidf_matrix       = vectorizer.tfidf_matrix,
            variance_threshold = config.variance_threshold
        )

        clusterer = HierarchicalClusterer(
            coordinates  = reducer.coordinates,
            document_ids = vectorizer.document_ids
        )
        cluster_labels = clusterer.labels(
            feature_names = vectorizer.feature_names,
            tfidf_matrix  = vectorizer.tfidf_matrix,
            top_n         = 50
        )

        network = CooccurrenceNetwork(
            binary_matrix    = vectorizer.binary_matrix,
            feature_names    = vectorizer.feature_names,
            min_cooccurrence = config.min_cooccurrence,
            random_seed      = config.random_seed
        )

        sector_labels     = compute_sector_labels(
            document_ids     = vectorizer.document_ids,
            extracted_skills = extracted_skills,
            occupation_index = occupation_index
        )
        geometry_pipeline = compose_geometry(reducer, vectorizer)

        profiles = build_profiles(
            cluster_labels   = cluster_labels,
            clusterer        = clusterer,
            extracted_skills = extracted_skills,
            occupation_index = occupation_index,
            sector_labels    = sector_labels,
            trades           = trades
        )

        graph = CareerPathwayGraph(
            max_density = config.max_graph_density,
            network     = network,
            profiles    = profiles
        )

        router = CareerRouter(
            graph    = graph.graph,
            profiles = profiles,
            trades   = trades
        )

        return cls(
            cluster_labels    = cluster_labels,
            clusterer         = clusterer,
            config            = config,
            extracted_skills  = extracted_skills,
            extractor         = extractor,
            geometry_pipeline = geometry_pipeline,
            graph             = graph,
            ppmi_df           = network.ppmi_dataframe(),
            profiles          = profiles,
            router            = router,
            trades            = trades
        )

    @classmethod
    def load(
        cls,
        config       : PipelineConfig,
        artifact_dir : Path | None = None
    ) -> Self:
        """
        Load a fitted pipeline from serialized artifacts.

        Rebuilds the `SkillExtractor` from lexicon files because the
        AhoCorasick automaton is not serializable. Validates sklearn
        objects via `check_is_fitted()` to ensure artifact integrity.

        Args:
            config       : Pipeline configuration for lexicon and
                           reference data paths.
            artifact_dir : Directory containing serialized artifacts.
                           Defaults to `config.pipeline_dir`.

        Returns:
            A fitted `Chalkline` ready for `match()` calls.
        """
        d = artifact_dir or config.pipeline_dir

        lexicons          = LexiconLoader(config.lexicon_dir)
        geometry_pipeline = load(d / "geometry.joblib")
        check_is_fitted(geometry_pipeline)

        clusterer = load(d / "clusterer.joblib")
        cluster_labels = [
            ClusterLabel(**raw)
            for raw in loads((d / "cluster_labels.json").read_text())
        ]
        extracted_skills = loads((d / "extracted_skills.json").read_text())
        profiles = {
            int(k): ClusterProfile(**v)
            for k, v in loads(
                (d / "profiles.json").read_text()
            ).items()
        }

        graph = CareerPathwayGraph(
            graph    = load(d / "graph.joblib"),
            profiles = profiles
        )

        trades = _load_trades(config)

        return cls(
            cluster_labels    = cluster_labels,
            clusterer         = clusterer,
            config            = config,
            extracted_skills  = extracted_skills,
            extractor         = SkillExtractor(_build_registry(lexicons)),
            geometry_pipeline = geometry_pipeline,
            graph             = graph,
            ppmi_df           = load(d / "ppmi.joblib"),
            profiles          = profiles,
            router            = CareerRouter(
                graph    = graph.graph,
                profiles = profiles,
                trades   = trades
            ),
            trades            = trades
        )

    def match(
        self,
        resume_text : str,
        top_k       : int | None = None
    ) -> MatchResult:
        """
        Project a resume into the fitted career landscape and return
        a full match result with gap analysis.

        Extracts skills from the resume text via the fitted
        `SkillExtractor`, delegates projection and gap ranking to
        `ResumeMatcher.match()`, and enriches the result with the
        matched cluster's sector from the profile data.

        Args:
            resume_text : Raw resume text (post-PDF extraction).
            top_k       : Override for the default `top_k_gaps`.

        Returns:
            Enriched `MatchResult` with sector annotation.
        """
        result = self.matcher.match(
            resume_skills = self.extractor.extract({"resume": resume_text})["resume"],
            top_k         = top_k
        )

        return result.model_copy(update={
            "sector": self.profiles[result.cluster_id].sector
        })

    def save(self, artifact_dir: Path | None = None):
        """
        Persist fitted artifacts to disk.

        Writes sklearn objects and the career graph via `joblib`,
        JSON for Pydantic models, and a manifest tracking provenance
        via `geometry_pipeline.get_params(deep=True)`.

        Args:
            artifact_dir : Target directory for serialized artifacts.
                           Defaults to `config.pipeline_dir`.
        """
        d = artifact_dir or self.config.pipeline_dir
        d.mkdir(parents=True, exist_ok=True)

        dump(self.clusterer, d / "clusterer.joblib")
        dump(self.geometry_pipeline, d / "geometry.joblib")
        dump(self.ppmi_df, d / "ppmi.joblib")

        (d / "extracted_skills.json").write_text(
            dumps(self.extracted_skills, indent=2)
        )
        (d / "cluster_labels.json").write_text(dumps(
            [cl.model_dump(mode="json") for cl in self.cluster_labels],
            indent=2
        ))
        (d / "profiles.json").write_text(dumps(
            {str(k): v.model_dump(mode="json")
             for k, v in self.profiles.items()},
            indent=2
        ))

        dump(self.graph.graph, d / "graph.joblib")

        (d / "manifest.json").write_text(
            PipelineManifest(
                corpus_size     = len(self.extracted_skills),
                geometry_params = self.geometry_pipeline.get_params(deep=True),
                posting_ids     = sorted(self.extracted_skills),
                timestamp       = datetime.now(timezone.utc).isoformat()
            ).model_dump_json(indent=2)
        )

        logger.info(f"Pipeline artifacts saved to {d}")
