"""
End-to-end pipeline orchestrator for Chalkline career mapping.

Coordinates the geometry track (extraction, vectorization, PCA,
clustering) and co-occurrence track (PMI network, Louvain communities)
into a single `fit()` / `match()` interface. Fitted artifacts are
cached in memory and optionally persisted to disk via `save()` so
that `match()` can project resumes without re-fitting.
"""

import pandas as pd

from collections               import Counter, defaultdict
from contextlib                import contextmanager
from datetime                  import datetime, timezone
from joblib                    import dump, load
from json                      import dumps, loads
from logging                   import getLogger
from pathlib                   import Path
from pydantic                  import TypeAdapter
from sklearn.pipeline          import Pipeline
from sklearn.utils.validation  import check_is_fitted
from time                      import perf_counter
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
    cluster_skills: dict[int, set[str]] = defaultdict(set)
    for doc, cid in zip(clusterer.document_ids, clusterer.assignments):
        cluster_skills[cid].update(extracted_skills.get(doc, []))

    by_cluster: dict[int, list[str]] = defaultdict(list)
    for soc, cid in zip(sector_labels, clusterer.assignments):
        by_cluster[cid].append(
            occupation_index.get(soc).sector
        )

    size_map  = {cl.cluster_id: cl.size for cl in cluster_labels}
    terms_map = {cl.cluster_id: cl.terms for cl in cluster_labels}

    profiles: dict[int, ClusterProfile] = {}
    for cid, skills in cluster_skills.items():
        terms       = terms_map.get(cid, [])
        apps, progs = trades.match([*terms, *skills])

        profiles[cid] = ClusterProfile(
            cluster_id = cid,
            job_zone   = occupation_index.job_zone_for_skills(
                {s.lower() for s in skills}
            ),
            sector = Counter(by_cluster[cid]).most_common(1)[0][0],
            size   = size_map[cid],
            skills = skills,
            terms  = terms,

            apprenticeship = next(iter(apps), None),
            programs       = progs
        )

    return profiles

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

class Chalkline:
    """
    End-to-end orchestrator for the Chalkline career mapping pipeline.

    Coordinates two parallel tracks (geometry and co-occurrence) that
    merge at the pathway graph stage, caches all fitted transforms,
    and provides `match()` for single-resume inference without
    re-fitting. Fitted artifacts are accessible as instance attributes
    for direct consumption by the Marimo notebook.
    """

    def __init__(self, config: PipelineConfig):
        """
        Args:
            config: End-to-end pipeline configuration.
        """
        self.config = config

        self.cluster_labels    : list[ClusterLabel]              = []
        self.clusterer         : HierarchicalClusterer | None    = None
        self.extractor         : SkillExtractor | None           = None
        self.extracted_skills  : dict[str, list[str]]            = {}
        self.geometry_pipeline : Pipeline | None          = None
        self.graph             : CareerPathwayGraph | None       = None
        self.matcher           : ResumeMatcher | None            = None
        self.ppmi_df           : pd.DataFrame | None             = None
        self.profiles          : dict[int, ClusterProfile]       = {}
        self.router            : CareerRouter | None             = None
        self.trades            : TradeIndex | None               = None

    @property
    def fitted(self) -> bool:
        """
        Whether the pipeline has been fitted and is ready for
        `match()` calls.
        """
        return self.geometry_pipeline is not None

    def _build_matcher(self) -> ResumeMatcher:
        """
        Construct a `ResumeMatcher` from the pipeline's fitted
        artifacts.

        Centralizes the 8-argument wiring so that both `fit()` and
        `load()` share the same construction path.
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

    def _load_corpus(self) -> dict[str, str]:
        """
        Load the posting corpus as a document-ID-to-text mapping.
        """
        postings = CorpusStorage(self.config.postings_dir).load()
        return {p.id: p.description for p in postings}

    def _load_lexicons(self) -> LexiconLoader:
        """
        Load all lexicon files from the configured directory.
        """
        return LexiconLoader(self.config.lexicon_dir)

    def _load_trades(self) -> TradeIndex:
        """
        Load apprenticeship and program reference data into a
        `TradeIndex`.
        """
        return TradeIndex(
            apprenticeships = TypeAdapter(list[ApprenticeshipContext]).validate_json(
                (self.config.reference_dir / "apprenticeships.json").read_bytes()
            ),
            programs = TypeAdapter(list[ProgramRecommendation]).validate_json(
                (self.config.lexicon_dir / "programs.json").read_bytes()
            )
        )

    @contextmanager
    def _timed(self, label: str):
        """
        Context manager that logs elapsed time for a pipeline step.
        """
        start = perf_counter()
        yield
        logger.info(f"{label}: {perf_counter() - start:.1f}s")

    def _validate_data(self):
        """
        Verify that all required data files exist before processing.

        Raises:
            FileNotFoundError: When any required file or directory
                is missing.
        """
        missing = []
        d = self.config.lexicon_dir
        for name in ("certifications.json", "onet.json",
                     "osha.json", "supplement.json"):
            if not (d / name).exists():
                missing.append(str(d / name))

        r = self.config.reference_dir
        for name in ("apprenticeships.json", "cc_programs.json",
                     "onet_codes.json", "umaine_programs.json"):
            if not (r / name).exists():
                missing.append(str(r / name))

        if not list(self.config.postings_dir.glob("*.json")):
            missing.append(f"{self.config.postings_dir}/*.json")

        if missing:
            raise FileNotFoundError(
                f"Missing required data files: {', '.join(missing)}"
            )

    def fit(self) -> Self:
        """
        Execute all pipeline steps in dependency order.

        Runs the geometry track (extraction, vectorization, PCA,
        clustering) and co-occurrence track (PMI network) in
        sequence, enriches clusters with stakeholder reference data,
        builds the career pathway graph and router, and composes the
        geometry pipeline for resume projection. All fitted artifacts
        are cached as instance attributes.

        Returns:
            The fitted pipeline instance for chaining.
        """
        self._validate_data()

        lexicons         = self._load_lexicons()
        registry         = LexiconRegistry(
            certifications   = lexicons.certifications,
            occupations      = lexicons.occupations,
            osha_terms       = lexicons.osha_terms,
            supplement_terms = lexicons.supplement_terms
        )

        self.extractor   = SkillExtractor(registry)
        self.trades      = self._load_trades()
        corpus           = self._load_corpus()
        occupation_index = OccupationIndex(lexicons.occupations)

        with self._timed("Extraction"):
            self.extracted_skills = self.extractor.extract(corpus)

        with self._timed("Vectorization"):
            vectorizer = SkillVectorizer(self.extracted_skills)

        with self._timed("PCA reduction"):
            reducer = PcaReducer(
                max_components     = self.config.max_components,
                random_seed        = self.config.random_seed,
                tfidf_matrix       = vectorizer.tfidf_matrix,
                variance_threshold = self.config.variance_threshold
            )

        with self._timed("Clustering"):
            self.clusterer = HierarchicalClusterer(
                coordinates  = reducer.coordinates,
                document_ids = vectorizer.document_ids
            )
            self.cluster_labels = self.clusterer.labels(
                feature_names = vectorizer.feature_names,
                tfidf_matrix  = vectorizer.tfidf_matrix,
                top_n         = 50
            )

        with self._timed("Co-occurrence"):
            network = CooccurrenceNetwork(
                binary_matrix    = vectorizer.binary_matrix,
                feature_names    = vectorizer.feature_names,
                min_cooccurrence = self.config.min_cooccurrence,
                random_seed      = self.config.random_seed
            )

        sector_labels = compute_sector_labels(
            document_ids     = vectorizer.document_ids,
            extracted_skills = self.extracted_skills,
            occupation_index = occupation_index
        )

        self.geometry_pipeline = compose_geometry(reducer, vectorizer)

        with self._timed("Profile enrichment"):
            self.profiles = build_profiles(
                cluster_labels   = self.cluster_labels,
                clusterer        = self.clusterer,
                extracted_skills = self.extracted_skills,
                occupation_index = occupation_index,
                sector_labels    = sector_labels,
                trades           = self.trades
            )

        with self._timed("Pathway graph"):
            self.graph = CareerPathwayGraph(
                max_density = self.config.max_graph_density,
                network     = network,
                profiles    = self.profiles
            )

        self.router = CareerRouter(
            graph    = self.graph.graph,
            profiles = self.profiles,
            trades   = self.trades
        )

        self.ppmi_df = network.ppmi_dataframe()
        self.matcher = self._build_matcher()

        return self

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
                           Defaults to `config.output_dir / "pipeline"`.

        Returns:
            A fitted `Chalkline` ready for `match()` calls.
        """
        d = artifact_dir or config.output_dir / "pipeline"

        pipe                   = cls(config)
        lexicons               = pipe._load_lexicons()
        pipe.extractor         = SkillExtractor(LexiconRegistry(
            certifications   = lexicons.certifications,
            occupations      = lexicons.occupations,
            osha_terms       = lexicons.osha_terms,
            supplement_terms = lexicons.supplement_terms
        ))
        pipe.geometry_pipeline = load(d / "geometry.joblib")
        pipe.extracted_skills  = loads(
            (d / "extracted_skills.json").read_text()
        )

        check_is_fitted(pipe.geometry_pipeline)

        pipe.clusterer      = load(d / "clusterer.joblib")
        pipe.cluster_labels = [
            ClusterLabel(**raw)
            for raw in loads((d / "cluster_labels.json").read_text())
        ]
        pipe.profiles = {
            int(k): ClusterProfile(**v)
            for k, v in loads(
                (d / "profiles.json").read_text()
            ).items()
        }

        pipe.graph = CareerPathwayGraph(
            graph    = load(d / "graph.joblib"),
            profiles = pipe.profiles
        )

        pipe.trades = pipe._load_trades()
        pipe.ppmi_df = load(d / "ppmi.joblib")

        pipe.router = CareerRouter(
            graph    = pipe.graph.graph,
            profiles = pipe.profiles,
            trades   = pipe.trades
        )

        pipe.matcher = pipe._build_matcher()

        return pipe

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

        Raises:
            RuntimeError: If the pipeline has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError(
                "Pipeline is not fitted; call fit() or load() first"
            )

        skills = self.extractor.extract(
            {"resume": resume_text}
        )["resume"]
        result = self.matcher.match(resume_skills=skills, top_k=top_k)

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
                           Defaults to `config.output_dir / "pipeline"`.
        """
        if not self.fitted:
            raise RuntimeError(
                "Pipeline is not fitted; call fit() first"
            )

        d = artifact_dir or self.config.output_dir / "pipeline"
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

        manifest = PipelineManifest(
            corpus_size     = len(self.extracted_skills),
            geometry_params = self.geometry_pipeline.get_params(
                deep=True
            ),
            posting_ids     = sorted(self.extracted_skills),
            timestamp       = datetime.now(timezone.utc).isoformat()
        )
        (d / "manifest.json").write_text(
            manifest.model_dump_json(indent=2)
        )

        logger.info(f"Pipeline artifacts saved to {d}")
