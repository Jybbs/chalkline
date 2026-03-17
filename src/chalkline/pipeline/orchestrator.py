"""
End-to-end pipeline orchestrator for Chalkline career mapping.

Coordinates the geometry track (extraction, vectorization, PCA,
clustering) and co-occurrence track (PMI network, Louvain communities)
into a single `fit()` / `match()` interface. Fitted artifacts are
cached in memory and optionally persisted to disk via `save()` so
that `match()` can project resumes without re-fitting.
"""

import joblib

from collections               import Counter, defaultdict
from contextlib                import contextmanager
from datetime                  import datetime, timezone
from json                      import dumps, loads
from logging                   import getLogger
from pathlib                   import Path
from sklearn.pipeline          import Pipeline as SklearnPipeline
from sklearn.utils.validation  import check_is_fitted
from time                      import perf_counter
from typing                    import Self

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.clustering.hierarchical  import HierarchicalClusterer
from chalkline.clustering.schemas       import ClusterLabel
from chalkline.collection.storage       import CorpusStorage
from chalkline.extraction.lexicons      import LexiconRegistry
from chalkline.extraction.loaders       import load_certifications, load_onet
from chalkline.extraction.loaders       import load_osha, load_supplement
from chalkline.extraction.occupations   import OccupationIndex
from chalkline.extraction.skills        import SkillExtractor
from chalkline.extraction.vectorize     import SkillVectorizer
from chalkline.matching.matcher         import ResumeMatcher
from chalkline.matching.schemas         import MatchResult
from chalkline.pathways.graph           import CareerPathwayGraph
from chalkline.pathways.routing         import CareerRouter
from chalkline.pipeline.enrichment      import EnrichmentContext, prefix_set
from chalkline.pipeline.programs        import load_programs
from chalkline.pipeline.schemas         import ApprenticeshipContext, ClusterProfile
from chalkline.pipeline.schemas         import PipelineConfig, PipelineManifest
from chalkline.pipeline.schemas         import ProgramRecommendation
from chalkline.reduction.pca            import PcaReducer


logger = getLogger(__name__)


# -------------------------------------------------------------------------
# Reusable composition helpers
# -------------------------------------------------------------------------

def build_profiles(
    cluster_labels   : list[ClusterLabel],
    clusterer        : HierarchicalClusterer,
    enrichment       : EnrichmentContext,
    extracted_skills : dict[str, list[str]],
    occupation_index : OccupationIndex,
    sector_labels    : list[str]
) -> dict[int, ClusterProfile]:
    """
    Enrich HAC clusters with sector, Job Zone, apprenticeship, and
    program annotations.

    Aggregates the union skill set per cluster from extracted skills,
    resolves sector via majority vote on SOC-matched labels, assigns
    Job Zone via overlap coefficient against concrete O*NET profiles,
    and links apprenticeships and programs via the shared
    `EnrichmentContext` prefix lookups.

    Args:
        cluster_labels   : TF-IDF centroid labels per cluster.
        clusterer        : Fitted hierarchical clusterer.
        enrichment       : Precomputed prefix lookup context.
        extracted_skills : Skills per document identifier.
        occupation_index : O*NET occupation index for Job Zone.
        sector_labels    : SOC codes aligned with document order.

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
        terms   = terms_map.get(cid, [])
        node_pf = {
            p for text in (*terms, *skills)
            for p in prefix_set(text)
        }

        profiles[cid] = ClusterProfile(
            cluster_id = cid,
            job_zone   = occupation_index.job_zone_for_skills(
                {s.lower() for s in skills}
            ),
            sector = Counter(by_cluster[cid]).most_common(1)[0][0],
            size   = size_map[cid],
            skills = skills,
            terms  = terms,

            apprenticeship = enrichment.find_apprenticeship(node_pf),
            programs       = enrichment.find_programs(node_pf)
        )

    return profiles


def compose_geometry(
    reducer    : PcaReducer,
    vectorizer : SkillVectorizer
) -> SklearnPipeline:
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
    return SklearnPipeline([
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


def deduplicate_apprenticeships(
    profiles: dict[int, ClusterProfile]
) -> list[ApprenticeshipContext]:
    """
    Collect unique apprenticeship contexts across all cluster
    profiles, keyed by RAPIDS code.
    """
    return list({
        p.apprenticeship.rapids_code: p.apprenticeship
        for p in profiles.values()
        if p.apprenticeship
    }.values())


def deduplicate_programs(
    profiles: dict[int, ClusterProfile]
) -> list[ProgramRecommendation]:
    """
    Collect unique program recommendations across all cluster
    profiles, keyed by (institution, program name).
    """
    return list({
        (p.institution, p.program): p
        for profile in profiles.values()
        for p in profile.programs
    }.values())


# -------------------------------------------------------------------------
# Pipeline
# -------------------------------------------------------------------------

class Pipeline:
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
        self.enrichment        : EnrichmentContext | None        = None
        self.extractor         : SkillExtractor | None           = None
        self.extracted_skills  : dict[str, list[str]]            = {}
        self.geometry_pipeline : SklearnPipeline | None          = None
        self.graph             : CareerPathwayGraph | None       = None
        self.matcher           : ResumeMatcher | None            = None
        self.profiles          : dict[int, ClusterProfile]       = {}
        self.router            : CareerRouter | None             = None

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """
        Whether the pipeline has been fitted and is ready for
        `match()` calls.
        """
        return self.geometry_pipeline is not None

    # -----------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------

    def _build_matcher(self, ppmi_df) -> ResumeMatcher:
        """
        Construct a `ResumeMatcher` from the pipeline's fitted
        artifacts.

        Centralizes the 8-argument wiring so that both `fit()` and
        `load()` share the same construction path.
        """
        return ResumeMatcher(
            cluster_labels    = self.cluster_labels,
            clusterer         = self.clusterer,
            enrichment        = self.enrichment,
            extracted_skills  = self.extracted_skills,
            geometry_pipeline = self.geometry_pipeline,
            metric            = self.config.distance_metric,
            ppmi_df           = ppmi_df,
            top_k_gaps        = self.config.top_k_gaps
        )

    def _load_apprenticeships(self) -> list[ApprenticeshipContext]:
        """
        Load apprenticeship reference data from the stakeholder
        directory.
        """
        path = self.config.reference_dir / "apprenticeships.json"
        return [
            ApprenticeshipContext(**raw)
            for raw in loads(path.read_text())
        ]

    def _load_corpus(self) -> dict[str, str]:
        """
        Load the posting corpus as a document-ID-to-text mapping.
        """
        postings = CorpusStorage(self.config.postings_dir).load()
        return {p.id: p.description for p in postings}

    def _load_registry(self) -> LexiconRegistry:
        """
        Build a `LexiconRegistry` from the four lexicon files.
        """
        d = self.config.lexicon_dir
        return LexiconRegistry(
            certifications   = load_certifications(d / "certifications.json"),
            occupations      = load_onet(d / "onet.json"),
            osha_terms       = load_osha(d / "osha.json"),
            supplement_terms = load_supplement(d / "supplement.json")
        )

    def _log_density(self, vectorizer: SkillVectorizer):
        """
        Log feature density metrics for extraction evaluation.

        Reports vocabulary size, mean skills per posting, and the
        fraction of posting pairs with zero skill overlap. A
        zero-overlap fraction above 50% signals that the lexicon-
        matching paradigm may lack sufficient feature density for
        meaningful clustering discrimination.
        """
        stats = vectorizer.statistics
        logger.info(f"Vocabulary: {stats.vocabulary_size} features")
        logger.info(
            f"Mean skills/posting: "
            f"{stats.mean_skills_per_posting:.1f}"
        )

        B            = vectorizer.binary_matrix
        intersection = B @ B.T
        n            = intersection.shape[0]
        total_pairs  = n * (n - 1) // 2
        # nnz includes diagonal (one per posting); off-diagonal
        # entries are symmetric, so halve after subtracting diagonal
        nonzero_pairs = (intersection.nnz - n) // 2
        zero_frac     = (
            1 - nonzero_pairs / total_pairs if total_pairs else 0.0
        )

        logger.info(f"Zero-overlap fraction: {zero_frac:.1%}")

        if zero_frac > 0.5:
            logger.warning(
                "Zero-overlap fraction exceeds 50%; consider "
                "sentence-embedding alternative for the geometry "
                "track"
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

    # -----------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------

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

        d            = self.config.lexicon_dir
        occupations  = load_onet(d / "onet.json")
        registry     = LexiconRegistry(
            certifications   = load_certifications(d / "certifications.json"),
            occupations      = occupations,
            osha_terms       = load_osha(d / "osha.json"),
            supplement_terms = load_supplement(d / "supplement.json")
        )

        self.extractor   = SkillExtractor(registry)
        corpus           = self._load_corpus()
        apprenticeships  = self._load_apprenticeships()
        programs         = load_programs(self.config.reference_dir)
        occupation_index = OccupationIndex(occupations)

        with self._timed("Extraction"):
            self.extracted_skills = self.extractor.extract(corpus)

        with self._timed("Vectorization"):
            vectorizer = SkillVectorizer(self.extracted_skills)

        with self._timed("PCA reduction"):
            reducer = PcaReducer(
                document_ids       = vectorizer.document_ids,
                feature_names      = vectorizer.feature_names,
                max_components     = self.config.max_components,
                random_seed        = self.config.random_seed,
                tfidf_matrix       = vectorizer.tfidf_matrix,
                variance_threshold = self.config.variance_threshold
            )

        with self._timed("Clustering"):
            self.clusterer = HierarchicalClusterer(
                coordinates  = reducer.coordinates,
                document_ids = reducer.document_ids
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
            document_ids     = reducer.document_ids,
            extracted_skills = self.extracted_skills,
            occupation_index = occupation_index
        )

        self.geometry_pipeline = compose_geometry(reducer, vectorizer)
        self.enrichment = EnrichmentContext(
            apprenticeships = apprenticeships,
            programs        = programs
        )

        with self._timed("Profile enrichment"):
            self.profiles = build_profiles(
                cluster_labels   = self.cluster_labels,
                clusterer        = self.clusterer,
                enrichment       = self.enrichment,
                extracted_skills = self.extracted_skills,
                occupation_index = occupation_index,
                sector_labels    = sector_labels
            )

        deduped_apps  = deduplicate_apprenticeships(self.profiles)
        deduped_progs = deduplicate_programs(self.profiles)

        with self._timed("Pathway graph"):
            self.graph = CareerPathwayGraph(
                apprenticeships = deduped_apps,
                max_density     = self.config.max_graph_density,
                network         = network,
                profiles        = self.profiles,
                programs        = deduped_progs
            )

        self.router = CareerRouter(
            enrichment = self.enrichment,
            graph      = self.graph.graph,
            profiles   = self.profiles
        )

        self.matcher = self._build_matcher(
            ppmi_df = network.association_dataframe("ppmi")
        )

        self._log_density(vectorizer)

        return self

    @classmethod
    def load(
        cls,
        config       : PipelineConfig,
        artifact_dir : Path | None = None
    ) -> "Pipeline":
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
            A fitted `Pipeline` ready for `match()` calls.
        """
        d = artifact_dir or config.output_dir / "pipeline"

        pipe                   = cls(config)
        pipe.extractor         = SkillExtractor(pipe._load_registry())
        pipe.geometry_pipeline = joblib.load(d / "geometry.joblib")
        pipe.extracted_skills  = loads(
            (d / "extracted_skills.json").read_text()
        )

        check_is_fitted(pipe.geometry_pipeline)

        pipe.clusterer      = joblib.load(d / "clusterer.joblib")
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

        pipe.graph = CareerPathwayGraph.from_serialized(
            json_path = d / "career_graph.json",
            profiles  = pipe.profiles
        )

        pipe.enrichment = EnrichmentContext(
            apprenticeships = pipe.graph.apprenticeships,
            programs        = pipe.graph.programs
        )

        pipe.router = CareerRouter(
            enrichment = pipe.enrichment,
            graph      = pipe.graph.graph,
            profiles   = pipe.profiles
        )

        pipe.matcher = pipe._build_matcher(
            ppmi_df = joblib.load(d / "ppmi.joblib")
        )

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

        Writes sklearn objects via `joblib`, the career graph in dual
        formats (JSON for pipeline reload, GraphML for interop), and
        a manifest tracking provenance via
        `geometry_pipeline.get_params(deep=True)`.

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

        joblib.dump(self.clusterer, d / "clusterer.joblib")
        joblib.dump(self.geometry_pipeline, d / "geometry.joblib")
        joblib.dump(self.matcher.ppmi_df, d / "ppmi.joblib")

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

        self.graph.export(d)

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
