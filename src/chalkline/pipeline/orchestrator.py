"""
End-to-end pipeline orchestrator for Chalkline career mapping.

Coordinates the geometry track (extraction, vectorization, PCA,
clustering) and co-occurrence track (PMI network, Louvain communities)
into a fitted `Chalkline` instance constructed via `fit()` or `load()`.
All fitted artifacts are guaranteed present by construction, and
`match()` projects resumes without re-fitting.
"""

import pandas as pd

from dataclasses              import dataclass, fields
from datetime                 import datetime, timezone
from hamilton                 import driver
from joblib                   import dump, load
from logging                  import getLogger
from pathlib                  import Path
from sklearn.pipeline         import Pipeline
from sklearn.utils.validation import check_is_fitted
from typing                   import Self

from chalkline.clustering.hierarchical import HierarchicalClusterer
from chalkline.clustering.schemas      import ClusterLabel
from chalkline.extraction.skills       import SkillExtractor
from chalkline.matching.matcher        import ResumeMatcher
from chalkline.matching.schemas        import MatchResult
from chalkline.pathways.graph          import CareerPathwayGraph
from chalkline.pathways.routing        import CareerRouter
from chalkline.pipeline                import steps
from chalkline.pipeline.schemas        import ClusterProfile, PipelineConfig
from chalkline.pipeline.schemas        import PipelineManifest
from chalkline.pipeline.trades         import TradeIndex

logger = getLogger(__name__)


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
    matcher           : ResumeMatcher
    ppmi_df           : pd.DataFrame
    profiles          : dict[int, ClusterProfile]
    router            : CareerRouter
    trades            : TradeIndex

    @classmethod
    def fit(cls, config: PipelineConfig) -> Self:
        """
        Execute all pipeline steps and return a fitted pipeline.

        Builds a Hamilton DAG from the node functions in `steps`
        and executes the terminal nodes needed for a fully fitted
        `Chalkline` instance. Hamilton resolves the dependency
        graph from function parameter names, forking the geometry
        and co-occurrence tracks after vectorization and merging
        them at the pathway graph.

        Args:
            config: End-to-end pipeline configuration.

        Returns:
            A fully fitted `Chalkline` instance.
        """
        result = driver.Builder().with_modules(steps).build().execute(
            [f.name for f in fields(cls)],
            inputs={"pipeline_config": config}
        )
        return cls(**result)

    @classmethod
    def load(
        cls,
        config       : PipelineConfig,
        artifact_dir : Path | None = None
    ) -> Self:
        """
        Load a fitted pipeline from serialized artifacts.

        Deserializes the artifacts file, passes its contents as
        Hamilton overrides, and lets the DAG rebuild the derived
        objects (extractor, matcher, router, trades) from config.
        The `SkillExtractor` is rebuilt because the AhoCorasick
        automaton is not serializable.

        Args:
            config       : Pipeline configuration for lexicon and
                           reference data paths.
            artifact_dir : Directory containing serialized artifacts.
                           Defaults to `config.pipeline_dir`.

        Returns:
            A fitted `Chalkline` ready for `match()` calls.
        """
        d = artifact_dir or config.pipeline_dir
        artifacts = load(d / "artifacts.joblib")
        check_is_fitted(artifacts["geometry_pipeline"])

        result = driver.Builder().with_modules(steps).build().execute(
            [f.name for f in fields(cls)],
            inputs={"pipeline_config": config},
            overrides=artifacts
        )
        return cls(**result)

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

        Serializes all fields except those Hamilton can rebuild
        from config (extractor, matcher, router, trades) into a
        single joblib file. A separate JSON manifest tracks
        provenance via `geometry_pipeline.get_params(deep=True)`.

        Args:
            artifact_dir : Target directory for serialized artifacts.
                           Defaults to `config.pipeline_dir`.
        """
        d = artifact_dir or self.config.pipeline_dir
        d.mkdir(parents=True, exist_ok=True)

        dump(
            {
                f.name: getattr(self, f.name) for f in fields(self)
                if f.name not in
                ("config", "extractor", "matcher", "router", "trades")
            },
            d / "artifacts.joblib"
        )

        (d / "manifest.json").write_text(
            PipelineManifest(
                corpus_size     = len(self.extracted_skills),
                geometry_params = self.geometry_pipeline.get_params(deep=True),
                posting_ids     = sorted(self.extracted_skills),
                timestamp       = datetime.now(timezone.utc).isoformat()
            ).model_dump_json(indent=2)
        )

        logger.info(f"Pipeline artifacts saved to {d}")
