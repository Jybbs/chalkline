"""
End-to-end pipeline orchestrator for Chalkline career mapping.

Coordinates the geometry track (extraction, vectorization, PCA,
clustering) and co-occurrence track (PMI network, Louvain communities)
into a fitted `Chalkline` instance constructed via `fit()`. Hamilton
resolves the DAG from function parameter names, caches every node
result, and serves from cache on subsequent calls with unchanged
code and config.
"""

import pandas as pd

from dataclasses                   import dataclass, fields
from hamilton                      import driver
from hamilton.plugins.h_threadpool import FutureAdapter
from hamilton.plugins.h_tqdm       import ProgressBar
from logging                       import getLogger
from sklearn.pipeline              import Pipeline
from typing                        import Self

from chalkline                         import SkillMap
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
    communities) into a fitted landscape. Call `fit()` to compute
    from scratch or restore from cache, then access fitted artifacts
    directly or call `match()` for single-resume inference.
    """

    cluster_labels    : list[ClusterLabel]
    clusterer         : HierarchicalClusterer
    config            : PipelineConfig
    extracted_skills  : SkillMap
    extractor         : SkillExtractor
    geometry_pipeline : Pipeline
    graph             : CareerPathwayGraph
    manifest          : PipelineManifest
    matcher           : ResumeMatcher
    ppmi_df           : pd.DataFrame
    profiles          : dict[int, ClusterProfile]
    router            : CareerRouter
    trades            : TradeIndex

    def match(
        self,
        resume_text : str,
        top_k       : int | None = None
    ) -> MatchResult:
        """
        Project a resume into the fitted career landscape and
        return a full match result with gap analysis.

        Extracts skills from the resume text via the fitted
        `SkillExtractor`, then delegates projection, cluster
        assignment, and gap ranking to the `ResumeMatcher`.

        Args:
            resume_text : Raw resume text (post-PDF extraction).
            top_k       : Override for the default `top_k_gaps`.

        Returns:
            `MatchResult` with cluster, gaps, and sector.
        """
        return self.matcher.match(
            skills = self.extractor.extract({"resume": resume_text})["resume"],
            top_k  = top_k
        )


    @staticmethod
    def fit(config: PipelineConfig) -> Self:
        """
        Execute all pipeline steps and return a fitted pipeline.

        Builds a Hamilton DAG from the node functions in `steps`
        with disk caching and parallel thread execution enabled.
        First call computes every node and persists results.
        Subsequent calls with unchanged code and config serve
        from cache, making `fit()` idempotent. Independent
        branches (geometry and co-occurrence tracks) run
        concurrently via `FutureAdapter`.

        Args:
            config: End-to-end pipeline configuration.

        Returns:
            A fully fitted `Chalkline` instance.
        """
        return Chalkline(**(
            driver.Builder()
            .with_modules(steps)
            .with_adapters(FutureAdapter(), ProgressBar("chalkline"))
            .with_cache(path=str(config.pipeline_dir / ".cache"))
            .build()
            .execute(
                [f.name for f in fields(Chalkline)],
                inputs={"pipeline_config": config}
            )
        ))
