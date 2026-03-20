"""
End-to-end pipeline orchestrator for Chalkline career mapping.

Coordinates the embedding pipeline (sentence-transformer encoding, SVD
reduction, Ward HAC clustering, stepwise k-NN career graph with per-edge
credential enrichment) into a fitted `Chalkline` instance constructed via
`fit()`. Hamilton resolves the DAG from function parameter names, caches
every node result, and serves from cache on subsequent calls with unchanged
code and config.
"""

from dataclasses             import dataclass, fields
from hamilton                import driver
from hamilton.plugins.h_tqdm import ProgressBar
from sentence_transformers   import SentenceTransformer

from chalkline.matching.matcher import ResumeMatcher
from chalkline.matching.schemas import MatchResult
from chalkline.pipeline         import steps
from chalkline.pipeline.graph   import CareerPathwayGraph
from chalkline.pipeline.schemas import ClusterProfile, Encoder
from chalkline.pipeline.schemas import PipelineConfig, PipelineManifest
from chalkline.pipeline.trades  import TradeIndex


@dataclass(kw_only=True)
class Chalkline:
    """
    Fitted career mapping pipeline.

    Coordinates embedding, clustering, career graph construction, and
    credential enrichment into a fitted landscape. Call `fit()` to compute
    from scratch or restore from cache, then call `match()` for
    single-resume inference with neighborhood exploration.
    """

    config   : PipelineConfig
    graph    : CareerPathwayGraph
    manifest : PipelineManifest
    matcher  : ResumeMatcher
    profiles : dict[int, ClusterProfile]
    trades   : TradeIndex

    def __repr__(self) -> str:
        """
        Compact one-line summary of fitted pipeline dimensions.
        """
        return (
            f"Chalkline("
            f"{len(self.profiles):,} clusters, "
            f"{self.graph.graph.number_of_edges()} edges, "
            f"{self.manifest.corpus_size:,} postings)"
        )

    def __str__(self) -> str:
        """
        Multi-line summary of fitted pipeline metrics with ANSI bold
        formatting and Unicode bullets.
        """
        bold, reset = "\033[1m", "\033[0m"
        return "\n".join([
            f"{bold}Chalkline{reset}",
            f"  \u00b7 {bold}{self.manifest.corpus_size:,}{reset}"
            f" postings encoded"
            f" with {self.manifest.embedding_model}",
            f"  \u00b7 {bold}{len(self.profiles):,}{reset}"
            f" clusters at d={self.manifest.component_count}",
            f"  \u00b7 {bold}"
            f"{self.graph.graph.number_of_edges()}{reset}"
            f" graph edges with credential enrichment"
        ])

    @staticmethod
    def fit(config: PipelineConfig) -> Chalkline:
        """
        Execute all pipeline steps and return a fitted pipeline.

        Loads the sentence transformer model outside the Hamilton DAG to
        avoid caching the ~400MB model weights. The model is passed as an
        input alongside the config, and all encoding node outputs (numpy
        arrays) cache normally via Hamilton's disk persistence.

        Args:
            config: Hyperparameters, directory paths, and embedding model name.

        Returns:
            A fully fitted `Chalkline` instance.
        """
        results = (driver.Builder()
            .with_modules(steps)
            .with_adapters(ProgressBar("chalkline"))
            .with_cache(str(config.hamilton_cache_dir))
            .build()
            .execute(
                final_vars = [f.name for f in fields(Chalkline)],
                inputs     = {
                    "config" : config,
                    "model"  : Encoder(SentenceTransformer(config.embedding_model))
                }
            )
        )
        return Chalkline(**results)

    def match(self, resume_text: str) -> MatchResult:
        """
        Project a resume into the fitted career landscape and return a full
        match result with gap analysis and neighborhood view.

        Args:
            resume_text: Raw resume text (post-PDF extraction).

        Returns:
            `MatchResult` with cluster, gaps, and neighborhood.
        """
        return self.matcher.match(resume_text)
