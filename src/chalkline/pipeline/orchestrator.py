"""
End-to-end pipeline orchestrator for Chalkline career mapping.

Coordinates the embedding pipeline (ONNX sentence encoding, SVD reduction,
Ward HAC clustering, stepwise k-NN career graph with per-edge credential
enrichment) into a fitted `Chalkline` instance constructed via `fit()`.
Hamilton resolves the DAG from function parameter names, caches every node
result, and serves from cache on subsequent calls with unchanged code and
config.
"""

from dataclasses import dataclass, fields

from chalkline.matching.matcher import ResumeMatcher
from chalkline.matching.schemas import MatchResult
from chalkline.pathways.graph   import CareerPathwayGraph
from chalkline.pathways.schemas import Clusters
from chalkline.pipeline.encoder import SentenceEncoder
from chalkline.pipeline.schemas import PipelineConfig


@dataclass(kw_only=True)
class Chalkline:
    """
    Fitted career mapping pipeline.

    Coordinates embedding, clustering, career graph construction, and
    credential enrichment into a fitted landscape. Call `fit()` to compute
    from scratch or restore from cache, then call `match()` for
    single-resume inference with neighborhood exploration.
    """

    clusters : Clusters
    config   : PipelineConfig
    graph    : CareerPathwayGraph
    matcher  : ResumeMatcher

    def __repr__(self) -> str:
        """
        Compact one-line summary of fitted pipeline dimensions.
        """
        return (
            f"Chalkline({len(self.clusters):,} clusters, "
            f"{self.graph.edge_count} edges, {self.corpus_size:,} postings)"
        )

    @property
    def corpus_size(self) -> int:
        """
        Total postings across all clusters.
        """
        return sum(c.size for c in self.clusters.values())

    @property
    def sector_count(self) -> int:
        """
        Number of distinct sectors across clusters.
        """
        return len({c.sector for c in self.clusters.values()})

    @staticmethod
    def fit(config: PipelineConfig, log_level: str = "INFO") -> Chalkline:
        """
        Execute all pipeline steps and return a fitted pipeline.

        Loads the ONNX encoder outside the Hamilton DAG to avoid caching the
        ~430MB model file. The encoder is passed as an input alongside the
        config, and all encoding node outputs (numpy arrays) cache normally
        via Hamilton's disk persistence.

        Args:
            config    : Hyperparameters, directory paths, and embedding model name.
            log_level : Minimum loguru level during execution. Pass `"DEBUG"` for
                        verbose output.

        Returns:
            A fully fitted `Chalkline` instance.
        """
        from hamilton.driver import Builder

        from chalkline.pathways.loaders  import LexiconLoader
        from chalkline.pipeline          import steps
        from chalkline.pipeline.progress import MarimoDisplay, RichDisplay

        display = MarimoDisplay.detect() or RichDisplay(level=log_level)
        encoder = SentenceEncoder(
            name       = config.embedding_model,
            tqdm_class = display.make_download_tqdm()
        )
        display.encoder = encoder

        results = (Builder()
            .with_modules(steps)
            .with_adapters(display)
            .with_cache(
                config.hamilton_cache_dir,
                ignore = ["encoder", "matcher"]
            )
            .build()
            .execute(
                final_vars = [f.name for f in fields(Chalkline)],
                inputs     = {
                    "config"   : config,
                    "encoder"  : encoder,
                    "lexicons" : LexiconLoader(config.lexicon_dir)
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
