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
from pathlib     import Path

from chalkline.matching.matcher  import ResumeMatcher
from chalkline.matching.schemas  import MatchResult
from chalkline.pathways.clusters import Clusters
from chalkline.pathways.graph    import CareerPathwayGraph
from chalkline.pipeline.encoder  import SentenceEncoder
from chalkline.pipeline.schemas  import PipelineConfig


@dataclass(kw_only=True)
class Chalkline:
    """
    Fitted career mapping pipeline.

    Coordinates embedding, clustering, career graph construction, and
    credential enrichment into a fitted landscape. Call `fit()` to compute
    from scratch or restore from cache, then call `match()` for
    single-resume inference with reach exploration.
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
    def embed_dim(self) -> int:
        """
        Dimensionality of the sentence embeddings from the encoder.
        """
        return self.matcher.encoder.dimension

    @property
    def substitutions(self) -> dict[str, str]:
        """
        Template variables for display-layer content rendering.
        """
        return {
            "embed_dim"     : str(self.embed_dim),
            "model_name"    : self.matcher.encoder.name.split("/")[-1],
            "n_clusters"    : str(len(self.clusters)),
            "n_occupations" : str(len({c.soc_title for c in self.clusters.values()})),
            "n_postings"    : f"{self.corpus_size:,}",
            "n_sectors"     : str(len(self.clusters.sectors)),
            "svd_components": str(self.config.component_count),
        }

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
        display.begin_display()
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

    def match(self, pdf_bytes: bytes, label: str = "resume") -> MatchResult:
        """
        Extract text from a PDF, clean it, and match against the fitted
        career landscape.

        Handles the full chain from raw upload bytes to match result,
        writing to a temporary file for `pdfplumber` extraction.

        Args:
            label     : Display name for debug logging.
            pdf_bytes : Raw PDF file contents.

        Returns:
            `MatchResult` with cluster, gaps, and reach.
        """
        from tempfile import NamedTemporaryFile

        from chalkline.matching.reader import clean_text, extract_pdf

        with NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            text = clean_text(extract_pdf(Path(tmp.name), label=label))

        return self.matcher.match(text)
