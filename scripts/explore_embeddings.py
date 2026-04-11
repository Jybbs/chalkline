"""
Reproduce the key decision points from the CL-13 embedding migration.

Walks through each architectural choice documented in
docs/comments/CL-13.md, printing the metrics that informed each decision.
Sections follow the comment's narrative arc from dimensionality selection
through credential enrichment and resume validation. All encoding is done
once at initialization and reused across sections.

    uv run python scripts/embedding_exploration.py
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import networkx as nx
import numpy    as np

from collections              import Counter
from json                     import loads
from pathlib                  import Path
from sklearn.cluster          import AgglomerativeClustering
from sklearn.decomposition    import TruncatedSVD
from sklearn.metrics          import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing    import normalize

from chalkline.collection.storage import CorpusStorage
from chalkline.pathways.loaders   import LexiconLoader
from chalkline.pipeline.encoder   import SentenceEncoder
from chalkline.pipeline.schemas   import PipelineConfig


class EmbeddingExploration:
    """
    Reproduce CL-13 embedding evaluation decision points.

    Loads the AGC Maine corpus and credential catalog, encodes everything
    with the pipeline's configured sentence model, then walks through
    each architectural choice from the evaluation, printing the metrics
    that informed each decision.
    """

    def __init__(self, config: PipelineConfig):
        """
        Load corpus, lexicons, and encode all text.

        Args:
            config: Pipeline paths for postings and lexicons.
        """
        raw = list(CorpusStorage(config.postings_dir).load())
        self.descriptions = {p.id: p.description for p in raw}
        self.doc_ids      = sorted(self.descriptions)
        self.titles       = {p.id: p.title for p in raw}

        loader           = LexiconLoader(config.lexicon_dir)
        self.occupations = loader.occupations

        self.credentials = loads(
            (config.lexicon_dir / "credentials.json").read_text()
        )

        self.model = SentenceEncoder(name=config.embedding_model)
        self.texts = [self.descriptions[d] for d in self.doc_ids]
        print(f"Encoding {len(self.texts)} postings...")
        self.raw_emb = self.model.encode(self.texts, unit=False)
        self.normed  = normalize(self.raw_emb)

        self._encode_occupations()
        self._encode_credentials()
        self._prepare_base()

        print(
            f"{len(self.doc_ids)} postings, "
            f"{len(self.cred_labels)} credentials ready\n")

    def _cluster(self, coords, k):
        """
        Run HAC Ward clustering at the given k.
        """
        return AgglomerativeClustering(
            linkage="ward", n_clusters=k).fit_predict(coords)

    def _cluster_embs_768(self, k, labels):
        """
        Mean posting embedding per cluster in the full 768-dim space,
        L2-normalized for cosine similarity.
        """
        return np.stack([
            normalize(
                self.raw_emb[labels == c].mean(axis=0, keepdims=True)
            )[0] for c in range(k)
        ])

    def _encode_credentials(self):
        """
        Encode the unified credential catalog via its pre-computed
        `embedding_text` field.
        """
        self.cred_labels = [c["label"]          for c in self.credentials]
        self.cred_types  = [c["kind"]           for c in self.credentials]
        cred_texts       = [c["embedding_text"] for c in self.credentials]

        print(f"Encoding {len(cred_texts)} credentials...")
        self.cred_embs = normalize(
            self.model.encode(cred_texts, unit=False)
        )

    def _encode_occupations(self):
        """
        Encode O*NET occupations using uncapped Task+DWA text.
        """
        occ_texts = [
            f"{occ.title}: {', '.join(
                s.name for s in occ.skills
                if s.type.value in ('task', 'dwa')
            )}" for occ in self.occupations
        ]

        print(f"Encoding {len(occ_texts)} occupations...")
        self.occ_embs = normalize(
            self.model.encode(occ_texts, unit=False)
        )

    def _hub_fraction(self, labels):
        """
        Fraction of postings in the largest cluster.
        """
        return Counter(labels).most_common(1)[0][1] / len(labels)

    def _job_zone_map(self, cluster_embs, k):
        """
        Assign Job Zones to clusters via top-3 median cosine against O*NET
        Task+DWA embeddings.
        """
        csims = cosine_similarity(cluster_embs, self.occ_embs)
        return {
            c: int(np.median([
                self.occupations[i].job_zone
                for i in np.argsort(csims[c])[-3:]
            ])) for c in range(k)
        }

    def _prepare_base(self):
        """
        Precompute the d=10/k=20 base state shared by SOC assignment,
        edge selection, credential enrichment, and resume validation.
        """
        self.coords_10, self.svd_10 = self._reduce(10)
        self.labels_20  = self._cluster(self.coords_10, 20)
        self.labels_5   = self._cluster(self.coords_10, 5)
        self.cl_embs    = self._cluster_embs_768(20, self.labels_20)
        self.centroids  = np.stack([
            self.coords_10[self.labels_20 == c].mean(axis=0)
            for c in range(20)
        ])
        self.cluster_sim = cosine_similarity(self.centroids)
        np.fill_diagonal(self.cluster_sim, 0)
        self.job_zone_map = self._job_zone_map(self.cl_embs, 20)
        self.cred_to_cl   = cosine_similarity(self.cred_embs, self.cl_embs)

        job_zone_levels    = sorted(set(self.job_zone_map.values()))
        self.next_job_zone = dict(zip(job_zone_levels, job_zone_levels[1:]))

    def _reduce(self, d):
        """
        L2 normalize then TruncatedSVD to d components. Returns the reduced
        coordinates and the fitted SVD.
        """
        svd    = TruncatedSVD(n_components=d, random_state=42)
        coords = svd.fit_transform(self.normed)
        return coords, svd

    def _sector_ints(self, occ_embs):
        """
        Assign each posting to its nearest occupation's sector and return
        integer-encoded sector labels. The occupation embeddings determine
        which encoding variant is used.
        """
        sims    = cosine_similarity(self.normed, occ_embs)
        nearest = np.argmax(sims, axis=1)
        sectors = [self.occupations[i].sector for i in nearest]
        mapping = {s: i for i, s in enumerate(sorted(set(sectors)))}
        return np.array([mapping[s] for s in sectors])

    def _top_title(self, c, labels):
        """
        Most common posting title in cluster c.
        """
        return Counter(
            self.titles[self.doc_ids[i]]
            for i in np.where(labels == c)[0]
        ).most_common(1)[0][0]

    def credential_enrichment(self):
        """
        Show the weight scale mismatch between cluster edges and credential
        edges, then sweep the per-edge metadata filter parameters across
        dest_pct and src_floor.
        """
        print("CREDENTIAL ENRICHMENT")
        print("=" * 60)

        cluster_sim_768 = cosine_similarity(self.cl_embs)
        np.fill_diagonal(cluster_sim_768, 0)
        cc_vals  = cluster_sim_768[cluster_sim_768 > 0]
        cred_all = self.cred_to_cl.flatten()
        print(f"\n  Weight scale mismatch (768-dim)")
        print(
            f"    Cluster-to-cluster:    "
            f"{cc_vals.min():.3f} – {cc_vals.max():.3f}")
        print(
            f"    Credential-to-cluster: "
            f"{cred_all.min():.3f} – {cred_all.max():.3f}")

        edges = [
            (c, j) for c in range(20) if self.job_zone_map[c] in self.next_job_zone
            for j in range(20)
            if self.job_zone_map[j] == self.next_job_zone[self.job_zone_map[c]]
            and self.cluster_sim[c, j] > 0.5
        ]
        src_idxs = np.array([s for s, _ in edges])
        tgt_idxs = np.array([t for _, t in edges])

        print(f"\n  Parameter sweep on {len(edges)} advancement edges")
        print(
            f"  {'dest_pct':>9} {'src_floor':>11} "
            f"{'mean':>6} {'median':>7} {'zero':>5}")
        print("  " + "-" * 45)

        for dest_pct in [3, 5, 8, 10, 15]:
            thresholds = np.percentile(
                self.cred_to_cl, 100 - dest_pct, axis=0)
            for floor_pct in [50, 60, 70, 75, 80]:
                src_floor = np.percentile(cred_all, floor_pct)
                tgt_pass  = self.cred_to_cl[:, tgt_idxs] >= thresholds[tgt_idxs]
                src_pass  = self.cred_to_cl[:, src_idxs] >= src_floor
                counts    = (tgt_pass & src_pass).sum(axis=0)
                zeros     = int(np.sum(counts == 0))
                print(
                    f"  top-{dest_pct:<4}  "
                    f"p{floor_pct}({src_floor:.3f})  "
                    f"{counts.mean():>5.1f} "
                    f"{int(np.median(counts)):>7} "
                    f"{zeros:>5}")
        print()

    def dimensionality_sweep(self):
        """
        Sweep d from 5 to 50, showing variance, silhouette, hub fraction,
        and ARI at each component count.
        """
        print("DIMENSIONALITY SWEEP")
        print("=" * 60)

        sector_ints = self._sector_ints(self.occ_embs)

        print(
            f"  {'d':>3}  {'var':>7}  {'sil(k=20)':>10}  "
            f"{'hub(k=20)':>10}  {'ARI(k=5)':>9}")
        print("  " + "-" * 48)

        for d in [5, 7, 10, 12, 15, 20, 50]:
            coords, svd = self._reduce(d)
            var       = svd.explained_variance_ratio_.sum()
            labels_20 = self._cluster(coords, 20)
            labels_5  = self._cluster(coords, 5)
            sil       = silhouette_score(coords, labels_20)
            hub       = self._hub_fraction(labels_20)
            ari       = adjusted_rand_score(sector_ints, labels_5)
            print(
                f"  {d:>3}  {var:>7.1%}  {sil:>10.3f}  "
                f"{hub:>10.1%}  {ari:>9.3f}")
        print()

    def dk_interaction(self):
        """
        Grid of silhouette and hub fraction across d and k, confirming d=10
        dominates.
        """
        print("d-k INTERACTION GRID (silhouette / hub)")
        print("=" * 60)

        ks     = [10, 15, 20, 25, 30]
        header = "  ".join(f"{'k=' + str(k):>10}" for k in ks)
        print(f"  {'d':>3}  {header}")
        print("  " + "-" * 58)

        for d in [10, 15, 20]:
            coords, _ = self._reduce(d)
            cells = []
            for k in ks:
                lab = self._cluster(coords, k)
                sil = silhouette_score(coords, lab)
                hub = self._hub_fraction(lab)
                cells.append(f"{sil:.3f}/{hub:.0%}")
            row = "  ".join(f"{c:>10}" for c in cells)
            print(f"  {d:>3}  {row}")
        print()

    def edge_selection(self):
        """
        Compare percentile thresholding, mutual k-NN, and stepwise k-NN for
        backbone edge selection at d=10, k=20.
        """
        print("EDGE SELECTION (d=10, k=20)")
        print("=" * 60)

        vals = self.cluster_sim[self.cluster_sim > 0]
        for pct in [25, 50]:
            threshold = np.percentile(vals, pct)
            n_edges   = int(np.sum(self.cluster_sim >= threshold))
            density   = n_edges / (20 * 19)
            print(
                f"  Percentile p{pct}: {n_edges} directed edges, "
                f"density={density:.2f}")

        rank_order = np.argsort(self.cluster_sim, axis=1)
        for knn in [2, 3, 5]:
            G = nx.Graph()
            G.add_nodes_from(range(20))
            for c in range(20):
                for j in rank_order[c, -knn:]:
                    if c in rank_order[j, -knn:]:
                        G.add_edge(c, j)
            comps = nx.number_connected_components(G)
            print(
                f"  Mutual k-NN (k={knn}): {G.number_of_edges()} edges, "
                f"{comps} components")

        G = nx.DiGraph()
        G.add_nodes_from(range(20))

        for c in range(20):
            my_job_zone = self.job_zone_map[c]
            same        = [
                j for j in range(20)
                if self.job_zone_map[j] == my_job_zone and j > c
            ]
            same.sort(
                key=lambda j: self.cluster_sim[c, j], reverse=True)
            for j in same[:2]:
                G.add_edge(c, j)
                G.add_edge(j, c)

            if my_job_zone in self.next_job_zone:
                up = [
                    j for j in range(20)
                    if self.job_zone_map[j] == self.next_job_zone[my_job_zone]
                ]
                up.sort(
                    key=lambda j: self.cluster_sim[c, j], reverse=True)
                for j in up[:2]:
                    G.add_edge(c, j)

        comps         = nx.number_weakly_connected_components(G)
        job_zone_dist = dict(sorted(Counter(self.job_zone_map.values()).items()))
        print(
            f"  Stepwise k-NN: {G.number_of_edges()} edges, "
            f"{comps} component(s)")
        print(f"  Job Zone distribution: {job_zone_dist}")
        print()

    def resume_validation(self):
        """
        Encode the Walt Amper test resume and project into the clustering
        space. Shows cluster assignment, distance ranking, and demonstrated
        vs gap skill split.
        """
        print("RESUME VALIDATION")
        print("=" * 60)

        resume_path = Path("tests/fixtures/parsing/sample.pdf")
        if not resume_path.exists():
            print("  (sample.pdf not found, skipping)\n")
            return

        try:
            import pdfplumber
        except ImportError:
            print("  (pdfplumber not installed, skipping)\n")
            return

        with pdfplumber.open(resume_path) as pdf:
            text = "\n".join(
                page.extract_text() or "" for page in pdf.pages)

        if not text.strip():
            print("  (empty PDF text, skipping)\n")
            return

        resume_emb    = self.model.encode([text], unit=False)
        resume_normed = normalize(resume_emb)
        resume_coords = self.svd_10.transform(resume_normed)[0]

        dists  = np.linalg.norm(
            self.centroids - resume_coords, axis=1)
        ranked = np.argsort(dists)

        print(f"\n  Top 5 nearest clusters:")
        for rank, c in enumerate(ranked[:5]):
            title = self._top_title(c, self.labels_20)
            n     = int((self.labels_20 == c).sum())
            print(
                f"    {rank + 1}. {title[:35]:<37} "
                f"dist={dists[c]:.3f}  n={n}")

        nearest       = ranked[0]
        cluster_dists = np.linalg.norm(
            self.coords_10[self.labels_20 == nearest]
            - self.centroids[nearest], axis=1)
        median_dist = float(np.median(cluster_dists))
        tightness   = (
            "tighter" if dists[nearest] < median_dist else "looser")
        print(f"\n  Cluster median dist: {median_dist:.3f}")
        print(
            f"  Resume dist:         {dists[nearest]:.3f} "
            f"({tightness} than median)")

        occ_csim = cosine_similarity(
            self.cl_embs[nearest:nearest + 1], self.occ_embs)
        best_occ = self.occupations[np.argmax(occ_csim)]
        tasks    = [
            s for s in best_occ.skills
            if s.type.value in ("task", "dwa")]

        if tasks:
            task_embs = normalize(self.model.encode(
                [t.name for t in tasks], unit=False))
            task_sims  = cosine_similarity(resume_normed, task_embs)[0]
            median_sim = float(np.median(task_sims))
            demonstrated = [tasks[i].name
                            for i in np.argsort(task_sims)[::-1]
                            if task_sims[i] >= median_sim]
            gaps = [tasks[i].name for i in np.argsort(task_sims)
                    if task_sims[i] < median_sim]

            print(
                f"\n  {best_occ.title} ({len(tasks)} Tasks+DWAs, "
                f"threshold={median_sim:.3f})")
            print(f"\n  Demonstrated ({len(demonstrated)}):")
            for name in demonstrated:
                print(f"    + {name[:65]}")
            print(f"\n  Gaps ({len(gaps)}):")
            for name in gaps:
                print(f"    - {name[:65]}")
        print()

    def run_all(self):
        """
        Run all evaluation sections in narrative order.
        """
        self.dimensionality_sweep()
        self.dk_interaction()
        self.soc_assignment()
        self.edge_selection()
        self.credential_enrichment()
        self.resume_validation()

        print("=" * 60)
        print("EXPLORATION COMPLETE")
        print("=" * 60)

    def soc_assignment(self):
        """
        Compare skills[:20] vs Task+DWA encoding for occupation matching and
        single vs top-3 median JZ assignment.
        """
        print("SOC ASSIGNMENT (d=10)")
        print("=" * 60)

        occ_embs_20 = normalize(self.model.encode([
            f"{occ.title}: "
            f"{', '.join(s.name for s in occ.skills[:20])}"
            for occ in self.occupations
        ], unit=False))

        print(f"\n  ARI by encoding variant:")
        for label, embs in [
            ("skills[:20]", occ_embs_20),
            ("Task+DWA", self.occ_embs)
        ]:
            si     = self._sector_ints(embs)
            ari_5  = adjusted_rand_score(si, self.labels_5)
            ari_20 = adjusted_rand_score(si, self.labels_20)
            print(
                f"    {label:>12}  ARI(k=5)={ari_5:.3f}  "
                f"ARI(k=20)={ari_20:.3f}")

        print(f"\n  Job Zone distribution by encoding and method:")
        for label, occ_embs in [
            ("Task+DWA", self.occ_embs),
            ("skills[:20]", occ_embs_20)
        ]:
            csims = cosine_similarity(self.cl_embs, occ_embs)
            job_zone_single = {
                c: self.occupations[np.argmax(csims[c])].job_zone
                for c in range(20)
            }
            job_zone_top3 = {
                c: int(np.median([
                    self.occupations[i].job_zone
                    for i in np.argsort(csims[c])[-3:]
                ])) for c in range(20)
            }
            sd = dict(sorted(Counter(job_zone_single.values()).items()))
            td = dict(sorted(Counter(job_zone_top3.values()).items()))
            print(f"    {label:>12}  single={sd}  top3={td}")
        print()


if __name__ == "__main__":

    EmbeddingExploration(
        config=PipelineConfig(
            lexicon_dir  = Path("data/lexicons"),
            postings_dir = Path("data/postings")
        )
    ).run_all()
