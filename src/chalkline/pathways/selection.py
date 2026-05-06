"""
Candidate-set selection algorithms over the pathways domain.

Two stateless dataclasses whose output drives a final selection against a
candidate pool. `SOCScorer` enables SOC assignment per cluster via
ColBERTv2-style late-interaction MaxSim across per-posting and per-task
embeddings. `CredentialSelector` enables credential-stack selection per
route via waste-aware Pareto-knee greedy search, picking a credential
combination that maximizes gap coverage with minimal redundant reach.
"""

import numpy as np
import pandas as pd

from dataclasses           import dataclass, field
from itertools             import accumulate
from kneed                 import KneeLocator
from math                  import ceil
from sklearn.preprocessing import normalize

from chalkline.pathways.schemas import EncodedOccupation, SelectedCredential


@dataclass(kw_only=True)
class CredentialSelector:
    """
    Waste-aware Pareto-knee picker for credential stacks per route.

    Sweeps α across `[0, 5]` in 100 steps and runs vectorized greedy
    `Δgaps − α · Δwaste` at each setting. The unique points trace a Pareto
    frontier on the (gaps_filled, waste) plane, and Kneedle picks the
    knee where added waste stops buying enough coverage. Stack length is
    data-driven because the greedy stops once marginal score ≤ 0,
    yielding a short stack on sparse candidate pools without a tuned cap.
    The coverage floor anchors the knee to qualifying points when
    reachable, falling back to the highest-coverage frontier point
    otherwise.
    """

    coverage_floor: float

    def select_stack(
        self,
        coverage  : dict[str, dict[int, float]],
        gap_set   : frozenset[int],
        max_picks : int
    ) -> list[SelectedCredential]:
        """
        Pareto-knee credential stack over the candidate coverage map.

        For `max_picks == 1`, returns the single credential filling the
        most gaps with ties broken by least waste, bypassing the α sweep
        because a single pick has no stack-shape tradeoff. Otherwise
        sweeps α ∈ `[0, 5]` across 100 steps, takes the upper-left Pareto
        frontier on the (gaps, waste) plane, and runs Kneedle over
        coverage-floor-eligible points, falling back to the
        highest-coverage frontier point when none qualify.

        Args:
            coverage  : Credential label to `{task_index: affinity}` over every
                        cluster task the credential reaches.
            gap_set   : Task indices the resume has not demonstrated.
            max_picks : Hard cap on stack size.
        """
        coverage = {label: scored for label, scored in coverage.items() if scored}
        if not coverage or not gap_set or max_picks <= 0:
            return []

        hits = (
            pd.DataFrame.from_dict(coverage, orient="index")
              .sort_index()
              .sort_index(axis=1)
        )
        cols       = hits.columns.to_numpy()
        labels     = hits.index.tolist()
        reach      = hits.notna().to_numpy()
        gap_mask   = np.isin(cols, list(gap_set))
        cred_reach = reach.sum(axis=1)
        per_gaps   = (reach & gap_mask).sum(axis=1)

        if max_picks == 1:
            if not per_gaps.any():
                return []
            i = np.lexsort((cred_reach - per_gaps, -per_gaps))[0]
            return [SelectedCredential.from_hits(labels[i], reach[i] & gap_mask, cols)]

        points = []
        for alpha in np.linspace(0.0, 5.0, 100):
            weights = np.where(gap_mask, 1.0, -alpha)
            covered = np.zeros(reach.shape[1], dtype=bool)
            picks   = []
            for _ in range(max_picks):
                scores = (reach & ~covered) @ weights
                i      = scores.argmax()
                if scores[i] <= 0:
                    break
                picks.append(i)
                covered |= reach[i]
            gaps_filled = (covered & gap_mask).sum()
            waste       = cred_reach[picks].sum() - gaps_filled
            points.append((gaps_filled, picks, waste))

        points.sort(key=lambda p: (-p[0], p[2]))
        prior_min = accumulate((p[2] for p in points), min, initial=float("inf"))
        frontier  = [p for p, m in zip(points, prior_min) if p[2] < m]
        floor     = ceil(self.coverage_floor * len(gap_set))
        eligible  = [p for p in frontier if p[0] >= floor] or frontier

        if len(eligible) == 1:
            chosen = eligible[0]
        else:
            by_gaps = sorted(eligible, key=lambda p: p[0])
            knee    = KneeLocator(
                [p[0] for p in by_gaps],
                [p[2] for p in by_gaps],
                curve     = "convex",
                direction = "increasing"
            ).knee
            chosen = next((p for p in by_gaps if p[0] == knee), by_gaps[-1])

        picks_idx = chosen[1]
        pick_rows = reach[picks_idx] & gap_mask
        gains     = (pick_rows.cumsum(axis=0) == 1) & pick_rows

        return [
            SelectedCredential.from_hits(labels[i], row, cols)
            for i, row in zip(picks_idx, gains)
            if row.any()
        ]


@dataclass(kw_only=True)
class SOCScorer:
    """
    ColBERTv2-style MaxSim mean-of-max scorer for clusters against O*NET
    occupations, referencing Santhanam et al. 2022's late-interaction
    formulation translated to the occupation-matching setting.

    Stacks the per-occupation task matrices into one flat matrix at
    construction so `score()` resolves every cluster-occupation pair with a
    single BLAS-backed matmul and a contiguous `np.maximum.reduceat`, rather
    than a Python double loop of per-pair matmuls. Per-occupation boundaries
    are preserved via `owner_starts`, a row-aligned index into the flat
    matrix suitable for reduceat segmentation.

    Every O*NET occupation in the curated reference set lists at least one
    task element, so the scorer requires non-empty task matrices and does
    not handle degenerate zero-task occupations.
    """

    occupations: list[EncodedOccupation]

    flat_tasks   : np.ndarray = field(init=False)
    owner_starts : np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Flatten per-occupation task matrices into one contiguous array plus
        a starting-index array for `np.maximum.reduceat` segmentation.
        """
        tasks = [e.tasks for e in self.occupations]
        sizes = np.array([t.shape[0] for t in tasks])
        if not sizes.all():
            raise ValueError(
                "SOCScorer requires every occupation to list at least one "
                "task; empty task matrix found."
            )
        self.flat_tasks   = np.vstack(tasks)
        self.owner_starts = np.cumsum(sizes) - sizes

    def score(
        self,
        assignments : np.ndarray,
        raw_vectors : np.ndarray
    ) -> np.ndarray:
        """
        Compute mean-of-max similarity for every (cluster, occupation) pair.

        Normalizes posting vectors once, runs a single matmul against the
        flat task matrix, reduces columnwise per occupation via
        `np.maximum.reduceat` to obtain each posting's best-matching task
        score per occupation, then averages within each cluster.

        Args:
            assignments : Cluster label per posting in `raw_vectors` order.
            raw_vectors : Unnormalized posting embeddings from the encoder, shape
                          `(n_postings, embedding_dim)`.

        Returns:
            Similarity matrix of shape `(n_clusters, n_occupations)` with cluster rows
            in sorted assignment-id order and occupation columns in `self.occupations`
            order.
        """
        per_task       = normalize(raw_vectors) @ self.flat_tasks.T
        per_occupation = np.maximum.reduceat(per_task, self.owner_starts, axis=1)

        return (
            pd.DataFrame(per_occupation)
              .groupby(assignments).mean()
              .to_numpy(dtype=np.float32)
        )
