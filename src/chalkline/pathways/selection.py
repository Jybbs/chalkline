"""
Candidate-set selection algorithms over the pathways domain.

Two stateless dataclasses whose output drives a final selection against a
candidate pool. `SOCScorer` enables SOC assignment per cluster via
ColBERTv2-style late-interaction MaxSim across per-posting and per-task
embeddings. `CredentialSelector` enables credential-stack selection per
route via waste-aware Pareto-knee optimization, picking a credential
combination that maximizes gap coverage with minimum redundant reach onto
tasks the user already demonstrates.
"""

import numpy as np
import pandas as pd

from dataclasses           import dataclass, field
from itertools             import accumulate
from kneed                 import KneeLocator
from math                  import ceil
from operator              import attrgetter
from sklearn.preprocessing import normalize

from chalkline.pathways.schemas import EncodedOccupation, SelectedCredential
from chalkline.pathways.schemas import SelectorConfig, SelectorFrontier


@dataclass(kw_only=True)
class CredentialSelector:
    """
    Waste-aware Pareto-knee picker for credential stacks per route.

    Sweeps a penalty coefficient across a fixed range, runs vectorized
    greedy `Δgaps − α · Δwaste` for each α, keeps the Pareto-dominant
    points, and selects the knee via the Kneedle algorithm. The coverage
    floor on `SelectorConfig` anchors the knee to the eligible high-coverage
    region when the pool can satisfy it, with graceful fallback to the
    unconstrained efficiency point otherwise.

    Args:
        config: Sweep resolution, penalty range, and coverage floor.
    """

    config: SelectorConfig = field(default_factory=SelectorConfig)

    def _greedy(
        self,
        alpha     : float,
        gap_mask  : np.ndarray,
        max_picks : int,
        reach     : np.ndarray
    ) -> tuple[list[int], np.ndarray]:
        """
        Vectorized greedy for one α, returning pick row indices. Per-column
        weights fold the penalty into a single `(reach & ~covered) @
        weights` matmul that scores every active credential in one BLAS
        sweep, where each gap column contributes +1 and each non-gap column
        contributes −α; the argmax wins while the marginal score stays
        positive.

        Args:
            alpha     : Penalty on new reach onto already-covered tasks.
            gap_mask  : Boolean flag per `reach` column marking gap tasks.
            max_picks : Hard cap on stack size.
            reach     : Boolean matrix, shape `(n_credentials, n_tasks)`.
        """
        covered = np.zeros(reach.shape[1], dtype=bool)
        weights = np.where(gap_mask, 1.0, -alpha)
        picks   = []

        for _ in range(max_picks):
            scores = (reach & ~covered) @ weights
            i      = int(scores.argmax())
            if scores[i] <= 0:
                break
            picks.append(i)
            covered |= reach[i]

        return picks, covered

    def select_stack(
        self,
        coverage  : dict[str, dict[int, float]],
        gap_set   : frozenset[int],
        max_picks : int
    ) -> tuple[list[SelectedCredential], bool]:
        """
        Waste-aware credential stack anchored to the coverage floor when
        reachable, with graceful fallback to the efficiency-optimal frontier
        point otherwise.

        Returns each pick's incremental gap positions for shelf rendering
        along with a bool signalling whether the floor was met.

        Args:
            coverage  : Credential label to `{task_index: affinity}` over every cluster
                        task the credential reaches, not just gap tasks.
            gap_set   : Task indices the resume has not demonstrated.
            max_picks : Hard cap on stack size.
        """
        coverage = {label: scored for label, scored in coverage.items() if scored}
        if not coverage or not gap_set:
            return [], False

        hits = (
            pd.DataFrame.from_dict(coverage, orient="index")
              .notna()
              .sort_index(axis=1)
        )
        reach      = hits.to_numpy()
        gap_mask   = hits.columns.isin(gap_set)
        cred_reach = np.count_nonzero(reach, axis=1)

        seen: dict[tuple[int, int], SelectorFrontier] = {}
        for alpha in self.config.alphas:
            indices, covered = self._greedy(alpha, gap_mask, max_picks, reach)
            point = SelectorFrontier(
                alpha       = alpha,
                gaps_filled = int((covered & gap_mask).sum()),
                picks       = tuple(indices),
                total_reach = int(cred_reach[indices].sum())
            )
            seen.setdefault((point.gaps_filled, point.waste), point)

        points     = sorted(seen.values(), key=lambda p: (-p.gaps_filled, p.waste))
        prior_min  = accumulate((p.waste for p in points), min, initial=float("inf"))
        frontier   = [p for p, m in zip(points, prior_min) if p.waste < m]
        qualifying = [
            p for p in frontier 
            if p.gaps_filled >= ceil(self.config.coverage_floor * len(gap_set))
        ]

        eligible = qualifying or frontier
        if len(eligible) < 2:
            chosen = eligible[0]
        else:
            by_gaps = sorted(eligible, key=attrgetter("gaps_filled"))
            knee    = KneeLocator(
                [p.gaps_filled for p in by_gaps],
                [p.waste       for p in by_gaps],
                curve     = "convex",
                direction = "increasing"
            ).knee
            chosen  = next((p for p in by_gaps if p.gaps_filled == knee), by_gaps[-1])

        pick_rows = reach[list(chosen.picks)] & gap_mask
        prior     = np.zeros_like(pick_rows)
        prior[1:] = np.logical_or.accumulate(pick_rows[:-1], axis=0)
        gains     = pick_rows & ~prior
        task_axis = hits.columns.to_numpy()
        
        return (
            [
                SelectedCredential.from_hits(hits.index[i], row, task_axis)
                for i, row in zip(chosen.picks, gains)
            ], 
            bool(qualifying)
        )


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
        if any(t.size == 0 for t in tasks):
            raise ValueError(
                "SOCScorer requires every occupation to list at least one "
                "task; empty task matrix found."
            )
        sizes             = np.array([t.shape[0] for t in tasks])
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
        postings       = normalize(raw_vectors)
        per_task       = postings @ self.flat_tasks.T
        per_occupation = np.maximum.reduceat(per_task, self.owner_starts, axis=1)

        return (
            pd.DataFrame(per_occupation)
              .groupby(assignments).mean()
              .to_numpy().astype(np.float32)
        )
