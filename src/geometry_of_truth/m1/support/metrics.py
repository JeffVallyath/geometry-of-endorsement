"""
Frozen metric definitions. Import these; do not reimplement them inline.

THREE STATISTICS, IN ORDER
    PRIMARY      standardized reciprocal interaction contrast  I_b
    SECONDARY    signed exact-board resolution                 B_b
    DESCRIPTIVE  both-ways accuracy, ALWAYS with its chance level

Both-ways is descriptive only. Measured 2026-08-04: a pure random scorer gets
0.258 and a random additive scorer gets exactly 0.000, so the metric is NOT
monotonic in model quality --

    additive        = 0.000
    random          ~ 0.17-0.26
    perfect         = 1.000

-- and a partially additive model can learn something real and still score
below random. Useful as an exact descriptive number, invalid as an endpoint.

THE SCALING TRAP
I_b is a difference of raw scores, so it inherits the scorer's units. Multiply
a probe's weights by ten and I_b multiplies by ten while every prediction is
unchanged. A difference-in-means direction, a logistic decision function and an
SBERT probability are on arbitrary and mutually incomparable scales. So every
scorer is standardized with statistics computed ONCE on the pilot/selection
split and then frozen:

    f~ = (f - mu_select) / sigma_select

Standardizing on the evaluation split instead would leak. Standardizing per
model at evaluation time would silently rescale the effect being measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ==========================================================================
# frozen standardization
# ==========================================================================

@dataclass(frozen=True)
class Scaler:
    """Location and scale fitted on the selection split, then frozen."""
    mu: float
    sigma: float
    fitted_on: str

    def __call__(self, scores) -> np.ndarray:
        return (np.asarray(scores, dtype=float) - self.mu) / self.sigma


def fit_scaler(pilot_scores, name: str = "pilot_select") -> Scaler:
    s = np.asarray(pilot_scores, dtype=float)
    sd = float(s.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("degenerate scorer: zero variance on the pilot split")
    return Scaler(mu=float(s.mean()), sigma=sd, fitted_on=name)


# ==========================================================================
# PRIMARY: standardized reciprocal interaction contrast
# ==========================================================================

def interaction_contrast(board, scores) -> float:
    """I_b = [f~(s1,cA) - f~(s1,cB)] + [f~(s2,cB) - f~(s2,cA)]

    `scores` must ALREADY be standardized. Orientation: higher = Supports, and
    the cells are arranged so both brackets are positive for a scorer that
    tracks the relation.

        additive f(s,c) = a(s) + b(c)  ->  exactly 0
        random / permuted scores       ->  expectation 0
        correct interaction            ->  > 0
        systematically reversed        ->  < 0
    """
    return ((scores[board["i_s1_c1"]] - scores[board["i_s1_c2"]])
            + (scores[board["i_s2_c2"]] - scores[board["i_s2_c1"]]))


# ==========================================================================
# SECONDARY: signed exact-board resolution
# ==========================================================================

def signed_board_score(board, scores) -> int:
    """B_b in {+1, 0, -1}: both rankings right / mixed-or-tied / both wrong.

    Scale-free and intuitive. An additive scorer gets exactly 0 on every board
    (its two gaps are opposites, including the zero-gap tie). A symmetric
    random scorer has expectation 0.
    Lower sensitivity than I_b, which is why it is secondary rather than
    primary, but it needs no standardization at all.
    """
    d1 = scores[board["i_s1_c1"]] - scores[board["i_s1_c2"]]
    d2 = scores[board["i_s2_c2"]] - scores[board["i_s2_c1"]]
    if d1 > 0 and d2 > 0:
        return 1
    if d1 < 0 and d2 < 0:
        return -1
    return 0


# ==========================================================================
# DESCRIPTIVE: both-ways, never reported without its chance level
# ==========================================================================

BOTH_WAYS_CHANCE = 0.258          # measured, pure random scorer, 382 boards
BOTH_WAYS_ADDITIVE = 0.000        # exact


def both_ways(board, scores) -> int:
    d1 = scores[board["i_s1_c1"]] - scores[board["i_s1_c2"]]
    d2 = scores[board["i_s2_c2"]] - scores[board["i_s2_c1"]]
    return int(d1 > 0 and d2 > 0)


def format_both_ways(rate: float) -> str:
    """Anything printing both-ways must print it like this."""
    return (f"{rate:.3f}  (chance {BOTH_WAYS_CHANCE:.3f}, additive "
            f"{BOTH_WAYS_ADDITIVE:.3f} -- DESCRIPTIVE ONLY, not monotonic "
            f"in model quality)")


# ==========================================================================
# the confirmatory quantity
# ==========================================================================

def delta_ib(boards, probe_scaled, baseline_scaled) -> np.ndarray:
    """Per-board Delta I_b = I_b(activation probe) - I_b(frozen text baseline).

    Both scorers evaluated on the SAME audited boards, both standardized with
    their own frozen pilot statistics. Return per-board values so the caller
    can apply the dyadic-robust estimator -- never collapse to a mean here.
    """
    return np.array([interaction_contrast(b, probe_scaled)
                     - interaction_contrast(b, baseline_scaled)
                     for b in boards])
