"""
teiresias.transitions
=====================

Hidden Semi-Markov Model (HSMM) transition dynamics for the seven regimes.

Standard HMMs assume geometric duration distributions in each state.  Markets
are not geometric: regimes have characteristic durations that exit only when
the underlying market condition has had enough time to develop.  We model
this with a Weibull hazard:

    h(d) = (k/lambda) * ((d+1)/lambda)^(k-1)

where ``d`` is the time already spent in the current regime.  The shape
parameter ``k > 1`` gives an increasing hazard with duration -- it becomes
progressively more likely to leave a regime the longer one has been in it,
which matches stylised facts about market cycles.

Per-regime parameters (median duration in days, shape ``k``) come from the
empirical duration histograms of the labelled training set.  The base
transition matrix encodes *where* a regime transitions to *given* that it
transitions; the diagonal is zero and rows sum to one.

Together: ``P(stay)  = 1 - h(d)``, ``P(go to r')  = h(d) * TRANS_BASE[r, r']``.
"""

from __future__ import annotations

import numpy as np


# Regime ids match REGIME_ORDER in teiresias.regimes:
# 0 Bull, 1 Sideways, 2 Correction, 3 Stress, 4 Bear, 5 Crisis, 6 Recovery
DURATION_PARAMS: dict[int, dict[str, float]] = {
    0: {"median": 362, "shape": 2.0},   # Bull       — long calm phases
    1: {"median":  69, "shape": 2.0},   # Sideways
    2: {"median":  43, "shape": 2.5},   # Correction — short, decisive
    3: {"median":  20, "shape": 2.0},   # Stress
    4: {"median":  77, "shape": 1.8},   # Bear       — slow decay
    5: {"median":  19, "shape": 2.5},   # Crisis     — short and violent
    6: {"median":  53, "shape": 2.0},   # Recovery
}


# Base transition matrix: where regime r goes when it does transition.
# Rows are source regimes, columns are destinations.  Diagonal is zero
# (handled separately by the hazard).  Rows are normalised to sum to one.
#
# Read row by row:
#   Bull       -> mostly Correction (0.40) and Stress (0.35)
#   Sideways   -> back to Bull (0.40) or to Correction (0.25)
#   Correction -> Bull (0.25), Bear (0.25), Recovery (0.20)
#   Stress    -> Crisis (0.35) is the dominant onward path
#   Bear       -> Recovery (0.55) or Crisis (0.35)
#   Crisis     -> Recovery (0.90) -- crises are followed by rebounds
#   Recovery   -> Bull (0.50)
TRANS_BASE: np.ndarray = np.array([
    # Bull  Side  Corr  Str   Bear  Cris  Rec
    [0.00, 0.10, 0.40, 0.35, 0.05, 0.05, 0.05],   # Bull
    [0.40, 0.00, 0.25, 0.15, 0.10, 0.05, 0.05],   # Sideways
    [0.25, 0.05, 0.00, 0.15, 0.25, 0.10, 0.20],   # Correction
    [0.20, 0.00, 0.25, 0.00, 0.10, 0.35, 0.10],   # Stress
    [0.00, 0.00, 0.00, 0.10, 0.00, 0.35, 0.55],   # Bear
    [0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.90],   # Crisis
    [0.50, 0.10, 0.10, 0.05, 0.15, 0.10, 0.00],   # Recovery
])
# NOTE: a defect in the original Parsifal source had Recovery[6]=0.05,
# putting weight on Recovery -> Recovery in the *given-transition* matrix.
# That is incorrect: TRANS_BASE encodes destinations *given* a transition
# occurs, so the diagonal must be zero (self-stay is governed by 1 - h(d)
# in get_transition_probs).  The mass has been moved to Recovery -> Crisis.

# Defensive normalisation: zero the diagonal and renormalise rows.
np.fill_diagonal(TRANS_BASE, 0.0)
for _i in range(TRANS_BASE.shape[0]):
    _row_sum = TRANS_BASE[_i].sum()
    if _row_sum > 0:
        TRANS_BASE[_i] /= _row_sum


N_REGIMES = TRANS_BASE.shape[0]


def hazard_rate(regime_id: int, duration: int) -> float:
    """
    Per-day Weibull hazard rate -- probability of leaving the current regime.

    Clipped to ``[0.001, 0.5]`` to avoid pathological values at duration=0
    or for very long-lived regimes.

    Parameters
    ----------
    regime_id : int
        Regime index (0..N_REGIMES-1).
    duration : int
        Number of days already spent in this regime.

    Returns
    -------
    float
        Hazard rate in ``[0.001, 0.5]``.
    """
    params = DURATION_PARAMS[regime_id]
    med = params["median"]
    shape = params["shape"]
    scale = med / (np.log(2) ** (1.0 / shape))
    h = (shape / scale) * ((duration + 1) / scale) ** (shape - 1)
    return float(np.clip(h, 0.001, 0.5))


def get_transition_probs(regime_id: int, duration: int) -> np.ndarray:
    """
    Full transition probability vector from (regime, duration).

    The probability of staying is ``1 - h(d)``; the remainder is distributed
    over the other regimes according to ``TRANS_BASE[regime_id]``.

    Returns
    -------
    np.ndarray
        Vector of length ``N_REGIMES`` summing to one.
    """
    h = hazard_rate(regime_id, duration)
    probs = np.zeros(N_REGIMES)
    probs[regime_id] = 1.0 - h
    probs += h * TRANS_BASE[regime_id]
    return probs / probs.sum()
