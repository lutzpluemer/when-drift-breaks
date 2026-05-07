"""
teiresias.loss_dynamics
=======================

Loss-dynamics features capturing how losses unfold over time.

While :mod:`teiresias.features` describes the *shape* of the return
distribution within a window (moments, quantiles, drawdown level), this
module describes the *temporal pattern* of losses:

* **Drawdown slope** -- how fast the drawdown is deepening
  (5-day change of the rolling max-relative drawdown).
* **Time-in-drawdown** -- the fraction of the rolling window during which
  the cumulative-from-end return is negative.
* **Maximum negative run** -- the longest uninterrupted streak of
  negative daily returns inside the window.
* **Tail rate** -- the fraction of days within the window whose return
  falls below the in-window 5%-quantile.

These features extend the Master-Setup pipeline.  They were added in the
*When Drift Breaks* paper as the first extension layer (April 2026) and
form Phase ``Shape`` of the main-result table.

Standard window sizes:

* drawdown slope, time-in-drawdown, max-neg-run: :math:`W \\in \\{20, 60\\}`
* tail rate:                                     :math:`W \\in \\{5, 20, 60\\}`

yielding 9 features in total per day.

Source: extracted from ``teiresias_phase_D1_setup.ipynb`` (5 May 2026)
without algorithmic modification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _max_neg_run(x: np.ndarray) -> float:
    """Length of the longest run of strictly negative values in x."""
    run, max_run = 0, 0
    for v in x:
        if v < 0:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return float(max_run)


def compute_loss_dynamics(
    prices: pd.Series,
    log_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute the 9 loss-dynamics features.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices, indexed by date.
    log_returns : pd.Series, optional
        Log-returns aligned with ``prices``.  If ``None`` they are computed
        internally as ``log(prices / prices.shift(1))``.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame indexed by the price index.  Columns:

        * ``dd_slope_5_20``, ``dd_slope_5_60``
        * ``time_in_dd_20``, ``time_in_dd_60``
        * ``max_neg_run_20``, ``max_neg_run_60``
        * ``tail_rate_5_5``, ``tail_rate_5_20``, ``tail_rate_5_60``
    """
    if log_returns is None:
        log_returns = np.log(prices / prices.shift(1))

    f = pd.DataFrame(index=prices.index)

    # --- Drawdown slope and time-in-drawdown (W = 20, 60) ---
    for W in [20, 60]:
        rolling_max = prices.rolling(W).max()
        drawdown = prices / rolling_max - 1.0
        f[f"dd_slope_5_{W}"] = drawdown - drawdown.shift(5)
        f[f"time_in_dd_{W}"] = log_returns.rolling(W).apply(
            lambda x: float(np.mean(np.cumsum(x[::-1]) < 0)), raw=True
        )

    # --- Maximum negative run (W = 20, 60) ---
    for W in [20, 60]:
        f[f"max_neg_run_{W}"] = log_returns.rolling(W).apply(
            _max_neg_run, raw=True
        )

    # --- Tail rate (W = 5, 20, 60) ---
    for W in [5, 20, 60]:
        f[f"tail_rate_5_{W}"] = log_returns.rolling(W).apply(
            lambda x: float(np.mean(x < np.quantile(x, 0.05))), raw=True
        )

    return f
