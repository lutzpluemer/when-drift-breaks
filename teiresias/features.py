"""
teiresias.features
==================

Multi-window distributional shape descriptors for financial time series.

The feature engine computes, for a set of rolling windows, statistics that
capture distributional shape rather than only level or volatility:

* Central moments: mean, std, skewness, kurtosis (excess)
* Quantile-based shape: q05, q95, IQR, Bowley skewness
* Tail behaviour: realised volatility, negative-return fraction, downside RMS
* Drawdown: rolling max-relative drawdown
* Bowley histogram: 6-bin classification of (sign, magnitude) of returns

Two preset window configurations are supported:

* ``windows='master'`` -> [5, 20, 60]      (the original Master-Setup pipeline)
* ``windows='octave'`` -> [10, 40, 80, 180] (the multi-scale octave stack
                                              described in the manuscript)

A custom list may also be passed.  Both presets return a DataFrame of features
indexed by the input price index, with one column per (statistic, window).

Source: adapted from the Master-Setup pipeline (February 2026).  See README
for the design rationale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, jarque_bera


WINDOW_PRESETS = {
    "master": [5, 20, 60],
    "octave": [10, 40, 80, 180],
}


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Logarithmic returns from a price series."""
    return np.log(prices / prices.shift(1))


def _resolve_windows(windows):
    """Accept a preset name, a list, or None."""
    if windows is None:
        return WINDOW_PRESETS["master"]
    if isinstance(windows, str):
        if windows not in WINDOW_PRESETS:
            raise ValueError(
                f"Unknown window preset {windows!r}. "
                f"Choose from {list(WINDOW_PRESETS)} or pass a list of ints."
            )
        return WINDOW_PRESETS[windows]
    return list(windows)


def compute_all_features(
    prices: pd.Series,
    vix: pd.Series | None = None,
    volume: pd.Series | None = None,
    windows: list[int] | str | None = "master",
) -> pd.DataFrame:
    """
    Compute the full set of multi-window distributional shape features.

    Parameters
    ----------
    prices : pd.Series
        Daily closing prices, indexed by date.
    vix : pd.Series, optional
        VIX (or other volatility index) aligned with ``prices``.  If supplied,
        VIX-derived columns are added.
    volume : pd.Series, optional
        Trading volume aligned with ``prices``.  If supplied, a volume-anomaly
        column is added.
    windows : list[int] or str or None
        Either a list of window sizes (e.g. ``[5, 20, 60]``), a preset name
        (``'master'`` or ``'octave'``), or ``None`` for the default
        (``'master'``).  Drawdown and Bowley features always use the two
        largest windows from this list.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame indexed by the price index.  Column names follow the
        pattern ``<statistic>_<window>``, e.g. ``ret_skew_20``.
    """
    W_list = _resolve_windows(windows)

    ret = compute_log_returns(prices)
    f = pd.DataFrame(index=prices.index)

    # --- Per-window distributional features ---
    for W in W_list:
        f[f"ret_mean_{W}"] = ret.rolling(W).mean()
        f[f"ret_std_{W}"] = ret.rolling(W).std()
        f[f"ret_skew_{W}"] = ret.rolling(W).apply(
            lambda x: skew(x), raw=True
        )
        f[f"ret_kurt_{W}"] = ret.rolling(W).apply(
            lambda x: kurtosis(x, fisher=True), raw=True
        )
        f[f"q05_{W}"] = ret.rolling(W).quantile(0.05)
        f[f"q95_{W}"] = ret.rolling(W).quantile(0.95)
        f[f"iqr_{W}"] = (
            ret.rolling(W).quantile(0.75) - ret.rolling(W).quantile(0.25)
        )
        f[f"rv_{W}"] = ret.rolling(W).apply(
            lambda x: np.sqrt(np.sum(x**2)), raw=True
        )
        f[f"neg_frac_{W}"] = ret.rolling(W).apply(
            lambda x: np.mean(x < 0), raw=True
        )
        f[f"downside_rms_{W}"] = ret.rolling(W).apply(
            lambda x: np.sqrt(np.mean(x[x < 0] ** 2)) if np.any(x < 0) else 0,
            raw=True,
        )

    # --- Drawdown and Bowley skewness on the two largest windows ---
    big_windows = sorted(W_list)[-2:]
    for W in big_windows:
        rolling_max = prices.rolling(W).max()
        f[f"drawdown_{W}"] = prices / rolling_max - 1.0
        q25 = ret.rolling(W).quantile(0.25)
        q50 = ret.rolling(W).quantile(0.50)
        q75 = ret.rolling(W).quantile(0.75)
        f[f"bowley_skew_{W}"] = (q75 + q25 - 2 * q50) / (q75 - q25 + 1e-8)

    # --- Volatility-of-volatility on the two smallest windows ---
    small_windows = sorted(W_list)[:2]
    if len(small_windows) >= 2:
        s, l = small_windows
        if f"rv_{s}" in f.columns and f"rv_{l}" in f.columns:
            f[f"vol_of_vol_{l}"] = f[f"rv_{s}"].rolling(l).std()
            f["vol_slope"] = f[f"rv_{s}"] - f[f"rv_{l}"]

    # --- Bowley histogram (6 bins of sign x magnitude) on the two largest ---
    abs_ret = ret.abs()
    a1, a2 = abs_ret.quantile(0.50), abs_ret.quantile(0.90)
    bow = pd.Series(0, index=ret.index)
    bow[(ret >= 0) & (abs_ret <= a1)] = 0
    bow[(ret >= 0) & (abs_ret > a1) & (abs_ret <= a2)] = 1
    bow[(ret >= 0) & (abs_ret > a2)] = 2
    bow[(ret < 0) & (abs_ret <= a1)] = 3
    bow[(ret < 0) & (abs_ret > a1) & (abs_ret <= a2)] = 4
    bow[(ret < 0) & (abs_ret > a2)] = 5
    bow_names = ["up_sm", "up_md", "up_bg", "dn_sm", "dn_md", "dn_bg"]
    for W in big_windows:
        for k, nm in enumerate(bow_names):
            f[f"bow_{nm}_{W}"] = bow.rolling(W).apply(
                lambda x, c=k: np.mean(x == c), raw=True
            )

    # --- Optional exogenous channels ---
    if vix is not None:
        f["vix"] = vix
        f["vix_chg_5"] = np.log(vix / vix.shift(5))
        f["vix_chg_20"] = np.log(vix / vix.shift(20))

    if volume is not None:
        vm = volume.rolling(20).mean()
        vs = volume.rolling(20).std()
        f["vol_anom"] = (volume - vm) / vs

    return f


def test_gaussianity(returns: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    Per-window non-Gaussianity diagnostics.

    Parameters
    ----------
    returns : pd.Series
        Log returns.
    window : int
        Rolling window size.

    Returns
    -------
    pd.DataFrame
        Columns: ``jarque_bera``, ``excess_kurtosis``, ``non_gaussian``
        (binary indicator at JB > 6).
    """
    f = pd.DataFrame(index=returns.index)
    f["jarque_bera"] = returns.rolling(window).apply(
        lambda x: jarque_bera(x)[0] if len(x) >= 20 else np.nan, raw=True
    )
    f["excess_kurtosis"] = returns.rolling(window).apply(
        lambda x: kurtosis(x, fisher=True), raw=True
    )
    f["non_gaussian"] = (f["jarque_bera"] > 6).astype(float)
    return f
