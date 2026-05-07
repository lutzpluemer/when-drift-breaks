"""
teiresias.spectral
==================

Cross-sectional spectral features from a panel of asset returns.

When markets enter a stress regime, individual asset returns become
strongly co-moving: the correlation matrix's leading eigenvalue swells
while smaller eigenvalues collapse.  This phenomenon -- the "spectral
condensation" of diversification -- has been studied in the absorption
ratio literature (Kritzman, Page, Turkington 2010; Billio et al. 2012).

This module computes three spectral descriptors on a rolling window of
log-returns from a panel of assets (e.g. nine S&P sector ETFs):

* ``absorption_ratio``        -- :math:`\\sum_{i=1}^{n_{\\mathrm{top}}} \\lambda_i / \\sum_i \\lambda_i`,
  the share of total variance explained by the top ``n_top`` modes.
* ``lambda1_ratio``           -- :math:`\\lambda_1 / \\sum_i \\lambda_i`,
  the share carried by the leading mode alone.
* ``absorption_ratio_chg_10`` -- the 10-day change of ``absorption_ratio``.

These features form Phase ``Shape+Skew+Spectral`` of the *When Drift
Breaks* main-result table.

Source: extracted from ``teiresias_phase_D1_setup.ipynb`` (5 May 2026)
without algorithmic modification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_absorption_features(
    sector_returns: pd.DataFrame,
    window: int = 60,
    n_top: int = 3,
) -> pd.DataFrame:
    """
    Compute rolling spectral features on a panel of asset returns.

    Parameters
    ----------
    sector_returns : pd.DataFrame
        Log-returns of a panel of assets, indexed by date.  Each column is
        one asset (e.g. one sector ETF).  Missing values are dropped per
        rolling window before computing the correlation matrix.
    window : int
        Length of the rolling correlation window in days.  Default 60.
    n_top : int
        Number of leading eigenvalues to aggregate for the absorption
        ratio.  Default 3.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame indexed by ``sector_returns.index`` with three
        columns:

        * ``absorption_ratio``
        * ``lambda1_ratio``
        * ``absorption_ratio_chg_10``

        Rows before the first full window are filled with NaN.
    """
    f = pd.DataFrame(index=sector_returns.index)
    ar_values: list[float] = []
    l1_values: list[float] = []

    for i in range(len(sector_returns)):
        if i < window:
            ar_values.append(np.nan)
            l1_values.append(np.nan)
            continue

        chunk = sector_returns.iloc[i - window:i].dropna(axis=1, how="any")
        if chunk.shape[1] < 4:
            # Need at least 4 assets for a meaningful spectral decomposition
            ar_values.append(np.nan)
            l1_values.append(np.nan)
            continue

        try:
            corr = chunk.corr().values
            eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]
            total = float(np.sum(eigenvalues))
            if total > 0:
                ar = float(np.sum(eigenvalues[:n_top]) / total)
                l1 = float(eigenvalues[0] / total)
            else:
                ar = np.nan
                l1 = np.nan
            ar_values.append(ar)
            l1_values.append(l1)
        except Exception:
            ar_values.append(np.nan)
            l1_values.append(np.nan)

    f["absorption_ratio"] = ar_values
    f["lambda1_ratio"] = l1_values
    f["absorption_ratio_chg_10"] = (
        f["absorption_ratio"] - f["absorption_ratio"].shift(10)
    )
    return f
