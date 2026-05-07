"""
teiresias.evaluation
====================

Diagnostic and validation utilities.

* ``separability_scan``    — per-feature ROC AUC against a binary label.
* ``compute_lead_time_soft`` — days between first warning and a known event.
* ``transition_analysis``   — tally regime transitions in a sequence.

These functions are not part of the inference pipeline; they assess how
well the pipeline does its job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def separability_scan(
    features_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    positive_labels: list,
    min_samples: int = 100,
) -> pd.DataFrame:
    """
    Univariate ROC AUC of each feature against a binary label.

    For each feature column we treat the value as a one-dimensional score
    and compute the AUC for distinguishing rows whose ``label_col`` is in
    ``positive_labels`` from the rest.  AUC is symmetric: we report
    ``max(auc, 1 - auc)`` so that the direction of the score is irrelevant.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain all ``feature_cols`` and the ``label_col``.
    feature_cols : list[str]
        Feature column names to score.
    label_col : str
        Column with discrete labels.
    positive_labels : list
        Labels that count as positive.
    min_samples : int
        Skip features with fewer finite values than this.

    Returns
    -------
    pd.DataFrame
        Columns ``feature``, ``auc``, sorted by AUC descending.
    """
    y = features_df[label_col].isin(positive_labels).astype(int)
    rows = []
    for feat in feature_cols:
        x = features_df[feat].values
        mask = np.isfinite(x)
        if mask.sum() < min_samples:
            continue
        try:
            auc = roc_auc_score(y.values[mask], x[mask])
            auc = max(auc, 1 - auc)
        except Exception:
            auc = 0.5
        rows.append({"feature": feat, "auc": auc})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def compute_lead_time_soft(
    regime_probs: pd.DataFrame,
    dates: pd.DatetimeIndex,
    crash_date: pd.Timestamp,
    warning_regimes: list[str],
    threshold: float = 0.3,
) -> dict:
    """
    Days between the first persistent warning and a known crash date.

    The warning signal at time ``t`` is the sum of posterior probabilities
    of the ``warning_regimes``.  The lead time is computed from the first
    pre-crash date at which this sum exceeds ``threshold``.

    Parameters
    ----------
    regime_probs : pd.DataFrame
        Per-regime posterior probabilities, columns named by regime.
    dates : pd.DatetimeIndex
        Index aligned with ``regime_probs`` rows.
    crash_date : pd.Timestamp
        Reference event.
    warning_regimes : list[str]
        Columns to sum (e.g. ``['Stress', 'Bear', 'Crisis']``).
    threshold : float
        Warning threshold.

    Returns
    -------
    dict
        ``{'lead_days': int or None, 'first_warning': pd.Timestamp or None}``
    """
    p_warn = regime_probs[warning_regimes].sum(axis=1)
    mask = dates < crash_date
    pre = p_warn.values[mask]
    pre_dates = dates[mask]
    above = pre > threshold
    if not above.any():
        return {"lead_days": None, "first_warning": None}
    idx = int(np.where(above)[0][0])
    first = pre_dates[idx]
    return {
        "lead_days": (crash_date - first).days,
        "first_warning": first,
    }


def transition_analysis(regimes) -> pd.DataFrame:
    """
    Tally regime-to-regime transitions in a sequence.

    Parameters
    ----------
    regimes : sequence of regime labels.

    Returns
    -------
    pd.DataFrame
        Columns ``from``, ``to``, ``count``, sorted by count descending.
    """
    trans: dict[tuple, int] = {}
    for i in range(len(regimes) - 1):
        if regimes[i] != regimes[i + 1]:
            key = (regimes[i], regimes[i + 1])
            trans[key] = trans.get(key, 0) + 1
    rows = [{"from": k[0], "to": k[1], "count": v} for k, v in trans.items()]
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
