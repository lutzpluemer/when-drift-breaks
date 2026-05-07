"""
teiresias.persistence
=====================

Save and load fitted pipeline artifacts.

Two complementary functions:

* ``save_nothung_state`` / ``load_nothung_state`` -- save the full fitted
  pipeline (scaler, KMeans, RF, features, Viterbi smoothing, posteriors).
  Useful for caching long-running training results.

* ``save_nothung_map`` / ``load_nothung_map`` -- save only the lightweight
  artifacts needed to *use* the model in inference: cluster cores per
  regime, scaler statistics, feature column names.  This is the shape
  expected by :func:`teiresias.observation.build_observation_model`.

Files use ``joblib`` for sklearn objects and pickle for plain dictionaries.
A ``config.json`` is written alongside, capturing parameters and a
timestamp.  This makes a fitted pipeline self-describing.
"""

from __future__ import annotations

import os
import json
import pickle
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Full state (all fitted estimators + features + diagnostics)
# ----------------------------------------------------------------------

def save_nothung_state(
    path: str | Path,
    scaler,
    kmeans,
    rf,
    features_df: pd.DataFrame,
    viterbi: np.ndarray,
    probs: np.ndarray,
    config: dict | None = None,
) -> None:
    """
    Save a fully fitted pipeline to disk.

    Creates ``{path}/models/`` and ``{path}/results/``; writes
    ``scaler.pkl``, ``kmeans.pkl``, ``rf.pkl`` (joblib),
    ``features_df.pkl`` (pandas pickle), ``viterbi.pkl`` (joblib of
    a dict), and ``config.json``.
    """
    path = Path(path)
    (path / "models").mkdir(parents=True, exist_ok=True)
    (path / "results").mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, path / "models" / "scaler.pkl")
    joblib.dump(kmeans, path / "models" / "kmeans.pkl")
    joblib.dump(rf, path / "models" / "rf.pkl")
    features_df.to_pickle(path / "results" / "features_df.pkl")
    joblib.dump(
        {"viterbi": viterbi, "probs": probs},
        path / "results" / "viterbi.pkl",
    )

    if config is not None:
        config = dict(config)
        config["_saved"] = datetime.now().isoformat()
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)


def load_nothung_state(path: str | Path) -> dict:
    """
    Load a previously saved pipeline.

    Returns a dict with keys ``scaler``, ``kmeans``, ``rf``,
    ``features_df``, ``viterbi``, ``probs``.
    """
    path = Path(path)
    state = {
        "scaler": joblib.load(path / "models" / "scaler.pkl"),
        "kmeans": joblib.load(path / "models" / "kmeans.pkl"),
        "rf":     joblib.load(path / "models" / "rf.pkl"),
        "features_df": pd.read_pickle(path / "results" / "features_df.pkl"),
    }
    state.update(joblib.load(path / "results" / "viterbi.pkl"))
    return state


# ----------------------------------------------------------------------
# Map only (lightweight, what the inference layer actually needs)
# ----------------------------------------------------------------------

def save_nothung_map(
    path: str | Path,
    cluster_cores: dict[int, np.ndarray],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    feature_cols: list[str],
    regime_names: dict[int, str] | list[str] | None = None,
) -> None:
    """
    Save the lightweight inference artifact (the "Nothung map").

    Parameters
    ----------
    path : Path-like
        Output directory.
    cluster_cores : dict
        ``{regime_id: ndarray of cluster centroids in feature space}``.
    scaler_mean, scaler_scale : np.ndarray
        Statistics from the RobustScaler used at training time.
    feature_cols : list[str]
        Feature column names (in the order used at training time).
    regime_names : dict or list, optional
        ``{regime_id: name}`` or list indexed by regime id.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    karte = {
        "cluster_cores": cluster_cores,
        "scaler_mean":   np.asarray(scaler_mean),
        "scaler_scale":  np.asarray(scaler_scale),
        "feature_cols":  list(feature_cols),
    }
    if regime_names is not None:
        karte["regime_names"] = regime_names
    with open(path / "nothung_karte.pkl", "wb") as f:
        pickle.dump(karte, f)


def load_nothung_map(path: str | Path) -> dict:
    """
    Load a previously saved map.

    Accepts either a directory containing ``nothung_karte.pkl`` or a direct
    path to a ``.pkl`` file.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "nothung_karte.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
