"""
teiresias.clustering
====================

The map-construction layer: from feature vectors to discrete state labels.

Pipeline:

1.  Robust-tanh scaling of features (RobustScaler followed by tanh squash).
2.  KMeans codebook with k=21 micro-clusters (the "Nothung map").
3.  Random-forest soft classifier that yields per-day cluster posteriors.
4.  Viterbi smoothing in two flavours:
    * ``viterbi_smooth``: a minimum-duration filter (no transition matrix).
    * ``viterbi_full``  : full HMM Viterbi with an explicit transition matrix.
5.  Empirical transition-matrix estimation from a state sequence.
6.  Aggregation of cluster-level probabilities into the seven semantic
    regimes (Bull, Sideways, ...).

The 21-cluster, 7-regime structure is intentional: clusters give resolution,
regimes give interpretable labels.  The mapping lives in :mod:`teiresias.regimes`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler


def scale_features(
    X: np.ndarray,
    method: str = "robust_tanh",
) -> tuple[np.ndarray, RobustScaler]:
    """
    Robustly scale features and squash extremes through tanh.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    method : {'robust_tanh', 'robust'}
        ``'robust_tanh'`` applies tanh after the RobustScaler.  ``'robust'``
        applies only the RobustScaler.

    Returns
    -------
    X_scaled, scaler
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    if method == "robust_tanh":
        X_scaled = np.tanh(X_scaled)
    return X_scaled, scaler


def fit_codebook(
    X: np.ndarray,
    k: int = 21,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit a k-means codebook on scaled features.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    k : int
        Number of clusters (default 21).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    km : KMeans
        Fitted estimator with ``km.cluster_centers_`` available.
    labels : np.ndarray
        Cluster assignment per row.
    """
    km = KMeans(
        n_clusters=k, n_init=20, random_state=random_state, max_iter=500
    )
    labels = km.fit_predict(X)
    return km, labels


def fit_soft_classifier(
    X: np.ndarray,
    labels: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 15,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, np.ndarray]:
    """
    Fit a random-forest soft classifier mapping features to cluster posteriors.

    The classifier is trained on the codebook labels and yields
    ``predict_proba`` per cluster.  It enables out-of-sample inference at
    points where the codebook itself is not directly applicable.

    Returns
    -------
    rf : RandomForestClassifier
    probs : np.ndarray
        In-sample posterior probabilities, shape ``(n_samples, k)``.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, labels)
    probs = rf.predict_proba(X)
    return rf, probs


def viterbi_smooth(probs: np.ndarray, min_duration: int = 3) -> np.ndarray:
    """
    Minimum-duration smoother: greedy Viterbi without a transition matrix.

    Replaces runs shorter than ``min_duration`` with the preceding state.
    Useful as a fast, parameter-free smoother.

    Parameters
    ----------
    probs : np.ndarray
        Posterior probabilities, shape ``(T, K)``.
    min_duration : int
        Minimum run length.

    Returns
    -------
    np.ndarray
        Smoothed state sequence, shape ``(T,)``.
    """
    states = np.argmax(probs, axis=1)
    smoothed = states.copy()
    i = 0
    while i < len(states):
        j = i
        while j < len(states) and smoothed[j] == smoothed[i]:
            j += 1
        if j - i < min_duration and i > 0:
            smoothed[i:j] = smoothed[i - 1]
        i = j
    return smoothed


def viterbi_full(
    probs: np.ndarray,
    transition_matrix: np.ndarray | None = None,
    min_duration: int = 1,
) -> np.ndarray:
    """
    Full Viterbi decoding under a Markov chain.

    Parameters
    ----------
    probs : np.ndarray
        Emission probabilities, shape ``(T, K)``.
    transition_matrix : np.ndarray, optional
        Row-stochastic K-by-K transition matrix.  If None, a default with
        0.7 self-loop probability and uniform transitions to other states is
        used.
    min_duration : int
        Reserved (currently passed but not used; kept for signature
        compatibility with ``viterbi_smooth``).

    Returns
    -------
    np.ndarray
        MAP state sequence.
    """
    T, K = probs.shape
    if transition_matrix is None:
        transition_matrix = np.full((K, K), 0.3 / (K - 1))
        np.fill_diagonal(transition_matrix, 0.7)

    log_p = np.log(probs + 1e-10)
    log_t = np.log(transition_matrix + 1e-10)

    V = np.zeros((T, K))
    bp = np.zeros((T, K), dtype=int)
    V[0] = log_p[0]

    for t in range(1, T):
        for k in range(K):
            cand = V[t - 1] + log_t[:, k]
            bp[t, k] = np.argmax(cand)
            V[t, k] = cand[bp[t, k]] + log_p[t, k]

    states = np.zeros(T, dtype=int)
    states[T - 1] = np.argmax(V[T - 1])
    for t in range(T - 2, -1, -1):
        states[t] = bp[t + 1, states[t + 1]]
    return states


def estimate_transition_matrix(
    states: np.ndarray,
    n_states: int = 21,
    smoothing: float = 0.1,
) -> np.ndarray:
    """
    Empirical transition matrix from a state sequence with Laplace smoothing.

    Parameters
    ----------
    states : np.ndarray
        Discrete state sequence.
    n_states : int
        Total number of states.
    smoothing : float
        Additive smoothing constant per cell.

    Returns
    -------
    np.ndarray
        Row-stochastic ``(n_states, n_states)`` matrix.
    """
    counts = np.full((n_states, n_states), smoothing)
    for i in range(len(states) - 1):
        counts[int(states[i]), int(states[i + 1])] += 1
    return counts / counts.sum(axis=1, keepdims=True)


def fit_regime_model(
    X: np.ndarray,
    k: int = 21,
    min_duration: int = 3,
    random_state: int = 42,
) -> dict:
    """
    Convenience wrapper: scale, fit codebook, fit RF, smooth.

    Returns
    -------
    dict
        Keys: ``scaler``, ``kmeans``, ``rf``, ``probs``, ``viterbi``.
    """
    X_scaled, scaler = scale_features(X)
    km, labels = fit_codebook(X_scaled, k=k, random_state=random_state)
    rf, probs = fit_soft_classifier(X_scaled, labels, random_state=random_state)
    viterbi = viterbi_smooth(probs, min_duration=min_duration)
    return {
        "scaler": scaler,
        "kmeans": km,
        "rf": rf,
        "probs": probs,
        "viterbi": viterbi,
    }


def map_clusters_to_regimes(
    labels: np.ndarray,
    mapping: dict,
) -> np.ndarray:
    """Translate cluster IDs to regime names via a mapping dictionary."""
    return np.array([mapping.get(c, "Unknown") for c in labels])


def aggregate_probs_to_regimes(
    probs: np.ndarray,
    mapping: dict,
    regime_order: list[str],
) -> pd.DataFrame:
    """
    Aggregate per-cluster posteriors into per-regime posteriors.

    Parameters
    ----------
    probs : np.ndarray
        Cluster posteriors, shape ``(T, K)``.
    mapping : dict
        ``cluster_id -> regime_name``.
    regime_order : list[str]
        Output column order.

    Returns
    -------
    pd.DataFrame
        Row-stochastic per-regime probabilities, shape ``(T, len(regime_order))``.
    """
    n = probs.shape[0]
    df = pd.DataFrame(0.0, index=range(n), columns=regime_order)
    for cid in range(probs.shape[1]):
        regime = mapping.get(cid)
        if regime and regime in regime_order:
            df[regime] += probs[:, cid]
    return df.div(df.sum(axis=1) + 1e-10, axis=0)
