"""
teiresias.observation
=====================

The observation model: how a current feature vector relates to each regime.

For every regime ``r`` the map stores a small set of cluster cores -- the
KMeans centroids of the clusters that belong to that regime (e.g. for
"Bull" the centroids of clusters 0, 6, 15).  At inference time we measure
the distance of the current feature vector to its k nearest cores in each
regime.  The median of those k distances is used as the per-regime distance
``delta_r``, fed downstream into the particle filter's ``update`` step.

Why k-NN and not a single centroid?  Regimes are *unions* of clusters by
construction (21 clusters distributed over 7 regimes).  A single regime
mean would be a poor summary of a multi-modal cloud.  Median-of-k preserves
robustness while not collapsing structure.

The observation model is built once from a fitted Nothung map and then
stays constant for the lifetime of the filter.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_observation_model(
    karte: dict,
    k_obs: int = 7,
) -> tuple[dict[int, NearestNeighbors], np.ndarray, np.ndarray, list[str]]:
    """
    Construct per-regime k-NN models from a fitted Nothung map.

    Parameters
    ----------
    karte : dict
        A Nothung map dictionary.  Required keys:

        * ``cluster_cores``: dict ``{regime_id: ndarray of centroids}``
        * ``scaler_mean``:   ndarray, shape ``(n_features,)``
        * ``scaler_scale``:  ndarray, shape ``(n_features,)``
        * ``feature_cols``:  list of feature column names

    k_obs : int
        Number of nearest neighbours per regime.  If a regime has fewer than
        ``k_obs+1`` cores, this is reduced to the available count.

    Returns
    -------
    nn_models : dict
        ``{regime_id: NearestNeighbors}``.
    scaler_mean : np.ndarray
    scaler_scale : np.ndarray
    feature_cols : list[str]
    """
    nn_models: dict[int, NearestNeighbors] = {}
    for regime_id, cores in karte["cluster_cores"].items():
        cores = np.asarray(cores)
        k = min(k_obs, max(1, len(cores) - 1))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(cores)
        nn_models[regime_id] = nn

    return (
        nn_models,
        np.asarray(karte["scaler_mean"]),
        np.asarray(karte["scaler_scale"]),
        list(karte["feature_cols"]),
    )


def scale_observation(
    x_raw: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    method: str = "robust_tanh",
) -> np.ndarray:
    """
    Apply the same scaling pipeline used during map construction.

    Parameters
    ----------
    x_raw : np.ndarray
        Raw feature vector, shape ``(n_features,)``.
    scaler_mean, scaler_scale : np.ndarray
        Centring and scaling vectors from the map.
    method : {'robust_tanh', 'robust'}
        Must match the method used in ``scale_features`` at training time.
        ``'robust_tanh'`` applies tanh after the RobustScaler step;
        ``'robust'`` applies only the RobustScaler.  The default
        ``'robust_tanh'`` matches the default in ``fit_regime_model`` and
        the example notebook.

    Returns
    -------
    np.ndarray
        Scaled feature vector, shape ``(n_features,)``.

    Notes
    -----
    The training-time and inference-time scaling pipelines must agree.  If
    you fit with ``method='robust_tanh'`` (the default), you must also pass
    ``method='robust_tanh'`` here -- otherwise distances measured against
    the cluster cores in the map are computed in different feature spaces,
    and inference is silently wrong.
    """
    if method not in ("robust_tanh", "robust"):
        raise ValueError(
            f"Unknown method {method!r}. Use 'robust_tanh' or 'robust'."
        )
    x = (x_raw - scaler_mean) / scaler_scale
    if method == "robust_tanh":
        x = np.tanh(x)
    return x


def compute_distances(
    x_scaled: np.ndarray,
    nn_models: dict[int, NearestNeighbors],
) -> dict[int, float]:
    """
    Per-regime median distance from the current feature vector.

    Parameters
    ----------
    x_scaled : np.ndarray
        Scaled feature vector, shape ``(n_features,)``.
    nn_models : dict
        Output of ``build_observation_model``.

    Returns
    -------
    dict
        ``{regime_id: median_distance_to_k_cores}``.
    """
    x_2d = x_scaled.reshape(1, -1)
    distances: dict[int, float] = {}
    for regime_id, nn in nn_models.items():
        dists, _ = nn.kneighbors(x_2d)
        distances[regime_id] = float(np.median(dists[0]))
    return distances
