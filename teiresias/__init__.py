"""
teiresias
=========

Particle-based real-time regime detection in financial time series.

This package implements the pipeline described in *When Drift Breaks:
Particle-Based Real-Time Regime Detection* (Plümer 2026).  It has three layers:

1.  **Map construction** -- multi-window distributional shape features
    plus k-means codebook plus random-forest soft classifier plus Viterbi
    smoothing.  See :mod:`teiresias.features`, :mod:`teiresias.clustering`,
    :mod:`teiresias.regimes`.

2.  **Particle-filter inference** -- HSMM-driven sequential Monte Carlo over
    (regime, intensity, duration) triples, with informed injection.  See
    :mod:`teiresias.particle_filter`, :mod:`teiresias.transitions`,
    :mod:`teiresias.observation`.

3.  **Evaluation and persistence** -- separability, lead-time, save/load.
    See :mod:`teiresias.evaluation`, :mod:`teiresias.persistence`.

A typical end-to-end workflow lives in ``notebooks/00_reproduce_results.ipynb``.

Citation
--------
Plümer, L. (2026). When Drift Breaks: Particle-Based Real-Time Regime
Detection.

Code repository: https://github.com/lutzpluemer/when-drift-breaks
"""

from __future__ import annotations

__version__ = "1.0.0"

# --- Layer 1: features ---
from .features import (
    compute_log_returns,
    compute_all_features,
    test_gaussianity,
    WINDOW_PRESETS,
)

# --- Layer 1: clustering and Viterbi ---
from .clustering import (
    scale_features,
    fit_codebook,
    fit_soft_classifier,
    viterbi_smooth,
    viterbi_full,
    estimate_transition_matrix,
    fit_regime_model,
    map_clusters_to_regimes,
    aggregate_probs_to_regimes,
)

# --- Layer 1: regimes ---
from .regimes import (
    REGIME_MAP,
    REGIME_ORDER,
    REGIME_COLORS,
    SAFE_REGIMES,
    WARNING_REGIMES,
    DANGER_REGIMES,
    N_REGIMES,
    N_CLUSTERS,
    regime_for_cluster,
    regime_index,
)

# --- Layer 2: HSMM transitions ---
from .transitions import (
    DURATION_PARAMS,
    TRANS_BASE,
    hazard_rate,
    get_transition_probs,
)

# --- Layer 2: observation model ---
from .observation import (
    build_observation_model,
    scale_observation,
    compute_distances,
)

# --- Layer 2: particle filter ---
from .particle_filter import NothungParticleFilter

# --- Layer 3: evaluation and persistence ---
from .evaluation import (
    separability_scan,
    compute_lead_time_soft,
    transition_analysis,
)
from .persistence import (
    save_nothung_state,
    load_nothung_state,
    save_nothung_map,
    load_nothung_map,
)


__all__ = [
    "__version__",
    # features
    "compute_log_returns", "compute_all_features", "test_gaussianity",
    "WINDOW_PRESETS",
    # clustering
    "scale_features", "fit_codebook", "fit_soft_classifier",
    "viterbi_smooth", "viterbi_full", "estimate_transition_matrix",
    "fit_regime_model", "map_clusters_to_regimes",
    "aggregate_probs_to_regimes",
    # regimes
    "REGIME_MAP", "REGIME_ORDER", "REGIME_COLORS",
    "SAFE_REGIMES", "WARNING_REGIMES", "DANGER_REGIMES",
    "N_REGIMES", "N_CLUSTERS",
    "regime_for_cluster", "regime_index",
    # transitions
    "DURATION_PARAMS", "TRANS_BASE", "hazard_rate", "get_transition_probs",
    # observation
    "build_observation_model", "scale_observation", "compute_distances",
    # particle filter
    "NothungParticleFilter",
    # evaluation
    "separability_scan", "compute_lead_time_soft", "transition_analysis",
    # persistence
    "save_nothung_state", "load_nothung_state",
    "save_nothung_map", "load_nothung_map",
]
