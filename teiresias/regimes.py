"""
teiresias.regimes
=================

Regime taxonomy and the mapping from k-means clusters to interpretable
regime labels.

The seven regimes describe market states by both volatility level and
direction:

* **Bull**       — calm, positive-drift growth phase.
* **Sideways**   — low-volatility ranging.
* **Correction** — short, moderate drawdown without panic.
* **Stress**     — elevated volatility, no clear direction.
* **Bear**       — sustained negative drift.
* **Crisis**     — panic, extreme volatility, deep drawdown.
* **Recovery**   — post-crisis rebound.

The mapping from 21 clusters to these 7 regimes is the result of expert
analysis of the cluster centroids in the Master-Setup pipeline; it should
be regarded as a calibrated prior that can be revisited if the codebook is
re-trained on different data or features.
"""

from __future__ import annotations

# Cluster-id -> regime-name. 21 clusters, 7 regimes.
REGIME_MAP: dict[int, str] = {
    0: "Bull",       6: "Bull",       15: "Bull",
    2: "Sideways",   3: "Sideways",   10: "Sideways",
    5: "Stress",    11: "Stress",    18: "Stress",   19: "Stress",
    1: "Correction", 13: "Correction",
    4: "Bear",      14: "Bear",      16: "Bear",     20: "Bear",
    7: "Crisis",    12: "Crisis",    17: "Crisis",
    8: "Recovery",   9: "Recovery",
}

REGIME_ORDER: list[str] = [
    "Bull", "Sideways", "Correction", "Stress", "Bear", "Crisis", "Recovery"
]

REGIME_COLORS: dict[str, str] = {
    "Bull":       "#27ae60",
    "Sideways":   "#85c1e9",
    "Correction": "#a569bd",
    "Stress":     "#f1c40f",
    "Bear":       "#e67e22",
    "Crisis":     "#c0392b",
    "Recovery":   "#1abc9c",
}

# Coarser groupings used by the alarm/diagnostic logic
SAFE_REGIMES:    list[str] = ["Bull", "Sideways"]
WARNING_REGIMES: list[str] = ["Stress", "Correction"]
DANGER_REGIMES:  list[str] = ["Bear", "Crisis"]

N_REGIMES: int = 7
N_CLUSTERS: int = 21


def regime_for_cluster(cluster_id: int) -> str:
    """Look up the regime label for a cluster id, with explicit error."""
    if cluster_id not in REGIME_MAP:
        raise KeyError(
            f"Cluster id {cluster_id} not in REGIME_MAP. "
            f"Known clusters: {sorted(REGIME_MAP)}"
        )
    return REGIME_MAP[cluster_id]


def regime_index(regime_name: str) -> int:
    """Return the integer index of a regime in REGIME_ORDER."""
    if regime_name not in REGIME_ORDER:
        raise ValueError(
            f"Unknown regime {regime_name!r}. Known: {REGIME_ORDER}"
        )
    return REGIME_ORDER.index(regime_name)
