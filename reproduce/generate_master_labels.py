"""
generate_master_labels.py
=========================
Erzeugt master_labels.parquet — die kanonische Diagnose-Tabelle pro Test-Tag.

Spalten:
    date                       (DatetimeIndex)
    cluster_id                 RF-Top-Cluster (0..20)
    cluster_regime             Aggregat-Regime des Top-Clusters
    rf_top_regime              RF posterior top regime
    rf_top_prob                RF posterior top probability
    rf_stress_prob             Sum p(Stress|Correction|Bear|Crisis)
    viterbi_test_regime        Viterbi-Decoding auf Test (für Diagnose)
    pf_top_regime              PF top regime
    pf_top_prob                PF top probability
    pf_stress_prob             Sum aggregated stress states (PF)
    pf_eta                     PF intensity scalar
    pf_ess                     PF effective sample size

Plus annotated crisis windows (4 Test-Krisen + Dotcom/Lehman als illustrativ).

Aufruf:
    python generate_master_labels.py \
        --snapshot /path/to/20260505_0732_phaseD4_eigenvalue_success \
        --output /path/to/master_labels.parquet
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLUSTER_TO_REGIME_IDX = {
    0: 0, 6: 0, 15: 0,
    2: 1, 3: 1, 10: 1,
    1: 2, 13: 2,
    5: 3, 11: 3, 18: 3, 19: 3,
    4: 4, 14: 4, 16: 4, 20: 4,
    7: 5, 12: 5, 17: 5,
    8: 6, 9: 6,
}
REGIME_NAMES = [
    "Bull", "Sideways", "Correction", "Stress",
    "Bear", "Crisis", "Recovery",
]
STRESS_REGIMES = {"Correction", "Stress", "Bear", "Crisis"}

# Reference crises
TEST_CRISES = [
    ("COVID 2020",     "2020-03-23"),
    ("Inflation 2022", "2022-10-12"),
    ("Hormuz 2025",    "2025-06-22"),
]
ILLUSTRATIVE_CRISES = [
    ("Dotcom 2002",    "2002-10-09"),
    ("Lehman 2009",    "2009-03-09"),
]


def viterbi_decode(emission_probs, transition_log, init_log):
    """Standard Viterbi over cluster-emission posteriors."""
    T = len(emission_probs)
    K = transition_log.shape[0]
    log_emit = np.log(np.maximum(emission_probs, 1e-12))
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = init_log + log_emit[0]
    for t in range(1, T):
        for j in range(K):
            scores = delta[t-1] + transition_log[:, j]
            psi[t, j] = scores.argmax()
            delta[t, j] = scores.max() + log_emit[t, j]
    path = np.zeros(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True,
                        help="Path to Mai snapshot directory")
    parser.add_argument("--output", required=True,
                        help="Output parquet path for master labels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    snapshot = Path(args.snapshot)
    out_path = Path(args.output)

    # Load Mai snapshot
    karte_path = snapshot / "models" / "karte_D4.pkl"
    viterbi_path = snapshot / "models" / "viterbi_anchor.pkl"
    features_path = snapshot / "features" / "features_D4.parquet"
    pf_path = snapshot / "pf_results" / "pf_D4.parquet"

    for p in [karte_path, viterbi_path, features_path, pf_path]:
        if not p.exists():
            print(f"FEHLT: {p}", file=sys.stderr)
            sys.exit(1)

    with open(karte_path, "rb") as f:
        karte = pickle.load(f)
    with open(viterbi_path, "rb") as f:
        viterbi_anchor = pickle.load(f)
    features = pd.read_parquet(features_path)
    pf = pd.read_parquet(pf_path)

    print(f"Features:       {features.shape}")
    print(f"PF output:      {pf.shape}")
    print(f"viterbi_anchor: {viterbi_anchor.shape}")

    # Training-Set für RF
    TRAIN_END = pd.Timestamp("2014-12-31")
    train_features = features.loc[:TRAIN_END]
    viterbi_d4 = viterbi_anchor[-len(train_features):]
    scaler_mean = np.asarray(karte["scaler_mean"])
    scaler_scale = np.asarray(karte["scaler_scale"])
    X_train = np.tanh((train_features.values - scaler_mean) / scaler_scale)

    print("\nTraining RF (n_estimators=500, seed={})...".format(args.seed))
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, random_state=args.seed, n_jobs=-1,
    )
    rf.fit(X_train, viterbi_d4)
    print(f"  Train accuracy: {rf.score(X_train, viterbi_d4):.3f}")

    # Test-Set: 2015-01-01 onwards
    TEST_START = pd.Timestamp("2015-01-01")
    test_features = features.loc[TEST_START:]
    X_test = np.tanh((test_features.values - scaler_mean) / scaler_scale)

    # RF on test
    print("\nRF inference on test set...")
    cluster_probs = rf.predict_proba(X_test)  # (n_test, 21)
    n_test = len(test_features)

    # Aggregate cluster probs → regime probs
    regime_probs = np.zeros((n_test, 7))
    for ci, cid in enumerate(rf.classes_):
        regime_probs[:, CLUSTER_TO_REGIME_IDX[int(cid)]] += cluster_probs[:, ci]
    rf_top_regime_idx = regime_probs.argmax(axis=1)
    rf_top_prob = regime_probs.max(axis=1)
    rf_stress_prob = regime_probs[:, [2, 3, 4, 5]].sum(axis=1)  # Correction, Stress, Bear, Crisis

    # Cluster top
    cluster_top_idx = cluster_probs.argmax(axis=1)
    cluster_top_id = rf.classes_[cluster_top_idx]
    cluster_regime_idx = np.array([CLUSTER_TO_REGIME_IDX[int(c)] for c in cluster_top_id])

    # Viterbi on test (diagnostic only — not in pipeline)
    print("\nViterbi decoding on test set (diagnostic)...")
    n_clusters = 21
    trans = np.zeros((n_clusters, n_clusters)) + 1.0
    for i in range(len(viterbi_d4) - 1):
        trans[viterbi_d4[i], viterbi_d4[i+1]] += 1
    trans = trans / trans.sum(axis=1, keepdims=True)
    init = np.full(n_clusters, 1.0 / n_clusters)
    viterbi_test = viterbi_decode(cluster_probs, np.log(trans), np.log(init))
    viterbi_test_regime_idx = np.array([CLUSTER_TO_REGIME_IDX[int(c)] for c in viterbi_test])

    # PF top regime (from pf_D4.parquet — alignment)
    pf_aligned = pf.reindex(test_features.index)
    pf_regime_arr = pf_aligned[REGIME_NAMES].values
    pf_top_regime_idx = pf_regime_arr.argmax(axis=1)
    pf_top_prob = pf_regime_arr.max(axis=1)
    pf_stress_prob = pf_aligned[["Stress", "Correction", "Bear", "Crisis"]].sum(axis=1).values

    # Assemble
    master = pd.DataFrame({
        "cluster_id":           cluster_top_id,
        "cluster_regime":       [REGIME_NAMES[i] for i in cluster_regime_idx],
        "rf_top_regime":        [REGIME_NAMES[i] for i in rf_top_regime_idx],
        "rf_top_prob":          rf_top_prob,
        "rf_stress_prob":       rf_stress_prob,
        "viterbi_test_regime":  [REGIME_NAMES[i] for i in viterbi_test_regime_idx],
        "pf_top_regime":        [REGIME_NAMES[i] for i in pf_top_regime_idx],
        "pf_top_prob":          pf_top_prob,
        "pf_stress_prob":       pf_stress_prob,
        "pf_eta":               pf_aligned["eta"].values,
        "pf_ess":               pf_aligned["ess"].values,
    }, index=test_features.index)
    master.index.name = "date"

    # Crisis annotations
    crisis_anno = pd.Series("", index=master.index, dtype=object)
    for name, peak in TEST_CRISES + ILLUSTRATIVE_CRISES:
        p = pd.Timestamp(peak)
        win = pd.date_range(p - pd.Timedelta(days=60), p + pd.Timedelta(days=30), freq="B")
        present = master.index.isin(win)
        for d in master.index[present]:
            if crisis_anno.loc[d]:
                crisis_anno.loc[d] = f"{crisis_anno.loc[d]}; {name}"
            else:
                crisis_anno.loc[d] = name
    master["crisis_window"] = crisis_anno

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_parquet(out_path)
    print(f"\n✓ Saved: {out_path}")
    print(f"  Rows: {len(master)}")
    print(f"  Date range: {master.index.min().date()} to {master.index.max().date()}")

    # Manifest
    manifest = {
        "purpose": "Diagnostic master-labels table per test day for Paper 1 (canonical Mai pipeline)",
        "rows": int(len(master)),
        "columns": list(master.columns),
        "n_features": int(features.shape[1]),
        "rf": {
            "n_estimators": 500,
            "max_depth": None,
            "seed": args.seed,
            "train_accuracy": float(rf.score(X_train, viterbi_d4)),
            "n_train": int(len(X_train)),
        },
        "test_range": [
            str(master.index.min().date()),
            str(master.index.max().date()),
        ],
        "crises_test": [{"name": n, "trough": d} for n, d in TEST_CRISES],
        "crises_illustrative": [{"name": n, "trough": d} for n, d in ILLUSTRATIVE_CRISES],
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
