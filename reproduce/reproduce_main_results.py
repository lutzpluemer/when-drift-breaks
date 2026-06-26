"""
reproduce_main_results.py
=========================

End-to-end reproduction script for the main result table of

    Plümer, L. (2026).  When Drift Breaks: Particle-Based Real-Time Regime
    Detection.

The script runs three nested feature configurations:

* **Shape**                -- distributional shape features + loss dynamics
                              (57 features, SPY only)
* **Shape+Skew**            -- adds the CBOE Skew Index             (60 features)
* **Shape+Skew+Spectral**   -- adds rolling spectral features on the
                              nine S&P sector ETFs                  (63 features)

Each configuration is fit on a 1993--2014 training window and evaluated
walk-forward on 2015-01-01 -- 2026-04-30.  The particle filter is run
with ``n_particles=2000`` over 10 random seeds (mode A: clustering held
fixed at ``random_state=42``, only PF seed varies).

Output is the main-result table (mean +/- std over 10 seeds).

Usage
-----

Local::

    python scripts/reproduce_main_results.py --data-dir ./data

Colab with Drive mounted::

    python scripts/reproduce_main_results.py \\
        --data-dir /content/drive/MyDrive/Teiresias/cache

Quick mode (load cached seed results, just print table)::

    python scripts/reproduce_main_results.py --mode quick \\
        --seed-results-dir ./data/seed_results

Force a specific data source::

    python scripts/reproduce_main_results.py --source yahoo
    python scripts/reproduce_main_results.py --source eodhd

Dependencies
------------

Core:    numpy, pandas, scipy, scikit-learn, pyarrow
Yahoo:   yfinance         (pip install yfinance)
EODHD:   requests + EODHD_API_KEY environment variable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the local ``teiresias`` package importable regardless of how this
# script is launched.  Running ``python reproduce/reproduce_main_results.py``
# puts the *script* directory (reproduce/) on sys.path, not the repo root,
# so ``import teiresias`` would fail.  Insert the repo root explicitly.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

TRAIN_END  = pd.Timestamp("2014-12-31")
TEST_START = pd.Timestamp("2015-01-01")
TEST_END   = pd.Timestamp("2026-04-30")

CRISES_TEST = [
    {"name": "china_2015",     "peak": "2015-08-24"},
    {"name": "covid",          "peak": "2020-03-23"},
    {"name": "inflation_2022", "peak": "2022-10-12"},
    {"name": "hormuz_2025",    "peak": "2025-06-22"},
]

STRESS_COLS = ["Stress", "Correction", "Bear", "Crisis"]

DEFAULT_SEEDS = [42, 7, 13, 99, 123, 256, 512, 1024, 2048, 4096]


# ---------------------------------------------------------------------------
# Map / inference helpers
# ---------------------------------------------------------------------------

def build_karte(features_train: pd.DataFrame, k: int = 21,
                random_state: int = 42) -> tuple[dict, dict]:
    """
    Fit a Nothung map: k-means codebook + scaler + cluster cores per regime.

    Mirrors the ``build_karte`` helper from the development notebook.  The
    clustering layer is held fixed (``random_state=42``) for the main
    result; this is the single calibrated step of the pipeline.
    """
    import teiresias as tei

    res = tei.fit_regime_model(
        features_train.values, k=k, random_state=random_state
    )
    X_scaled, _ = tei.scale_features(features_train.values, method="robust_tanh")
    viterbi = res["viterbi"]

    cluster_cores: dict[int, np.ndarray] = {}
    for regime_name in tei.REGIME_ORDER:
        cluster_ids = [c for c, r in tei.REGIME_MAP.items() if r == regime_name]
        centroids = []
        for cid in cluster_ids:
            mask = viterbi == cid
            if mask.sum() > 0:
                centroids.append(X_scaled[mask].mean(axis=0))
        if centroids:
            cluster_cores[tei.regime_index(regime_name)] = np.array(centroids)

    karte = {
        "cluster_cores": cluster_cores,
        "scaler_mean":   res["scaler"].center_,
        "scaler_scale":  res["scaler"].scale_,
        "feature_cols":  list(features_train.columns),
    }
    return karte, res


def run_pf(features_test: pd.DataFrame, karte: dict,
           n_particles: int = 2000, obs_sigma: float = 1.5,
           injection_fraction: float = 0.07, ess_threshold: float = 0.5,
           seed: int = 42) -> pd.DataFrame:
    """Run the Nothung particle filter over a test window."""
    import teiresias as tei

    np.random.seed(seed)
    nn_models, sm, ss, _ = tei.build_observation_model(karte, k_obs=2)
    pf = tei.NothungParticleFilter(
        n_particles=n_particles, n_regimes=7,
        injection_fraction=injection_fraction,
        ess_threshold=ess_threshold,
        obs_sigma=obs_sigma,
    )
    pf.initialize()

    rows = []
    for i, dt in enumerate(features_test.index):
        x_raw = features_test.iloc[i].values
        x_scaled = tei.scale_observation(x_raw, sm, ss)
        distances = tei.compute_distances(x_scaled, nn_models)
        regime_probs = pf.step(distances)
        h = pf.history[-1]

        row = {"date": dt}
        for i_r, r in enumerate(tei.REGIME_ORDER):
            row[r] = float(regime_probs[i_r])
        row["eta"]    = float(h["eta"])
        row["ess"]    = float(h["ess"])
        row["p_max"]  = float(regime_probs.max())
        row["p_warn"] = float(regime_probs[3] + regime_probs[4] + regime_probs[5])
        rows.append(row)

    return pd.DataFrame(rows).set_index("date")


def _crisis_metrics(pf_df: pd.DataFrame,
                    crises: list[dict] = CRISES_TEST) -> dict[str, float]:
    """
    Compute pre-peak (60d) maxima, global TPR/FPR over crisis windows
    (+/-90d business days), and the FPR for calendar-year 2017.
    """
    stress = pf_df[STRESS_COLS].sum(axis=1)
    out: dict[str, float] = {}

    for c in crises:
        peak = pd.Timestamp(c["peak"])
        win_start = peak - pd.Timedelta(days=60)
        if (stress.index >= win_start).any():
            out[f"prepeak_{c['name']}"] = float(stress.loc[win_start:peak].max())
        else:
            out[f"prepeak_{c['name']}"] = float("nan")

    # Global TPR / FPR over the union of crisis windows
    crisis_window = pd.Index([])
    for c in crises:
        peak = pd.Timestamp(c["peak"])
        crisis_window = crisis_window.union(
            pd.date_range(peak - pd.Timedelta(days=90),
                          peak + pd.Timedelta(days=90), freq="B")
        )
    is_crisis = stress.index.isin(crisis_window)
    y_true = is_crisis.astype(int)
    y_pred = (stress.values >= 0.30).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    out["fpr_pct"] = fp / (fp + tn) * 100 if (fp + tn) > 0 else float("nan")
    out["tpr_pct"] = tp / (tp + fn) * 100 if (tp + fn) > 0 else float("nan")
    out["tpr_minus_fpr"] = out["tpr_pct"] - out["fpr_pct"]

    # FPR restricted to calendar-year 2017
    m17 = (stress.index >= "2017-01-01") & (stress.index <= "2017-12-31")
    if m17.any():
        out["fpr_2017_pct"] = (stress[m17] >= 0.30).mean() * 100
    else:
        out["fpr_2017_pct"] = float("nan")

    return out


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_features_shape(spy: pd.Series) -> pd.DataFrame:
    """Master features + loss dynamics on SPY only (Phase Shape, 57 cols)."""
    import teiresias as tei
    from teiresias.loss_dynamics import compute_loss_dynamics

    log_ret = np.log(spy / spy.shift(1))
    f_master = tei.compute_all_features(spy, windows="master")
    f_loss   = compute_loss_dynamics(spy, log_returns=log_ret)
    return pd.concat([f_master, f_loss], axis=1).dropna()


def build_features_skew(spy: pd.Series, skew: pd.Series) -> pd.DataFrame:
    """Phase Shape+Skew (60 cols) -- adds three SKEW-based columns."""
    base = build_features_shape(spy)
    skew_aligned = skew.reindex(base.index).ffill()

    base = base.copy()
    base["skew"]        = skew_aligned
    base["skew_chg_5"]  = np.log(skew_aligned / skew_aligned.shift(5))
    base["skew_chg_20"] = np.log(skew_aligned / skew_aligned.shift(20))

    return base.dropna()


def build_features_spectral(spy: pd.Series, skew: pd.Series,
                            sectors: pd.DataFrame) -> pd.DataFrame:
    """Phase Shape+Skew+Spectral (63 cols) -- adds three spectral columns."""
    from teiresias.spectral import compute_absorption_features

    base = build_features_skew(spy, skew)
    sector_returns = np.log(sectors / sectors.shift(1))
    eig = compute_absorption_features(sector_returns, window=60, n_top=3)
    eig_aligned = eig.reindex(base.index).ffill()

    return pd.concat([base, eig_aligned], axis=1).dropna()


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------

def run_phase(name: str, features: pd.DataFrame,
              viterbi_anchor: np.ndarray | None,
              anchor_index: pd.DatetimeIndex | None,
              seeds: list[int],
              n_particles: int = 2000,
              verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Build the karte for one phase, run the PF over multiple seeds, and
    return (records DataFrame, karte).

    If ``viterbi_anchor`` is provided, training labels are taken from it
    instead of running k-means again.  This is used for phases Shape+Skew
    and Shape+Skew+Spectral so all phases share the same regime taxonomy.
    """
    import teiresias as tei

    train = features.loc[:TRAIN_END]
    test  = features.loc[TEST_START:TEST_END]

    # When a viterbi_anchor is supplied, intersect indexes so the anchor
    # labels align with the (possibly shorter) training set of this phase.
    if viterbi_anchor is not None and anchor_index is not None:
        common = anchor_index.intersection(train.index)
        train = train.loc[common]
        mask = anchor_index.isin(common)
        viterbi_local = viterbi_anchor[mask]
    else:
        viterbi_local = None

    if verbose:
        print(f"\n  Phase: {name}")
        print(f"    train: {train.shape},  test: {test.shape}")

    # Build karte (k-means seed=42, fixed across all seeds of this phase)
    if viterbi_local is None:
        karte, res = build_karte(train, k=21, random_state=42)
        anchor_out = res["viterbi"]
        anchor_idx_out = train.index
    else:
        # Fixed-label karte: cluster cores from the anchor labels
        X_scaled, _ = tei.scale_features(train.values, method="robust_tanh")
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler().fit(train.values)

        cluster_cores: dict[int, np.ndarray] = {}
        for regime_name in tei.REGIME_ORDER:
            cluster_ids = [c for c, r in tei.REGIME_MAP.items() if r == regime_name]
            centroids = []
            for cid in cluster_ids:
                mask_c = viterbi_local == cid
                if mask_c.sum() > 0:
                    centroids.append(X_scaled[mask_c].mean(axis=0))
            if centroids:
                cluster_cores[tei.regime_index(regime_name)] = np.array(centroids)

        karte = {
            "cluster_cores": cluster_cores,
            "scaler_mean":   scaler.center_,
            "scaler_scale":  scaler.scale_,
            "feature_cols":  list(train.columns),
        }
        anchor_out = viterbi_local
        anchor_idx_out = train.index

    # Run PF over all seeds
    records = []
    for seed in seeds:
        t0 = time.time()
        pf_df = run_pf(test, karte, n_particles=n_particles, seed=seed)
        m = _crisis_metrics(pf_df)
        m["seed"] = seed
        records.append(m)
        if verbose:
            print(
                f"    seed={seed:>5d}: "
                f"china={m['prepeak_china_2015']:.3f}  "
                f"covid={m['prepeak_covid']:.3f}  "
                f"infl={m['prepeak_inflation_2022']:.3f}  "
                f"hormuz={m['prepeak_hormuz_2025']:.3f}  "
                f"TPR-FPR={m['tpr_minus_fpr']:+.1f}  "
                f"FPR2017={m['fpr_2017_pct']:.1f}%  "
                f"({time.time()-t0:.0f}s)"
            )

    return pd.DataFrame(records), {
        "karte": karte,
        "anchor": anchor_out,
        "anchor_index": anchor_idx_out,
    }


# ---------------------------------------------------------------------------
# Pretty-print main results
# ---------------------------------------------------------------------------

def print_results_table(results: dict[str, pd.DataFrame]) -> None:
    """Pretty-print the headline mean-+/-std table to stdout."""
    print()
    print("=" * 130)
    print("MAIN RESULT TABLE (mean ± std over seeds, fixed clustering)")
    print("=" * 130)
    cols = ["prepeak_china_2015", "prepeak_covid", "prepeak_inflation_2022",
            "prepeak_hormuz_2025", "tpr_minus_fpr", "fpr_2017_pct"]
    headers = ["China 2015", "COVID 2020", "Infl 2022", "Hormuz 2025",
               "TPR-FPR", "FPR 2017"]

    print(f"\n{'Phase':<22s}  " + "  ".join(f"{h:>14s}" for h in headers))
    print("-" * 130)
    for phase_name, df in results.items():
        row = f"  {phase_name:<20s}"
        for c in cols:
            m = df[c].mean()
            s = df[c].std(ddof=0)
            if "fpr" in c or "tpr" in c:
                row += f"  {m:>6.1f}±{s:<5.1f}  "
            else:
                row += f"  {m:>6.3f}±{s:<5.3f}  "
        print(row)


# ---------------------------------------------------------------------------
# Quick mode: load cached seed results
# ---------------------------------------------------------------------------

def load_seed_results(seed_results_dir: str,
                      stamp: str = "20260505_1334") -> dict[str, pd.DataFrame]:
    """Load the three Parquet files of a saved seed-hardening run."""
    sdir = Path(seed_results_dir)
    paths = {
        "Shape":                sdir / f"{stamp}_shape_seeds.parquet",
        "Shape+Skew":           sdir / f"{stamp}_skew_seeds.parquet",
        "Shape+Skew+Spectral":  sdir / f"{stamp}_spectral_seeds.parquet",
    }
    out: dict[str, pd.DataFrame] = {}
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing seed-results file: {p}")
        out[name] = pd.read_parquet(p)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_replay(karte_path: str, features_path: str,
               seeds=DEFAULT_SEEDS, n_particles: int = 2000) -> dict[str, pd.DataFrame]:
    """
    Replay mode: reproduce the headline (full 63-feature) result from the
    *frozen* Nothung map and the *frozen* feature matrix, without re-fitting
    the clustering layer.  The particle filter is re-run from scratch over
    every seed, but the codebook is loaded rather than re-estimated, so the
    published numbers are reproduced exactly (up to PF seed variation).
    """
    import pickle

    with open(karte_path, "rb") as fh:
        karte = pickle.load(fh)

    feats = pd.read_parquet(features_path)
    feats.index = pd.to_datetime(feats.index)
    cols = karte["feature_cols"]
    missing = [c for c in cols if c not in feats.columns]
    if missing:
        raise KeyError(
            f"Feature matrix is missing {len(missing)} karte columns, "
            f"e.g. {missing[:5]}"
        )
    feats_test = feats.loc[TEST_START:TEST_END, cols]
    print(
        f"[replay] {len(cols)} features | "
        f"test {feats_test.index.min().date()}..{feats_test.index.max().date()} "
        f"({len(feats_test)} days) | {len(seeds)} seeds"
    )

    records = []
    for sd in seeds:
        pf_df = run_pf(feats_test, karte, n_particles=n_particles, seed=sd)
        m = _crisis_metrics(pf_df)
        m["seed"] = sd
        records.append(m)
        print(
            f"  seed {sd:>5d}: china {m['prepeak_china_2015']:.3f}  "
            f"covid {m['prepeak_covid']:.3f}  infl {m['prepeak_inflation_2022']:.3f}  "
            f"hormuz {m['prepeak_hormuz_2025']:.3f}  "
            f"TPR-FPR {m['tpr_minus_fpr']:.2f}  FPR2017 {m['fpr_2017_pct']:.2f}"
        )

    return {"Shape+Skew+Spectral (replay)": pd.DataFrame(records)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the main result table of the Teiresias paper."
    )
    parser.add_argument(
        "--mode", choices=["full", "quick", "replay"], default="full",
        help="full: run the full pipeline (hours).  quick: load cached "
             "seed results and just print the table (seconds).",
    )
    parser.add_argument(
        "--data-dir", default="./data",
        help="Directory holding cached Parquet files for SPY, SKEW, "
             "sectors.  In Colab pass e.g. /content/drive/MyDrive/Teiresias/cache",
    )
    parser.add_argument(
        "--seed-results-dir", default="./data/seed_results",
        help="Directory holding the saved seed-hardening Parquet files "
             "(used by --mode quick).",
    )
    parser.add_argument(
        "--seed-stamp", default="20260505_1334",
        help="Timestamp prefix of the seed-hardening run files.",
    )
    parser.add_argument(
        "--source", choices=["auto", "eodhd", "yahoo"], default="auto",
        help="Data source for the live fetch path.",
    )
    parser.add_argument(
        "--karte", default="./data/snapshot_mai/models/karte_D4.pkl",
        help="Frozen Nothung map (pickle), used by --mode replay.",
    )
    parser.add_argument(
        "--features", default="./data/snapshot_mai/features/features_D4.parquet",
        help="Frozen feature matrix (parquet), used by --mode replay.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="PF random seeds (default: 10 mixed seeds).",
    )
    parser.add_argument(
        "--n-particles", type=int, default=2000,
        help="Number of particles for the PF (default 2000).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="If set, write per-phase Parquet files of the seed records "
             "and a JSON summary.",
    )
    args = parser.parse_args(argv)

    # ---- Replay mode -------------------------------------------------
    if args.mode == "replay":
        print(f"[replay mode] karte    = {args.karte}")
        print(f"[replay mode] features = {args.features}")
        results = run_replay(args.karte, args.features,
                             seeds=args.seeds, n_particles=args.n_particles)
        print_results_table(results)
        return 0

    # ---- Quick mode --------------------------------------------------
    if args.mode == "quick":
        print(f"[quick mode] Loading seed results from {args.seed_results_dir}")
        results = load_seed_results(args.seed_results_dir, stamp=args.seed_stamp)
        for name, df in results.items():
            print(f"  ✓ {name:<22s} {df.shape}")
        print_results_table(results)
        return 0

    # ---- Full mode: load data ---------------------------------------
    from teiresias.data import load_spy, load_skew, load_sectors

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[full mode] Data directory: {data_dir.resolve()}")
    print(f"[full mode] Source: {args.source}")

    print("\n  Loading SPY...")
    spy = load_spy(
        cache_path=str(data_dir / "spy.parquet"),
        source=args.source,
    )
    print(f"    SPY: {len(spy)} days, {spy.index.min().date()} to {spy.index.max().date()}")

    print("  Loading SKEW...")
    skew = load_skew(
        cache_path=str(data_dir / "skew.parquet"),
        source=args.source,
    )
    print(f"    SKEW: {len(skew)} days, {skew.index.min().date()} to {skew.index.max().date()}")

    print("  Loading sector ETFs...")
    sectors = load_sectors(
        cache_path=str(data_dir / "sectors.parquet"),
        source=args.source,
    )
    print(f"    Sectors: {sectors.shape}, {sectors.index.min().date()} to {sectors.index.max().date()}")

    # ---- Build features ---------------------------------------------
    print("\n  Building feature matrices...")
    feat_shape    = build_features_shape(spy)
    feat_skew     = build_features_skew(spy, skew)
    feat_spectral = build_features_spectral(spy, skew, sectors)
    print(f"    Shape:               {feat_shape.shape}")
    print(f"    Shape+Skew:          {feat_skew.shape}")
    print(f"    Shape+Skew+Spectral: {feat_spectral.shape}")

    # ---- Phase Shape (anchor karte) ---------------------------------
    print("\n" + "=" * 78)
    print("PHASE 1/3: Shape")
    print("=" * 78)
    df_shape, ph_shape = run_phase(
        "Shape", feat_shape,
        viterbi_anchor=None, anchor_index=None,
        seeds=args.seeds, n_particles=args.n_particles,
    )
    viterbi_anchor = ph_shape["anchor"]
    anchor_index   = ph_shape["anchor_index"]

    # ---- Phase Shape+Skew -------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 2/3: Shape+Skew")
    print("=" * 78)
    df_skew, _ = run_phase(
        "Shape+Skew", feat_skew,
        viterbi_anchor=viterbi_anchor, anchor_index=anchor_index,
        seeds=args.seeds, n_particles=args.n_particles,
    )

    # ---- Phase Shape+Skew+Spectral ----------------------------------
    print("\n" + "=" * 78)
    print("PHASE 3/3: Shape+Skew+Spectral")
    print("=" * 78)
    df_spectral, _ = run_phase(
        "Shape+Skew+Spectral", feat_spectral,
        viterbi_anchor=viterbi_anchor, anchor_index=anchor_index,
        seeds=args.seeds, n_particles=args.n_particles,
    )

    # ---- Results table ----------------------------------------------
    results = {
        "Shape":               df_shape,
        "Shape+Skew":          df_skew,
        "Shape+Skew+Spectral": df_spectral,
    }
    print_results_table(results)

    # ---- Optional save ----------------------------------------------
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        df_shape.to_parquet(out / f"{stamp}_shape_seeds.parquet")
        df_skew.to_parquet(out / f"{stamp}_skew_seeds.parquet")
        df_spectral.to_parquet(out / f"{stamp}_spectral_seeds.parquet")

        meta = {
            "timestamp":   stamp,
            "n_particles": args.n_particles,
            "seeds":       args.seeds,
            "source":      args.source,
            "shape":               df_shape.to_dict("records"),
            "shape_skew":          df_skew.to_dict("records"),
            "shape_skew_spectral": df_spectral.to_dict("records"),
        }
        with open(out / f"{stamp}_FINAL.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"\nResults written to {out.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
