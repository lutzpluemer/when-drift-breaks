# Data Provenance

This file documents the origin, coverage, and processing of the data series in
this repository, and the relationship between the two data sources involved.

## Two sources

- **EODHD** (eodhd.com) was used for the figures and tables in the manuscript.
  EODHD data is licensed and is *not* redistributed here.
- **Yahoo Finance**, accessed through the `yfinance` package, is the open
  source shipped in `data/yahoo/`. It lets anyone reproduce the headline
  results without an EODHD subscription.

We verified that the two sources agree on the quantities that actually drive
the regime-detection pipeline: the cross-sectional spectral features
(absorption ratio and leading-eigenvalue share) computed from the two inputs
over their overlapping window have a Pearson correlation above 0.9999. The
published numbers are therefore reproducible from the open Yahoo data.

## Instruments

| Symbol | Instrument | Role |
|--------|------------|------|
| SPY | SPDR S&P 500 ETF | benchmark / shape features |
| SKEW | CBOE SKEW Index | implied tail-risk feature |
| XLB | Materials Select Sector SPDR | sector / spectral |
| XLE | Energy Select Sector SPDR | sector / spectral |
| XLF | Financials Select Sector SPDR | sector / spectral |
| XLI | Industrials Select Sector SPDR | sector / spectral |
| XLK | Technology Select Sector SPDR | sector / spectral |
| XLP | Consumer Staples Select Sector SPDR | sector / spectral |
| XLU | Utilities Select Sector SPDR | sector / spectral |
| XLV | Health Care Select Sector SPDR | sector / spectral |
| XLY | Consumer Discretionary Select Sector SPDR | sector / spectral |

## Download settings (Yahoo)

All Yahoo series were downloaded with `yfinance` using `auto_adjust=False`, so
that the raw `Close` and the dividend/split-adjusted `Adj Close` are preserved
as separate columns. Returns and all downstream features are computed from the
adjusted series.

## Windows

- Training window: up to and including 2014-12-31 (codebook calibration).
- Out-of-sample test window: 2015-01-01 -- 2026-04-30.

SPY-only shape features extend back to SPY's 1993 inception; the full 63-feature
spectral matrix begins in 1999, limited by the December 1998 launch of the
Select Sector SPDR ETFs. The clustering layer is calibrated once on the training
window and then frozen (see `--mode replay`).

## Frozen artifacts

The exact codebook and feature matrix behind the published table are committed
under `data/snapshot_mai/`:

- `models/karte_D4.pkl` -- frozen Nothung map (cluster cores, scaler, feature columns).
- `models/viterbi_anchor.pkl` -- anchor Viterbi labelling.
- `features/features_D4.parquet` -- 6807 x 63 feature matrix, 1999-04-07 to 2026-04-29.
- `pf_results/pf_D4.parquet` -- particle-filter output.

The seed-hardening results behind the published mean +/- std table are in
`data/seed_results/` (run stamp `20260505_1334`, ten PF seeds).

## Redistribution

Code in this repository is released under the MIT license (see `LICENSE`).
Market data redistributed in `data/yahoo/` originates from Yahoo Finance and
remains subject to Yahoo's terms of use; it is included solely to enable
reproduction of the published results.
