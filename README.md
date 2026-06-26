# When Drift Breaks

Companion code for the paper

> *When Drift Breaks: Particle-Based Real-Time Regime Detection*
> Lutz Plümer (2026), manuscript submitted for publication.

The repository contains the full pipeline used to produce the main result table of the paper, plus the data needed to reproduce it from a public source.

## What is in here

- `teiresias/` — Core Python package
  - `features.py` — distributional shape features
  - `loss_dynamics.py` — drawdown / time-in-drawdown / tail rate
  - `spectral.py` — cross-sectional eigenvalue features
  - `clustering.py` — k-means + Viterbi label generation
  - `observation.py` — core-distance (kernelized) observation model
  - `particle_filter.py` — Rao-Blackwellized particle filter
  - `regimes.py` — 7-regime taxonomy
  - `transitions.py` — HSMM transition matrix
  - `evaluation.py` — TPR / FPR / lead-time metrics
  - `data.py` — multi-source data loading
- `reproduce/`
  - `reproduce_main_results.py` — end-to-end pipeline
- `data/`
  - `yahoo/` — SPY, SKEW, 9 sector ETFs (Yahoo Finance)
- `figure1_precursors.pdf`, `figure2_separation.pdf` -- main paper figures

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Reproduce the headline table from the cached seed-hardening results (takes about 30 seconds):

```bash
python reproduce/reproduce_main_results.py --mode quick \
    --seed-results-dir data/seed_results
```

Regenerate the headline numbers from the *frozen* model -- this loads the
calibrated codebook and feature matrix, then re-runs the particle filter over
all seeds (a few minutes per seed):

```bash
python reproduce/reproduce_main_results.py --mode replay
```

This is the recommended way to verify the published figures: the clustering
layer is *loaded*, not re-estimated, so the result reproduces the paper to the
reported precision (FPR 2017 = 0.0, TPR-FPR ~ 41.8).

Rebuild everything from raw data (advanced; takes ~3.5 hours):

```bash
python reproduce/reproduce_main_results.py --mode full --source yahoo
```

> **Note.** `--mode full` re-fits the clustering layer from scratch on freshly
> downloaded data. Because the regime codebook is re-estimated, the absolute
> figures can differ from the published table -- small changes in the input
> series propagate through the k-means labelling. For an exact reproduction of
> the paper, use `--mode quick` (cached seed results) or `--mode replay`
> (frozen codebook). `--mode full` is provided for transparency of the full
> pipeline, not as the canonical reproduction path.

## Three nested feature configurations

The paper compares three nested models:

| Phase                  | Features | Data sources                  |
|------------------------|----------|-------------------------------|
| Shape                  | 57       | SPY                           |
| Shape+Skew             | 60       | SPY + CBOE Skew Index         |
| Shape+Skew+Spectral    | 63       | SPY + Skew + 9 SPDR sector ETFs |

Phase Shape captures distributional shape deformation and loss dynamics. Phase Shape+Skew adds implied tail-risk information from the options market. Phase Shape+Skew+Spectral adds cross-sectional spectral structure (absorption ratio, leading eigenvalue share) from the sector correlation matrix.

## Reproducibility

All data in `data/yahoo/` was downloaded from Yahoo Finance using `yfinance` with `auto_adjust=False` (preserving `Adj Close` separately). For the manuscript itself we used EODHD data; we verified that the spectral features that drive the regime-detection pipeline reproduce between the two sources with correlations above 0.9999, see `data/PROVENANCE.md`.

## License

Code: MIT (see `LICENSE`).

Data redistributed from Yahoo Finance under their terms of use.

## Citation

```bibtex
@unpublished{pluemer2026drift,
  author = {Plümer, Lutz},
  title  = {When Drift Breaks: Particle-Based Real-Time Regime Detection},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```
