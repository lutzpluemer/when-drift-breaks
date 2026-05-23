<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/e1d1f940-74c6-453f-bc26-7f6339559667" /><img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/d34ae1b2-94b9-422e-899c-359262572635" /><img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/00c00945-4f17-4893-9a06-6d1613446fab" /># When Drift Breaks

Companion code for the paper

> *When Drift Breaks: Particle-Based Regime Inference for Crash Detection*
> Lutz Plümer (2026), manuscript under preparation.

The repository contains the full pipeline used to produce the main result table of the paper, plus the data needed to reproduce it from a public source.

## What is in here

## What is in here

- `teiresias/` — Core Python package
  - `features.py` — distributional shape features
  - `loss_dynamics.py` — drawdown / time-in-drawdown / tail rate
  - `spectral.py` — cross-sectional eigenvalue features
  - `clustering.py` — k-means + Viterbi label generation
  - `observation.py` — k-NN observation model
  - `particle_filter.py` — Rao-Blackwellized particle filter
  - `regimes.py` — 7-regime taxonomy
  - `transitions.py` — HSMM transition matrix
  - `evaluation.py` — TPR / FPR / lead-time metrics
  - `data.py` — multi-source data loading
- `reproduce/`
  - `reproduce_main_results.py` — end-to-end pipeline
- `data/`
  - `yahoo/` — SPY, SKEW, 9 sector ETFs (Yahoo Finance)
- `pyproject.toml`

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

Run the full pipeline from raw data (takes ~3.5 hours):

```bash
python reproduce/reproduce_main_results.py --mode full --source yahoo
```

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

Code: MIT (to be added).

Data redistributed from Yahoo Finance under their terms of use.

## Citation

```bibtex
@unpublished{pluemer2026drift,
  author = {Plümer, Lutz},
  title  = {When Drift Breaks: Particle-Based Regime Inference for Crash Detection},
  year   = {2026},
  note   = {Manuscript under preparation}
}
```
