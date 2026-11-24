# Algebra of Personality Types (ARC)

Reproducible analytical, Monte Carlo, and empirical code accompanying the study on the algebraic and geometric emergence of Big Five personality types (Resilient, Overcontrolled, Undercontrolled — ARC — and Gerlach et al., 2018 density-enrichment types) under unimodal correlated trait distributions.

## Overview

This repository provides:

- **Core library (`src/arc_study/`)**:
  - `matrices.py`: Observed meta-analytic Big Five correlation matrices (van der Linden et al., 2010; Busseri & Erb, 2024) in strict `(N, E, O, A, C)` ordering (`N` unreversed) and SPD validation.
  - `theory.py`: Closed-form spectral geometry, non-collinearity threshold ($\lambda_2 / \lambda_1 > 0.50764$), and density-ratio identities.
  - `generators.py`: Reproducible null data-generating processes (Null 1: isotropic normal / column-wise empirical permutation; Null 2: correlated multivariate normal $\mathcal{N}_5(\mathbf{0}, \mathbf{R})$ with optional Inverse-Wishart SPD heterogeneity; Null 3: Pearson-calibrated Gaussian copula with empirical or skewed marginals and KDE joint unimodality diagnostics).
  - `clustering.py`: Ward + *k*-means ($K=3$), diagonal Latent Profile Analysis (LPA), full-covariance Gaussian Mixture Models (GMM), and the Gerlach et al. (2018) GMM + KDE density-enrichment procedure (supporting fixed-bandwidth isotropic and Scott KDEs with mandatory $p_{\min} < \alpha$ validation).
  - `metrics.py`: Optimal centroid matching (exhaustive / Hungarian), Cronbach–Gleser $D^2$ decomposition (elevation, scatter, shape), Tucker congruence ($\phi$), Asendorpf split-half double cross-validation ($\kappa$, ARI, NMI), and leak-free out-of-sample *k*-fold incremental predictive validity.
- **Simulation, empirical, and verification scripts (`scripts/`)**:
  - `run_montecarlo.py`: Full factorial Monte Carlo study (7 null conditions $\times$ $N \in \{300, 1{,}000, 5{,}000, 50{,}000\}$, 1,000 replicates/block).
  - `run_empirical_comparison.py`: Empirical vs. Null 1 / Null 2 / Null 3 contrast pipeline for any Big Five CSV dataset.
  - `run_gerlach_addon.py` & `run_gerlach_faithful.py`: Corrected multi-null Gerlach et al. (2018) density-enrichment add-on and faithful $K=13$ replication against progressively stricter unimodal nulls (Permutation, MVN, Gaussian copula, Student-$t$ copula).
  - `run_r_sufficiency_cross_instrument.py`, `run_r_sufficiency_2x2_ward.py`, `run_r_sufficiency_2x2_multistart.py`: Causal sufficiency tests of the correlation matrix $\mathbf{R}$ for cross-instrument ARC replicability.
  - `run_soto2019_predictive.py`: Out-of-sample incremental predictive validity on Soto (2019) LOOPR ($N = 6{,}126$).
  - `run_final_via_a_checks.py`, `run_p0_audit_fixes.py`, `run_v5_final_checks.py`: Multi-start Mean-Shift topography, positive-control power checks, orthogonal-rotation vs. whitening representation experiments, and final audit verifications.
  - `build_figures_and_tables.py`, `build_multidataset_figures.py`, `build_via_a_figures.py`: Generation of publication-ready `.csv`/`.tex` tables, `.pdf`/`.svg` vector figures, and audit summaries.
- **Unit tests (`tests/`)**: Automated verification of analytic control values, matrix generators, profile metrics, and Gerlach procedure guards.
- **Lightweight results (`results/`)**: Summary tables (`.csv`, `.tex`), JSON summaries, Markdown audit reports, publication vector figures (`results/figures/`), and `results/requirements_lock.txt`.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite to verify the installation:

```bash
pytest -v
```

## Methodological & Reproducibility Standards

- **Strict trait order**: `('N', 'E', 'O', 'A', 'C')` with Neuroticism (`N`) unreversed throughout the entire codebase.
- **Observed correlations only**: No disattenuated correlation matrices are used to simulate observed scores.
- **Deterministic RNG**: Master seed `20261121` routed through `numpy.random.SeedSequence` across all parallel workers.
- Exact package versions used to produce the reported results are frozen in `results/requirements_lock.txt`.

## Data Availability

Raw and processed empirical datasets are excluded from this repository due to file size and third-party licensing. To reproduce the empirical scripts, place the publicly available datasets under `data/raw/` and run the preparation scripts:

1. **Spanish BFI-2 Study 1** ($N = 1{,}146$): Place processed domain scores at `data/processed/bfi2_study1_domains.csv` with columns `N,E,O,A,C`.
2. **IPIP-NEO-300 & IPIP-NEO-120 (Johnson, 2014)**: Download from OSF ([https://osf.io/wxvth/](https://osf.io/wxvth/)), place under `data/raw/ipip300/` and `data/raw/ipip120/`, and run:
   ```bash
   python scripts/prepare_ipip300.py
   python scripts/prepare_ipip120_gerlach.py
   ```
3. **Soto (2019) LOOPR** ($N = 6{,}126$): Download from OSF ([https://osf.io/3d485/](https://osf.io/3d485/)) and place under `data/raw/soto2019/`.
4. **Cross-Domain Clinical Psychopathology (HiTOP-SR & HiTOP MASEM)**:
   - **Ringwald, Forbes & Wright (2023)** meta-analytic MASEM matrices: OSF ([https://osf.io/pqmtc/](https://osf.io/pqmtc/)).
   - **US University of Kansas HiTOP-SR** ($N = 411$, 405 items): Distributed in the official R package `hitop` v0.2.0 ([https://github.com/jmgirard/hitop](https://github.com/jmgirard/hitop)).
   - **Czech HiTOP-CZ** ($N = 443$, 76 scale scores; Doubková & Rossi, 2026): Download `dataset.csv` from OSF ([https://osf.io/jx8h7/](https://osf.io/jx8h7/)) and place under `data/raw/hitop_empirical/czech_hitop/`.

## Reproducing the Analyses

```bash
# 1. Factorial Monte Carlo simulation (use --mode smoke for a fast check)
python scripts/run_montecarlo.py --mode full --reps 1000 --n-jobs -2 --seed 20261121 --resume

# 2. Corrected Gerlach et al. (2018) add-on (MC + Spanish BFI-2)
python scripts/run_gerlach_addon.py --mode full --n-jobs -2 --seed 20261121 --resume

# 3. Empirical comparison against Null 1, Null 2, and Null 3
python scripts/run_empirical_comparison.py \
  --input data/processed/bfi2_study1_domains.csv \
  --dataset-name bfi2_es_study1 \
  --cols-neoac N E O A C \
  --reps 1000 --n-jobs -2 --seed 20261121

# 4. Cross-domain individual-level HiTOP-SR validation (KU US + Czech Republic)
Rscript r/score_hitopsr.R
python scripts/hitop_01_score_and_qc.py
python scripts/hitop_03_run_hitopsr.py
python scripts/hitop_03b_revision_checks.py

# 5. Build tables, vector figures, and AUDIT_SUMMARY.md
python scripts/build_figures_and_tables.py
python scripts/build_multidataset_figures.py
python scripts/build_via_a_figures.py
```

## License

Released under the [MIT License](LICENSE).
