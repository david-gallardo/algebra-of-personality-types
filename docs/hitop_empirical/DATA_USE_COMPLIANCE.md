# Data Use, Licensing & Compliance Governance (`DATA_USE_COMPLIANCE.md`)

1. **Repository & Privacy Isolation**
   - `data/` (including `data/raw/hitop_empirical/` and `data/processed/hitop_empirical/`) is permanently excluded by `.gitignore`.
   - No respondent-level psychiatric data (`dataset.csv` from Czech HiTOP-CZ, CPES microdata, or HBN phenotypes) will ever be committed to Git, pushed to GitHub, or embedded in public reports/artifacts.
   - All outputs written to `results/empirical/hitop/` and `reports/` will be strictly **aggregate statistics, simulation replicates, and summary tables** (programmatically verified by unit tests asserting zero participant IDs or row-level exports).

2. **Source-Specific License & DUA Compliance**
   - **KU HiTOP-SR (`hitop 0.2.0`)**: Distributed publicly under **GPL-3.0**. Local scoring in R is fully permitted; cite Girard & Gray / HiTOP Society and pin package version `0.2.0` (commit `404f466`).
   - **Czech HiTOP-CZ (`https://osf.io/jx8h7/`)**: Distributed publicly under **CC-BY-NC 4.0 International** (non-commercial research use with attribution to Doubková & Rossi, 2026). Local analysis is permitted; redistribution of `dataset.csv` is avoided (users download directly from OSF).
   - **CPES (ICPSR 20240 v9)**: **Not downloaded automatically.** Requires human researcher login and acceptance of ICPSR Public-Use Terms of Use (or formal DUA for restricted files).
   - **HBN**: **Not accessed.** Strictly requires an institutional Data Usage Agreement (`https://redcap.link/HBN_DUA`) signed by PI + distinct authorized institutional signatory, plus approved compute infrastructure.

3. **Manuscript & Main Branch Protection**
   - All work is isolated on git branch `feature/hitop-empirical-validation`.
   - Neither `main`, nor `reference/manuscript_current.pdf`, nor the existing 28 Monte Carlo Parquet blocks will be modified without explicit author authorization.
