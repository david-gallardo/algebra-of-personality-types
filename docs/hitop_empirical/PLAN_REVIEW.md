# Phase 0 — Implementation Plan & Human Approval Gates (`PLAN_REVIEW.md`)

**Branch:** `feature/hitop-empirical-validation`  
**Current Gate:** **Gate G0 (Data Provenance & Permission Audit) — COMPLETE.** Awaiting author review before Gate G1/G2 lock and Phase 3 confirmatory execution.

## Architecture & Exact Files to Create (Phases 1–5)

```text
src/hitop_empirical/
  __init__.py
  sources.py          # Provenance, SHA-256 verification, authorization gates
  harmonization.py    # 76-scale crosswalk (KU <-> CZ), missingness & floor QC
  psychometrics.py    # Spectrum composites, omega/alpha, cross-cohort comparability
  geometry.py         # R, eigenvalues, det(R), |R|^-1/2, lambda2/lambda1
  generators.py       # I0 (indep MVN), I1 (indep emp), C0 (corr MVN), C1 (calibrated copula)
  clustering.py       # Matched Ward + 9 k-means++ (K=3,4) & single-start Ward sensitivity
  matching.py         # Hungarian matching, full & elevation-removed Tucker phi
  topography.py       # Exact KDE mean-shift, basin persistence, valleys, positive controls
  inference.py        # Exact MC p=(1+k)/(B+1), Wilson CIs, participant-level bootstrap, Holm
  predict.py          # Leak-free nested 10-fold CV (PANAS + clinical history in CZ)
  visualization.py    # Publication-ready vector PDF + PNG figures

r/
  score_hitopsr.R     # Official hitop 0.2.0 scoring of ku_hitopsr + reliability/invariance checks

scripts/
  hitop_00_audit_sources.py
  hitop_01_score_and_qc.py
  hitop_02_lock_measurement.py
  hitop_03_run_hitopsr.py
  hitop_04_run_cpes.py             # Conditional on human ICPSR download
  hitop_05_run_hbn_if_authorized.py # Blocked until institutional DUA
  hitop_06_verify_and_report.py

tests/
  test_hitop_synthetic.py          # All 10 mandatory synthetic unit tests (a)-(j) + privacy guard

config/
  hitop_study.yaml                 # Locked preregistration config (SHA-256 hashed at Gate G2)
```

## Mandatory Human Approval Gates

- **Gate G0 (NOW — Review Phase 0 Audit):** Confirm Tier A (KU + CZ) scope, acknowledge Tier B (CPES requires human ICPSR login) and Tier C (HBN blocked pending DUA), and approve moving to Phase 1 (KU R scoring + scale-level QC + spectrum crosswalk + synthetic unit tests).
- **Gate G1/G2 (Pre-Confirmatory Lock):** Author reviews `VARIABLE_CROSSWALK.csv`, `PSYCHOMETRIC_COMPARABILITY.md`, and `PREREGISTRATION_LOCKED.md` (choosing 5 vs. 6 spectra and locking `config/hitop_study.yaml`) **before** any real-data clustering or confirmatory inference is executed.
- **Gate G5/G6 (Adversarial Audit & Manuscript Integration):** Independent recomputation audit before deciding whether verified results belong in NHB main text, Extended Data, Supplement, or a standalone paper.
