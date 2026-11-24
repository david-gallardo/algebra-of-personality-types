# Phase 0 — Data Access & Source Audit (`DATA_ACCESS_AUDIT.md`)

**Audit date:** 2026-11-24  
**Git branch:** `feature/hitop-empirical-validation` (isolated from `main`; existing manuscript and prior results untouched)  
**Status:** Phase 0 complete — **NO empirical clustering or confirmatory inference executed.**

## Executive Data Availability Table

| Tier | Cohort / Source | Access Status | License / Governance | Verified $N$ | Verified Individual-Level Variables | Key Blockers / Limitations |
| :--- | :--- | :--- | :--- | ---: | :--- | :--- |
| **A1 (Core)** | **KU HiTOP-SR** (`hitop::ku_hitopsr`, `jmgirard/hitop` v0.2.0, commit `404f466`) | **OPEN — VERIFIED & INSTALLED** | GPL-3.0 | 411 | 405 raw items (`hsr_001`–`hsr_405`, 1–4 scale), `participant`, `biosex` (117 M / 294 F); officially scored into **76 scales** (`1–4` mean) via `score_hitopsr()` | University sample (non-clinical); `0.092%` item NA (`407/411` complete across 76 scales). |
| **A2 (Core)** | **Czech HiTOP-CZ** (Doubková & Rossi, 2026; `https://osf.io/jx8h7/`) | **OPEN — VERIFIED & DOWNLOADED** | CC-BY-NC 4.0 International | 443 | `ResponseId`, `Age` (18–79, $M=35.2$), `Gender`, `Education_level`, `Mental_health` (69.1% Yes), `Treatment` (conditional), **76 official HiTOP-CZ scale scores** (`1–4` mean, $\le 25\%$ item NA rule), **20 raw PANAS items** | **No raw 405 HiTOP items** on OSF (only official respondent-level 76 scale scores); item-level re-scoring & item-level MI not possible. Sexual scales have 6.8–15.1% NA (`337/443` complete on 76 scales; `384/443` on 68 non-sexual scales). Strong clinical/treatment enrichment vs. KU. |
| **B (External)** | **CPES / ICPSR 20240 v9** (`https://www.icpsr.umich.edu/web/ICPSR/studies/20240`) | **CONDITIONAL — HUMAN DOWNLOAD REQUIRED** | ICPSR Public-Use Terms of Use (login required); Restricted files require DUA | Pending download | Pending public-use codebook audit (NCS-R, NSAL, NLAAS harmonized diagnostic/symptom variables + weights/PSU/strata) | Automated HTTP fetch returned `403 Forbidden`. Requires human login + click-through Terms of Use at ICPSR. Must never bypass access gates. |
| **C (Clinical Dev.)** | **Healthy Brain Network (HBN)** (`https://redcap.link/HBN_DUA`) | **RESTRICTED — DUA REQUIRED (BLOCKED)** | Institutional DUA (PI + distinct authorized institutional signatory) | Not accessed | Public FAQ instrument scope: CBCL/YSR, SCARED, MFQ, E-SWAN, KSADS/clinician consensus (cross-sectional) | No DUA on file; automated FAQ fetch also failed SSL hostname verification (`fcon_1000.projects.nitrc.org`). Arm explicitly stopped (`HBN_ACCESS_BLOCKER.md`). |
| **Bridge (Aggregate)** | **Ringwald, Forbes & Wright (2023)** (`https://osf.io/pqmtc/`) | **OPEN — VERIFIED & DOWNLOADED** | OSF Public Archive | 120,596 ($N_h = 10,502$) | **Aggregate only**: Mplus 8.5 MASEM output (`hitop_masem_output.out`) + `hitop_supplement.docx` | **Not individual-level data.** Reconciled 23 vs. 21 diagnoses: `NAMES ARE` has 23 DSM disorders; `USEVARIABLES ARE` excludes **`bip` (Bipolar)** and **`dpd` (Dependent PD)** from primary EFA (DPD sensitivity in Supplement Table 4). |

## Critical Phase 0 Findings

1. **Tier A (KU + Czech) is immediately feasible and scientifically strong** for direct HiTOP-SR instrument replication across two independent, non-overlapping cohorts in two languages (English US university vs. Czech clinically enriched community), sharing **all 76 HiTOP-SR primary scales on the exact same 1–4 arithmetic mean metric**.
2. **Exact 76-scale naming reconciliation**: 74/76 scale names match verbatim between `hitop 0.2.0` (`hsr_*`) and Czech (`HiTOP_*`); two benign label aliases were identified:
   - `hsr_appearanceFocus` $\leftrightarrow$ `HiTOP_Body Focus`
   - `hsr_nonSuicidalSelfInjury` $\leftrightarrow$ `HiTOP_NSSI`
3. **Cross-cultural & sampling asymmetry (Critical Risk)**: KU is a US student sample (71.5% female), whereas Czech is an older (18–79), majority-male (57.3% men) community sample with **69.1% lifetime mental health history** (`306/443`) and **60.3% treatment history** (`267/443`). Mean-level profile elevation differences are therefore confounded with clinical selection and potential scalar non-invariance; **within-cohort standardized and elevation-removed shape comparisons** are mandatory alongside pooled-reference checks.
4. **Tier B (CPES) and Tier C (HBN)** cannot be ingested automatically without human authorization (ICPSR login for CPES; institutional DUA for HBN). Both arms are cleanly gated without fabricating or bypassing access.
