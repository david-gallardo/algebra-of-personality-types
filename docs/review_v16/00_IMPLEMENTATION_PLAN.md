# v16 Implementation Plan — NHB Strong Box 1, Provenance, and Novelty

**Branch:** `review/v16-strong-box-provenance-novelty`  
**Authoritative v15 sources (preserved untouched):**
- `/Users/davidgallardo-pujol/Downloads/nature_human_behaviour_manuscript_v15_nhb_narrative.tex`
- `/Users/davidgallardo-pujol/Downloads/nature_human_behaviour_manuscript_v15_nhb_narrative.pdf`

**Non-negotiable safeguards:**
- Never overwrite v15 files, canonical raw data (`data/raw/`), or primary archived simulation results (`results/simulations/block_*.parquet`, `results/empirical/`).
- Never push commits or submit the manuscript without explicit user authorization.
- Never fabricate results, tune cutoffs post-hoc, or execute the prospective P3 benchmark before `07_OPTIONAL_BENCHMARK_PREREG_PLAN.md` is reviewed and approved.

---

## Phased Execution Plan

### Phase P0 — Mandatory Spanish BFI-2 Source & Provenance Audit (BLOCKING GATE)
- **Objective:** Audit OSF project `https://osf.io/kp572/` (Gallardo-Pujol et al., 2022, *PTAD*, DOI `10.1027/2698-1866/a000020`), inventory every accessible file/version (URL, size, date, checksum, license, codebook), and reconcile `N = 1,000` (published Study 1) vs. `N = 1,419` (IRW BFI-2 table) vs. `N = 1,146` (`data/processed/bfi2_study1_domains.csv` in v15).
- **Files to inspect/create:**
  - Inspect `data/raw/bfi2_study1_raw.csv` and `data/processed/bfi2_study1_domains.csv`.
  - Query OSF API v2 (`https://api.osf.io/v2/nodes/kp572/...`) to list and download publicly accessible files into `data/raw/bfi2_osf_kp572/` (gitignored).
  - Create portable preprocessing script `scripts/prepare_bfi2_es.py` (no absolute local paths, no copyrighted item text committed).
  - Deliverables:
    - `docs/review_v16/01_BFI2_PROVENANCE_AND_REPRODUCIBILITY.md`
    - `docs/review_v16/01_bfi2_provenance_audit.json`
    - `docs/review_v16/01_bfi2_discrepancy_table.csv`
    - Clear PASS / FAIL provenance gate decision for `N = 1,146`.

### Phase P1 — Strong Box 1 Redesign & Adversarial Stress-Testing
- **Objective:** Preserve Box 1 as a **strong minimum evidential/reporting standard** for clustering-based claims of discrete kinds while protecting against counterexamples (same-mean variance mixtures, heavily overlapping mixtures, response-style mixtures, substantive clinical groups without incremental prediction).
- **Deliverables:**
  - `docs/review_v16/02_BOX1_REVISED.md` (Version A ~190 words & Version B ~130 words + selection recommendation)
  - `docs/review_v16/03_BOX1_ADVERSARIAL_REVIEW.md` (independent stress-tests from a mixture-model statistician, psychometrician, and clinical taxonomist)

### Phase P2 — Exact Corrections to v15 Narrative, Tables, Abstract, and Cross-References
- **Objective:** Apply all verified corrections A–J from `ANTIGRAVITY_V16_NHB_STRONG_BOX_NOVELTY.md` to create `v16` without altering primary statistical estimates unless a documented audit discrepancy requires it:
  - Fix `all major Big Five domain benchmarks` $\to$ `five of six Big Five domain benchmarks` (BFI-2 ES = `1.229 < 1.25`).
  - Refine ~150-word Abstract (distinguishing proof vs. geometric argument vs. empirical finding; distinguishing faithful Gerlach IPIP-300 replication vs. parallel IPIP-120 topographic check; verifying NHB official Abstract guidance).
  - Fix `Supplementary Table S1` $\to$ `Supplementary Table S2` (keep `S1` for HiTOP 37-scale specification).
  - Temper `isolate the causal roles` $\to$ `experimentally decompose the effects within the specified simulation design`.
  - Temper clinical DSM Cluster A–C comparison to qualitative resemblance.
  - Ensure Czech HiTOP-CZ is never called a clinical patient cohort and row-centering is never equated to a formal latent $p$-factor.
  - Align `three sequential tests` phrasing with the revised strong Box 1.
  - Verify Appendix F `25/25 collapse to 1 mode` directly against `results/empirical/hitop/hitop_v12_revision_checks.json`.
  - Update Data Availability with exact OSF `kp572` provenance and 2022 citation/DOI.
- **Deliverables:**
  - `docs/review_v16/04_CHANGELOG_V15_TO_V16.md`
  - `docs/review_v16/05_ABSTRACT_OPTIONS.md`
  - `/Users/davidgallardo-pujol/Downloads/nature_human_behaviour_manuscript_v16_submission_ready.tex` & `.pdf`
  - `docs/review_v16/04_crossref_and_number_audit.log`

### Phase P3 — Scientific Novelty Matrix, Prospective Benchmark Plan, and Editorial Positioning
- **Objective:** Ground novelty scientifically (not rhetorically), freeze a prospective simulation benchmark design for user review (do NOT execute before approval), and build editorial positioning.
- **Deliverables:**
  - `docs/review_v16/06_NOVELTY_MATRIX.md`
  - `docs/review_v16/07_OPTIONAL_BENCHMARK_PREREG_PLAN.md` (frozen prospective plan awaiting review/approval before coding or execution)
  - `docs/review_v16/09_EDITORIAL_POSITIONING.md` (100–150 word significance pitch, interdisciplinary reach, 5 likely editor objections & evidence-based responses, Article vs. Methods framing)

### Phase P4 — Final Quality, Math/Numeric, Test, and Layout Integrity Gate
- **Objective:** Independently verify mathematical identities (`0.50764047`, `|R|^-1/2`, Phipson–Smyth $p$, factorial interactions, Wilson CIs), run the full `pytest` suite with exact environment versions, compile and visually inspect the `v16` PDF, and produce a candid Blocking / Important / Optional readiness decision table.
- **Deliverables:**
  - `docs/review_v16/10_FINAL_AUDIT_AND_READINESS_DECISION.md`
  - `docs/review_v16/10_independent_math_verification.json`
