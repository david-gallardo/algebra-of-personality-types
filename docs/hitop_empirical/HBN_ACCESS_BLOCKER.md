# Tier C Access Blocker — Healthy Brain Network (`HBN_ACCESS_BLOCKER.md`)

**Status:** **NOT EXECUTED — RESTRICTED BY INSTITUTIONAL DUA**  
**Audit date:** 2026-11-24

## Why Tier C Is Stopped
Healthy Brain Network (HBN) phenotypic and clinical diagnostic data are governed by a mandatory **Data Usage Agreement (DUA)** (`https://redcap.link/HBN_DUA`) requiring:
1. Principal Investigator (PI) signature, **and**
2. Signature of a **distinct authorized institutional official** (e.g., Universitat de Barcelona Vice-Rectorate / Legal Office).
3. Verification of approved compute/storage infrastructure prohibiting upload of individual-level phenotypic data to public repositories or unapproved third-party services.

Additionally, automated HTTP inspection of `https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/FAQ.html` failed TLS certificate hostname verification; in compliance with security and integrity rules, TLS verification was not bypassed.

## Checklist to Unlock Tier C (Human Action Required)
- [ ] Submit institutional DUA at `https://redcap.link/HBN_DUA` with PI + distinct UB authorized signatory.
- [ ] Confirm approved local/institutional storage path (outside OneDrive cloud sync if prohibited by DUA).
- [ ] Obtain authorized release metadata and verify exact informant/age-band instruments (CBCL/YSR, SCARED, MFQ, E-SWAN, KSADS clinician consensus, impairment measures).
- [ ] Authorize `scripts/hitop_05_run_hbn_if_authorized.py` in a separate DUA-compliant local directory.

**No clinical HBN results have been generated or simulated.**
