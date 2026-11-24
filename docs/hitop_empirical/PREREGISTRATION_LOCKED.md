# Preregistration Lock — Empirical HiTOP Validation (`PREREGISTRATION_LOCKED.md`)

**Status:** READY FOR GATE G2 AUTHOR APPROVAL — **NO confirmatory clustering or inferential tests have been executed on KU or Czech data.**  
**Branch:** `feature/hitop-empirical-validation`  
**Config file:** `config/hitop_study.yaml`

## 1. Frozen Measurement Specification (Verified in Phase 1)
- **Tier A Cohorts**: KU HiTOP-SR ($N = 411$, scored natively via `hitop 0.2.0::score_hitopsr()`) and Czech HiTOP-CZ ($N = 439$ complete on 5/6 spectra, OSF `jx8h7`, SHA-256 verified).
- **Primary Dimensionality**: **5 core HiTOP spectra** (`INT, DET, TD, A_EXT, D_EXT`), each showing strong cross-language 1-factor loading congruence ($\phi \in [0.972, 0.999]$) and acceptable-to-high internal consistency ($\alpha \in [0.687, 0.905]$, $\omega_t \in [0.698, 0.909]$).
- **Sensitivity Dimensionality**: **6 spectra** adding **Somatoform (`SOM`)** ($\phi = 0.998$, $\alpha = 0.795/0.828$).
- **Standardization**: **Within-cohort z-standardization (primary)** because Czech is clinically enriched (+0.28 to +0.81 SD higher raw symptom elevation across all 6 spectra); pooled-reference standardization reported as labeled sensitivity.

## 2. Frozen Primary Confirmatory Endpoint & Multiplicity Rules
- **Single Primary Endpoint**: Cross-cohort matched **full-profile Tucker congruence ($\bar{\phi}_{\text{full}}$) at $K = 3$** under matched Ward + 9 *k*-means++ (min SSE), evaluated against **C0 (Correlated MVN preserving $\hat{\mathbf{R}}_{\text{KU}}$ and $\hat{\mathbf{R}}_{\text{CZ}}$)** with $B = 5{,}000$ independent pairs, exact $p_{\text{MC}} = (1 + k)/(B + 1)$, and 95% Wilson CI.
- **Secondary Family (Holm–Bonferroni adjusted)**:
  1. Elevation-removed shape congruence ($\bar{\phi}_{\text{shape}}$) at $K = 3$ and $K = 4$ under I0, I1, C0, C1.
  2. Within-cohort split-half $\kappa$, ARI, NMI ($K = 3, 4$) under I0, I1, C0, C1.
  3. Matched single-start Ward + *k*-means sensitivity ($K = 3, 4$).
  4. 5D/6D KDE mean-shift topography across $h \in \{0.80, 1.00, 1.15, 1.25, 1.50\} \times h_{\text{NN}}$ calibrated against 50-replicate positive controls at exact $N = 411$ and $N = 443$ ($\Delta \in [1.5, 4.0]$ SD).
  5. Leak-free 10-fold CV incremental prediction in Czech cohort (`PANAS_PA`, `PANAS_NA` — explicitly labeled *same-source self-report affect* — plus binary `Mental_health` and `Treatment`).

## 3. Pre-Specified Interpretation Framework
- **(A)** Merely consistent with continuous alternatives;
- **(B)** Continuous alternatives (C0/C1) account for the pre-specified clustering/replicability signatures;
- **(C)** Empirically distinguishable multimodality or latent structure beyond C0/C1;
- **(D)** Indeterminate owing to sample size ($N \approx 411\text{–}439$), measurement, or power limits.

Non-rejection under C0/C1 is **never** interpreted as ontological proof against all latent clinical subtypes.
