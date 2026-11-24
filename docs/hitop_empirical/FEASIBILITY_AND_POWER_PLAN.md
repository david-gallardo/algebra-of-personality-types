# Feasibility, Power & Minimum Detectable Alternatives (`FEASIBILITY_AND_POWER_PLAN.md`)

## 1. Sample Size Reality Check ($N_{\text{KU}} = 411$, $N_{\text{CZ}} = 443$)

Unlike the Big Five cohorts ($N = 145{,}388$ and $410{,}376$), Tier A direct HiTOP-SR cohorts have **moderate sample sizes ($N \approx 400$ per cohort)**. This imposes strict statistical boundaries that must be pre-specified honestly:

| Analysis Family | Statistical Power at $N \approx 410\text{–}440$ | Pre-Specified Minimum Detectable Effect / Strategy |
| :--- | :--- | :--- |
| **Q2 (Primary): Cross-cohort congruence ($\bar{\phi}_{\text{full}}, \bar{\phi}_{\text{shape}}$)** | **High** for rejecting independence (I0/I1 vs. C0/C1); **Moderate** for separating C0 (MVN) from C1 (copula) or subtle latent classes | Use $B = 5{,}000$ independent pairs per null ($p_{\min} = 1/5001 = 0.00020$, Monte Carlo SE at $p=0.05$ is $\pm 0.003$). Report full null quantiles `[2.5%, 50%, 97.5%]` and participant-level bootstrap 95% CIs for empirical $\bar{\phi}$. |
| **Q3: 5D/6D KDE Mean-Shift Topography** | **Low for closely overlapping mixtures ($\Delta \le 2.5\text{ SD}$)**; adequate only for well-separated macroscopic modes ($\Delta \ge 3.0\text{–}3.5\text{ SD}$) | Run positive-control power curves at **exact $N = 411$ and $N = 443$** across $\Delta \in \{1.5, 2.0, 2.5, 3.0, 3.5, 4.0\}\text{ SD}$ (50 replicates/cell) with both balanced and imbalanced (70/20/10) weights and floor-skewed marginals. Explicitly report the **detectable separation threshold**; never claim high-power ruling-out of subtle modes at $N \approx 411$. |
| **Psychometric Measurement Invariance (KU vs. CZ)** | **Adequate** for 5–6 spectrum composites (multi-group CFA on ~3–6 indicators per spectrum); **Insufficient** for a joint 76-scale CFA | Restrict invariance testing to pre-specified spectrum composites (configural $\to$ metric $\to$ scalar via `lavaan` / Python). If scalar invariance fails (expected given clinical enrichment in CZ), prohibit raw-elevation comparisons and anchor primary inference on within-cohort standardized profiles and shape congruence. |
| **Q4: External Predictive Validity (Czech cohort)** | **Moderate** ($N \approx 380\text{–}430$ complete cases) for continuous PANAS PA/NA and binary `Mental_health` / `Treatment` | Strictly leak-free nested 10-fold CV with participant-level paired bootstrap CIs. Clearly label PANAS as same-source self-report affect (not independent clinical outcome). |

## 2. Scientific Risk Register (Ranked)

| Rank | Risk | Mitigation |
| :---: | :--- | :--- |
| **CRITICAL** | **Cross-cultural & clinical selection confounding** (CZ 69.1% mental health history vs. KU undergraduates) | Test configural/metric/scalar invariance in Phase 1; use within-cohort standardization as primary and pooled-reference as sensitivity; evaluate shape ($\bar{\phi}_{\text{shape}}$) separately from full profile ($\bar{\phi}_{\text{full}}$). |
| **CRITICAL** | **Floor effects & zero-inflation** in low-base-rate clinical scales (e.g., `Reality Distortion`, `Suicidality`, `NSSI`, `Paraphilias`) creating artificial density ridges | Diagnose floor percentages per scale in Phase 1; include floor-matched empirical-marginal copula (C1) and never claim C1 is strictly unimodal; run positive-control null calibration on floor-skewed synthetic data. |
| **HIGH** | **Low power of 5D KDE topography at $N \approx 411$** | Calibrate exact detectable $\Delta$ via 50-rep positive controls at $N=411/443$; frame non-detection strictly as *"no persistent macroscopic modes above detectable separation $\Delta^*$"*, never *"proof of unimodality"*. |
| **HIGH** | **Absence of raw 405 items in Czech OSF** | Audit `analysis.R` and `codebook.rtf`; document that item-level re-scoring and item-level MI are impossible in CZ; verify 1–4 scale equivalence and $\le 25\%$ NA rule against KU `score_hitopsr()`. |
| **HIGH** | **Sexual-scale missingness in Czech (6.8–15.1% NA)** | Exclude sexual-dysfunction scales from primary 5/6-spectrum composites (keeping $N$ high without imputation); document complete sample-flow diagram (`SAMPLE_FLOW.csv`). |
| **MODERATE** | **Adaptive leakage in null calibration** | Enforce programmatic rule: every standardization, Ward subsample, *k*-means++ start, bandwidth rule, and Hungarian alignment is identically executed inside every empirical and null replicate. |
| **MODERATE** | **CPES & HBN access blockers** | Keep Tier B and Tier C cleanly modular and gated; never fabricate or impute inaccessible arms. |
