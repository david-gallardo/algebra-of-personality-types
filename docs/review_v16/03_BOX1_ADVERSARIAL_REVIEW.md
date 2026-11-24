# P1 — Adversarial Stress-Test of Strong Box 1 (Versions A & B)

Three adversarial specialist reviewers independently attempted to break Candidate Versions A and B of Box 1 using known statistical, psychometric, and clinical counterexamples.

---

## 1. Mixture-Model Statistician Review

### Attack Vectors Tested
1. **Same-mean Gaussian scale (variance) mixtures**: $f(\mathbf{x}) = 0.5\,\mathcal{N}(\mathbf{0}, \mathbf{I}_5) + 0.5\,\mathcal{N}(\mathbf{0}, 4\mathbf{I}_5)$ has two genuine latent classes, identical means, and strictly one density peak with zero valleys.
2. **Heavily overlapping location mixtures**: Equiproportionate mixtures with Mahalanobis separation $\Delta \in [1.5, 2.5]$ SD are genuine discrete latent classes yet appear unimodal under 5D KDE smoothing at $N \le 10,000$ (and up to $\Delta = 4.0$ SD at $N \approx 400$).
3. **Non-unimodal continuous copulas**: An empirical-marginal Gaussian copula (C1) is a continuous generator without explicit latent classes, yet discretized or bimodal empirical marginals can induce slight ripples or multi-component full-GMM BIC wins.
4. **"Passing the checklist proves classes"**: A skewed non-Gaussian distribution with two ridges or an omitted nonlinear latent factor can beat a Gaussian copula and show local density bumps without discrete latent classes.

### Verdict on Version A vs. Version B
- **Version A withstands all four attacks:**
  - *Scope* explicitly limits the Box to *warranted inference from clustering data*, not logical properties of all mixtures.
  - *Audit 2* explicitly names *"overlapping or same-mean scale mixtures"* and mandates the verdict ***uninformative*, never *unimodality confirmed*** when exact-$N$, exact-$d$ positive controls fail to resolve modes.
  - *Audit 1* specifies *"continuous nulls preserving covariance ($\mathbf{R}$) and empirical marginals or tail dependence"* rather than falsely calling empirical copulas "strictly unimodal."
  - *Interpretation* explicitly states *"passing all three never by itself proves ontology."*
- **Version B** survives attacks 1–4 in compressed form, but omits explicit mention of tail dependence and Monte Carlo uncertainty, making it slightly easier for a hostile reviewer to ask *"which continuous nulls?"*

---

## 2. Psychometrician Review

### Attack Vectors Tested
1. **Response-style and careless-responding mixtures**: Online survey panels routinely contain a genuine discrete latent class of careless, midpoint, or extreme responders (Meade & Craig, 2012). Such a mixture *can* beat continuous nulls, *can* improve GMM-4 fit, and *can* even add small incremental prediction for socially undesirable or low-base-rate outcomes (as observed in Soto 2019 Survey 1 Young Adults)—without implying substantive personality types!
2. **Elevation vs. shape confounding**: Cross-cultural or cross-instrument profile congruence ($\bar{\phi} > 0.90$) can be driven almost entirely by overall severity/elevation (as in HiTOP-SR US–Czech: $0.935 \to 0.699$ once elevation is removed), rather than qualitative profile shapes.
3. **Optimizer and preprocessing mismatch**: Comparing multistart empirical $k$-means against single-start simulated $k$-means, or comparing raw domain scores against whitened factor scores, creates spurious rejections of continuous nulls.

### Verdict on Version A vs. Version B
- **Version A withstands all three attacks:**
  - *Audit 1* explicitly mandates **both full and elevation-removed congruence** and **identical scoring, initialization, and sample sizes** across empirical and simulated data.
  - *Audit 3* asks for reproducible predictive, biological, or mechanistic information, and the companion Discussion paragraph explicitly warns that **response-style mixtures** can mimic substantive classes even when they show small incremental predictive gains.
- **Version B** covers full vs. shape-only and identical preprocessing/optimization, though it relies on the main text for the response-style warning.

---

## 3. Clinical Taxonomist / Psychiatrist Review

### Attack Vectors Tested
1. **Substantive clinical subgroups with null incremental prediction**: Two clinically distinct conditions or etiologically distinct subtypes (e.g., two different biological pathways leading to similar cross-sectional symptom severity, or a rare clinical subgroup defined by genetic/biomarker etiology) may show **zero incremental predictive utility** over continuous symptom severity for generic short-term outcomes, yet still represent real clinical kinds.
2. **Separate non-clustering evidentiary pathways**: Psychiatry and medicine often establish discrete taxa via genetics, pathogen identification, treatment-response RCT interactions, or causal neurobiology—not unsupervised symptom clustering. A dogmatic "must pass all three clustering audits" rule would absurdly block non-clustering evidence.
3. **Underpowered clinical samples ($N \approx 200\text{--}500$)**: Clinical cohorts rarely have $N = 100,000$. If Audit 2 treated $N = 400$ single-mode KDE output as evidence against subtypes, it would systematically bias clinical nosology toward continua.

### Verdict on Version A vs. Version B
- **Version A withstands all three attacks:**
  - *Scope* explicitly carves out *"nor a bar to convergent causal, genetic, or biological evidence advanced through separate evidentiary pathways."*
  - *Audit 2* protects moderate-$N$ clinical samples by mandating exact-$N$ positive controls and an *uninformative* verdict whenever the protocol lacks power (directly matching the manuscript's own HiTOP-SR $N = 411/439$ finding).
  - *Audit 3* explicitly states *"Null incremental prediction bounds demonstrated utility, not the logical existence of classes."*
  - *Interpretation* cleanly separates **mandatory reporting** of all three audits from **requiring positive outcomes** on all three.

---

## Summary Matrix of Counterexample Resilience

| Counterexample / Objection | Version A (~201w) | Version B (~134w) | Protected by |
|---|---|---|---|
| Same-mean variance/scale mixture (1 mode) | **PASS** | **PASS** | Scope + Audit 2 (*uninformative*) |
| Heavily overlapping location mixture ($\Delta \le 2.5$ SD) | **PASS** | **PASS** | Audit 2 exact-$N$ positive control |
| Underpowered clinical cohort ($N \approx 400$) | **PASS** | **PASS** | Audit 2 (*uninformative*, never *unimodality confirmed*) |
| Careless / response-style mixture | **PASS** | **PARTIAL** (via text) | Audit 3 + Companion Discussion guardrail |
| Clinical subtype with no incremental prediction | **PASS** | **PASS** | Audit 3 + Interpretation |
| Non-clustering genetic / causal / biological evidence | **PASS** | **PASS** | Scope explicit carve-out |
| "Passing all three proves ontology" | **PASS** | **PASS** | Interpretation explicit denial |

**Final Decision:** Adopt **Version A** as Box 1 in `v16`, paired with the companion Discussion guardrail paragraph.
