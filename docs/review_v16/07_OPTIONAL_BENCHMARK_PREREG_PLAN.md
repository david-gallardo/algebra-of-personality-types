# P3 — Prospective Preregistered Simulation Benchmark Plan (FROZEN FOR REVIEW — NOT YET EXECUTED)

**Status:** **PROSPECTIVE DESIGN — FROZEN BEFORE CODING OR EXECUTION.**  
Do **not** execute this benchmark or insert any benchmark results into the manuscript until the author reviews and explicitly approves this frozen design and its pre-specified decision rules.

---

## 1. Scientific Objective

Evaluate the **operating characteristics** (false-positive declaration rate, true-positive detection power, *uninformative* rate, and joint signature reproduction rate with exact 95% Wilson CIs) of the **three-audit evidential standard (Box 1)** across a controlled factorial of continuous and genuinely discrete data-generating processes (DGPs).

**Primary new scientific question:** Can a single pre-specified family of continuous nulls **jointly reproduce** all three conventional typology signatures simultaneously (split-half/cross-sample replicability, multi-component mixture BIC selection, and independence-referenced density enrichment), and where does the three-audit standard succeed versus intentionally abstain (*uninformative*)?

---

## 2. Pre-Specified Data-Generating Processes (DGPs; $d = 5$)

All DGPs use $d = 5$ standardized indicators with empirical Big Five (IPIP-NEO-300) or HiTOP-SR (KU) correlation structure $\mathbf{R}$ where applicable:

1. **DGP-1 — Smooth Correlated Unimodal Normal (Null Truth):** $\mathcal{N}_5(\mathbf{0}, \mathbf{R})$. Strictly unimodal, zero latent classes.
2. **DGP-2 — Skewed / Floor-Affected Continuous Copula (Null Truth, Explicitly Audited for Modality):** Empirical-marginal Gaussian copula (C1 calibrated to HiTOP-SR KU marginals & $\mathbf{R}$, plus a smooth parametric skewed-marginal copula using skew-normal/log-normal marginals with analytically verified unimodality). We explicitly audit C1's fine-bandwidth modality rather than assuming strict unimodality.
3. **DGP-3a — Authentic Well-Separated Location Mixture (Discrete Truth, Identifiable):** Equiproportionate $K = 3$ and $K = 4$ isotropic/correlated Gaussian mixtures with pairwise separation $\Delta = 3.5$ SD.
4. **DGP-3b — Authentic Overlapping Location Mixture (Discrete Truth, Hard/Boundary):** Equiproportionate $K = 3$ and $K = 4$ mixtures with $\Delta = 2.0$ SD (genuinely discrete, expected to fall below 5D KDE resolution).
5. **DGP-4 — Authentic Same-Mean Scale (Variance) Mixture (Counterexample — Discrete Truth, Strictly Unimodal):** $0.5\,\mathcal{N}_5(\mathbf{0}, 0.5\mathbf{I}_5) + 0.5\,\mathcal{N}_5(\mathbf{0}, 2.0\mathbf{I}_5)$ (re-standardized). Two genuine latent classes, identical mean, strictly 1 density mode, 0 valleys—explicitly demonstrating what Audit 2 **cannot** identify.

---

## 3. Experimental Factors & Sample Sizes

- **Sample sizes:** $N \in \{400, 1{,}000, 10{,}000\}$ (matching HiTOP-SR, BFI-2 ES, and large-scale positive-control regimes).
- **Cluster counts evaluated:** $K \in \{3, 4\}$.
- **External outcome generator (for Audit 3):** Two pre-specified outcome regimes per DGP:
  - *Linear-trait outcome:* $y = \boldsymbol{\beta}^\top \mathbf{x} + \varepsilon$ ($R^2_{\mathrm{true}} = 0.15$; no class-specific effect).
  - *Class-specific outcome (for DGP-3a/3b/4):* $y = \boldsymbol{\beta}^\top \mathbf{x} + \gamma_c + \varepsilon$ ($\Delta R^2_{\mathrm{true}} = 0.05$ from true latent class $c$; plus a *null-utility clinical counterexample* where $\gamma_c = 0$ despite genuine latent classes).
- **Replicates:** $R = 50$ independent datasets per cell at $N \in \{400, 1{,}000\}$ and $R = 25$ per cell at $N = 10{,}000$ (fixed master `SeedSequence(20261125)`).

---

## 4. Frozen Decision Rules (Specified Before Execution)

For each simulated dataset, run the exact three Box 1 audits with **no post-hoc tuning**:

- **Audit 1 (Continuous-null challenge):**
  - *Pass (rejects continuous null)* iff empirical full-profile AND shape-only congruence (or full-GMM $\Delta\mathrm{BIC}_{K\text{ vs }1}$) exceeds the 95th percentile ($p_{\mathrm{MC}} < 0.05$, Phipson–Smyth, $B = 200$ matched C1/C0 nulls) under identical Ward + 9 $k$-means++ optimization.
- **Audit 2 (Calibrated topography):**
  - First check exact-$N$, exact-$d$ positive-control power at $\Delta = 3.5$ SD ($B_{\mathrm{pc}} = 25$):
    - If positive-control exact-$K$ recovery $< 50\%$, Audit 2 verdict is **`UNINFORMATIVE`** (never `unimodal`).
    - Otherwise, *Pass (multimodal)* iff $\ge 2$ persistent macroscopic modes ($\ge 5\%$ basin mass across $1.00\text{--}1.15\,h_{\mathrm{NN}}$) with pairwise valley depth $\delta_{\mathrm{valley}} > 0.05$.
- **Audit 3 (Leak-free external distinction):**
  - 10-fold CV Ridge regression comparing `Traits Linear` vs. `Traits + Typology (Hard & GMM-Prob)` strictly inside folds.
  - *Pass (incremental utility)* iff mean $\Delta R^2_{\mathrm{CV}} > 0.01$ and paired fold-level 95% CI excludes zero.
- **Composite Evidential Verdicts (pre-specified):**
  - *Strong clustering-based support:* Audit 1 Pass + Audit 2 Pass (+ Audit 3 reported).
  - *False-positive discrete-type declaration (under DGP-1 / DGP-2):* Rate of datasets passing Audit 1 + Audit 2 (or passing conventional uncalibrated signatures: $\kappa \ge 0.60$, diagonal LPA $K > 1$, permutation enrichment $e > 1.25$).
  - *Counterexample verification:* Explicitly report Audit 2 `UNINFORMATIVE` rate at $N = 400$ and Audit 2 non-detection on DGP-4 (scale mixture) and DGP-3b ($\Delta = 2.0$ SD), plus Audit 3 null rate on genuine classes with $\gamma_c = 0$.

---

## 5. Planned Outputs (Only After Approval)

- Script: `scripts/run_box1_benchmark.py` + unit tests `tests/test_box1_benchmark.py`
- Results report: `docs/review_v16/08_BENCHMARK_RESULTS.md` + `results/box1_benchmark/summary.json`
- Optional Extended Data Figure (if substantively informative): 3-panel operating-characteristic heatmap comparing conventional criteria vs. Box 1 audits across DGP-1 to DGP-4 and $N = 400, 1{,}000, 10{,}000$.
