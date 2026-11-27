# 3. Empirical Algebraic Audit Design & User Execution Guide

**Strict compliance with `PROJECT_RULES.md`:**  
No terminal commands (`python`, `pytest`, `lualatex`, or simulations) were executed during this algebraic revision. All newly written verification code and the v17 LaTeX source are ready for execution by the user. **No unexecuted empirical pentad or third-cumulant numbers are reported as verified in v17.**

---

## 1. Scientific Design & Cost Justification

| Component | Target Question | Inference Method | Cost & Precision |
|---|---|---|---|
| **A. Exact Symbolic & Rational Checks** (`output/algebraic_derivation/verify_algebra.py` + `tests/test_algebraic_complexity.py`) | Verify 12-term pentad cancellation, weighted-simplex covariance equality, $3 \times 3$ Heywood boundary counterexample, and exact rational third-order simplex norm. | Exact symbolic/rational arithmetic + unit tests (`pytest`). | $< 2$ seconds; exact (zero Monte Carlo error). |
| **B. Individual-Level Empirical Audit** (`bfi2_es_study1`, `ipip300_domains`, `ipip120_domains`, `ku_hitopsr`, `czech_hitop`) | Do observed full-precision 5D correlation matrices exclude exact $K \le 3$ diagonal representation under joint sampling uncertainty, and how does empirical $B$-whitened $\|T\|_F^2$ compare to the homoscedastic bound $r(r-1) = 6$? | Whole-respondent nonparametric bootstrap ($B = 500$, seed `20261121`); never perturbs correlation entries independently; propagates joint uncertainty of $(\hat{D}, \hat{B}, \hat{W})$ with conditioning guard $\lambda_{\min}(\hat{B}) \ge 10^{-3}$. | ~30–60 seconds total; 95% joint confidence radius $\varepsilon_{0.95}$ and bootstrap CIs. |
| **C. Meta-Analytic Matrices** (`van_der_linden_2010`, `busseri_erb_2024`) | Separate rounded-matrix pentad diagnostics from joint sampling variability at effective $N$. | SPD-preserving Wishart sampling ($W_5(N_{\mathrm{eff}}-1, R/(N_{\mathrm{eff}}-1))$ standardized to correlation matrices, $B = 500$). | $< 3$ seconds; explicitly tagged as rounded-matrix Wishart approximations, not population tests. |
| **D. Sampling-Aware Complexity Upper Bound $d_K(R; \tau)$** | Quantify approximation distance to $\mathcal{C}_{K,\tau}$ ($\tau = 0.01$, $K = 1\dots 5$). | Multistart L-BFGS-B (12 starts). | Explicitly reported as an **upper bound** on $d_K$; never interpreted as proof of impossibility when a local optimizer fails, and never conflated with minimum trace. |

---

## 2. Exact Commands for the User to Run

Run these commands from the project root:
`/Users/davidgallardo-pujol/Library/CloudStorage/OneDrive-UniversitatdeBarcelona/Estudis/2026 Algebra of Personality Types`

```bash
# 1. Run exact symbolic pentad & rational 3rd-order verification
python3 output/algebraic_derivation/verify_algebra.py

# 2. Run the full unit test suite (including new tests/test_algebraic_complexity.py)
/opt/anaconda3/bin/pytest -v

# 3. Run the empirical algebraic audit (bootstrap + Wishart + d_K upper bounds + 3rd-order diagnostic)
/opt/anaconda3/bin/python scripts/run_algebraic_audit.py --n-boot 500 --tau 0.01 --seed 20261121 --out results/algebraic_audit/algebraic_audit_summary.json

# 4. Compile v17 LaTeX draft to PDF (two passes for cross-references)
lualatex -interaction=nonstopmode -output-directory=/Users/davidgallardo-pujol/Downloads /Users/davidgallardo-pujol/Downloads/nature_human_behaviour_manuscript_v17_algebraic_draft.tex
lualatex -interaction=nonstopmode -output-directory=/Users/davidgallardo-pujol/Downloads /Users/davidgallardo-pujol/Downloads/nature_human_behaviour_manuscript_v17_algebraic_draft.tex

# 5. Check compilation log for any errors or undefined references
grep -E "^!|Warning|duplicate destination" /Users/davidgallardo-pujol/Downloads/nature_human_behaviour_manuscript_v17_algebraic_draft.log || echo "CLEAN_LOG_ZERO_ERRORS"
```
