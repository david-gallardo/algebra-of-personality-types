# 5. v17 → v18 Hybrid Revision: Resolution of Editorial Objections & Changelog

**New manuscript source:** `/Users/davidgallardo-pujol/Downloads/nature_human_behaviour_manuscript_v18_hybrid.tex` (`v17` preserved untouched).

| Review Objection / Issue | Exact Resolution in `v18_hybrid` |
|---|---|
| **Objection 1 ($K = 4$ overclaim on IPIP)** | Replaced *"pinning representational diagonal complexity to $K_{\mathrm{diag}}^{(2)}(\mathbf{R}) = 4$"* with *"excluding exact $K \le 3$ diagonal covariance representations, while a $K = 4$ representation approximates the observed correlation matrix to within numerical tolerance ($d_4 < 4 \times 10^{-6}$). Because nonconvex optimization provides an upper bound on $d_4$ rather than an exact population existence certificate, we treat $K = 4$ as numerically compatible rather than mathematically certified as the exact population minimum."* |
| **Objection 2 (Non-uniqueness of $\mathbf{R} = \mathbf{D} + \mathbf{B}$ in 3rd cumulant)** | Explicitly scoped the empirical Big Five finding to *"the fitted well-conditioned rank-3 covariance decomposition"*, noted in both Results and Appendix A that $p = 5, r = 3$ forms a 2D feasible manifold where near-boundary decompositions ($\lambda_3(\mathbf{B}) \to 0$) can inflate $\|\hat{\mathbf{T}}\|_F^2$, and added `sweep_feasible_rank3_decompositions()` in `src/arc_study/algebraic_complexity.py` + unit test in `tests/test_algebraic_complexity.py`. |
| **Title too weak / defensive** | Updated to **`The algebra of apparent psychological types: from covariance constraints to replicable profiles`**. |
| **Abstract (~140w) missing $\mathbf{B}$-whitening** | Tightened to ~140 words ($\le 150$, unstructured, unreferenced) and explicitly stated *"after whitening by between-component covariance $\mathbf{B}$"*. |
| **Box 1 weakened in v17** | Restored strong **Box 1 | A minimum evidential standard for clustering-based subtype claims** and upgraded Audit 1 to explicitly **separate covariance from distributional equivalence**, directly linking Theorems 1 & 2 to Box 1. |
| **Visual / LaTeX bugs in v17 PDF** | Replaced literal `**covariance**`, `**distribution**`, `**interpretation**` with `\textbf{...}`; removed `DRAFT` header/supplement labels; wrapped Data Availability URLs with `\sloppy` to eliminate margin overflows. |
| **GitHub 404s on public `main`** | Exact `git` commands provided below to commit and push the new algebraic module, audit script, tests, JSON summary, and docs once approved by the author. |
