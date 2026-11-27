# 1. Mathematical Audit — Covariance Component Complexity and Third-Order Restrictions

**Scope:** Independent line-by-line verification of `output/algebraic_derivation/covariance_component_complexity.tex` and `output/algebraic_derivation/verify_algebra.py`.  
**Execution status:** Per `PROJECT_RULES.md`, no scripts were executed in the terminal; all algebraic proofs, counterexamples, and symbolic checks below were audited analytically, and executable verification commands are listed in `03_EMPIRICAL_AUDIT_AND_EXECUTION_GUIDE.md`.

---

## Summary Table of Audited Claims

| Claim / Result | Mathematical Status | Key Condition / Correction Added |
|---|---|---|
| **Theorem 1:** $K_{\mathrm{diag}}^{(2)}(R) = 1 + r_+(R)$ | **VERIFIED** (constructive equality) | Added explicit proof that the $r+1$ simplex means are pairwise distinct ($\|z_i - z_j\|_2^2 = \pi_i^{-1} + \pi_j^{-1} > 0$) and affinely independent. Holds for *any* positive weight vector $\pi \in \Delta_{>0}^r$. |
| **Boundary distinction:** $D \succ 0$ vs. $D \succeq 0$ | **VERIFIED** (essential) | Constructed an explicit $3 \times 3$ strictly positive-definite correlation matrix $R \succ 0$ with $r_0(R) = 1$ (Heywood boundary $d_1 = 0$) but $r_+(R) = 2$, proving $K_{\mathrm{diag},0}^{(2)}(R) = 2 < 3 = K_{\mathrm{diag},+}^{(2)}(R)$. |
| **Universal bounds:** $1 \le K_{\mathrm{diag}}^{(2)}(R) \le p$ | **VERIFIED** | Attained at $K=1 \iff R = I_p$; attained at $K=p$ for negative equicorrelation $-1/(p-1) < \rho < 0$ (Perron–Frobenius proof verified). Discontinuity at $I_p$ confirmed. |
| **Generic five-trait bound:** $4 \le K_{\mathrm{diag}}^{(2)}(R) \le 5$ a.e. | **VERIFIED** (generic, NOT universal) | Holds for Lebesgue-almost every $5 \times 5$ $R \succ 0$ (Ledermann/Shapiro dimension count: rank 2 spans at most 9 off-diagonal dimensions vs. 10 free correlations). Non-generic exact 1- or 2-factor matrices have $K \in \{1, 2, 3\}$. |
| **Pentad certificate:** $P(R) \neq 0 \implies K_{\mathrm{diag}}^{(2)}(R) \ge 4$ | **VERIFIED** (necessary, NOT sufficient) | All 12 terms in `verify_algebra.py` verified as the 12 Hamiltonian 5-cycles of $K_5$ with alternating signs (Drton et al., 2007). $P(R)=0$ does *not* guarantee real PSD completion or $D \succ 0$. |
| **Theorem 2:** $\|T\|_F^2 = \sum_{k=1}^{r+1}\pi_k^{-1} - 3(r+1) + 2 \ge r(r-1)$ | **VERIFIED** (strictly scoped) | Valid **only** for minimal ($K = r+1$) **homoscedastic** Gaussian mixtures ($\epsilon \perp C$) whitened by $B$ ($WBW^\top = I_r$), **not** $R$. Yields new corollary $K_{\mathrm{diag,hom}}^{(3)}(R) \ge r_+(R) + 2$ for $r_+(R) \ge 2$ when matching both $R$ and $\cum_3 = 0$. Fails for heteroscedastic LPA and for over-complete ($K = 2r$) symmetric mixtures. |
| **Unrestricted multimodality:** same $R$ allows $1$ or arbitrarily many modes | **VERIFIED** | Full-covariance observed-space construction $X = R^{1/2} Z$ with $m \ge 2$ separated scalar bumps along $e_1$ preserves $\Cov(X) = R$ and creates $m$ modes as $t \to 1$, with no contradiction to Theorem 1. |
| **Sampling-aware diagnostic:** $d_K(R;\tau)$ is 1-Lipschitz | **VERIFIED** (with optimizer caveat) | Compactness of $\mathcal{C}_{K,\tau}$ and triangle-inequality 1-Lipschitzness verified. Local nonconvex optimization yields only an *upper bound* on $d_K$, never a certificate of impossibility; minimum trace $\neq$ minimum rank. |

---

## Detailed Proofs, Refinements, and Counterexamples

### 1. Theorem 1 — Constructive Equality & Distinctness of Means
Let $R \succ 0$ be a $p \times p$ correlation matrix and
$$r_+(R) = \min\{\operatorname{rank}(R - D) : D \succ 0 \text{ diagonal},\; R - D \succeq 0\}.$$
- **Necessity:** Any centered $K$-component mixture with positive weights $\pi_k > 0$ and strictly positive diagonal within-component covariances $D_k \succ 0$ satisfies $R = D + B$ with $D = \sum_k \pi_k D_k \succ 0$ diagonal and $B = M M^\top$ where $M = [\sqrt{\pi_1}\mu_1, \dots, \sqrt{\pi_K}\mu_K]$. Since $Ms = 0$ for $s = (\sqrt{\pi_k})_k \neq 0$, $\operatorname{rank}(B) = \operatorname{rank}(M) \le K - 1$. Thus $K \ge r_+(R) + 1$.
- **Attainment & Sufficiency:** Because $\lambda_{\min}(R) I_p$ is feasible and $\operatorname{rank}(R - D) \in \{0, 1, \dots, p-1\}$ is finite, the minimum $r = r_+(R)$ is attained at some $D^* \succ 0$ without requiring compactness. For $r = 0$, $R = I_p$ and $K = 1$ suffices. For $r \ge 1$, factor $R - D^* = L L^\top$ with $L \in \mathbb{R}^{p \times r}$ of full column rank $r$. Fix $K = r + 1$ and any positive probability vector $\pi$. Let $Q \in \mathbb{R}^{r \times K}$ have orthonormal rows spanning $s^\perp$ ($QQ^\top = I_r$, $Qs = 0$, hence $Q^\top Q = I_K - ss^\top$). Setting $z_k = q_k/\sqrt{\pi_k}$ and $\mu_k = L z_k$ gives $\sum_k \pi_k \mu_k = 0$ and $\sum_k \pi_k \mu_k \mu_k^\top = L Q Q^\top L^\top = R - D^*$.
- **Refinement (Pairwise Distinctness):** Because $z_i^\top z_j = \delta_{ij}/\pi_i - 1$, for all $i \neq j$:
  $$\|z_i - z_j\|_2^2 = (\pi_i^{-1} - 1) + (\pi_j^{-1} - 1) + 2 = \pi_i^{-1} + \pi_j^{-1} > 0.$$
  Since $L$ has full column rank $r$, $\mu_i \neq \mu_j$ for all $i \neq j$, so the $r + 1$ components are strictly non-degenerate and affinely independent.

### 2. Explicit Counterexample for the Boundary $D \succ 0$ vs. $D \succeq 0$
Define $r_0(R) = \min\{\operatorname{rank}(R - D) : D \succeq 0 \text{ diagonal},\; R - D \succeq 0\}$. Consider the $3 \times 3$ correlation matrix
$$R = \begin{pmatrix} 1 & 1/2 & 1/2 \\ 1/2 & 1 & 1/4 \\ 1/2 & 1/4 & 1 \end{pmatrix}.$$
- **Strict positive definiteness:** Leading principal minors are $1$, $3/4$, and $\det(R) = 9/16 > 0$, so $R \succ 0$.
- **Boundary rank $r_0(R) = 1$:** With $D_0 = \operatorname{diag}(0, 3/4, 3/4) \succeq 0$, $R - D_0 = u u^\top$ for $u = (1, 1/2, 1/2)^\top$, so $r_0(R) = 1$.
- **Strictly positive rank $r_+(R) = 2$:** Any rank-1 decomposition $R - D = v v^\top$ must satisfy $v_1 v_2 = 1/2$, $v_1 v_3 = 1/2$, $v_2 v_3 = 1/4$, which uniquely forces $v_1^2 = (v_1 v_2)(v_1 v_3)/(v_2 v_3) = 1$ and hence $d_1 = 1 - v_1^2 = 0$, violating $D \succ 0$. Meanwhile $D = \lambda_{\min}(R) I_3 \succ 0$ achieves rank $\le 2$. Thus **$r_0(R) = 1 < 2 = r_+(R)$**, demonstrating that Heywood boundary matrices separate degenerate ($D \succeq 0$) from non-degenerate ($D \succ 0$) diagonal Gaussian mixtures.

### 3. Theorem 2 — Third-Order Cumulant Identity, Corollary, and Counterexamples
Under $X = \mu_C + \epsilon$ with $K = r + 1 \ge 2$, $\mathbb{E}[\mu_C] = 0$, $\operatorname{rank}(B) = r$, $\epsilon \sim \mathcal{N}_p(0, D)$ independent of $C$, and $W B W^\top = I_r$:
- Cumulant additivity + vanishing Gaussian third cumulant yield $T = \cum_3(WX) = \sum_{k=1}^{r+1} \pi_k z_k^{\otimes 3}$.
- Orthogonality of $\begin{bmatrix} H \\ s^\top \end{bmatrix}$ yields $z_i^\top z_j = \delta_{ij}/\pi_i - 1$, and exact cancellation of $\pm \sum_i \pi_i^2$ yields
  $$\|T\|_F^2 = \sum_{k=1}^{r+1} \pi_k^{-1} - 3(r+1) + 2 \ge r(r-1),$$
  with equality iff $\pi_k = 1/(r+1)$.
- **Corollary (Homoscedastic Third-Moment Complexity):** If $r_+(R) \ge 2$, no minimal ($K = r_+(R) + 1$) homoscedastic diagonal Gaussian mixture can simultaneously match $\Cov(X) = R$ and $\cum_3(X) = 0$ (the third cumulant of $\mathcal{N}_p(0, R)$). Hence matching both second and third cumulants under homoscedastic diagonal LPA requires $K_{\mathrm{diag,hom}}^{(3)}(R) \ge r_+(R) + 2$.
- **Counterexample 1 (Over-complete symmetric mixture, $K = 2r$):** Take $K = 2r$ equal-weight components with $z_{2j-1} = +\sqrt{r}\,e_j$, $z_{2j} = -\sqrt{r}\,e_j$ ($j = 1, \dots, r$). Then $\mathbb{E}[Z] = 0$, $\Cov(Z) = I_r$, and by central symmetry **$T = \cum_3(WX) = 0$ identically**. Thus Theorem 2 constrains *minimal* ($K = r+1$) representations, not all homoscedastic mixtures.
- **Counterexample 2 (Heteroscedastic mixtures):** When $D_k$ depends on $C = k$, $\epsilon$ is no longer independent of $C$ and $\cum_3(WX) = \sum_k \pi_k z_k^{\otimes 3} + \sum_k \pi_k \operatorname{sym}(z_k \otimes W D_k W^\top)$, where the mean–variance cross-term can offset the mean-simplex skewness. Thus Theorem 2 **cannot** be used to reject heteroscedastic LPA or general GMMs.
- **Whitening distinction ($B$ vs. $R$):** $W$ whitens $B = R - D$, not $R$. Observed-score Mardia skewness (whitened by $R$) is attenuated by noise $D$ and depends on signal-to-noise eigenvalues; conversely, empirical estimation of $W$ from $\hat{B} = \hat{R} - \hat{D}$ is sensitive to small nonzero eigenvalues of $\hat{B}$ and requires joint bootstrap uncertainty over $(\hat{D}, \hat{B}, \hat{W})$.
