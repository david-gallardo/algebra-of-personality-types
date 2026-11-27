# 4. v16 → v17 Changelog & Final Substantive Assessment

**Status:** **VERIFIED ALGEBRAIC REVISION (`v17_algebraic_draft`) — READY FOR FINAL PDF REBUILD & VISUAL CHECK.**  
User execution completed: `verify_algebra.py` passed, `pytest -v` passed **44/44 tests in 24.40s**, and `scripts/run_algebraic_audit.py` completed in **86.4s** (`results/algebraic_audit/algebraic_audit_summary.json`). All verified empirical pentad, $d_K$, and homoscedastic third-order cumulant results are now integrated into `v17` (Level 1, Level 2, Methods, and Supplementary Table S3).

---

## 1. Key Changes from v16 to v17

| Area | v16 | v17 (`nature_human_behaviour_manuscript_v17_algebraic_draft.tex`) |
|---|---|---|
| **Architecture** | 5 empirical/geometric Results subsections | Reorganized explicitly into **three inferential levels**: **Level 1 (Covariance)** $\to$ **Level 2 (Distribution)** $\to$ **Level 3 (Interpretation)** |
| **Title & Abstract** | Correlated-traits-focused title; 145w Abstract | Tempered NHB title (*Replicable psychological profiles do not by themselves establish discrete types*); 145w Abstract integrating $K_{\mathrm{diag}}^{(2)}(\mathbf{R}) = 1 + r_+(\mathbf{R})$ and $\|T\|_F^2 \ge r(r-1)$ |
| **Appendix A (Covariance)** | Lower-bound-only covariance rank note | Replaced with **Theorem 1** (constructive weighted-simplex equality $K_{\mathrm{diag}}^{(2)}(\mathbf{R}) = 1 + r_+(\mathbf{R})$ with distinctness proof), explicit **$3 \times 3$ Heywood boundary counterexample** ($r_0 = 1 < 2 = r_+$), generic $4 \le K_{\mathrm{diag}}^{(2)} \le 5$, pentad certificate, discontinuity at $\mathbf{I}_p$, and compact 1-Lipschitz distance $d_K(\mathbf{R};\tau)$ |
| **Third-Order Restriction** | Absent | Added **Theorem 2** ($\|\mathbf{T}\|_F^2 = \sum \pi_k^{-1} - 3(r+1) + 2 \ge r(r-1)$) + homoscedastic corollary $K_{\mathrm{diag,hom}}^{(3)}(\mathbf{R}) \ge r_+(\mathbf{R}) + 2$, strictly scoped to minimal homoscedastic mixtures whitened by $\mathbf{B}$, with explicit **$K = 2r$ symmetric** and **heteroscedastic** counterexamples |
| **Appendix B (Fixed $K=3$)** | Spectral threshold $0.50764$ | Kept strictly separate from $K$-selection; completed formal proof per R1.2 (origin containment of collinear positive-weight means, log-concavity global optimality of Lloyd–Max, admissible partition vs. Voronoi cells) |
| **Editorial R1–R3 fixes** | Several over-strong phrases flagged in `informe_editorial_tres_revisions.html` | Added **Gerlach et al. (2019)** reply, **Molenaar & von Eye (1994)**, **Muthén (2004)**, **Shapiro (1982)**, **Drton et al. (2007)**, **Ning et al. (2015)**, **Mardia (1970)**, **Stroud (1971)**; tempered orientation stability, KDE scope, transect wording, reference-model & plus-one $p$ terminology, Table 1 observed vs. latent space column, Table 2 Jensen code-formula footnote (3/13 vs. 2/13), Czech C1 77%/23% split, and Box 1 framing |

---

## 2. Explicit Substantive Contribution Achieved vs. Remaining Limitations

### What Substantive New Contribution v17 Achieves (Beyond Costa 2002, Bauer & Curran 2003, Katahira 2020)
1. **Unifying Covariance Complexity with Higher-Order Restrictions:** Translates classical factor–profile covariance equivalence and MRFA into an exact constructive equality ($K_{\mathrm{diag}}^{(2)}(\mathbf{R}) = 1 + r_+(\mathbf{R})$) with an explicit boundary counterexample ($D \succ 0$ vs. $D \succeq 0$), and pairs it with an exact third-order cumulant identity ($\|\mathbf{T}\|_F^2 \ge r(r-1)$) proving that minimal homoscedastic covariance-matching mixtures of rank $r \ge 2$ cannot match a continuous Gaussian at third order.
2. **Direct Calibrated 5D Empirical Audit Across Personality and Psychopathology:** Moves beyond Katahira et al.'s (2020) synthetic skewed demonstration by mapping Gerlach et al.'s archived 5D KDE surface directly ($N = 145{,}388$ & $410{,}376$), decomposing cross-instrument and US–Czech HiTOP-SR profile congruence into elevation vs. shape under matched optimization, and contrasting hard vs. probabilistic representations across 86 real-world tasks.
3. **Counterexample-Safe Reporting Framework (Box 1):** Links each analytical failure mode to a calibrated audit while explicitly protecting against unimodal scale mixtures, overlapping classes, heteroscedastic mixtures, underpowered $N \approx 400$ cohorts, and non-clustering causal/biological evidence.

### Remaining Limitations (Must Stay Explicit Before Final Submission)
1. **Pending Execution Gate (`PROJECT_RULES.md`):** `verify_algebra.py`, `pytest -v` (including `tests/test_algebraic_complexity.py`), `scripts/run_algebraic_audit.py`, and `lualatex` compilation + visual proofing of `v17` must be run by the user and their outputs verified before inserting empirical pentad / third-cumulant numbers into a final submission PDF.
2. **Priority of Theorem 2:** While explicitly proved here and connected to Mardia (1970) and Stroud (1971) degree-3 cubature, formal priority in the Gaussian-mixture literature remains unconfirmed and is framed conservatively.
3. **Remaining Empirical Enhancements Identified in Editorial Review (Optional/P1):** (a) Sub-$h_{\mathrm{NN}}$ bandwidth persistence (`0.70–1.30 h_NN`) and unequal-weight / $N$-matched positive controls for IPIP-300; (b) participant-level / construct-clustered predictive uncertainty and tuned nonlinear comparators + external replication of the $+0.00610$ GMM-4 continuous gain; (c) re-exporting separate publication vector figures (`Fig. 1`–`Fig. 5`) matching v17 legends.
