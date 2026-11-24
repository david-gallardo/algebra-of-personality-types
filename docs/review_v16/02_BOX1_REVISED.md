# P1 — Revised Strong Box 1: Two Candidate Versions & Selection

**Design objective:** Preserve Box 1 as an explicit, **strong minimum evidential and reporting standard** for claims that statistically discovered clusters represent discrete psychological or clinical kinds—without committing ontological overclaiming, conflating reporting requirements with logical necessities, or failing against known statistical and clinical counterexamples.

---

## Candidate Version A — Firm Full Standard (Recommended; 201 words)

> **Box 1 | A minimum evidential standard for claims of discrete psychological and clinical subtypes**
>
> **Scope.** Whenever discrete natural kinds are inferred primarily from unsupervised cluster profiles, density-enriched centres, or mixture-component counts, authors should complete and transparently report all three audits below. This is a minimum reporting standard for *warranted inference from clustering data*, not a claim that genuine latent classes must be multimodal or independently predictive, nor a bar to convergent causal, genetic, or biological evidence advanced through separate evidentiary pathways.
>
> **1. Challenge calibrated continuous alternatives.** Benchmark profile stability ($\kappa$), cross-sample congruence (both full and elevation-removed), density enrichment, and mixture selection against distribution-aware continuous nulls preserving covariance ($\mathbf{R}$) and empirical marginals or tail dependence. Apply identical scoring, initialization, and sample sizes to empirical and simulated data, reporting effect sizes and Monte Carlo uncertainty.
>
> **2. Test density topography only where identifiable.** Where a subtype hypothesis predicts distinct density peaks, map persistent modes and intervening valleys across modest bandwidth perturbations ($1.00\text{--}1.25\,h_{\mathrm{NN}}$), calibrated with positive-control mixtures at the exact observed $N$ and dimensionality $d$. If genuine overlapping or same-mean scale mixtures cannot be resolved at that $N$, report *uninformative*, never *unimodality confirmed*.
>
> **3. Evaluate independent distinction or utility.** Using held-out data and leak-free cross-validation, test whether subtype assignments add reproducible predictive, biological, or mechanistic information beyond continuous dimensional baselines. Null incremental prediction bounds demonstrated utility, not the logical existence of classes.
>
> **Interpretation.** Replicable clusters or multi-component fits justify descriptive summaries, not discrete kinds. Requiring all three audits to be *reported* does not require positive outcomes on every audit, and passing all three never by itself proves ontology.

---

## Candidate Version B — Firm Compact Standard (134 words)

> **Box 1 | A minimum evidential standard for clustering-based subtype claims**
>
> When discrete psychological or clinical kinds are inferred primarily from unsupervised profiles, enriched centres, or mixture fits, all three audits below should be reported as a minimum evidential standard (without implying that real latent classes must be multimodal or incrementally predictive, or excluding separate causal/biological evidence):
>
> **1. Continuous-null challenge.** Benchmark replicability (full and shape-only), enrichment, and mixture selection against continuous nulls preserving $\mathbf{R}$ and empirical marginals under identical preprocessing and optimization.
>
> **2. Calibrated topography.** Test persistent modes and valleys across bandwidths only alongside exact-$N$, exact-$d$ positive controls; report *uninformative* whenever separation is unresolvable (e.g., overlapping or same-mean scale mixtures).
>
> **3. Leak-free external distinction.** Test incremental predictive, biological, or mechanistic value beyond continuous baselines on held-out data.
>
> **Interpretation.** Reporting all three audits is mandatory for clustering-based ontological claims, whereas passing them is neither logically necessary for all latent classes nor sufficient proof of discrete kinds.

---

## Companion Main-Text Guardrail Paragraph (for Discussion)

To reinforce the distinction between *mandatory reporting* and *logical necessity*, the Discussion paragraph introducing Box 1 should state explicitly:

> Box 1 formalizes these findings as a minimum evidential and reporting standard whenever discrete kinds are inferred primarily from unsupervised clustering, enrichment tests, or mixture-component counts. Requiring all three audits to be *reported* is distinct from requiring positive outcomes on all three: genuine latent classes can share a single density mode (as in same-mean variance mixtures or heavily overlapping classes), can reflect non-substantive measurement artefacts (such as careless or extreme response-style mixtures), or can define clinically meaningful subgroups that do not improve short-term outcome prediction beyond continuous severity scores. Equally, passing all three audits strengthens empirical support against continuous baselines without logically proving ontology, and independent causal, genetic, or neurobiological evidence may establish subtypes via separate evidentiary pathways.

---

## Selection Recommendation

- **Choose Version A (201 words)** for `v16`.
- **Rationale:** Version A remains comfortably within the 180–240-word target, fits cleanly in a single half-page shaded box in `v16`, and explicitly names every epistemic guardrail (*uninformative* under low power, same-mean scale mixtures, full vs. elevation-removed congruence, separate biological/causal pathways, and the reporting-vs.-positive-outcome distinction) directly inside the Box so that a reader who copies or cites Box 1 alone cannot misread it as an oversimplified "checklist for ontology." Version B is retained as a fallback only if strict journal layout constraints require shortening the Box by ~70 words.
