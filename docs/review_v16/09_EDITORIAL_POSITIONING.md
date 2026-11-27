# P3 — Editorial Positioning for *Nature Human Behaviour*

## 1. Format Recommendation: Article vs. Methods-Oriented Contribution

- **Recommended format:** **Article** (broad interdisciplinary empirical + theoretical contribution) rather than a narrow *Technical Report* or *Matters Arising*.
- **Rationale:** Under current *Nature Human Behaviour* author guidelines, **Articles** are original research reports of broad significance across the behavioural and social sciences (typical main text ~3,000–4,500 words, ~150-word unreferenced Abstract, Methods at the end, up to ~6 main display items + Extended Data). Our manuscript combines (i) general mathematical theory, (ii) a $N = 563{,}886$ cross-domain empirical test spanning both personality psychology and clinical psychopathology (HiTOP-SR), (iii) an external predictive benchmark across 86 real-world life outcomes, and (iv) a generalizable three-audit minimum evidential standard (Box 1) applicable to psychiatry, neuroscience, economics, and behavioural science. Framing it as an **Article** about a universal geometric inference trap—rather than a single-paper rebuttal—best matches NHB's editorial mandate. *(Note: No acceptance probability is asserted as fact; editorial decisions depend on competing submissions and reviewer evaluations.)*

---

## 2. 130-Word Significance Pitch (for Cover Letter)

> Across psychology, psychiatry, neuroscience, and behavioural economics, researchers routinely infer discrete human types—personality classes, clinical subtypes, or neurophysiological biotypes—whenever clustering algorithms replicate across samples or exceed shuffled null distributions. Here we prove analytically and demonstrate empirically across $N = 563{,}886$ participants from six personality and psychopathology cohorts that correlated continuous distributions naturally generate replicable non-collinear clusters, apparent density enrichment, and multi-class mixture solutions without discrete subpopulations. Direct five-dimensional topographic mapping shows that all 13 centres in the landmark Gerlach et al. (2018) *Nature Human Behaviour* personality space collapse to a single density mode, while matched continuous nulls reproduce both cross-instrument Big Five profiles and US--Czech HiTOP-SR psychopathology profiles. Because five linear traits decisively outperform hard typologies across 86 life outcomes, we introduce a calibrated three-audit evidential standard to prevent premature reification of clusters across the behavioural sciences.

---

## 3. Why This Matters Outside Personality Psychology

1. **Psychiatry & Clinical Nosology:** Unsupervised LPA/GMM subtyping of symptom scales (DSM clusters, depression/psychosis subtypes, HiTOP profiles) routinely treats correlated, right-skewed symptom continua as discrete syndromes. Our HiTOP-SR validation shows that full-profile congruence ($\bar{\phi} = 0.935$) and full-GMM $K \ge 2$ extraction (100/100 copula draws) emerge without latent classes, and that removing overall symptom elevation collapses shape replicability.
2. **Neuroimaging & Biotyping:** High-dimensional resting-state connectivity and structural MRI features exhibit strong intercorrelations and non-Gaussian tails; applying $k$-means, canonical correlation, or uncalibrated mixture models produces replicable "biotypes" via the exact same rotational symmetry breaking and covariance-splitting mechanics (Drysdale et al., 2017; Dinga et al., 2019).
3. **Behavioural Economics & Computational Social Science:** Studies clustering players in economic games or digital-trace behaviours (Poncela-Casasnovas et al., 2016) frequently benchmark against independent permutations, which inevitably inflate $|\mathbf{R}|^{-1/2}$ density ratios whenever behaviours co-vary.

---

## 4. Five Likely Editor Objections & Evidence-Based Responses

1. **"Didn't Costa et al. (2002) and Bauer & Curran (2003) already show this 20 years ago?"**
   - *Response:* They showed empirically/via simulation that continuous data *can* yield clusters or over-extracted mixture classes, but did not derive the exact spectral quantization threshold ($\lambda_2/\lambda_1 > 0.50764$) and $|\mathbf{R}|^{-1/2}$ enrichment identity, did not map the 5D KDE topography of Gerlach et al.'s $N > 550{,}000$ space with calibrated positive controls, did not run matched $2 \times 2$ cross-instrument and cross-cultural HiTOP-SR decompositions separating elevation from shape, and did not synthesize these failure modes into a unified, counterexample-safe evidential standard.
2. **"Isn't this just a *Matters Arising* critique of Gerlach et al. (2018)?"**
   - *Response:* Gerlach et al. (2018) is only one of four converging empirical tests (Section 2 of Results) illustrating a general geometric law that equally governs ARC typologies, BFI-2 vs. IPIP-300 cross-instrument replication, meta-analytic and individual-level HiTOP psychopathology, and 86 external life outcomes.
3. **"Your Box 1 standard could be criticized as demanding too much—real latent classes can be unimodal (e.g., variance mixtures) or lack incremental prediction."**
   - *Response:* Revised Box 1 explicitly distinguishes *mandatory reporting* of three audits from *requiring positive outcomes* on all three, explicitly carves out same-mean scale mixtures, overlapping mixtures, response-style mixtures, and non-clustering causal/genetic/biological pathways, and mandates reporting *uninformative* whenever exact-$N$ positive controls lack power.
4. **"Your HiTOP-SR cohorts ($N = 411$ and $439$) are community samples, not diagnosed clinical patients, and are underpowered for 5D KDE."**
   - *Response:* We transparently label them as US undergraduate and Czech community cohorts (with 69.1% self-reported mental-health difficulties), pair them with the $N = 120{,}596$ clinical HiTOP MASEM benchmark (Ringwald et al., 2023), and turn the $N \approx 400$ KDE limitation into a central methodological lesson backed by 400 archived positive-control replicates (25/25 collapse to 1 mode even at $\Delta = 4.0$ SD).
5. **"Why should NHB care if probabilistic GMM-4 actually adds $+0.0061$ $R^2$ in Soto (2019)?"**
   - *Response:* Reporting that modest, subsample-concentrated continuous gain transparently strengthens credibility: it shows we are not dogmatically anti-mixture, while proving that hard canonical typologies (ARC, Gerlach-4) lose $>50\%$ of predictive variance and add zero increment over five linear traits.
