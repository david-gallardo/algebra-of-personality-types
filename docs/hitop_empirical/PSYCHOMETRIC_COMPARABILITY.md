# Phase 1 — Psychometric Comparability & Measurement Audit (`PSYCHOMETRIC_COMPARABILITY.md`)

**Status:** Phase 1 QC complete — **NO empirical clustering executed.**

## 1. Sample Flow & Scale Crosswalk

- **Exact 76/76 scale match** verified between KU (`hitop 0.2.0`) and Czech (`dataset.csv`), with 0 duplicate IDs in either cohort.
- **Retention**: KU retains `411/411` (`100%` for 5/6 spectra); CZ retains `439/443` for 5 spectra and `439/443` for 6 spectra (excluding high-NA sexual scales).
- **Floor-effect warning**: `17/76` individual scales have $\ge 50\%$ of respondents at the exact floor (`1.0`) in at least one cohort (e.g., `Antisocial Behavior`, `Reality Distortion`, `Suicidality`, `NSSI`, `Paraphilias`), confirming why 76-scale raw clustering is inappropriate and why spectrum composites + floor-aware copulas (C1) are required.

## 2. Spectrum Composite Reliability & Cross-Language Loading Comparability

| Spectrum | # Scales | KU $\alpha$ / $\omega_t$ | CZ $\alpha$ / $\omega_t$ | Loading Tucker $\phi$ | KU Mean (1–4) | CZ Mean (1–4) | CZ $-$ KU ($d$) |
| :--- | ---: | :---: | :---: | ---: | ---: | ---: | ---: |
| **INT** | 11 | 0.866 / 0.879 | 0.905 / 0.909 | 0.995 | 1.63 | 2.03 | +0.81 |
| **DET** | 4 | 0.687 / 0.698 | 0.763 / 0.767 | 0.997 | 1.79 | 2.11 | +0.58 |
| **TD** | 6 | 0.851 / 0.869 | 0.856 / 0.866 | 0.999 | 1.47 | 1.77 | +0.68 |
| **A_EXT** | 8 | 0.842 / 0.859 | 0.861 / 0.867 | 0.999 | 1.59 | 1.70 | +0.28 |
| **D_EXT** | 8 | 0.711 / 0.737 | 0.758 / 0.784 | 0.972 | 1.47 | 1.63 | +0.47 |
| **SOM** | 5 | 0.795 / 0.822 | 0.828 / 0.840 | 0.998 | 1.52 | 1.88 | +0.75 |

## 3. Descriptive Correlation Geometry (Theoretical Benchmark Only — No Clustering)

- **5 spectra (['INT', 'DET', 'TD', 'A_EXT', 'D_EXT'])**:
  - KU ($N=411$): $\lambda_1=3.337$, $\lambda_2=0.830$, $\lambda_2/\lambda_1=0.249$, $\det(\mathbf{R})=0.048$, $|\mathbf{R}|^-0.5=4.583$.
  - CZ ($N=439$): $\lambda_1=3.353$, $\lambda_2=0.907$, $\lambda_2/\lambda_1=0.271$, $\det(\mathbf{R})=0.039$, $|\mathbf{R}|^-0.5=5.070$.
  - Cross-cohort off-diagonal correlation similarity: $r(\mathbf{R}_{\mathrm{KU}}, \mathbf{R}_{\mathrm{CZ}}) = 0.974$ ($\max |\Delta r| = 0.086$).
- **6 spectra (['INT', 'DET', 'TD', 'A_EXT', 'D_EXT', 'SOM'])**:
  - KU ($N=411$): $\lambda_1=3.841$, $\lambda_2=0.891$, $\lambda_2/\lambda_1=0.232$, $\det(\mathbf{R})=0.022$, $|\mathbf{R}|^-0.5=6.696$.
  - CZ ($N=439$): $\lambda_1=3.948$, $\lambda_2=0.932$, $\lambda_2/\lambda_1=0.236$, $\det(\mathbf{R})=0.014$, $|\mathbf{R}|^-0.5=8.360$.
  - Cross-cohort off-diagonal correlation similarity: $r(\mathbf{R}_{\mathrm{KU}}, \mathbf{R}_{\mathrm{CZ}}) = 0.957$ ($\max |\Delta r| = 0.086$).

## 4. Measurement Recommendation for Gate G2 Preregistration Lock

- **Configural / Metric Comparability**: All 6 pre-specified spectrum composites achieve very high 1-factor loading congruence across US English and Czech cohorts (see Tucker $\phi$ column above) and acceptable-to-high internal consistency ($\alpha, \omega_t$).
- **Scalar / Elevation Non-Comparability**: Because CZ has 69.1% mental health history vs. KU undergraduates, raw mean elevation differences ($d$) reflect clinical selection as well as potential scalar shifts. Therefore, **within-cohort z-standardization MUST be the primary representation**, with **elevation-removed shape congruence ($\bar{\phi}_{\text{shape}}$)** and **pooled-reference standardization** reported as pre-specified sensitivities.
- **Dimensionality**: Both **5 core spectra (`INT, DET, TD, A_EXT, D_EXT`)** and **6 spectra (`+SOM`)** have near-100% case retention and strong reliability. We recommend locking **5 core spectra** as primary (directly paralleling the 5-dimensional Big Five and HiTOP-5 space) and **6 spectra (`+SOM`)** as a pre-specified sensitivity.
