# Phase 0 — Methods & Measurement Decisions (`METHODS_DECISIONS.md`)

**Status:** PROPOSED FOR SCIENTIFIC REVIEW (Pre-lock; no empirical clustering run).

## 1. Realistic Construct Harmonization for Direct KU vs. Czech HiTOP-SR (Tier A)

Because $N_{\text{KU}} = 411$ and $N_{\text{CZ}} = 443$ cannot support unconstrained 76-indicator factor models or 76D density estimation, and because we must **not** force Ringwald et al.'s 5 DSM Geomin factors onto HiTOP-SR, we propose **pre-specified, theory-anchored unit-weighted (standardized-scale) HiTOP spectrum composites** built strictly from published HiTOP-SR construct definitions (Simms et al., 2022; Kotov et al., 2017; HiTOP Society 2024):

| Proposed Spectrum Composite | Candidate Shared HiTOP-SR Primary Scales (exact in both KU & CZ) | Expected Coverage & Missingness |
| :--- | :--- | :--- |
| **1. Internalizing (INT)** | `Distress-Dysphoria`, `Social Anxiety`, `Panic`, `Agoraphobia`, `Trauma Reactions`, `Insomnia`, `Nightmares`, `Suicidality`, `NSSI` (`nonSuicidalSelfInjury`), `Appetite Loss`, `Well-being` (reverse-keyed) | Excellent coverage; very low NA (<1% in both cohorts). |
| **2. Detachment (DET)** | `Social Aloofness`, `Restricted Affectivity`, `Romantic Disinterest`, `Submissiveness` (check loading/cross-loading in Phase 1) | Good coverage; low NA (<1%). |
| **3. Thought Disorder (TD)** | `Reality Distortion`, `Eccentricity`, `Fantasy Proneness`, `Disorganization`, `Dissociation`, `Mistrust` | Excellent coverage; low NA (<1%). |
| **4. Antagonistic Externalizing (A-EXT)** | `Callousness`, `Dishonesty`, `Domineering`, `Entitlement`, `Exhibitionism`, `Grandiosity`, `Social Aggression`, `Oppositionality` | Excellent coverage; low NA (<1%). |
| **5. Disinhibited Externalizing (D-EXT)** | `Antisocial Behavior`, `Non-persistence`, `Non-planfulness`, `Risk Taking`, `Risky Sex`, `Problematic Shopping`, `Gambling`, `Gaming` (note: core SUD/AUD scales are absent from this 76-scale HiTOP-SR release except `Sex-Related Substance Use`, which has 4.1% NA in CZ!) | **Moderate/partial coverage — flag in Phase 1!** Must verify reliability ($\omega$, $\alpha$) and check whether 5 spectra vs. 6 spectra (adding **Somatoform**: `Bodily Distress`, `Conversion Symptoms`, `Disease Conviction`, `Health Anxiety`, `Somatic Preoccupation`) is better supported psychometrically. |
| **6. Somatoform (SOM) — Candidate 6th Spectrum** | `Bodily Distress`, `Conversion Symptoms`, `Disease Conviction`, `Health Anxiety`, `Somatic Preoccupation` | Very clean 5-scale composite with low NA! |

**Decision required from authors at Gate G1 (before locking `config/study.yaml`):**
1. Confirm exact scale-to-spectrum assignment from the official HiTOP-SR structural mapping (and decide whether primary dimensionality is **5 core spectra [INT, DET, TD, A-EXT, D-EXT]** or **6 spectra [adding SOM]**; sexual functioning scales with 6.8–15.1% NA in CZ should be excluded from primary composites to preserve $N \approx 407$ KU / $N \approx 415+$ CZ without imputation).
2. Confirm primary standardization reference: **within-cohort z-standardization** (primary, because CZ is 69.1% clinically enriched vs. KU university sample) with **pooled-reference standardization** as a labeled sensitivity analysis.

## 2. Primary Endpoint & Multiplicity Control

- **Single Primary Confirmatory Endpoint (Q2):** Cross-cohort matched **full-profile Tucker congruence** ($\bar{\phi}_{\text{full}}$) at **$K = 3$** under matched Ward + multistart *k*-means (1 Ward + 9 *k*-means++, min SSE) evaluated against **C0 (Correlated MVN preserving $\hat{\mathbf{R}}_{\text{KU}}$ and $\hat{\mathbf{R}}_{\text{CZ}}$)** using $B = 5{,}000$ independent pairs, exact $p_{\text{MC}} = (1 + k)/(B + 1)$ and 95% Wilson CI.
- **Key Secondary / Corroborative Endpoints (Holm–Bonferroni controlled within family):**
  - Cross-cohort **elevation-removed shape congruence** ($\bar{\phi}_{\text{shape}}$) at $K = 3$ and $K = 4$ under I0, I1, C0, C1.
  - Within-cohort split-half $\kappa$, ARI, NMI ($K = 3, 4$) against I0, I1, C0, C1.
  - 5D/6D KDE mean-shift topography across $h \in \{0.8, 1.0, 1.15, 1.25, 1.5\} \times h_{\text{NN}}$ with positive-control calibration at exact $N = 411$ and $N = 443$.
  - Single-start Ward sensitivity protocol (matched strictly against its own empirical benchmark).
  - External incremental prediction in Czech cohort (PANAS Positive/Negative Affect — clearly labeled as *same-source self-report affect*, plus `Mental_health` / `Treatment` clinical history as binary criteria) using leak-free nested 10-fold CV vs. Ridge, Poly-2, Spline-GAM, and HGB.

## 3. Reconciliation of Ringwald et al. (2023) 23 vs. 21 Diagnoses

Verified directly from `data/raw/hitop_ringwald2023/hitop_masem_output.out` (lines 13–20) and `hitop_supplement.docx`:
- The pooled meta-analytic dataset includes **23 DSM diagnoses** (`NAMES ARE`: `alc arx asp avp bip bpd dpd drg dys gad hpd mdd npd ocd ocp pnc ppd psy pts scl sph stp szp`).
- The primary Mplus EFA (`USEVARIABLES ARE`) excludes **2 diagnoses**: **`bip` (Bipolar disorder)** and **`dpd` (Dependent personality disorder)** (sensitivity with `dpd` is reported in Supplement Table 4).
- Ringwald et al.'s matrices remain **strictly an aggregate meta-analytic benchmark**, never equated with individual-level HiTOP-SR scores.
