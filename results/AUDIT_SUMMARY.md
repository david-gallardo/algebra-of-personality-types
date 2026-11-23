# Auditoria Final dels Resultats — Algebra of Personality Types (ARC)

- **Data de generació**: 2026-11-21 21:42:01
- **Llavor base (PROJECT_RULES.md)**: `20261121` (empíric: `20261121`; add-on Gerlach: `[20261121, 2018]`)
- **Blocs Monte Carlo carregats**: 28 fitxers (`28,000` rèpliques totals; mides mostrals $N$: [300, 1000, 5000, 50000])
- **Add-on Gerlach corregit**: `2,800` rèpliques MC ($N \in [1000, 5000]$, `B=200`, $\alpha=0.01$) + BFI-2 empíric (`B=500`) i `300` rèpliques nul·les BFI-2
- **Contrast empíric**: `bfi2_es_study1` ($N = 1,146$, `1000` rèpliques per model nul; `3,000` rèpliques totals)

## 1. Diagnòstics empírics del BFI-2 espanyol

- **Mida mostral neta**: $N = 1,146$ (files eliminades per NA: `0`)
- **Ordre estricte de dimensions**: `['N', 'E', 'O', 'A', 'C']` (`N` sense invertir)
- **Valors propis de $\mathbf{R}_{\mathrm{emp}}$**: `1.79233`, `1.04305`, `0.85277`, `0.77927`, `0.53258`
- **Ràtio $\lambda_2 / \lambda_1$**: `0.58195` vs. llindar suficient de no-col·linealitat `0.50764` (**supera el llindar**, marge `+0.07431`)
- **Determinant $\det(\mathbf{R}_{\mathrm{emp}})$**: `0.66165` (definida positiva: `True`)
- **Asimetria marginal**: N=`+0.027`, E=`-0.286`, O=`-0.355`, A=`-0.539`, C=`-0.092`
- **Curtosi d'excés marginal**: N=`-0.382`, E=`-0.100`, O=`-0.338`, A=`+0.475`, C=`-0.315`
- **Calibració còpula Nul 3**: error màxim absolut `9.23e-05`, convergit=`True` (2 iteracions), unimodalitat conjunta=`True` (modes marginals=[1, 1, 1, 1, 1], PC12=1)

## 2. Centroides empírics alineats a ARC (Ward + k-means, $K=3$)

| Tipus ARC | Proporció | N | E | O | A | C | $D^2$ | Tucker $\phi$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Resilient | 35.8% | -0.46 | +0.69 | +0.65 | +0.58 | +0.57 | 0.203 | 0.948 |
| Overcontrolled | 34.1% | -0.30 | -0.21 | -0.94 | -0.10 | -0.01 | 2.009 | 0.093 |
| Undercontrolled | 30.1% | +0.88 | -0.59 | +0.29 | -0.58 | -0.67 | 1.122 | 0.681 |

## 3. Xifres clau del Monte Carlo (resum a $N = 50,000$ i add-on Gerlach a $N = 5,000$)

### 3a. Mètriques principals a $N = 50,000$ (1,000 rèpliques/bloc)

| Condició | $D^2_{\mathrm{ARC}}$ | Tucker $\bar{\phi}$ | $\kappa$ split-half | $\Delta$BIC (3-diag $-$ 1-full) |
| :--- | ---: | ---: | ---: | ---: |
| Null 1: Isotropic N(0, I) | 1.489 [0.720, 2.153] | 0.439 | 0.290 | +258.8 |
| Null 2: MVN (vdL) | 0.792 [0.620, 0.970] | 0.695 | 0.604 | +3637.8 |
| Null 2: MVN (BE) | 0.935 [0.642, 1.274] | 0.636 | 0.587 | +3067.8 |
| Null 2: MVN hetero (vdL) | 0.985 [0.407, 1.757] | 0.643 | 0.641 | +8371.1 |
| Null 2: MVN hetero (BE) | 1.062 [0.472, 1.808] | 0.616 | 0.649 | +7808.8 |
| Null 3: Copula (vdL) | 0.710 [0.638, 0.783] | 0.773 | 0.949 | -7315.3 |
| Null 3: Copula (BE) | 0.758 [0.614, 1.107] | 0.787 | 0.749 | -7873.5 |

### 3b. Add-on Gerlach et al. (2018) corregit ($B=200$, $\alpha=0.01$, $h=0.50$ isotròpic, 200 rèpliques/bloc)

| Condició | $N$ | Perm ($e>1.25$) [$\% \ge 1$] | Nul 2 ($e>1.25$) [$\% \ge 1$] | Nul 3 ($e>1.25$) [$\% \ge 1$] | Perm ($e>1.00$) | Scott Perm ($e>1.25$) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Null 1: Isotropic N(0, I) | 1,000 | 0.10 [10%] | 0.07 [7%] | 0.06 [6%] | 0.12 | 0.12 |
| Null 1: Isotropic N(0, I) | 5,000 | 0.00 [0%] | 0.00 [0%] | 0.00 [0%] | 0.07 | 0.03 |
| Null 2: MVN (vdL) | 1,000 | 2.31 [98%] | 0.07 [7%] | 0.03 [2%] | 2.35 | 2.62 |
| Null 2: MVN (vdL) | 5,000 | 2.15 [100%] | 0.00 [0%] | 0.00 [0%] | 3.72 | 3.10 |
| Null 2: MVN (BE) | 1,000 | 2.40 [100%] | 0.01 [2%] | 0.01 [1%] | 2.44 | 2.75 |
| Null 2: MVN (BE) | 5,000 | 2.13 [100%] | 0.00 [0%] | 0.00 [0%] | 3.69 | 2.98 |
| Null 2: MVN hetero (vdL) | 1,000 | 2.68 [94%] | 0.04 [4%] | 0.03 [3%] | 2.69 | 3.06 |
| Null 2: MVN hetero (vdL) | 5,000 | 3.04 [98%] | 0.00 [0%] | 0.00 [0%] | 3.89 | 3.67 |
| Null 2: MVN hetero (BE) | 1,000 | 2.74 [96%] | 0.05 [5%] | 0.03 [3%] | 2.77 | 3.13 |
| Null 2: MVN hetero (BE) | 5,000 | 3.06 [99%] | 0.00 [0%] | 0.00 [0%] | 3.90 | 3.65 |
| Null 3: Copula (vdL) | 1,000 | 2.14 [98%] | 1.17 [100%] | 0.02 [2%] | 2.23 | 2.66 |
| Null 3: Copula (vdL) | 5,000 | 2.17 [100%] | 1.00 [100%] | 0.00 [0%] | 3.75 | 3.41 |
| Null 3: Copula (BE) | 1,000 | 2.22 [98%] | 1.15 [100%] | 0.03 [2%] | 2.31 | 2.82 |
| Null 3: Copula (BE) | 5,000 | 2.20 [100%] | 1.00 [100%] | 0.01 [0%] | 3.75 | 3.34 |

## 4. Contrast empíric BFI-2 espanyol ($N = 1,146$)

| Mètrica | Empíric | Nul 1 ($p_{\mathrm{emp}}$) | Nul 2 ($p_{\mathrm{emp}}$) | Nul 3 ($p_{\mathrm{emp}}$) |
| :--- | ---: | ---: | ---: | ---: |
| Global ARC D^2 (Ward+k-means) | 1.111 | 1.550 (`0.182`) | 1.151 (`0.523`) | 1.123 (`0.578`) |
| Elevation (Nivell) | 0.195 | 0.294 (`0.467`) | 0.312 (`0.476`) | 0.295 (`0.449`) |
| Scatter (Dispersio) | 0.111 | 0.070 (`0.797`) | 0.058 (`0.891`) | 0.066 (`0.831`) |
| Shape (Forma) | 0.805 | 1.187 (`0.211`) | 0.781 (`0.524`) | 0.761 (`0.555`) |
| Tucker congruence (mean) | 0.574 | 0.439 (`0.221`) | 0.572 (`0.549`) | 0.603 (`0.666`) |
| Split-half kappa (Asendorpf) | 0.698 | 0.290 (`0.000`) | 0.543 (`0.060`) | 0.548 (`0.067`) |
| Delta BIC (LPA 3-diag - GMM 1-full) | 64.158 | 4.953 (`0.000`) | 186.105 (`1.000`) | 107.567 (`0.958`) |
| Delta BIC (LPA 3-diag - 1-diag) | -338.724 | 65.660 (`0.000`) | -227.363 (`0.001`) | -307.221 (`0.225`) |
| Delta BIC (GMM 3-full - 1-full) | 150.645 | 177.700 (`0.024`) | 253.304 (`0.000`) | 167.606 (`0.183`) |
| Gerlach iso h=0.50 e>1.25 (Perm) | 4.000 | 0.110 (`0.000`) | 2.110 (`0.010`) | 2.040 (`0.030`) |
| Gerlach iso h=0.50 e>1.25 (Null 2) | 1.000 | 0.050 (`0.050`) | 0.040 (`0.040`) | 0.100 (`0.100`) |
| Gerlach iso h=0.50 e>1.25 (Null 3) | 1.000 | 0.020 (`0.020`) | 0.010 (`0.010`) | 0.030 (`0.030`) |
| Gerlach iso h=0.50 e>1.00 (Perm) | 4.000 | 0.120 (`0.000`) | 2.200 (`0.040`) | 2.060 (`0.030`) |
| Gerlach iso h=0.50 e>1.00 (Null 2) | 1.000 | 0.050 (`0.050`) | 0.050 (`0.050`) | 0.100 (`0.100`) |
| Gerlach iso h=0.50 e>1.00 (Null 3) | 1.000 | 0.020 (`0.020`) | 0.010 (`0.010`) | 0.030 (`0.030`) |
| Gerlach Scott e>1.25 (Perm) | 4.000 | 0.120 (`0.000`) | 2.550 (`0.100`) | 2.380 (`0.100`) |
| Gerlach Scott e>1.25 (Null 2) | 1.000 | 0.040 (`0.040`) | 0.040 (`0.040`) | 0.040 (`0.040`) |
| Gerlach Scott e>1.25 (Null 3) | 1.000 | 0.030 (`0.030`) | 0.020 (`0.020`) | 0.040 (`0.040`) |

## 5. Estat de les hipòtesis

1. **H1 — Geometria espectral ($\lambda_2 / \lambda_1$ vs. `0.50764`)**: Al BFI-2 espanyol $\lambda_2 / \lambda_1 = 0.58195$ (supera el llindar suficient de no-col·linealitat `0.50764`).
2. **H2 — Emergència d'ARC sota nuls unimodals correlacionats**: **CONFIRMAT** — a $N=50,000$, $D^2_{\mathrm{ARC}}$ cau de `1.489` (Nul 1 isòtrop) a `0.944` (Nul 2 MVN) i `0.734` (Nul 3 còpula).
3. **H3 — Artefacte de selecció de models (LPA diagonal vs. GMM covariància plena) i sobreextracció per no-normalitat**: **CONFIRMAT (bifurcació Nul 2 vs. Nul 3)** — 
   - **Sota Nul 2 (normalitat correlacionada)**: GMM 1-full supera sistemàticament LPA 3-diag a totes les mides mostrals ($\Delta\mathrm{BIC} > 0$, arribant a `+3,637.8` per vdL i `+3,067.8` per BE a $N=50,000$, i `+64.2` al BFI-2 empíric).
   - **Sota Nul 3 (còpula unimodal amb asimetria moderada)**: a partir de $N=1{,}000$, LPA 3-diag supera fins i tot GMM 1-full ($\Delta\mathrm{BIC} < 0$, arribant a `-7,315.3` per vdL i `-7,873.5` per BE a $N=50,000$), il·lustrant la sobreextracció per no-normalitat de Bauer i Curran (2003).
4. **H4 — Falsos positius de Gerlach et al. (2018) sota nul independent vs. nuls correlacionats (add-on corregit, $B=200$, $\alpha=0.01$, KDE isotròpica $h=0.50$, $e>1.25$)**: **CONFIRMAT** — a $N=5,000$:
   - **Dades Nul 2 (MVN correlacionada)**: el nul de permutació independent detecta `2.60` tipus significatius de mitjana (`99.1%` de rèpliques amb $\ge 1$ fals positiu), mentre que el nul correlacionat Nul 2 el col·lapsa a `0.00` (`0.0%`) i Nul 3 a `0.00`.
   - **Dades Nul 3 (còpula asimètrica unimodal)**: el nul de permutació detecta `2.18` tipus, Nul 2 (MVN) encara genera `1.00` falsos positius per asimetria marginal, i Nul 3 (còpula calibrada) ho col·lapsa a `0.00` (`0.2%`).
   - **BFI-2 espanyol empíric ($B=500$)**: amb KDE isotròpica $h=0.50$ ($e>1.25$), Perm=`4/4`, Nul 2=`1/4`, Nul 3=`1/4` (amb Scott $e>1.25$: Perm=`4/4`, Nul 2=`1/4`, Nul 3=`1/4`).
5. **H5 — Contrast empíric BFI-2 contra nuls correlacionats**: **CONSISTENT AMB NUL UNIMODAL CORRELACIONAT** — per a $D^2_{\mathrm{ARC}}$, $p_{\mathrm{emp}}(\text{Nul 2}) = 0.523$ i $p_{\mathrm{emp}}(\text{Nul 3}) = 0.578$; per a $\kappa$ split-half, $p_{\mathrm{emp}}(\text{Nul 3}) = 0.067$.

## 6. Fitxers generats

- `results/table1_montecarlo_summary.csv`
- `results/table1_montecarlo_summary.tex`
- `results/table2_empirical_bfi2.csv`
- `results/table2_empirical_bfi2.tex`
- `results/figures/fig1_mc_arc_similarity.pdf`
- `results/figures/fig1_mc_arc_similarity.svg`
- `results/figures/fig2_mc_bic_and_kappa.pdf`
- `results/figures/fig2_mc_bic_and_kappa.svg`
- `results/figures/fig3_empirical_bfi2_contrast.pdf`
- `results/figures/fig3_empirical_bfi2_contrast.svg`
- `results/figures/fig4_gerlach_significant_types.pdf`
- `results/figures/fig4_gerlach_significant_types.svg`
