# Resum Empíric Multidataset (BFI-2 ES + IPIP-NEO-300)

| Dataset | $N$ | $\lambda_2/\lambda_1$ | $D^2_{\mathrm{ARC}}$ ($p_{\mathrm{N3}}$) | Tucker $\bar{\phi}$ ($p_{\mathrm{N3}}$) | $\kappa$ ($p_{\mathrm{N1}}$ / $p_{\mathrm{N3}}$) | $\Delta\mathrm{BIC}_{\mathrm{LPA3-GMM1}}$ | Gerlach ($h=0.50, e>1.25$) P/N2/N3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| BFI-2 ES | 1,146 | 0.582 | 1.111 (`0.578`) | 0.574 (`0.666`) | 0.698 (`0.001` / `0.068`) | +64.2 | 4/1/1 |
| IPIP-300 Domains | 145,388 | 0.611 | 0.911 (`0.796`) | 0.670 (`0.826`) | 0.661 (`0.005` / `0.502`) | +24,924.0 | 4/1/1 |
| IPIP-300 Varimax | 145,388 | 1.000 | 1.313 (`0.348`) | 0.520 (`0.328`) | 0.556 (`0.030` / `0.020`) | -14,129.7 | 1/2/1 |

## Fitxers nous generats
- `results/figures/fig3_empirical_multidataset_contrast.{pdf,svg,png}`
- `results/figures/fig5_empirical_ablation_and_gerlach.{pdf,svg,png}`
- `results/table2_empirical_all_datasets.{csv,tex}`
