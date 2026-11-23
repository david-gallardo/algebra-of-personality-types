#!/usr/bin/env python3
"""Generate final Via A figures:
  - results/figures/fig6_via_a_topography_and_control.{pdf,svg,png}
  - results/figures/fig7_via_a_soto2019_predictive.{pdf,svg,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.2,
    "axes.titlesize": 10.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "legend.fontsize": 8.2,
    "savefig.dpi": 300,
})

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def exact_kde(X: np.ndarray, pts: np.ndarray, bw: float, chunk: int = 50_000) -> np.ndarray:
    n, d = X.shape
    k = pts.shape[0]
    inv = 0.5 / (bw * bw)
    p2 = np.sum(pts * pts, axis=1)
    acc = np.zeros(k, dtype=np.float64)
    for s in range(0, n, chunk):
        xb = X[s : s + chunk]
        x2 = np.sum(xb * xb, axis=1, keepdims=True)
        sq = np.maximum(0.0, x2 + p2[None, :] - 2.0 * (xb @ pts.T))
        acc += np.sum(np.exp(-sq * inv), axis=0)
    return acc / (float(n) * ((2.0 * np.pi * bw * bw) ** (0.5 * d)))


def save_fig(fig: plt.Figure, stem: Path) -> None:
    for ext in (".pdf", ".svg", ".png"):
        fig.savefig(stem.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def build_fig6_topography_and_control() -> None:
    g_res = json.loads((ROOT / "results/gerlach_faithful/ipip300_faithful_k13.json").read_text())["gerlach_ipip300_fa_varimax"]
    v_res = json.loads((ROOT / "results/gerlach_faithful/final_via_a_checks.json").read_text())
    X = np.load(ROOT / "external/gerlach2018/data_filter/ipip300-no0_arr_pd_neoac_score-1.npy").astype(np.float64)

    centers = np.array(g_res["centers_neoac"], dtype=np.float64)
    bw = float(g_res["bw_nn"])
    ev = g_res["evaluations"]
    enr_perm = np.array(ev["permutation"]["enrichment_gerlach"])
    p_perm = np.array(ev["permutation"]["pval_gerlach"])
    order = np.argsort(enr_perm)[::-1]
    sig_idx = [int(i) for i in order if enr_perm[i] > 1.25 and p_perm[i] < 0.01]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8), constrained_layout=True)

    # --- Panel A: Faithful Gerlach K=13 top 6 clusters across 4 nulls ---
    ax = axes[0, 0]
    top6 = order[:6]
    null_specs = [
        ("permutation", "Perm. (Null 1)", "#7F7F7F"),
        ("mvnormal", "MVN (Null 2)", "#0072B2"),
        ("gaussian_copula", "Gauss Copula (Null 3)", "#009E73"),
        ("t_copula", "Student-t Copula (Null 4)", "#CC79A7"),
    ]
    x = np.arange(len(top6))
    w = 0.19
    ax.axhline(1.25, color="#D55E00", ls="--", lw=1.2, label="Threshold ($e=1.25$)")
    ax.axhline(1.00, color="#666666", ls=":", lw=0.9)
    for j, (nk, lbl, col) in enumerate(null_specs):
        vals = np.array(ev[nk]["enrichment_gerlach"])[top6]
        pvs = np.array(ev[nk]["pval_gerlach"])[top6]
        pos = x + (j - 1.5) * w
        bars = ax.bar(pos, vals, width=w * 0.92, color=col, alpha=0.88, edgecolor="black", lw=0.45, label=lbl)
        for b, v, p in zip(bars, vals, pvs):
            if v > 1.25 and p < 0.01:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.03, "*", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cl {i+1}" + ("*" if top6[i] in sig_idx else "") for i in range(len(top6))])
    ax.set_ylim(0.0, 2.15)
    ax.set_ylabel("Density enrichment ($\\hat{f}_{\\mathrm{emp}}/\\bar{f}_{\\mathrm{null}}$)")
    ax.set_title("A. Faithful Gerlach ($K=13$, $h_{\\mathrm{NN}}=0.254$, $B=1000$)\nIPIP-300 FA+Varimax ($N=145,388$)", loc="left")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=7.6, ncol=2)

    # --- Panel B: 6 Pairwise KDE density transects between the 4 significant centers ---
    ax = axes[0, 1]
    t_grid = np.linspace(0.0, 1.0, 41)
    pair_colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00", "#56B4E9"]
    p_i = 0
    for a in range(len(sig_idx)):
        for b in range(a + 1, len(sig_idx)):
            ia, ib = sig_idx[a], sig_idx[b]
            seg = (1.0 - t_grid)[:, None] * centers[ia][None, :] + t_grid[:, None] * centers[ib][None, :]
            dens = exact_kde(X, seg, bw) * 1e3
            ax.plot(t_grid, dens, lw=2.0, color=pair_colors[p_i], label=f"Cl {a+1}* <-> Cl {b+1}*")
            ax.scatter([0.0, 1.0], [dens[0], dens[-1]], s=24, color=pair_colors[p_i], zorder=4)
            p_i += 1
    ax.set_xlabel("Linear interpolation $t$ between significant centers")
    ax.set_ylabel("Exact KDE density ($\\times 10^{-3}$)")
    ax.set_title("B. Pairwise Density Transects Among 4 Significant Types\n0/6 valleys; all 13 centers climb to 1 global mode", loc="left")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(loc="lower center", ncol=2, frameon=True, framealpha=0.9, fontsize=7.6)

    # --- Panel C: Positive control (genuine K=3 and K=4 mixtures) ---
    ax = axes[1, 0]
    pc = pd.DataFrame(v_res["positive_control"])
    for k_true, col, mk in [(3, "#0072B2", "o"), (4, "#D55E00", "s")]:
        sub = pc[pc["K_true"] == k_true].sort_values("delta_sep")
        ax.plot(sub["delta_sep"], sub["modes_found"], marker=mk, lw=2.2, ms=7, color=col, label=f"Genuine mixture ($K_{{\\mathrm{{true}}}}={k_true}$)")
    ax.axhline(1.0, color="#009E73", ls="--", lw=1.4, label="Empirical IPIP-300 & IPIP-120 (1 mode)")
    ax.set_xticks([1.5, 2.5, 3.5, 4.5])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylim(0.6, 4.4)
    ax.set_xlabel("True component separation ($\\Delta$, SD units)")
    ax.set_ylabel("Distinct modes recovered by Mean-Shift")
    ax.set_title("C. Positive Control: Mean-Shift Mode Recovery\n($N=20,000, d=5$, exact $h_{\\mathrm{NN}}$)", loc="left")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.0)

    # --- Panel D: Higher-order nonlinear dependence (co-skewness & excess co-kurtosis) ---
    ax = axes[1, 1]
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    names = ["N", "E", "O", "A", "C"]
    mat = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            if i < j:
                mat[i, j] = float(np.mean((Z[:, i] ** 2) * (Z[:, j] ** 2)) - 1.0)  # excess co-kurtosis
            elif i > j:
                mat[i, j] = float(np.mean(Z[:, i] * (Z[:, j] ** 2)))              # co-skewness
            else:
                mat[i, j] = np.nan
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.18, vmax=0.18)
    for i in range(5):
        for j in range(5):
            if i == j:
                ax.text(j, i, names[i], ha="center", va="center", fontweight="bold", fontsize=9.5, color="#333333")
            else:
                ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=8.2, color="black")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Moment value (0 under independence)")
    ax.set_title("D. Nonlinear Dependence in Orthogonalized Varimax ($\\mathbf{R}\\approx\\mathbf{I}$)\nUpper: excess co-kurtosis | Lower: co-skewness $\\mathbb{E}[Z_i Z_j^2]$", loc="left")

    save_fig(fig, FIG_DIR / "fig6_via_a_topography_and_control")


def build_fig7_soto_and_representation() -> None:
    df_cv = pd.read_csv(ROOT / "results/soto2019/soto2019_incremental_cv.csv")
    v_res = json.loads((ROOT / "results/gerlach_faithful/final_via_a_checks.json").read_text())
    cont = df_cv[df_cv["type"] == "continuous"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)

    # --- Panel A: Mean out-of-sample R^2 across 72 continuous outcomes ---
    ax = axes[0]
    models = [
        ("Gerlach4_Hard", "Gerlach-4\n(Hard)", "#999999"),
        ("ARC_Hard", "ARC\n(Hard)", "#7F7F7F"),
        ("LPA3_Prob", "LPA-3\n(Prob)", "#56B4E9"),
        ("GMM4_Prob", "GMM-4\n(Prob)", "#0072B2"),
        ("Traits_Linear", "Traits\n(Linear)", "#D55E00"),
        ("Traits+ARC_Hard", "Traits +\nARC", "#E69F00"),
        ("Traits+Gerlach4", "Traits +\nGerlach-4", "#009E73"),
        ("Traits_Poly2", "Traits\nPoly-2", "#CC79A7"),
    ]
    means = [float(cont[m].mean()) for m, _, _ in models]
    sems = [float(cont[m].std(ddof=1) / np.sqrt(len(cont))) for m, _, _ in models]
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=sems, capsize=3, color=[c for _, _, c in models], alpha=0.88, edgecolor="black", lw=0.5)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in models], fontsize=7.6)
    ax.set_ylim(0.0, 0.155)
    ax.set_ylabel("Mean 10-fold CV out-of-sample $R^2$")
    ax.set_title("A. Soto (2019) Predictive Accuracy\n($k=72$ continuous outcomes, $N=6,126$)", loc="left")
    ax.grid(axis="y", ls=":", alpha=0.4)

    # --- Panel B: Distribution of incremental Delta R^2 over Traits_Linear ---
    ax = axes[1]
    deltas = [
        ("delta_ARC_over_Traits", "ARC Hard"),
        ("delta_Gerlach4_over_Traits", "Gerlach-4"),
        ("delta_Poly2_over_Traits", "Poly-2"),
        ("delta_LPA3_over_Traits", "LPA-3 Prob"),
        ("delta_GMM4_over_Traits", "GMM-4 Prob"),
    ]
    data_box = [cont[col].to_numpy() for col, _ in deltas]
    bp = ax.boxplot(
        data_box,
        vert=False,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3.2, alpha=0.45, markerfacecolor="#555555", markeredgecolor="none"),
        medianprops=dict(color="black", lw=1.6),
    )
    box_colors = ["#7F7F7F", "#009E73", "#CC79A7", "#56B4E9", "#0072B2"]
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax.axvline(0.0, color="#D55E00", ls="--", lw=1.2)
    ax.set_yticks(range(1, len(deltas) + 1))
    ax.set_yticklabels([lbl for _, lbl in deltas])
    ax.set_xlabel("Incremental $\\Delta R^2$ over 5 linear traits")
    ax.set_title("B. Incremental Validity Over Traits\n(ARC: 0/72 > 0.01; Gerlach-4: 1/72 > 0.01)", loc="left")
    ax.grid(axis="x", ls=":", alpha=0.4)

    # --- Panel C: Representation experiment on IPIP-300 ---
    ax = axes[2]
    rep = v_res["representation_experiment_ipip300"]
    short_lbls = ["Domains\n(z-scored)", "Pure Orth.\nRotation", "ZCA\nWhitening", "FA +\nVarimax"]
    d2_dom = [r["d2_domain_coords"] for r in rep]
    phi_dom = [r["phi_domain_coords"] for r in rep]
    xr = np.arange(len(rep))
    w = 0.34
    b1 = ax.bar(xr - w / 2, d2_dom, width=w, color="#0072B2", alpha=0.88, edgecolor="black", lw=0.5, label="$D^2_{\\mathrm{ARC}}$ (domain coords)")
    b2 = ax.bar(xr + w / 2, phi_dom, width=w, color="#009E73", alpha=0.88, edgecolor="black", lw=0.5, label="Tucker $\\bar{\\phi}$ (domain coords)")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=7.6)
    ax.set_xticks(xr)
    ax.set_xticklabels(short_lbls, fontsize=8.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric value (evaluated in common domain space)")
    ax.set_title("C. Representation Experiment (IPIP-300)\nPure rotation is invariant; whitening shifts metric", loc="left")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=7.8)

    save_fig(fig, FIG_DIR / "fig7_via_a_soto2019_predictive")


def main() -> int:
    build_fig6_topography_and_control()
    build_fig7_soto_and_representation()
    print("Figures generades correctament:")
    print("  - results/figures/fig6_via_a_topography_and_control.{pdf,svg,png}")
    print("  - results/figures/fig7_via_a_soto2019_predictive.{pdf,svg,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
