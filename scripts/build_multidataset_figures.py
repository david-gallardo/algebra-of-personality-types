#!/usr/bin/env python3
"""Generate unified multi-dataset empirical figures and tables (BFI-2 ES, IPIP-300 Domains, IPIP-300 Varimax)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from arc_study.metrics import ARC_LABELS, ARC_PROTOTYPES

TRAITS = ["N", "E", "O", "A", "C"]
DATASETS = [
    ("bfi2_es_study1", "BFI-2 ES", "N = 1,146"),
    ("ipip300_domains", "IPIP-300 Domains", "N = 145,388"),
    ("ipip300_varimax", "IPIP-300 Varimax", "N = 145,388"),
]
NULL_ORDER = [
    ("null1_permutation", ("null1_permutation", "null1", "permutation"), "Null 1: Permutation", "#7F7F7F"),
    ("null2_mvnormal", ("null2_mvnormal", "null2", "mvnormal"), "Null 2: Correlated MVN", "#0072B2"),
    ("null3_gaussian_copula", ("null3_gaussian_copula", "null3", "copula"), "Null 3: Gaussian Copula", "#009E73"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "savefig.dpi": 300,
})


def _find_null_col(df: pd.DataFrame) -> str:
    for c in ("null_model", "null_name", "null", "condition", "model", "null_family"):
        if c in df.columns:
            return c
    raise KeyError(f"Could not find null-model column in parquet columns: {list(df.columns)}")


def _get_null_array(df: pd.DataFrame, null_col: str, aliases: tuple[str, ...], metric: str) -> np.ndarray:
    mask = df[null_col].astype(str).isin(aliases)
    vals = df.loc[mask, metric].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        uniq = df[null_col].astype(str).unique().tolist()
        raise ValueError(f"No rows found for aliases {aliases} in column '{null_col}' (values={uniq}).")
    return vals


def _load_empirical_bundle(name: str) -> tuple[dict[str, Any], pd.DataFrame, str]:
    emp_dir = _PROJECT_ROOT / "results" / "empirical"
    with open(emp_dir / f"{name}_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    df = pd.read_parquet(emp_dir / f"{name}_null_reps.parquet")
    if "delta_bic_lpa3_vs_gmm1" not in df.columns and {"lpa3_bic", "gmm1_bic"}.issubset(df.columns):
        df["delta_bic_lpa3_vs_gmm1"] = df["lpa3_bic"] - df["gmm1_bic"]
    return summary, df, _find_null_col(df)


def _extract_bfi2_gerlach_iso050() -> dict[str, dict[str, Any]]:
    """Load BFI-2 Gerlach (iso h=0.50, B=500) component enrichments from JSON, or compute in ~2s if schema differs."""
    path = _PROJECT_ROOT / "results" / "gerlach_addon" / "bfi2_es_study1_gerlach.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        def _search(obj: Any) -> dict[str, Any] | None:
            if isinstance(obj, dict):
                if {"permutation", "null2", "null3"}.issubset(obj.keys()):
                    sample_v = obj["permutation"]
                    if isinstance(sample_v, dict) and "enrichment" in sample_v and "p_values" in sample_v:
                        return obj
                for pref in (
                    "iso_h050", "iso_0.50", "isotropic_h050", "isotropic_0.50", "isotropic_0.5",
                    "isotropic", "empirical", "evaluations", "modes", "configs",
                ):
                    if pref in obj:
                        res = _search(obj[pref])
                        if res is not None:
                            return res
                for v in obj.values():
                    res = _search(v)
                    if res is not None:
                        return res
            return None

        found = _search(raw)
        if found is not None:
            print("[INFO] BFI-2 Gerlach (iso h=0.50) carregat de bfi2_es_study1_gerlach.json")
            out: dict[str, dict[str, Any]] = {}
            for k in ("permutation", "null2", "null3"):
                enr = np.asarray(found[k]["enrichment"], dtype=float)
                pv = np.asarray(found[k]["p_values"], dtype=float)
                out[k] = {
                    "enrichment": enr.tolist(),
                    "p_values": pv.tolist(),
                    "n_sig_e125": int(np.sum((enr > 1.25) & (pv < 0.01))),
                    "n_sig_e100": int(np.sum((enr > 1.00) & (pv < 0.01))),
                }
            return out

    print("[INFO] Recalculant detall per component de BFI-2 Gerlach (iso h=0.50, B=500)...")
    from arc_study.clustering import run_gerlach_procedure

    bfi_path = _PROJECT_ROOT / "data" / "processed" / "bfi2_study1_domains.csv"
    df_bfi = pd.read_csv(bfi_path)[TRAITS].dropna()
    X = df_bfi.to_numpy(dtype=np.float64)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    res = run_gerlach_procedure(
        X,
        n_clusters=4,
        covariance_type="full",
        n_init=10,
        reg_covar=1e-6,
        n_null_samples=500,
        null_models=("permutation", "null2", "null3"),
        alpha=0.01,
        min_enrichment=1.25,
        kde_mode="isotropic",
        isotropic_bandwidth=0.50,
        max_kde_points=5000,
        random_state=20261121,
    )
    out = {}
    for k in ("permutation", "null2", "null3"):
        ev = res.evaluations[k]
        out[k] = {
            "enrichment": ev.enrichment.tolist(),
            "p_values": ev.p_values.tolist(),
            "n_sig_e125": ev.count_significant(min_enrichment=1.25, alpha=0.01),
            "n_sig_e100": ev.count_significant(min_enrichment=1.00, alpha=0.01),
        }
    return out


def _load_all_gerlach_iso050() -> dict[str, dict[str, dict[str, Any]]]:
    ipip_path = _PROJECT_ROOT / "results" / "gerlach_addon" / "ipip300_gerlach.json"
    with open(ipip_path, "r", encoding="utf-8") as f:
        ipip_raw = json.load(f)
    return {
        "bfi2_es_study1": _extract_bfi2_gerlach_iso050(),
        "ipip300_domains": ipip_raw["ipip300_domains"]["iso_h050"]["evaluations"],
        "ipip300_varimax": ipip_raw["ipip300_varimax"]["iso_h050"]["evaluations"],
    }


def _save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".svg", ".png"):
        fig.savefig(stem.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def build_fig3_multidataset(bundles: dict[str, tuple[dict[str, Any], pd.DataFrame, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)

    panels = [
        ("wk_arc_mean_d2", "A. Global ARC Distance ($D^2_{\\mathrm{ARC}}$)", "$D^2_{\\mathrm{ARC}}$ (lower = closer to ARC)", False, None),
        ("wk_arc_mean_tucker", "B. Mean Tucker Congruence ($\\bar{\\phi}$)", "Tucker $\\bar{\\phi}$ (higher = closer to ARC)", False, None),
        ("wk_split_half_kappa", "C. Split-Half Replicability (Asendorpf $\\kappa$)", "Split-half Cohen's $\\kappa$", False, 0.60),
        ("delta_bic_lpa3_vs_gmm1", "D. Model Selection ($\\Delta\\mathrm{BIC}_{\\mathrm{LPA3 - GMM1}} / N$)", "$\\Delta\\mathrm{BIC} / N$ ($>0$ favors 1-full)", True, 0.0),
    ]

    x = np.arange(len(DATASETS))
    offsets = [-0.22, 0.0, 0.22]
    xlabels = [f"{title}\n({n_str})" for _, title, n_str in DATASETS]

    for ax, (metric, title, ylabel, per_n, ref_line) in zip(axes.flat, panels):
        if ref_line is not None:
            ax.axhline(ref_line, color="#444444", linestyle="--", linewidth=1.1, alpha=0.8, zorder=1)

        for (_, aliases, null_label, color), dx in zip(NULL_ORDER, offsets):
            means, lo_err, hi_err = [], [], []
            for ds_key, _, _ in DATASETS:
                summary, df, null_col = bundles[ds_key]
                n_obs = float(summary["empirical_diagnostics"]["n"])
                sub = _get_null_array(df, null_col, aliases, metric)
                if per_n:
                    sub = sub / n_obs
                m = float(np.mean(sub))
                q025, q975 = np.quantile(sub, [0.025, 0.975])
                means.append(m)
                lo_err.append(max(0.0, m - q025))
                hi_err.append(max(0.0, q975 - m))

            ax.errorbar(
                x + dx,
                means,
                yerr=[lo_err, hi_err],
                fmt="o",
                color=color,
                capsize=3.5,
                markersize=5.5,
                linewidth=1.6,
                label=null_label,
                zorder=3,
            )

        emp_vals = []
        for ds_key, _, _ in DATASETS:
            summary, _, _ = bundles[ds_key]
            n_obs = float(summary["empirical_diagnostics"]["n"])
            em = summary["empirical_metrics"]
            val = (em["lpa3_bic"] - em["gmm1_bic"]) if metric == "delta_bic_lpa3_vs_gmm1" else em[metric]
            emp_vals.append(float(val) / n_obs if per_n else float(val))

        ax.scatter(
            x,
            emp_vals,
            marker="D",
            s=72,
            color="#D55E00",
            edgecolor="black",
            linewidth=0.9,
            label="Empirical",
            zorder=5,
        )

        for xi, yi in zip(x, emp_vals):
            ax.annotate(
                f"{yi:.2f}" if not per_n else f"{yi:+.3f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#A03D00",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.set_xlim(-0.55, 2.55)
        ax.set_title(title, loc="left", pad=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.45)

    axes[1, 0].text(
        0.99, 0.612, "Asendorpf threshold (0.60)",
        transform=axes[1, 0].get_yaxis_transform(),
        fontsize=7.6, color="#444444", ha="right", va="bottom",
    )
    axes[1, 1].text(
        0.99, 0.010, "GMM 1-full better",
        transform=axes[1, 1].get_yaxis_transform(),
        fontsize=7.6, color="#444444", ha="right", va="bottom",
    )
    axes[1, 1].text(
        0.99, -0.012, "LPA 3-diag better",
        transform=axes[1, 1].get_yaxis_transform(),
        fontsize=7.6, color="#444444", ha="right", va="top",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=4, frameon=False)
    _save_fig(fig, _PROJECT_ROOT / "results" / "figures" / "fig3_empirical_multidataset_contrast")


def build_fig5_ablation_and_gerlach(
    bundles: dict[str, tuple[dict[str, Any], pd.DataFrame, str]],
    gerlach_iso: dict[str, dict[str, dict[str, Any]]],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.3), constrained_layout=True)

    arc_colors = {
        "Resilient": "#009E73",
        "Overcontrolled": "#0072B2",
        "Undercontrolled": "#D55E00",
    }
    trait_x = np.arange(len(TRAITS))

    for col_idx, (ds_key, ds_title, n_str) in enumerate(DATASETS):
        ax = axes[0, col_idx]
        summary, _, _ = bundles[ds_key]
        centroids = summary["empirical_centroids"]["ward_kmeans_k3"]
        l2_l1 = summary["empirical_diagnostics"]["lambda2_over_lambda1"]

        for proto_idx, label in enumerate(ARC_LABELS):
            ax.plot(
                trait_x,
                ARC_PROTOTYPES[proto_idx],
                linestyle="--",
                linewidth=1.3,
                alpha=0.55,
                color=arc_colors[label],
            )

        for c in centroids:
            label = c["arc_type"]
            vec = [c["centroid_neoac"][t] for t in TRAITS]
            phi = c["tucker_congruence"]
            ax.plot(
                trait_x,
                vec,
                marker="o",
                markersize=4.8,
                linewidth=2.1,
                color=arc_colors[label],
                label=f"{label[:3]}. ($\\phi={phi:.2f}$)",
            )

        ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_xticks(trait_x)
        ax.set_xticklabels(TRAITS)
        ax.set_ylim(-1.25, 1.25)
        ax.set_title(f"{chr(65 + col_idx)}. {ds_title} ({n_str})\n$\\lambda_2/\\lambda_1 = {l2_l1:.3f}$", loc="left", fontsize=9.8)
        if col_idx == 0:
            ax.set_ylabel("Standardized centroid ($z$-score)")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(loc="lower right", frameon=True, framealpha=0.88, fontsize=7.6)

    null_specs = [
        ("permutation", "Perm. (Null 1)", "#7F7F7F"),
        ("null2", "MVN (Null 2)", "#0072B2"),
        ("null3", "Copula (Null 3)", "#009E73"),
    ]
    comp_x = np.arange(4)
    bar_w = 0.25

    for col_idx, (ds_key, ds_title, _) in enumerate(DATASETS):
        ax = axes[1, col_idx]
        g_ds = gerlach_iso[ds_key]
        ax.axhline(
            1.25,
            color="#D55E00",
            linestyle="--",
            linewidth=1.2,
            label="Gerlach threshold ($e = 1.25$)" if col_idx == 0 else None,
        )
        ax.axhline(1.00, color="#666666", linestyle=":", linewidth=0.9)

        sig_counts = []
        max_bar = 1.6
        for idx_n, (nkey, nlabel, color) in enumerate(null_specs):
            ev = g_ds[nkey]
            enr = np.asarray(ev["enrichment"], dtype=float)
            pvals = np.asarray(ev["p_values"], dtype=float)
            sig = (enr > 1.25) & (pvals < 0.01)
            sig_counts.append(int(np.sum(sig)))
            max_bar = max(max_bar, float(np.nanmax(enr)))

            xpos = comp_x + (idx_n - 1) * bar_w
            bars = ax.bar(
                xpos,
                enr,
                width=bar_w * 0.9,
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                label=nlabel if col_idx == 0 else None,
            )
            for b, is_s, val in zip(bars, sig, enr):
                if is_s:
                    ax.text(
                        b.get_x() + b.get_width() / 2.0,
                        val + 0.04,
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                        color="black",
                    )

        ax.set_xticks(comp_x)
        ax.set_xticklabels([f"Comp {i+1}" for i in range(4)])
        ax.set_ylim(0.0, max(2.65, max_bar + 0.35))
        ax.set_title(
            f"{chr(68 + col_idx)}. Gerlach KDE ($h=0.50$, $B=500$): {ds_title}\n"
            f"Sig. ($e>1.25, p<.01$): Perm {sig_counts[0]}/4 -> N2 {sig_counts[1]}/4 -> N3 {sig_counts[2]}/4",
            loc="left",
            fontsize=9.3,
        )
        if col_idx == 0:
            ax.set_ylabel("Density enrichment ($\\hat{f}_{\\mathrm{emp}} / \\bar{f}_{\\mathrm{null}}$)")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False)
    _save_fig(fig, _PROJECT_ROOT / "results" / "figures" / "fig5_empirical_ablation_and_gerlach")


def build_unified_table_and_audit(
    bundles: dict[str, tuple[dict[str, Any], pd.DataFrame, str]],
    gerlach_iso: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    rows = []
    alias_map = {k: al for k, al, _, _ in NULL_ORDER}

    for ds_key, ds_title, _ in DATASETS:
        summary, df, null_col = bundles[ds_key]
        d = summary["empirical_diagnostics"]
        em = summary["empirical_metrics"]
        g = gerlach_iso[ds_key]

        def _p_mc(metric: str, emp_val: float, direction: str, null_key: str) -> float:
            vals = _get_null_array(df, null_col, alias_map[null_key], metric)
            k = int(np.sum(vals <= emp_val)) if direction == "le" else int(np.sum(vals >= emp_val))
            return float((1 + k) / (len(vals) + 1))

        delta_lpa3_gmm1 = float(em["lpa3_bic"] - em["gmm1_bic"])
        rows.append({
            "dataset": ds_key,
            "dataset_label": ds_title,
            "n": int(d["n"]),
            "lambda2_over_lambda1": float(d["lambda2_over_lambda1"]),
            "det_r_emp": float(d["det_r_emp"]),
            "d2_emp": float(em["wk_arc_mean_d2"]),
            "d2_p_n1": _p_mc("wk_arc_mean_d2", em["wk_arc_mean_d2"], "le", "null1_permutation"),
            "d2_p_n2": _p_mc("wk_arc_mean_d2", em["wk_arc_mean_d2"], "le", "null2_mvnormal"),
            "d2_p_n3": _p_mc("wk_arc_mean_d2", em["wk_arc_mean_d2"], "le", "null3_gaussian_copula"),
            "tucker_emp": float(em["wk_arc_mean_tucker"]),
            "tucker_p_n3": _p_mc("wk_arc_mean_tucker", em["wk_arc_mean_tucker"], "ge", "null3_gaussian_copula"),
            "kappa_emp": float(em["wk_split_half_kappa"]),
            "kappa_p_n1": _p_mc("wk_split_half_kappa", em["wk_split_half_kappa"], "ge", "null1_permutation"),
            "kappa_p_n2": _p_mc("wk_split_half_kappa", em["wk_split_half_kappa"], "ge", "null2_mvnormal"),
            "kappa_p_n3": _p_mc("wk_split_half_kappa", em["wk_split_half_kappa"], "ge", "null3_gaussian_copula"),
            "lpa_delta_bic_3_vs_1": float(em["lpa_delta_bic_3_vs_1"]),
            "delta_bic_lpa3_vs_gmm1": delta_lpa3_gmm1,
            "gmm_delta_bic_3_vs_1": float(em["gmm_delta_bic_3_vs_1"]),
            "gerlach_h050_e125_perm": int(g["permutation"]["n_sig_e125"]),
            "gerlach_h050_e125_null2": int(g["null2"]["n_sig_e125"]),
            "gerlach_h050_e125_null3": int(g["null3"]["n_sig_e125"]),
        })

    out_df = pd.DataFrame(rows)
    res_dir = _PROJECT_ROOT / "results"
    out_df.to_csv(res_dir / "table2_empirical_all_datasets.csv", index=False)

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Multi-dataset empirical contrast against unimodal null models (BFI-2 ES and IPIP-NEO-300).}",
        r"\label{tab:empirical_all}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrc}",
        r"\toprule",
        r"Dataset & $N$ & $\lambda_2/\lambda_1$ & $D^2_{\mathrm{ARC}}$ ($p_{\mathrm{N3}}$) & $\bar{\phi}$ ($p_{\mathrm{N3}}$) & $\kappa$ ($p_{\mathrm{N1}}$ / $p_{\mathrm{N3}}$) & $\Delta\mathrm{BIC}_{\mathrm{LPA3-1}}$ & $\Delta\mathrm{BIC}_{\mathrm{LPA3-GMM1}}$ & $\Delta\mathrm{BIC}_{\mathrm{GMM3-1}}$ & Gerlach (P/N2/N3) \\",
        r"\midrule",
    ]
    for r in rows:
        tex_lines.append(
            f"{r['dataset_label']} & {r['n']:,} & {r['lambda2_over_lambda1']:.3f} & "
            f"{r['d2_emp']:.3f} ({r['d2_p_n3']:.3f}) & {r['tucker_emp']:.3f} ({r['tucker_p_n3']:.3f}) & "
            f"{r['kappa_emp']:.3f} ({r['kappa_p_n1']:.3f}/{r['kappa_p_n3']:.3f}) & "
            f"{r['lpa_delta_bic_3_vs_1']:,.1f} & {r['delta_bic_lpa3_vs_gmm1']:+,.1f} & "
            f"{r['gmm_delta_bic_3_vs_1']:+,.1f} & "
            f"{r['gerlach_h050_e125_perm']}/{r['gerlach_h050_e125_null2']}/{r['gerlach_h050_e125_null3']} \\\\"
        )
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    (res_dir / "table2_empirical_all_datasets.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    md_lines = [
        "# Resum Empíric Multidataset (BFI-2 ES + IPIP-NEO-300)",
        "",
        "| Dataset | $N$ | $\\lambda_2/\\lambda_1$ | $D^2_{\\mathrm{ARC}}$ ($p_{\\mathrm{N3}}$) | Tucker $\\bar{\\phi}$ ($p_{\\mathrm{N3}}$) | $\\kappa$ ($p_{\\mathrm{N1}}$ / $p_{\\mathrm{N3}}$) | $\\Delta\\mathrm{BIC}_{\\mathrm{LPA3-GMM1}}$ | Gerlach ($h=0.50, e>1.25$) P/N2/N3 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['dataset_label']} | {r['n']:,} | {r['lambda2_over_lambda1']:.3f} | "
            f"{r['d2_emp']:.3f} (`{r['d2_p_n3']:.3f}`) | {r['tucker_emp']:.3f} (`{r['tucker_p_n3']:.3f}`) | "
            f"{r['kappa_emp']:.3f} (`{r['kappa_p_n1']:.3f}` / `{r['kappa_p_n3']:.3f}`) | "
            f"{r['delta_bic_lpa3_vs_gmm1']:+,.1f} | "
            f"{r['gerlach_h050_e125_perm']}/{r['gerlach_h050_e125_null2']}/{r['gerlach_h050_e125_null3']} |"
        )
    md_lines += [
        "",
        "## Fitxers nous generats",
        "- `results/figures/fig3_empirical_multidataset_contrast.{pdf,svg,png}`",
        "- `results/figures/fig5_empirical_ablation_and_gerlach.{pdf,svg,png}`",
        "- `results/table2_empirical_all_datasets.{csv,tex}`",
    ]
    (res_dir / "AUDIT_SUMMARY_MULTIDATASET.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return out_df


def main() -> int:
    bundles = {k: _load_empirical_bundle(k) for k, _, _ in DATASETS}
    gerlach_iso = _load_all_gerlach_iso050()
    build_fig3_multidataset(bundles)
    build_fig5_ablation_and_gerlach(bundles, gerlach_iso)
    df_out = build_unified_table_and_audit(bundles, gerlach_iso)
    print("=== Taula Empírica Multidataset ===")
    print(df_out.to_string(index=False))
    print("\nFigures i taules generades correctament.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
