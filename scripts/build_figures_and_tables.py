#!/usr/bin/env python3
"""Build publication-ready tables (.csv, .tex), vector figures (.pdf, .svg),
and the final audit summary (results/AUDIT_SUMMARY.md) from Monte Carlo
simulation blocks and the Spanish BFI-2 empirical comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import gaussian_kde

# Allow running directly as `python scripts/build_figures_and_tables.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from arc_study.clustering import fit_ward_kmeans  # noqa: E402
from arc_study.generators import (  # noqa: E402
    _apply_gaussian_copula,
    calibrate_gaussian_copula,
    default_skewed_marginals,
    nearest_pd_correlation,
)
from arc_study.matrices import BASE_SEED, TRAIT_ORDER  # noqa: E402
from arc_study.metrics import ARC_LABELS, ARC_PROTOTYPES, match_centroids  # noqa: E402

NONCOLLINEARITY_THRESHOLD: float = 0.50764

CONDITION_ORDER: tuple[str, ...] = (
    "null1_isotropic",
    "null2_vdl",
    "null2_be",
    "null2_vdl_hetero",
    "null2_be_hetero",
    "null3_vdl_copula",
    "null3_be_copula",
)

CONDITION_LABELS: dict[str, str] = {
    "null1_isotropic": "Null 1: Isotropic N(0, I)",
    "null2_vdl": "Null 2: MVN (vdL)",
    "null2_be": "Null 2: MVN (BE)",
    "null2_vdl_hetero": "Null 2: MVN hetero (vdL)",
    "null2_be_hetero": "Null 2: MVN hetero (BE)",
    "null3_vdl_copula": "Null 3: Copula (vdL)",
    "null3_be_copula": "Null 3: Copula (BE)",
}

# Colorblind-safe Okabe-Ito palette
CONDITION_COLORS: dict[str, str] = {
    "null1_isotropic": "#7F7F7F",
    "null2_vdl": "#0072B2",
    "null2_be": "#56B4E9",
    "null2_vdl_hetero": "#009E73",
    "null2_be_hetero": "#66C2A5",
    "null3_vdl_copula": "#D55E00",
    "null3_be_copula": "#E69F00",
}

CONDITION_STYLES: dict[str, tuple[str, str]] = {
    "null1_isotropic": ("--", "s"),
    "null2_vdl": ("-", "o"),
    "null2_be": ("-.", "^"),
    "null2_vdl_hetero": (":", "D"),
    "null2_be_hetero": (":", "v"),
    "null3_vdl_copula": ("-", "P"),
    "null3_be_copula": ("-.", "X"),
}

EMP_NULL_ORDER: tuple[str, ...] = (
    "null1_permutation",
    "null2_mvnormal",
    "null3_gaussian_copula",
)

EMP_NULL_LABELS: dict[str, str] = {
    "null1_permutation": "Null 1: Column permutation",
    "null2_mvnormal": "Null 2: MVN N(0, R_emp)",
    "null3_gaussian_copula": "Null 3: Gaussian copula",
}

EMP_NULL_COLORS: dict[str, str] = {
    "null1_permutation": "#7F7F7F",
    "null2_mvnormal": "#0072B2",
    "null3_gaussian_copula": "#D55E00",
}


def _configure_matplotlib() -> None:
    """Apply clean academic typography and styling defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_vector_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> list[Path]:
    """Save a Matplotlib figure as both .pdf and .svg at 300 dpi."""
    paths: list[Path] = []
    for ext in ("pdf", "svg"):
        out_path = fig_dir / f"{stem}.{ext}"
        fig.savefig(out_path, format=ext, dpi=300, bbox_inches="tight")
        paths.append(out_path)
    plt.close(fig)
    return paths


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters for safe tabular cells."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _fmt_mean_ci(mean_val: float, q025: float, q975: float, decimals: int = 2) -> str:
    """Format mean [2.5%, 97.5%] concisely."""
    if not (math.isfinite(mean_val) and math.isfinite(q025) and math.isfinite(q975)):
        return "--"
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean_val)} [{fmt.format(q025)}, {fmt.format(q975)}]"


def _summarize_array(values: NDArray[np.float64]) -> tuple[float, float, float]:
    """Return (mean, q2.5, q97.5) ignoring non-finite entries."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(valid)),
        float(np.quantile(valid, 0.025)),
        float(np.quantile(valid, 0.975)),
    )


def _load_simulation_blocks(sim_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Read and concatenate all results/simulations/block_*.parquet files."""
    block_files = sorted(sim_dir.glob("block_*.parquet"))
    if not block_files:
        raise FileNotFoundError(f"No block_*.parquet files found in {sim_dir}.")

    frames = [pd.read_parquet(path) for path in block_files]
    df_sim = pd.concat(frames, ignore_index=True)
    df_sim["delta_bic_lpa3_vs_gmm1"] = df_sim["lpa3_bic"] - df_sim["gmm1_bic"]
    return df_sim, block_files


def _load_empirical_results(
    emp_reps_path: Path,
    emp_summary_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read empirical null replicates Parquet and summary JSON."""
    if not emp_reps_path.exists():
        raise FileNotFoundError(f"Empirical null reps file not found: {emp_reps_path}")
    if not emp_summary_path.exists():
        raise FileNotFoundError(f"Empirical summary JSON not found: {emp_summary_path}")

    df_emp = pd.read_parquet(emp_reps_path)
    df_emp["delta_bic_lpa3_vs_gmm1"] = df_emp["lpa3_bic"] - df_emp["gmm1_bic"]

    with emp_summary_path.open("r", encoding="utf-8") as fh:
        summary: dict[str, Any] = json.load(fh)

    emp_m = summary["empirical_metrics"]
    emp_m["delta_bic_lpa3_vs_gmm1"] = float(emp_m["lpa3_bic"]) - float(emp_m["gmm1_bic"])
    return df_emp, summary


def _load_gerlach_addon(
    gerlach_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the corrected Gerlach add-on Monte Carlo blocks and empirical BFI-2 outputs."""
    mc_files = sorted(gerlach_dir.glob("gerlach_*_N*.parquet"))
    if not mc_files:
        raise FileNotFoundError(f"No gerlach_*_N*.parquet files found in {gerlach_dir}.")
    df_g_mc = pd.concat([pd.read_parquet(p) for p in mc_files], ignore_index=True)

    emp_p = gerlach_dir / "bfi2_es_study1_gerlach.parquet"
    emp_j = gerlach_dir / "bfi2_es_study1_gerlach.json"
    if not emp_p.exists() or not emp_j.exists():
        raise FileNotFoundError(
            f"Missing BFI-2 Gerlach add-on files in {gerlach_dir}: {emp_p.name}, {emp_j.name}"
        )
    df_g_emp = pd.read_parquet(emp_p)
    with emp_j.open("r", encoding="utf-8") as fh:
        g_emp_summary: dict[str, Any] = json.load(fh)
    return df_g_mc, df_g_emp, g_emp_summary


def _ordered_conditions(present: list[str]) -> list[str]:
    """Order conditions canonically, appending any unknown conditions sorted."""
    ordered = [c for c in CONDITION_ORDER if c in present]
    extras = sorted(c for c in present if c not in ordered)
    return ordered + extras


# ---------------------------------------------------------------------------
# Table 1: Monte Carlo Summary (.csv and .tex)
# ---------------------------------------------------------------------------


def build_table1_montecarlo(
    df_sim: pd.DataFrame,
    df_gerlach_mc: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, Path, Path]:
    """Build Table 1 (Monte Carlo summary by condition and N) in CSV and LaTeX."""
    core_metrics: tuple[tuple[str, str, int], ...] = (
        ("wk_arc_mean_d2", "d2", 2),
        ("wk_arc_mean_elevation", "elev", 2),
        ("wk_arc_mean_scatter", "scat", 2),
        ("wk_arc_mean_shape", "shape", 2),
        ("wk_arc_mean_tucker", "tucker", 3),
        ("wk_split_half_kappa", "kappa", 3),
        ("delta_bic_lpa3_vs_gmm1", "delta_bic_lpa3_gmm1", 1),
    )

    gerlach_tex_metrics: tuple[tuple[str, str], ...] = (
        ("iso_h050_n_sig_e125_perm", "gerlach_iso050_e125_perm"),
        ("iso_h050_n_sig_e125_null2", "gerlach_iso050_e125_null2"),
        ("iso_h050_n_sig_e125_null3", "gerlach_iso050_e125_null3"),
    )
    gerlach_extra_csv_metrics: tuple[tuple[str, str], ...] = (
        ("iso_h050_n_sig_e100_perm", "gerlach_iso050_e100_perm"),
        ("iso_h050_n_sig_e100_null2", "gerlach_iso050_e100_null2"),
        ("iso_h050_n_sig_e100_null3", "gerlach_iso050_e100_null3"),
        ("scott_n_sig_e125_perm", "gerlach_scott_e125_perm"),
        ("scott_n_sig_e125_null2", "gerlach_scott_e125_null2"),
        ("scott_n_sig_e125_null3", "gerlach_scott_e125_null3"),
    )

    conditions = _ordered_conditions(df_sim["condition"].unique().tolist())
    n_values = sorted(int(v) for v in df_sim["n"].unique().tolist())

    rows: list[dict[str, Any]] = []
    tex_rows: list[str] = []

    for cond in conditions:
        for n_val in n_values:
            sub = df_sim.loc[(df_sim["condition"] == cond) & (df_sim["n"] == n_val)]
            if sub.empty:
                continue
            sub_g = df_gerlach_mc.loc[
                (df_gerlach_mc["condition"] == cond) & (df_gerlach_mc["n"] == n_val)
            ]

            row: dict[str, Any] = {
                "condition": cond,
                "condition_label": CONDITION_LABELS.get(cond, cond),
                "n": n_val,
                "n_reps": int(len(sub)),
                "gerlach_addon_reps": int(len(sub_g)),
            }
            tex_cells = [
                _latex_escape(CONDITION_LABELS.get(cond, cond)),
                f"{n_val:,}",
            ]

            for col, short, dec in core_metrics:
                mean_v, q025, q975 = _summarize_array(sub[col].to_numpy(dtype=np.float64))
                row[f"{short}_mean"] = mean_v
                row[f"{short}_q025"] = q025
                row[f"{short}_q975"] = q975
                tex_cells.append(_fmt_mean_ci(mean_v, q025, q975, decimals=dec))

            for col, short in gerlach_tex_metrics:
                if sub_g.empty or col not in sub_g.columns:
                    row[f"{short}_mean"] = float("nan")
                    row[f"{short}_q025"] = float("nan")
                    row[f"{short}_q975"] = float("nan")
                    row[f"{short}_rate_ge1"] = float("nan")
                    tex_cells.append("--")
                else:
                    arr = sub_g[col].to_numpy(dtype=np.float64)
                    mean_v, q025, q975 = _summarize_array(arr)
                    row[f"{short}_mean"] = mean_v
                    row[f"{short}_q025"] = q025
                    row[f"{short}_q975"] = q975
                    row[f"{short}_rate_ge1"] = float(np.mean(arr >= 1.0))
                    tex_cells.append(_fmt_mean_ci(mean_v, q025, q975, decimals=2))

            for col, short in gerlach_extra_csv_metrics:
                if sub_g.empty or col not in sub_g.columns:
                    row[f"{short}_mean"] = float("nan")
                    row[f"{short}_rate_ge1"] = float("nan")
                else:
                    arr = sub_g[col].to_numpy(dtype=np.float64)
                    row[f"{short}_mean"] = float(np.mean(arr))
                    row[f"{short}_rate_ge1"] = float(np.mean(arr >= 1.0))

            rows.append(row)
            tex_rows.append(" & ".join(tex_cells) + r" \\")

    df_table1 = pd.DataFrame(rows)
    csv_path = out_dir / "table1_montecarlo_summary.csv"
    df_table1.to_csv(csv_path, index=False)

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Monte Carlo summary by null condition and sample size $N$. Core columns report mean and [2.5\%, 97.5\%] percentile interval across 1,000 replicates per block. $\Delta\mathrm{BIC} = \mathrm{BIC}(\mathrm{LPA}_{3\text{-diag}}) - \mathrm{BIC}(\mathrm{GMM}_{1\text{-full}})$ (positive values favour 1-component full-covariance Gaussian; negative values under Null 3 reflect non-normality over-extraction). Gerlach et al.\ (2018) columns report the corrected add-on ($N \in \{1{,}000, 5{,}000\}$, 200 replicates/block, $B = 200$ internal nulls, $\alpha = 0.01$, isotropic Gaussian KDE $h = 0.50$, enrichment $> 1.25$; full breakdown with enrichment $> 1.00$ and Scott KDE is provided in the accompanying CSV).}",
        r"\label{tab:montecarlo_summary}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrcccccccccc}",
        r"\toprule",
        r"Condition & $N$ & $D^2_{\mathrm{ARC}}$ & Elevation & Scatter & Shape & Tucker $\bar{\phi}$ & Split-half $\kappa$ & $\Delta\mathrm{BIC}$ (3-diag $-$ 1-full) & Gerlach (Perm) & Gerlach (Null 2) & Gerlach (Null 3) \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    tex_path = out_dir / "table1_montecarlo_summary.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    return df_table1, csv_path, tex_path


# ---------------------------------------------------------------------------
# Table 2: Empirical BFI-2 Spanish Contrast (.csv and .tex)
# ---------------------------------------------------------------------------


def build_table2_empirical(
    df_emp: pd.DataFrame,
    summary: dict[str, Any],
    df_gerlach_emp: pd.DataFrame,
    gerlach_emp_summary: dict[str, Any],
    out_dir: Path,
) -> tuple[pd.DataFrame, Path, Path]:
    """Build Table 2 (Spanish BFI-2 empirical vs. Null 1, Null 2, Null 3) in CSV and LaTeX."""
    emp_m = summary["empirical_metrics"]
    n_emp = int(summary["empirical_diagnostics"]["n"])

    # (column, display_label, latex_label, direction, decimals)
    table2_core: tuple[tuple[str, str, str, str, int], ...] = (
        ("wk_arc_mean_d2", "Global ARC D^2 (Ward+k-means)", r"Global $D^2_{\mathrm{ARC}}$", "le", 3),
        ("wk_arc_mean_elevation", "Elevation (Nivell)", r"Elevation (Nivell)", "le", 3),
        ("wk_arc_mean_scatter", "Scatter (Dispersio)", r"Scatter (Dispersi\'o)", "le", 3),
        ("wk_arc_mean_shape", "Shape (Forma)", r"Shape (Forma)", "le", 3),
        ("wk_arc_mean_tucker", "Tucker congruence (mean)", r"Tucker $\bar{\phi}$", "ge", 3),
        ("wk_split_half_kappa", "Split-half kappa (Asendorpf)", r"Split-half $\kappa$", "ge", 3),
        ("delta_bic_lpa3_vs_gmm1", "Delta BIC (LPA 3-diag - GMM 1-full)", r"$\Delta\mathrm{BIC}$ (3-diag $-$ 1-full)", "ge", 1),
        ("lpa_delta_bic_3_vs_1", "Delta BIC (LPA 3-diag - 1-diag)", r"$\Delta\mathrm{BIC}_{\mathrm{LPA}}$ (3 $-$ 1)", "le", 1),
        ("gmm_delta_bic_3_vs_1", "Delta BIC (GMM 3-full - 1-full)", r"$\Delta\mathrm{BIC}_{\mathrm{GMM}}$ (3 $-$ 1)", "le", 1),
    )

    # Corrected Gerlach add-on rows: (mode, thresh_key, null_key, label, latex_label)
    gerlach_rows_spec: tuple[tuple[str, str, str, str, str], ...] = (
        ("iso_h050", "e125", "perm", "Gerlach iso h=0.50 e>1.25 (Perm)", r"Gerlach iso $h{=}0.50$, $e{>}1.25$ (Perm)"),
        ("iso_h050", "e125", "null2", "Gerlach iso h=0.50 e>1.25 (Null 2)", r"Gerlach iso $h{=}0.50$, $e{>}1.25$ (Null 2)"),
        ("iso_h050", "e125", "null3", "Gerlach iso h=0.50 e>1.25 (Null 3)", r"Gerlach iso $h{=}0.50$, $e{>}1.25$ (Null 3)"),
        ("iso_h050", "e100", "perm", "Gerlach iso h=0.50 e>1.00 (Perm)", r"Gerlach iso $h{=}0.50$, $e{>}1.00$ (Perm)"),
        ("iso_h050", "e100", "null2", "Gerlach iso h=0.50 e>1.00 (Null 2)", r"Gerlach iso $h{=}0.50$, $e{>}1.00$ (Null 2)"),
        ("iso_h050", "e100", "null3", "Gerlach iso h=0.50 e>1.00 (Null 3)", r"Gerlach iso $h{=}0.50$, $e{>}1.00$ (Null 3)"),
        ("scott", "e125", "perm", "Gerlach Scott e>1.25 (Perm)", r"Gerlach Scott, $e{>}1.25$ (Perm)"),
        ("scott", "e125", "null2", "Gerlach Scott e>1.25 (Null 2)", r"Gerlach Scott, $e{>}1.25$ (Null 2)"),
        ("scott", "e125", "null3", "Gerlach Scott e>1.25 (Null 3)", r"Gerlach Scott, $e{>}1.25$ (Null 3)"),
    )

    rows: list[dict[str, Any]] = []
    tex_rows: list[str] = []

    def _append_row(
        metric_id: str,
        label: str,
        latex_label: str,
        direction: str,
        dec: int,
        emp_val: float,
        source_df: pd.DataFrame,
        col_name: str,
    ) -> None:
        row: dict[str, Any] = {
            "metric": metric_id,
            "metric_label": label,
            "tail_direction": "P(sim <= emp)" if direction == "le" else "P(sim >= emp)",
            "empirical": emp_val,
        }
        emp_str = f"{emp_val:.{dec}f}" if dec > 0 else f"{emp_val:.0f}"
        tex_cells = [latex_label, emp_str]

        for null_name in EMP_NULL_ORDER:
            sub = source_df.loc[source_df["null_model"] == null_name]
            vals = sub[col_name].to_numpy(dtype=np.float64)
            valid = vals[np.isfinite(vals)]
            mean_v, q025, q975 = _summarize_array(valid)
            if valid.size > 0 and math.isfinite(emp_val):
                hits = int(np.sum(valid <= emp_val) if direction == "le" else np.sum(valid >= emp_val))
                p_emp = hits / float(valid.size)
                p_mc = (1.0 + hits) / float(valid.size + 1)
            else:
                p_emp = float("nan")
                p_mc = float("nan")

            row[f"{null_name}_mean"] = mean_v
            row[f"{null_name}_q025"] = q025
            row[f"{null_name}_q975"] = q975
            row[f"{null_name}_p_emp"] = p_emp
            row[f"{null_name}_p_mc"] = p_mc

            tex_cells.append(_fmt_mean_ci(mean_v, q025, q975, decimals=dec))
            tex_cells.append(f"{p_emp:.3f}" if math.isfinite(p_emp) else "--")

        rows.append(row)
        tex_rows.append(" & ".join(tex_cells) + r" \\")

    for col, label, latex_label, direction, dec in table2_core:
        _append_row(col, label, latex_label, direction, dec, float(emp_m[col]), df_emp, col)

    emp_g = gerlach_emp_summary["empirical_gerlach"]
    for mode_key, thresh_key, nl_key, label, latex_label in gerlach_rows_spec:
        col_name = f"{mode_key}_n_sig_{thresh_key}_{nl_key}"
        emp_val = float(emp_g[mode_key]["evaluations"][nl_key][f"n_sig_{thresh_key}"])
        _append_row(col_name, label, latex_label, "ge", 2, emp_val, df_gerlach_emp, col_name)

    df_table2 = pd.DataFrame(rows)
    csv_path = out_dir / "table2_empirical_bfi2.csv"
    df_table2.to_csv(csv_path, index=False)

    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Empirical contrast of the Spanish BFI-2 sample ($N = {n_emp:,}$) against Null 1 (independent column permutation), Null 2 (correlated multivariate normal $\mathcal{{N}}_5(\mathbf{{0}}, \mathbf{{R}}_{{\mathrm{{emp}}}})$), and Null 3 (calibrated Gaussian copula with empirical marginals). Core rows use 1,000 null replicates/model; Gerlach et al.\ (2018) rows use the corrected add-on ($B = 500$ internal nulls on the empirical sample, 100 null replicates/model with $B = 200$, $\alpha = 0.01$). Null cells report mean [2.5\%, 97.5\%] percentile interval and empirical tail probability $P(S_{{\mathrm{{sim}}}} \ge S_{{\mathrm{{emp}}}})$ (or $P(D^2_{{\mathrm{{sim}}}} \le D^2_{{\mathrm{{emp}}}})$ for distances).}}",
        r"\label{tab:empirical_bfi2}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrcccccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Null 1 (Permutation)} & \multicolumn{2}{c}{Null 2 (MVN $\mathbf{R}_{\mathrm{emp}}$)} & \multicolumn{2}{c}{Null 3 (Gaussian Copula)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Metric & Empirical & Mean [2.5\%, 97.5\%] & $p_{\mathrm{emp}}$ & Mean [2.5\%, 97.5\%] & $p_{\mathrm{emp}}$ & Mean [2.5\%, 97.5\%] & $p_{\mathrm{emp}}$ \\",
        r"\midrule",
        *tex_rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    tex_path = out_dir / "table2_empirical_bfi2.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    return df_table2, csv_path, tex_path


# ---------------------------------------------------------------------------
# Figure 1: Monte Carlo ARC Similarity (Global D^2, Elevation, Scatter, Shape)
# ---------------------------------------------------------------------------


def build_figure1_mc_arc_similarity(df_sim: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """Plot Monte Carlo ARC D^2 decomposition across null models and sample sizes N."""
    conditions = _ordered_conditions(df_sim["condition"].unique().tolist())
    n_values = sorted(int(v) for v in df_sim["n"].unique().tolist())
    x_pos = np.arange(len(n_values), dtype=np.float64)

    panels: tuple[tuple[str, str], ...] = (
        ("wk_arc_mean_d2", "(A) Global ARC Distance ($D^2$)"),
        ("wk_arc_mean_elevation", "(B) Elevation (Nivell)"),
        ("wk_arc_mean_scatter", "(C) Scatter (Dispersió)"),
        ("wk_arc_mean_shape", "(D) Shape (Forma)"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    offsets = np.linspace(-0.18, 0.18, max(len(conditions), 1))

    for ax, (col, title) in zip(axes.flat, panels):
        for idx, cond in enumerate(conditions):
            means: list[float] = []
            lo_errs: list[float] = []
            hi_errs: list[float] = []
            xs: list[float] = []
            for j, n_val in enumerate(n_values):
                sub = df_sim.loc[(df_sim["condition"] == cond) & (df_sim["n"] == n_val)]
                if sub.empty:
                    continue
                m, q025, q975 = _summarize_array(sub[col].to_numpy(dtype=np.float64))
                xs.append(x_pos[j] + offsets[idx])
                means.append(m)
                lo_errs.append(max(0.0, m - q025))
                hi_errs.append(max(0.0, q975 - m))

            ls, marker = CONDITION_STYLES.get(cond, ("-", "o"))
            color = CONDITION_COLORS.get(cond, "#333333")
            ax.errorbar(
                xs,
                means,
                yerr=[lo_errs, hi_errs],
                label=CONDITION_LABELS.get(cond, cond),
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=5.0,
                linewidth=1.4,
                capsize=2.5,
                alpha=0.92,
            )

        ax.set_title(title)
        ax.set_ylabel("Mean squared distance (lower = closer to ARC)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{n:,}" for n in n_values])

    axes[1, 0].set_xlabel("Sample size $N$")
    axes[1, 1].set_xlabel("Sample size $N$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Monte Carlo Distribution of ARC Profile Similarity (Cronbach–Gleser Decomposition)",
        fontsize=12.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    return _save_vector_figure(fig, fig_dir, "fig1_mc_arc_similarity")


# ---------------------------------------------------------------------------
# Figure 2: Delta BIC (GMM 1-full vs. LPA 3-diag) and Split-half Kappa
# ---------------------------------------------------------------------------


def build_figure2_bic_and_kappa(df_sim: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """Plot Delta BIC (LPA 3-diag vs. GMM 1-full) and split-half kappa across N."""
    conditions = _ordered_conditions(df_sim["condition"].unique().tolist())
    n_values = sorted(int(v) for v in df_sim["n"].unique().tolist())
    x_pos = np.arange(len(n_values), dtype=np.float64)
    offsets = np.linspace(-0.18, 0.18, max(len(conditions), 1))

    fig, (ax_bic, ax_kap) = plt.subplots(1, 2, figsize=(11.8, 4.7), sharex=True)

    for idx, cond in enumerate(conditions):
        ls, marker = CONDITION_STYLES.get(cond, ("-", "o"))
        color = CONDITION_COLORS.get(cond, "#333333")

        xs: list[float] = []
        bic_m: list[float] = []
        bic_lo: list[float] = []
        bic_hi: list[float] = []
        kap_m: list[float] = []
        kap_lo: list[float] = []
        kap_hi: list[float] = []

        for j, n_val in enumerate(n_values):
            sub = df_sim.loc[(df_sim["condition"] == cond) & (df_sim["n"] == n_val)]
            if sub.empty:
                continue
            bm, bq025, bq975 = _summarize_array(
                sub["delta_bic_lpa3_vs_gmm1"].to_numpy(dtype=np.float64)
            )
            km, kq025, kq975 = _summarize_array(
                sub["wk_split_half_kappa"].to_numpy(dtype=np.float64)
            )
            xs.append(x_pos[j] + offsets[idx])
            bic_m.append(bm)
            bic_lo.append(max(0.0, bm - bq025))
            bic_hi.append(max(0.0, bq975 - bm))
            kap_m.append(km)
            kap_lo.append(max(0.0, km - kq025))
            kap_hi.append(max(0.0, kq975 - km))

        ax_bic.errorbar(
            xs,
            bic_m,
            yerr=[bic_lo, bic_hi],
            label=CONDITION_LABELS.get(cond, cond),
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=5.0,
            linewidth=1.4,
            capsize=2.5,
        )
        ax_kap.errorbar(
            xs,
            kap_m,
            yerr=[kap_lo, kap_hi],
            label=CONDITION_LABELS.get(cond, cond),
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=5.0,
            linewidth=1.4,
            capsize=2.5,
        )

    ax_bic.axhline(0.0, color="#222222", linestyle=":", linewidth=1.1)
    ax_bic.set_yscale("symlog", linthresh=20.0)
    ax_bic.set_title("(A) Model Selection: $\\Delta$BIC (LPA 3-diag $-$ GMM 1-full)")
    ax_bic.set_ylabel("$\\Delta$BIC (> 0 favours 1-component full-covariance)")
    ax_bic.set_xlabel("Sample size $N$")
    ax_bic.set_xticks(x_pos)
    ax_bic.set_xticklabels([f"{n:,}" for n in n_values])

    ax_kap.axhline(0.60, color="#222222", linestyle=":", linewidth=1.1, label="Asendorpf threshold ($\\kappa = 0.60$)")
    ax_kap.set_title("(B) Split-Half Double Cross-Validation Replicability ($\\kappa$)")
    ax_kap.set_ylabel("Cohen's $\\kappa$")
    ax_kap.set_xlabel("Sample size $N$")
    ax_kap.set_ylim(-0.05, 1.02)
    ax_kap.set_xticks(x_pos)
    ax_kap.set_xticklabels([f"{n:,}" for n in n_values])

    handles, labels = ax_bic.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Model-Selection Artefact ($\\Delta$BIC) and Split-Half Replicability Across Null Conditions",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    return _save_vector_figure(fig, fig_dir, "fig2_mc_bic_and_kappa")


# ---------------------------------------------------------------------------
# Figure 3: Spanish BFI-2 Empirical Contrast + Centroid Profiles vs. Copula
# ---------------------------------------------------------------------------


def _solve_lognormal_sigma_for_skew(target_skew: float) -> float:
    """Solve (exp(s^2) + 2) * sqrt(exp(s^2) - 1) = |target_skew| for signed sigma."""
    if not math.isfinite(target_skew) or abs(target_skew) < 1e-4:
        return 0.0
    target = abs(float(target_skew))
    lo, hi = 0.0, 1.5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        es2 = math.exp(mid * mid)
        val = (es2 + 2.0) * math.sqrt(max(0.0, es2 - 1.0))
        if val < target:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    return sigma if target_skew > 0.0 else -sigma


def _simulate_copula_predicted_centroids(
    summary: dict[str, Any],
    n_draws: int = 40,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate Ward+k-means centroids under a Gaussian copula calibrated to R_emp
    and empirical marginal skewnesses to obtain predicted mean and [2.5%, 97.5%] bands.
    """
    diag = summary["empirical_diagnostics"]
    n_emp = int(diag["n"])
    r_emp = nearest_pd_correlation(np.asarray(diag["r_emp"], dtype=np.float64))
    skew_map = diag["marginal_skewness"]
    skew_params = tuple(_solve_lognormal_sigma_for_skew(float(skew_map[t])) for t in TRAIT_ORDER)
    marginals = default_skewed_marginals(skew_params)

    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 303]))
    cal = calibrate_gaussian_copula(
        target_corr=r_emp,
        marginals=marginals,
        rng=rng,
        n_cal=30_000,
        tol=1e-3,
    )

    matched_stack = np.empty((n_draws, 3, 5), dtype=np.float64)
    for b in range(n_draws):
        z = rng.standard_normal((n_emp, 5))
        sample = _apply_gaussian_copula(z, cal.latent_corr, marginals)
        wk = fit_ward_kmeans(
            sample,
            n_clusters=3,
            random_state=rng,
            ward_subsample=2000 if n_emp > 2000 else None,
        )
        m = match_centroids(wk.centers, ARC_PROTOTYPES, metric="d2")
        matched_stack[b] = m.matched_centroids

    return (
        np.mean(matched_stack, axis=0),
        np.quantile(matched_stack, 0.025, axis=0),
        np.quantile(matched_stack, 0.975, axis=0),
    )


def _plot_null_density(
    ax: plt.Axes,
    values: NDArray[np.float64],
    label: str,
    color: str,
) -> None:
    """Plot a smooth KDE curve (with safe fallback if variance is near zero)."""
    valid = values[np.isfinite(values)]
    if valid.size < 3:
        return
    if float(np.std(valid)) < 1e-9:
        ax.axvline(float(valid[0]), color=color, linewidth=1.8, label=label)
        return
    kde = gaussian_kde(valid)
    lo, hi = float(np.min(valid)), float(np.max(valid))
    pad = 0.08 * max(hi - lo, 1e-3)
    grid = np.linspace(lo - pad, hi + pad, 240)
    dens = kde(grid)
    ax.plot(grid, dens, color=color, linewidth=1.8, label=label)
    ax.fill_between(grid, 0.0, dens, color=color, alpha=0.14)


def build_figure3_empirical_contrast(
    df_emp: pd.DataFrame,
    summary: dict[str, Any],
    fig_dir: Path,
) -> list[Path]:
    """Plot Spanish BFI-2 empirical contrast (distributions + centroid profiles)."""
    emp_m = summary["empirical_metrics"]
    n_emp = int(summary["empirical_diagnostics"]["n"])

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))

    # Top row: Null 1 / Null 2 / Null 3 distributions with empirical vertical line
    dist_specs: tuple[tuple[str, str, str, str], ...] = (
        ("wk_arc_mean_d2", "(A) Global ARC Distance ($D^2$)", "$D^2$ (lower = closer to ARC)", "le"),
        ("wk_arc_mean_tucker", "(B) Tucker Congruence ($\\bar{\\phi}$)", "Mean Tucker $\\bar{\\phi}$", "ge"),
        ("wk_split_half_kappa", "(C) Split-Half Replicability ($\\kappa$)", "Cohen's $\\kappa$", "ge"),
    )

    for ax, (col, title, xlabel, direction) in zip(axes[0], dist_specs):
        emp_val = float(emp_m[col])
        p_lines: list[str] = []
        for null_name in EMP_NULL_ORDER:
            sub_vals = df_emp.loc[df_emp["null_model"] == null_name, col].to_numpy(dtype=np.float64)
            valid = sub_vals[np.isfinite(sub_vals)]
            _plot_null_density(
                ax,
                valid,
                label=EMP_NULL_LABELS[null_name],
                color=EMP_NULL_COLORS[null_name],
            )
            if valid.size > 0:
                hits = np.sum(valid <= emp_val) if direction == "le" else np.sum(valid >= emp_val)
                p_emp = float(hits) / float(valid.size)
                short_n = null_name.split("_")[0].replace("null", "N")
                p_lines.append(f"{short_n}: $p={p_emp:.3f}$")

        ax.axvline(
            emp_val,
            color="#111111",
            linestyle="--",
            linewidth=2.0,
            label=f"Empirical BFI-2 ({emp_val:.3f})",
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.text(
            0.03,
            0.95,
            "\n".join(p_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#CCCCCC"},
        )

    handles_top, labels_top = axes[0, 0].get_legend_handles_labels()
    axes[0, 2].legend(
        handles_top,
        labels_top,
        loc="upper right",
        fontsize=7.5,
        frameon=True,
        framealpha=0.88,
    )

    # Bottom row: Empirical Ward+k-means centroids vs. Copula-predicted centroids
    cop_mean, cop_q025, cop_q975 = _simulate_copula_predicted_centroids(summary, n_draws=40)
    emp_centroids_list = summary["empirical_centroids"]["ward_kmeans_k3"]
    emp_by_type = {entry["arc_type"]: entry for entry in emp_centroids_list}
    x_traits = np.arange(len(TRAIT_ORDER), dtype=np.float64)

    panel_letters = ("(D)", "(E)", "(F)")
    for k, (ax, arc_name, letter) in enumerate(zip(axes[1], ARC_LABELS, panel_letters)):
        entry = emp_by_type[arc_name]
        emp_vec = np.array([float(entry["centroid_neoac"][t]) for t in TRAIT_ORDER], dtype=np.float64)
        proto_vec = ARC_PROTOTYPES[k]

        ax.fill_between(
            x_traits,
            cop_q025[k],
            cop_q975[k],
            color="#D55E00",
            alpha=0.20,
            label="Null 3 copula 95% band",
        )
        ax.plot(
            x_traits,
            cop_mean[k],
            color="#D55E00",
            linestyle="-",
            marker="s",
            markersize=5.0,
            linewidth=1.8,
            label="Null 3 copula mean",
        )
        ax.plot(
            x_traits,
            emp_vec,
            color="#111111",
            linestyle="-",
            marker="o",
            markersize=5.5,
            linewidth=2.0,
            label=f"Empirical BFI-2 ({entry['proportion'] * 100:.1f}%)",
        )
        ax.plot(
            x_traits,
            proto_vec,
            color="#0072B2",
            linestyle=":",
            marker="^",
            markersize=4.5,
            linewidth=1.4,
            label="Canonical ARC prototype",
        )
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.8)
        ax.set_title(f"{letter} {arc_name} ($\\phi = {entry['tucker_congruence']:.2f}$)")
        ax.set_xticks(x_traits)
        ax.set_xticklabels(list(TRAIT_ORDER))
        ax.set_xlabel("Big Five dimension (N unreversed)")
        ax.set_ylim(-1.6, 1.6)
        if k == 0:
            ax.set_ylabel("Standardized centroid ($z$-score)")
        if k == 2:
            ax.legend(loc="lower left", fontsize=7.4, frameon=True, framealpha=0.88)

    fig.suptitle(
        f"Empirical Contrast of Spanish BFI-2 ($N = {n_emp:,}$) Against Unimodal Null Models",
        fontsize=12.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return _save_vector_figure(fig, fig_dir, "fig3_empirical_bfi2_contrast")


# ---------------------------------------------------------------------------
# Figure 4: Gerlach et al. (2018) Significant Types (Independent vs. Correlated)
# ---------------------------------------------------------------------------


def build_figure4_gerlach_types(
    df_gerlach_mc: pd.DataFrame,
    df_gerlach_emp: pd.DataFrame,
    gerlach_emp_summary: dict[str, Any],
    summary: dict[str, Any],
    fig_dir: Path,
) -> list[Path]:
    """Plot corrected Gerlach significant-type rates under independent vs. correlated nulls."""
    fig, (ax_mc, ax_emp) = plt.subplots(1, 2, figsize=(12.6, 5.1))

    # Panel A: Monte Carlo add-on at largest available N (N = 5,000), iso h=0.50, e > 1.25
    max_n = int(df_gerlach_mc["n"].max())
    sub_mc = df_gerlach_mc.loc[df_gerlach_mc["n"] == max_n]
    conditions = _ordered_conditions(sub_mc["condition"].unique().tolist())

    x_mc = np.arange(len(conditions), dtype=np.float64)
    width = 0.26
    test_specs = (
        ("iso_h050_n_sig_e125_perm", "Independent permutation null", "#CC79A7"),
        ("iso_h050_n_sig_e125_null2", "Correlated MVN null (Null 2)", "#0072B2"),
        ("iso_h050_n_sig_e125_null3", "Correlated copula null (Null 3)", "#009E73"),
    )

    for idx, (col, label, color) in enumerate(test_specs):
        means = [
            float(sub_mc.loc[sub_mc["condition"] == c, col].mean()) for c in conditions
        ]
        bars = ax_mc.bar(
            x_mc + (idx - 1) * width,
            means,
            width=width,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, c in zip(bars, conditions):
            rate_ge1 = float((sub_mc.loc[sub_mc["condition"] == c, col] >= 1).mean()) * 100.0
            ax_mc.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.06,
                f"{rate_ge1:.0f}%",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="#222222",
            )

    ax_mc.set_title(
        f"(A) Monte Carlo Gerlach Significant Types ($N = {max_n:,}$; labels = % ≥ 1 type)"
    )
    ax_mc.set_ylabel("Mean # significant types (out of $K = 4$)")
    ax_mc.set_xticks(x_mc)
    ax_mc.set_xticklabels(
        [CONDITION_LABELS.get(c, c).replace("Null ", "N") for c in conditions],
        rotation=24,
        ha="right",
        fontsize=8.0,
    )
    ax_mc.set_ylim(0.0, 4.35)
    ax_mc.legend(loc="upper left", fontsize=8.0, frameon=True, framealpha=0.92)

    # Panel B: Empirical BFI-2 null replicates — mean count + % >= 1 type
    n_emp = int(summary["empirical_diagnostics"]["n"])
    emp_iso = gerlach_emp_summary["empirical_gerlach"]["iso_h050"]["evaluations"]
    x_emp = np.arange(len(EMP_NULL_ORDER), dtype=np.float64)

    for idx, (col, label, color) in enumerate(test_specs):
        means = [
            float(df_gerlach_emp.loc[df_gerlach_emp["null_model"] == nm, col].mean())
            for nm in EMP_NULL_ORDER
        ]
        bars = ax_emp.bar(
            x_emp + (idx - 1) * width,
            means,
            width=width,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, nm in zip(bars, EMP_NULL_ORDER):
            rate_ge1 = float(
                (df_gerlach_emp.loc[df_gerlach_emp["null_model"] == nm, col] >= 1).mean()
            ) * 100.0
            ax_emp.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.06,
                f"{rate_ge1:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color="#222222",
            )

    emp_note = (
        f"Observed BFI-2 (B=500, h=0.50, e>1.25):\n"
        f"  Perm null: {int(emp_iso['perm']['n_sig_e125'])}/4 "
        f"(e>1.00: {int(emp_iso['perm']['n_sig_e100'])}/4)\n"
        f"  Null 2 (MVN): {int(emp_iso['null2']['n_sig_e125'])}/4 "
        f"(e>1.00: {int(emp_iso['null2']['n_sig_e100'])}/4)\n"
        f"  Null 3 (Copula): {int(emp_iso['null3']['n_sig_e125'])}/4 "
        f"(e>1.00: {int(emp_iso['null3']['n_sig_e100'])}/4)"
    )
    ax_emp.text(
        0.97,
        0.96,
        emp_note,
        transform=ax_emp.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "alpha": 0.92, "edgecolor": "#CCCCCC"},
    )
    ax_emp.set_title(f"(B) BFI-2 Spanish Null Replicates ($N = {n_emp:,}$; labels = % ≥ 1 type)")
    ax_emp.set_ylabel("Mean # significant types (out of $K = 4$)")
    ax_emp.set_xticks(x_emp)
    ax_emp.set_xticklabels([EMP_NULL_LABELS[nm] for nm in EMP_NULL_ORDER], rotation=12, ha="right")
    ax_emp.set_ylim(0.0, 4.45)

    fig.suptitle(
        "Gerlach et al. (2018) Density-Enrichment False Positives (Isotropic KDE $h=0.50$, $e>1.25$, $\\alpha=0.01$)",
        fontsize=11.3,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return _save_vector_figure(fig, fig_dir, "fig4_gerlach_significant_types")


# ---------------------------------------------------------------------------
# Final Audit Summary (results/AUDIT_SUMMARY.md)
# ---------------------------------------------------------------------------


def build_audit_summary(
    df_sim: pd.DataFrame,
    block_files: list[Path],
    df_emp: pd.DataFrame,
    summary: dict[str, Any],
    df_gerlach_mc: pd.DataFrame,
    df_gerlach_emp: pd.DataFrame,
    gerlach_emp_summary: dict[str, Any],
    df_table1: pd.DataFrame,
    df_table2: pd.DataFrame,
    generated_files: list[Path],
    out_dir: Path,
) -> Path:
    """Write comprehensive Markdown audit summary to results/AUDIT_SUMMARY.md."""
    diag = summary["empirical_diagnostics"]
    emp_m = summary["empirical_metrics"]
    meta = summary["metadata"]
    cal = summary["null3_copula_calibration"]

    n_emp = int(diag["n"])
    ratio = float(diag["lambda2_over_lambda1"])
    exceeds_thr = bool(diag["exceeds_noncollinearity_threshold"])
    max_n = int(df_sim["n"].max())

    # Evaluate hypothesis statuses from observed results
    mc_max = df_sim.loc[df_sim["n"] == max_n]
    d2_n1 = float(mc_max.loc[mc_max["null_family"] == "null1", "wk_arc_mean_d2"].mean())
    d2_n2 = float(mc_max.loc[mc_max["null_family"] == "null2", "wk_arc_mean_d2"].mean())
    d2_n3 = float(mc_max.loc[mc_max["null_family"] == "null3", "wk_arc_mean_d2"].mean())

    # H3 separated cleanly by Null 2 vs Null 3
    bic_n2_vdl = float(mc_max.loc[mc_max["condition"] == "null2_vdl", "delta_bic_lpa3_vs_gmm1"].mean())
    bic_n2_be = float(mc_max.loc[mc_max["condition"] == "null2_be", "delta_bic_lpa3_vs_gmm1"].mean())
    bic_n3_vdl = float(mc_max.loc[mc_max["condition"] == "null3_vdl_copula", "delta_bic_lpa3_vs_gmm1"].mean())
    bic_n3_be = float(mc_max.loc[mc_max["condition"] == "null3_be_copula", "delta_bic_lpa3_vs_gmm1"].mean())
    bic_n2_all_pos = bool(
        (df_sim.loc[df_sim["null_family"] == "null2"].groupby(["condition", "n"])["delta_bic_lpa3_vs_gmm1"].mean() > 0).all()
    )
    bic_n3_ge1k_neg = bool(
        (df_sim.loc[(df_sim["null_family"] == "null3") & (df_sim["n"] >= 1000)].groupby(["condition", "n"])["delta_bic_lpa3_vs_gmm1"].mean() < 0).all()
    )

    # H4 from corrected Gerlach add-on (N=5,000, iso_h050, e > 1.25)
    g_max_n = int(df_gerlach_mc["n"].max())
    g_mc_max = df_gerlach_mc.loc[df_gerlach_mc["n"] == g_max_n]
    g_corr_n2 = g_mc_max.loc[g_mc_max["null_family"] == "null2"]
    g_corr_n3 = g_mc_max.loc[g_mc_max["null_family"] == "null3"]
    g_n2_perm = float(g_corr_n2["iso_h050_n_sig_e125_perm"].mean())
    g_n2_n2 = float(g_corr_n2["iso_h050_n_sig_e125_null2"].mean())
    g_n2_n3 = float(g_corr_n2["iso_h050_n_sig_e125_null3"].mean())
    g_n2_perm_rate = float((g_corr_n2["iso_h050_n_sig_e125_perm"] >= 1).mean()) * 100.0
    g_n2_n2_rate = float((g_corr_n2["iso_h050_n_sig_e125_null2"] >= 1).mean()) * 100.0
    g_n3_perm = float(g_corr_n3["iso_h050_n_sig_e125_perm"].mean())
    g_n3_n2 = float(g_corr_n3["iso_h050_n_sig_e125_null2"].mean())
    g_n3_n3 = float(g_corr_n3["iso_h050_n_sig_e125_null3"].mean())
    g_n3_n3_rate = float((g_corr_n3["iso_h050_n_sig_e125_null3"] >= 1).mean()) * 100.0

    emp_g_iso = gerlach_emp_summary["empirical_gerlach"]["iso_h050"]["evaluations"]
    emp_g_scott = gerlach_emp_summary["empirical_gerlach"]["scott"]["evaluations"]

    p_d2_n2 = float(
        df_table2.loc[df_table2["metric"] == "wk_arc_mean_d2", "null2_mvnormal_p_emp"].iloc[0]
    )
    p_d2_n3 = float(
        df_table2.loc[df_table2["metric"] == "wk_arc_mean_d2", "null3_gaussian_copula_p_emp"].iloc[0]
    )
    p_kap_n3 = float(
        df_table2.loc[df_table2["metric"] == "wk_split_half_kappa", "null3_gaussian_copula_p_emp"].iloc[0]
    )

    lines: list[str] = [
        "# Auditoria Final dels Resultats — Algebra of Personality Types (ARC)",
        "",
        f"- **Data de generació**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Llavor base (PROJECT_RULES.md)**: `{BASE_SEED}` (empíric: `{meta.get('seed', BASE_SEED)}`; add-on Gerlach: `[{BASE_SEED}, 2018]`)",
        f"- **Blocs Monte Carlo carregats**: {len(block_files)} fitxers (`{len(df_sim):,}` rèpliques totals; mides mostrals $N$: {sorted(int(x) for x in df_sim['n'].unique())})",
        f"- **Add-on Gerlach corregit**: `{len(df_gerlach_mc):,}` rèpliques MC ($N \\in {sorted(int(x) for x in df_gerlach_mc['n'].unique())}$, `B=200`, $\\alpha=0.01$) + BFI-2 empíric (`B=500`) i `{len(df_gerlach_emp):,}` rèpliques nul·les BFI-2",
        f"- **Contrast empíric**: `{meta.get('dataset_name', 'bfi2_es_study1')}` ($N = {n_emp:,}$, `{meta.get('reps_per_null', 'N/A')}` rèpliques per model nul; `{len(df_emp):,}` rèpliques totals)",
        "",
        "## 1. Diagnòstics empírics del BFI-2 espanyol",
        "",
        f"- **Mida mostral neta**: $N = {n_emp:,}$ (files eliminades per NA: `{diag.get('n_dropped_na', 0)}`)",
        f"- **Ordre estricte de dimensions**: `{diag.get('trait_order', list(TRAIT_ORDER))}` (`N` sense invertir)",
        f"- **Valors propis de $\\mathbf{{R}}_{{\\mathrm{{emp}}}}$**: "
        + ", ".join(f"`{v:.5f}`" for v in diag["eigenvalues"]),
        f"- **Ràtio $\\lambda_2 / \\lambda_1$**: `{ratio:.5f}` vs. llindar suficient de no-col·linealitat `{NONCOLLINEARITY_THRESHOLD:.5f}` "
        f"({'**supera el llindar**' if exceeds_thr else '**per sota del llindar**'}, marge `{ratio - NONCOLLINEARITY_THRESHOLD:+.5f}`)",
        f"- **Determinant $\\det(\\mathbf{{R}}_{{\\mathrm{{emp}}}})$**: `{float(diag['det_r_emp']):.5f}` (definida positiva: `{diag.get('r_emp_is_positive_definite', True)}`)",
        "- **Asimetria marginal**: "
        + ", ".join(f"{t}=`{float(diag['marginal_skewness'][t]):+.3f}`" for t in TRAIT_ORDER),
        "- **Curtosi d'excés marginal**: "
        + ", ".join(f"{t}=`{float(diag['marginal_kurtosis_excess'][t]):+.3f}`" for t in TRAIT_ORDER),
        f"- **Calibració còpula Nul 3**: error màxim absolut `{float(cal['max_abs_error']):.2e}`, "
        f"convergit=`{cal['converged']}` ({cal['iterations']} iteracions), "
        f"unimodalitat conjunta=`{cal['unimodality']['is_unimodal']}` "
        f"(modes marginals={cal['unimodality']['marginal_modes']}, PC12={cal['unimodality']['pc12_joint_modes']})",
        "",
        "## 2. Centroides empírics alineats a ARC (Ward + k-means, $K=3$)",
        "",
        "| Tipus ARC | Proporció | N | E | O | A | C | $D^2$ | Tucker $\\phi$ |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for entry in summary["empirical_centroids"]["ward_kmeans_k3"]:
        c = entry["centroid_neoac"]
        lines.append(
            f"| {entry['arc_type']} | {entry['proportion']*100:.1f}% | "
            f"{c['N']:+.2f} | {c['E']:+.2f} | {c['O']:+.2f} | {c['A']:+.2f} | {c['C']:+.2f} | "
            f"{entry['d2']:.3f} | {entry['tucker_congruence']:.3f} |"
        )

    lines.extend(
        [
            "",
            f"## 3. Xifres clau del Monte Carlo (resum a $N = {max_n:,}$ i add-on Gerlach a $N = {g_max_n:,}$)",
            "",
            f"### 3a. Mètriques principals a $N = {max_n:,}$ (1,000 rèpliques/bloc)",
            "",
            "| Condició | $D^2_{\\mathrm{ARC}}$ | Tucker $\\bar{\\phi}$ | $\\kappa$ split-half | $\\Delta$BIC (3-diag $-$ 1-full) |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )

    t1_max = df_table1.loc[df_table1["n"] == max_n]
    for _, r in t1_max.iterrows():
        lines.append(
            f"| {r['condition_label']} | {r['d2_mean']:.3f} [{r['d2_q025']:.3f}, {r['d2_q975']:.3f}] | "
            f"{r['tucker_mean']:.3f} | {r['kappa_mean']:.3f} | {r['delta_bic_lpa3_gmm1_mean']:+.1f} |"
        )

    lines.extend(
        [
            "",
            f"### 3b. Add-on Gerlach et al. (2018) corregit ($B=200$, $\\alpha=0.01$, $h=0.50$ isotròpic, 200 rèpliques/bloc)",
            "",
            "| Condició | $N$ | Perm ($e>1.25$) [$\\% \\ge 1$] | Nul 2 ($e>1.25$) [$\\% \\ge 1$] | Nul 3 ($e>1.25$) [$\\% \\ge 1$] | Perm ($e>1.00$) | Scott Perm ($e>1.25$) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for cond in _ordered_conditions(df_gerlach_mc["condition"].unique().tolist()):
        for n_val in sorted(int(v) for v in df_gerlach_mc["n"].unique()):
            sg = df_gerlach_mc.loc[(df_gerlach_mc["condition"] == cond) & (df_gerlach_mc["n"] == n_val)]
            if sg.empty:
                continue
            p125 = sg["iso_h050_n_sig_e125_perm"]
            n2125 = sg["iso_h050_n_sig_e125_null2"]
            n3125 = sg["iso_h050_n_sig_e125_null3"]
            p100 = sg["iso_h050_n_sig_e100_perm"]
            sp125 = sg["scott_n_sig_e125_perm"]
            lines.append(
                f"| {CONDITION_LABELS.get(cond, cond)} | {n_val:,} | "
                f"{p125.mean():.2f} [{(p125 >= 1).mean()*100:.0f}%] | "
                f"{n2125.mean():.2f} [{(n2125 >= 1).mean()*100:.0f}%] | "
                f"{n3125.mean():.2f} [{(n3125 >= 1).mean()*100:.0f}%] | "
                f"{p100.mean():.2f} | {sp125.mean():.2f} |"
            )

    lines.extend(
        [
            "",
            f"## 4. Contrast empíric BFI-2 espanyol ($N = {n_emp:,}$)",
            "",
            "| Mètrica | Empíric | Nul 1 ($p_{\\mathrm{emp}}$) | Nul 2 ($p_{\\mathrm{emp}}$) | Nul 3 ($p_{\\mathrm{emp}}$) |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, r in df_table2.iterrows():
        lines.append(
            f"| {r['metric_label']} | {r['empirical']:.3f} | "
            f"{r['null1_permutation_mean']:.3f} (`{r['null1_permutation_p_emp']:.3f}`) | "
            f"{r['null2_mvnormal_mean']:.3f} (`{r['null2_mvnormal_p_emp']:.3f}`) | "
            f"{r['null3_gaussian_copula_mean']:.3f} (`{r['null3_gaussian_copula_p_emp']:.3f}`) |"
        )

    inc_val = summary.get("incremental_validity")
    if inc_val:
        lines.extend(["", "### Validesa predictiva incremental fora de mostra (CV)", ""])
        for crit, cinfo in inc_val.items():
            if cinfo.get("skipped"):
                lines.append(f"- `{crit}`: omès ({cinfo.get('reason')})")
                continue
            lines.append(f"- **`{crit}`** (tasca: `{cinfo['task']}`, $N={cinfo['n']}$, {cinfo['n_splits']}-fold CV):")
            for mname, minfo in cinfo["models"].items():
                lines.append(
                    f"  - `{mname}` ({minfo['metric_name']}): base=`{minfo['mean_baseline']:.4f}`, "
                    f"augmentat=`{minfo['mean_augmented']:.4f}`, $\\Delta$=`{minfo['mean_delta']:+.4f}` ± `{minfo['std_delta']:.4f}`"
                )

    h4_confirmed = (g_n2_perm > g_n2_n2) and (g_n3_perm > g_n3_n3)
    lines.extend(
        [
            "",
            "## 5. Estat de les hipòtesis",
            "",
            f"1. **H1 — Geometria espectral ($\\lambda_2 / \\lambda_1$ vs. `0.50764`)**: "
            f"Al BFI-2 espanyol $\\lambda_2 / \\lambda_1 = {ratio:.5f}$ "
            f"({'supera' if exceeds_thr else 'no supera'} el llindar suficient de no-col·linealitat `0.50764`).",
            f"2. **H2 — Emergència d'ARC sota nuls unimodals correlacionats**: "
            f"{'**CONFIRMAT**' if (d2_n2 < d2_n1 and d2_n3 < d2_n1) else '**A REVISAR**'} — "
            f"a $N={max_n:,}$, $D^2_{{\\mathrm{{ARC}}}}$ cau de `{d2_n1:.3f}` (Nul 1 isòtrop) a `{d2_n2:.3f}` (Nul 2 MVN) i `{d2_n3:.3f}` (Nul 3 còpula).",
            f"3. **H3 — Artefacte de selecció de models (LPA diagonal vs. GMM covariància plena) i sobreextracció per no-normalitat**: "
            f"{'**CONFIRMAT (bifurcació Nul 2 vs. Nul 3)**' if (bic_n2_all_pos and bic_n3_ge1k_neg) else '**A REVISAR**'} — "
            f"\n   - **Sota Nul 2 (normalitat correlacionada)**: GMM 1-full supera sistemàticament LPA 3-diag a totes les mides mostrals ($\\Delta\\mathrm{{BIC}} > 0$, arribant a `{bic_n2_vdl:+,.1f}` per vdL i `{bic_n2_be:+,.1f}` per BE a $N={max_n:,}$, i `{float(emp_m['delta_bic_lpa3_vs_gmm1']):+.1f}` al BFI-2 empíric)."
            f"\n   - **Sota Nul 3 (còpula unimodal amb asimetria moderada)**: a partir de $N=1{{,}}000$, LPA 3-diag supera fins i tot GMM 1-full ($\\Delta\\mathrm{{BIC}} < 0$, arribant a `{bic_n3_vdl:+,.1f}` per vdL i `{bic_n3_be:+,.1f}` per BE a $N={max_n:,}$), il·lustrant la sobreextracció per no-normalitat de Bauer i Curran (2003).",
            f"4. **H4 — Falsos positius de Gerlach et al. (2018) sota nul independent vs. nuls correlacionats (add-on corregit, $B=200$, $\\alpha=0.01$, KDE isotròpica $h=0.50$, $e>1.25$)**: "
            f"{'**CONFIRMAT**' if h4_confirmed else '**A REVISAR**'} — "
            f"a $N={g_max_n:,}$:\n"
            f"   - **Dades Nul 2 (MVN correlacionada)**: el nul de permutació independent detecta `{g_n2_perm:.2f}` tipus significatius de mitjana (`{g_n2_perm_rate:.1f}%` de rèpliques amb $\\ge 1$ fals positiu), mentre que el nul correlacionat Nul 2 el col·lapsa a `{g_n2_n2:.2f}` (`{g_n2_n2_rate:.1f}%`) i Nul 3 a `{g_n2_n3:.2f}`.\n"
            f"   - **Dades Nul 3 (còpula asimètrica unimodal)**: el nul de permutació detecta `{g_n3_perm:.2f}` tipus, Nul 2 (MVN) encara genera `{g_n3_n2:.2f}` falsos positius per asimetria marginal, i Nul 3 (còpula calibrada) ho col·lapsa a `{g_n3_n3:.2f}` (`{g_n3_n3_rate:.1f}%`).\n"
            f"   - **BFI-2 espanyol empíric ($B=500$)**: amb KDE isotròpica $h=0.50$ ($e>1.25$), Perm=`{int(emp_g_iso['perm']['n_sig_e125'])}/4`, Nul 2=`{int(emp_g_iso['null2']['n_sig_e125'])}/4`, Nul 3=`{int(emp_g_iso['null3']['n_sig_e125'])}/4` "
            f"(amb Scott $e>1.25$: Perm=`{int(emp_g_scott['perm']['n_sig_e125'])}/4`, Nul 2=`{int(emp_g_scott['null2']['n_sig_e125'])}/4`, Nul 3=`{int(emp_g_scott['null3']['n_sig_e125'])}/4`).",
            f"5. **H5 — Contrast empíric BFI-2 contra nuls correlacionats**: "
            f"{'**CONSISTENT AMB NUL UNIMODAL CORRELACIONAT**' if (p_d2_n3 > 0.05 and p_kap_n3 > 0.05) else '**DESVIACIÓ DETECTADA (REVISAR DETALLS)**'} — "
            f"per a $D^2_{{\\mathrm{{ARC}}}}$, $p_{{\\mathrm{{emp}}}}(\\text{{Nul 2}}) = {p_d2_n2:.3f}$ i $p_{{\\mathrm{{emp}}}}(\\text{{Nul 3}}) = {p_d2_n3:.3f}$; "
            f"per a $\\kappa$ split-half, $p_{{\\mathrm{{emp}}}}(\\text{{Nul 3}}) = {p_kap_n3:.3f}$.",
            "",
            "## 6. Fitxers generats",
            "",
            *[f"- `{path}`" for path in generated_files],
            "",
        ]
    )

    audit_path = out_dir / "AUDIT_SUMMARY.md"
    audit_path.write_text("\n".join(lines), encoding="utf-8")
    return audit_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build academic tables, vector figures, and AUDIT_SUMMARY.md."
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        default=Path("results/simulations"),
        help="Directory containing Monte Carlo block_*.parquet files (default: results/simulations).",
    )
    parser.add_argument(
        "--emp-reps",
        type=Path,
        default=Path("results/empirical/bfi2_es_study1_null_reps.parquet"),
        help="Path to empirical null replicates Parquet (default: results/empirical/bfi2_es_study1_null_reps.parquet).",
    )
    parser.add_argument(
        "--emp-summary",
        type=Path,
        default=Path("results/empirical/bfi2_es_study1_summary.json"),
        help="Path to empirical summary JSON (default: results/empirical/bfi2_es_study1_summary.json).",
    )
    parser.add_argument(
        "--gerlach-dir",
        type=Path,
        default=Path("results/gerlach_addon"),
        help="Directory containing corrected Gerlach add-on files (default: results/gerlach_addon).",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for vector figures (default: results/figures).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for tables and AUDIT_SUMMARY.md (default: results).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load results and generate all tables, vector figures, and the audit report."""
    args = _parse_args(argv)
    _configure_matplotlib()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    df_sim, block_files = _load_simulation_blocks(args.sim_dir)
    df_emp, summary = _load_empirical_results(args.emp_reps, args.emp_summary)
    df_gerlach_mc, df_gerlach_emp, gerlach_emp_summary = _load_gerlach_addon(args.gerlach_dir)

    generated: list[Path] = []

    df_table1, t1_csv, t1_tex = build_table1_montecarlo(df_sim, df_gerlach_mc, args.out_dir)
    generated.extend([t1_csv, t1_tex])

    df_table2, t2_csv, t2_tex = build_table2_empirical(
        df_emp, summary, df_gerlach_emp, gerlach_emp_summary, args.out_dir
    )
    generated.extend([t2_csv, t2_tex])

    generated.extend(build_figure1_mc_arc_similarity(df_sim, args.fig_dir))
    generated.extend(build_figure2_bic_and_kappa(df_sim, args.fig_dir))
    generated.extend(build_figure3_empirical_contrast(df_emp, summary, args.fig_dir))
    generated.extend(
        build_figure4_gerlach_types(
            df_gerlach_mc, df_gerlach_emp, gerlach_emp_summary, summary, args.fig_dir
        )
    )

    audit_path = build_audit_summary(
        df_sim=df_sim,
        block_files=block_files,
        df_emp=df_emp,
        summary=summary,
        df_gerlach_mc=df_gerlach_mc,
        df_gerlach_emp=df_gerlach_emp,
        gerlach_emp_summary=gerlach_emp_summary,
        df_table1=df_table1,
        df_table2=df_table2,
        generated_files=generated,
        out_dir=args.out_dir,
    )
    generated.append(audit_path)

    print("=== Generated Tables, Figures, and Audit Summary ===")
    for path in generated:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
