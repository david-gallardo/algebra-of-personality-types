#!/usr/bin/env python3
"""Empirical dataset comparison against ARC null models (Null 1, Null 2, Null 3).

Loads an empirical Big Five CSV in strict (N, E, O, A, C) order (with N unreversed),
computes correlation-matrix and marginal diagnostics, runs the exact Monte Carlo
clustering/validation pipeline on the empirical sample, generates parallel null
replicates under Null 1 (independent column permutation), Null 2 (correlated
multivariate normal), and Null 3 (calibrated Gaussian copula with empirical
marginals), estimates empirical tail probabilities, optionally evaluates
out-of-sample incremental predictive validity, and saves Parquet + JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Literal

from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import kurtosis, skew

# Allow running directly as `python scripts/run_empirical_comparison.py` without PYTHONPATH=src
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from arc_study.clustering import (  # noqa: E402
    fit_gmm_full,
    fit_lpa_diagonal,
    fit_ward_kmeans,
    run_gerlach_procedure,
)
from arc_study.generators import (  # noqa: E402
    _apply_gaussian_copula,
    calibrate_gaussian_copula,
    check_joint_unimodality,
    empirical_marginal_ppfs,
    nearest_pd_correlation,
    sample_null1_column_permutation,
    sample_null2_correlated_normal,
)
from arc_study.matrices import (  # noqa: E402
    BASE_SEED,
    TRAIT_ORDER,
    is_positive_definite,
)
from arc_study.metrics import (  # noqa: E402
    ARC_LABELS,
    ARC_PROTOTYPES,
    CentroidMatchResult,
    assignment_stability,
    incremental_predictive_validity,
    make_gmm_posterior_extractor,
    make_ward_kmeans_onehot_extractor,
    match_centroids,
    split_half_replicability,
)

NullModelName = Literal["null1_permutation", "null2_mvnormal", "null3_gaussian_copula"]
NULL_MODELS: tuple[NullModelName, ...] = (
    "null1_permutation",
    "null2_mvnormal",
    "null3_gaussian_copula",
)

# Theoretical sufficient non-collinearity threshold: c_L / (27 / (16 * pi)) - 1
NONCOLLINEARITY_THRESHOLD: float = 0.50764

# Metric comparison direction against null replicates:
# "le" -> P(sim <= emp) for distances / BIC differences where smaller is more ARC-like
# "ge" -> P(sim >= emp) for similarity / replicability where larger is more ARC-like
COMPARISON_METRICS: tuple[tuple[str, Literal["le", "ge"]], ...] = (
    ("wk_arc_mean_d2", "le"),
    ("wk_arc_mean_elevation", "le"),
    ("wk_arc_mean_scatter", "le"),
    ("wk_arc_mean_shape", "le"),
    ("wk_arc_mean_tucker", "ge"),
    ("wk_arc_min_tucker", "ge"),
    ("wk_split_half_kappa", "ge"),
    ("wk_split_half_ari", "ge"),
    ("wk_split_half_nmi", "ge"),
    ("wk_between_var", "ge"),
    ("lpa3_arc_mean_d2", "le"),
    ("lpa3_arc_mean_elevation", "le"),
    ("lpa3_arc_mean_scatter", "le"),
    ("lpa3_arc_mean_shape", "le"),
    ("lpa3_arc_mean_tucker", "ge"),
    ("lpa_delta_bic_3_vs_1", "le"),
    ("gmm3_arc_mean_d2", "le"),
    ("gmm3_arc_mean_elevation", "le"),
    ("gmm3_arc_mean_scatter", "le"),
    ("gmm3_arc_mean_shape", "le"),
    ("gmm3_arc_mean_tucker", "ge"),
    ("gmm_delta_bic_3_vs_1", "le"),
)

GERLACH_SIG_COLS: tuple[str, ...] = (
    "gerlach_n_sig_perm",
    "gerlach_n_sig_null2",
    "gerlach_n_sig_null3",
)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds into a compact human-readable string."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, rem_min = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(rem_min)}m {rem:.0f}s"


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert NumPy types, Paths, and non-finite floats for strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _load_and_prepare_data(
    input_path: Path,
    cols_neoac: list[str],
    criteria_cols: list[str] | None,
) -> tuple[pd.DataFrame, NDArray[np.float64], dict[str, Any], NDArray[np.float64]]:
    """Load CSV, drop NAs on (N, E, O, A, C), standardize, and compute diagnostics."""
    if len(cols_neoac) != 5:
        raise ValueError(
            f"--cols-neoac must specify exactly 5 columns in (N, E, O, A, C) order, got {len(cols_neoac)}."
        )
    if len(set(cols_neoac)) != 5:
        raise ValueError(f"--cols-neoac contains duplicate column names: {cols_neoac}.")

    df = pd.read_csv(input_path)
    missing_traits = [c for c in cols_neoac if c not in df.columns]
    if missing_traits:
        raise ValueError(
            f"Missing Big Five columns in {input_path}: {missing_traits}. "
            f"Available columns: {list(df.columns)}"
        )

    if criteria_cols:
        missing_criteria = [c for c in criteria_cols if c not in df.columns]
        if missing_criteria:
            raise ValueError(
                f"Missing criteria columns in {input_path}: {missing_criteria}."
            )

    n_rows_raw = int(len(df))
    df_work = df.copy()
    for col in cols_neoac:
        df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

    df_clean = df_work.dropna(subset=cols_neoac).reset_index(drop=True)
    x_raw = df_clean[cols_neoac].to_numpy(dtype=np.float64)
    finite_mask = np.all(np.isfinite(x_raw), axis=1)
    if not np.all(finite_mask):
        df_clean = df_clean.loc[finite_mask].reset_index(drop=True)
        x_raw = x_raw[finite_mask]

    n = int(x_raw.shape[0])
    if n < 20:
        raise ValueError(f"Too few complete observations after NA removal: N={n}.")

    means = x_raw.mean(axis=0)
    stds = x_raw.std(axis=0, ddof=1)
    if np.any(stds <= 0.0):
        zero_cols = [c for c, s in zip(cols_neoac, stds) if s <= 0.0]
        raise ValueError(f"Zero variance in Big Five column(s): {zero_cols}.")

    x_std = (x_raw - means) / stds

    r_emp = np.corrcoef(x_std, rowvar=False)
    r_emp = 0.5 * (r_emp + r_emp.T)
    np.fill_diagonal(r_emp, 1.0)

    r_is_pd = is_positive_definite(r_emp)
    r_emp_pd = r_emp if r_is_pd else nearest_pd_correlation(r_emp)

    eigvals = np.sort(np.linalg.eigvalsh(r_emp))[::-1]
    lambda1 = float(eigvals[0])
    lambda2 = float(eigvals[1])
    ratio = lambda2 / lambda1
    det_r = float(np.linalg.det(r_emp))

    marg_skew = skew(x_std, axis=0, bias=False)
    marg_kurt = kurtosis(x_std, axis=0, fisher=True, bias=False)

    diagnostics: dict[str, Any] = {
        "n_rows_raw": n_rows_raw,
        "n_dropped_na": n_rows_raw - n,
        "n": n,
        "trait_order": list(TRAIT_ORDER),
        "input_cols_neoac": list(cols_neoac),
        "r_emp": r_emp,
        "r_emp_is_positive_definite": r_is_pd,
        "eigenvalues": eigvals,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda2_over_lambda1": ratio,
        "noncollinearity_threshold": NONCOLLINEARITY_THRESHOLD,
        "exceeds_noncollinearity_threshold": bool(ratio > NONCOLLINEARITY_THRESHOLD),
        "threshold_margin": ratio - NONCOLLINEARITY_THRESHOLD,
        "det_r_emp": det_r,
        "marginal_skewness": {
            trait: float(val) for trait, val in zip(TRAIT_ORDER, marg_skew)
        },
        "marginal_kurtosis_excess": {
            trait: float(val) for trait, val in zip(TRAIT_ORDER, marg_kurt)
        },
    }
    return df_clean, x_std, diagnostics, r_emp_pd


def _summarize_matched_centroids(
    match: CentroidMatchResult,
    proportions: NDArray[np.float64],
) -> list[dict[str, Any]]:
    """Serialize matched centroids in canonical ARC order with profile decompositions."""
    entries: list[dict[str, Any]] = []
    for arc_idx, (c_idx, p_idx, decomp) in enumerate(
        zip(match.centroid_indices, match.prototype_indices, match.profile_decompositions)
    ):
        centroid_vec = match.matched_centroids[arc_idx]
        entries.append(
            {
                "arc_type": ARC_LABELS[p_idx],
                "original_cluster_index": int(c_idx),
                "proportion": float(proportions[c_idx]),
                "centroid_neoac": {
                    trait: float(val) for trait, val in zip(TRAIT_ORDER, centroid_vec)
                },
                "d2": decomp.d2,
                "elevation": decomp.elevation,
                "scatter": decomp.scatter,
                "shape": decomp.shape,
                "tucker_congruence": decomp.tucker_congruence,
            }
        )
    return entries


def _evaluate_sample_pipeline(
    data: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    gerlach_null_samples: int,
    split_half_splits: int,
    ward_subsample: int,
    max_kde_points: int,
    return_centroids: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run the exact Monte Carlo clustering and validation pipeline on ``data``."""
    n = int(data.shape[0])

    # 1. Ward + k-means (K = 3)
    wk = fit_ward_kmeans(
        data,
        n_clusters=3,
        random_state=rng,
        ward_subsample=ward_subsample if n > ward_subsample else None,
    )
    wk_match = match_centroids(wk.centers, ARC_PROTOTYPES, metric="d2")
    wk_elev = float(np.mean([d.elevation for d in wk_match.profile_decompositions]))
    wk_scat = float(np.mean([d.scatter for d in wk_match.profile_decompositions]))
    wk_shape = float(np.mean([d.shape for d in wk_match.profile_decompositions]))

    def _wk_half_fit(
        half: NDArray[np.float64], seed: int
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        res = fit_ward_kmeans(
            half,
            n_clusters=3,
            random_state=seed,
            ward_subsample=ward_subsample if half.shape[0] > ward_subsample else None,
        )
        return res.labels, res.centers

    wk_rep = split_half_replicability(
        data,
        fit_fn=_wk_half_fit,
        n_clusters=3,
        n_splits=split_half_splits,
        random_state=rng,
    )

    # 2. Diagonal LPA (K = 1 vs K = 3)
    lpa1 = fit_lpa_diagonal(data, n_clusters=1, n_init=10, reg_covar=1e-6, random_state=rng)
    lpa3 = fit_lpa_diagonal(data, n_clusters=3, n_init=10, reg_covar=1e-6, random_state=rng)
    lpa3_match = match_centroids(lpa3.means, ARC_PROTOTYPES, metric="d2")
    lpa3_elev = float(np.mean([d.elevation for d in lpa3_match.profile_decompositions]))
    lpa3_scat = float(np.mean([d.scatter for d in lpa3_match.profile_decompositions]))
    lpa3_shape = float(np.mean([d.shape for d in lpa3_match.profile_decompositions]))

    # 3. Full-covariance GMM (K = 1 vs K = 3)
    gmm1 = fit_gmm_full(data, n_clusters=1, n_init=10, reg_covar=1e-6, random_state=rng)
    gmm3 = fit_gmm_full(data, n_clusters=3, n_init=10, reg_covar=1e-6, random_state=rng)
    gmm3_match = match_centroids(gmm3.means, ARC_PROTOTYPES, metric="d2")
    gmm3_elev = float(np.mean([d.elevation for d in gmm3_match.profile_decompositions]))
    gmm3_scat = float(np.mean([d.scatter for d in gmm3_match.profile_decompositions]))
    gmm3_shape = float(np.mean([d.shape for d in gmm3_match.profile_decompositions]))

    # Cross-method assignment stability on the same sample
    stab_wk_lpa = assignment_stability(wk.labels, lpa3.labels)
    stab_wk_gmm = assignment_stability(wk.labels, gmm3.labels)
    stab_lpa_gmm = assignment_stability(lpa3.labels, gmm3.labels)

    # 4. Gerlach et al. (2018) GMM + KDE enrichment against Null 1, Null 2, Null 3
    gerlach = run_gerlach_procedure(
        data,
        n_clusters=4,
        covariance_type="full",
        n_init=10,
        reg_covar=1e-6,
        n_null_samples=gerlach_null_samples,
        null_models=("permutation", "null2", "null3"),
        alpha=0.01,
        min_enrichment=1.0,
        max_kde_points=max_kde_points if n > max_kde_points else None,
        copula_calibration_n=10_000,
        copula_calibration_tol=1e-3,
        random_state=rng,
    )
    ev_perm = gerlach.evaluations["permutation"]
    ev_n2 = gerlach.evaluations["null2"]
    ev_n3 = gerlach.evaluations["null3"]

    metrics: dict[str, Any] = {
        # Ward + k-means (K = 3)
        "wk_inertia": wk.inertia,
        "wk_between_var": wk.between_variance,
        "wk_arc_mean_d2": wk_match.mean_d2,
        "wk_arc_mean_elevation": wk_elev,
        "wk_arc_mean_scatter": wk_scat,
        "wk_arc_mean_shape": wk_shape,
        "wk_arc_mean_tucker": wk_match.mean_tucker,
        "wk_arc_min_tucker": float(np.nanmin(wk_match.tucker_congruences)),
        "wk_split_half_kappa": wk_rep.mean_kappa,
        "wk_split_half_ari": wk_rep.mean_ari,
        "wk_split_half_nmi": wk_rep.mean_nmi,
        # Diagonal LPA (K = 1 vs K = 3)
        "lpa1_bic": lpa1.bic,
        "lpa1_aic": lpa1.aic,
        "lpa1_ll": lpa1.log_likelihood,
        "lpa3_bic": lpa3.bic,
        "lpa3_aic": lpa3.aic,
        "lpa3_ll": lpa3.log_likelihood,
        "lpa3_converged": lpa3.converged,
        "lpa_delta_bic_3_vs_1": lpa3.bic - lpa1.bic,
        "lpa_prefers_3": bool(lpa3.bic < lpa1.bic),
        "lpa3_arc_mean_d2": lpa3_match.mean_d2,
        "lpa3_arc_mean_elevation": lpa3_elev,
        "lpa3_arc_mean_scatter": lpa3_scat,
        "lpa3_arc_mean_shape": lpa3_shape,
        "lpa3_arc_mean_tucker": lpa3_match.mean_tucker,
        # Full-covariance GMM (K = 1 vs K = 3)
        "gmm1_bic": gmm1.bic,
        "gmm1_aic": gmm1.aic,
        "gmm1_ll": gmm1.log_likelihood,
        "gmm3_bic": gmm3.bic,
        "gmm3_aic": gmm3.aic,
        "gmm3_ll": gmm3.log_likelihood,
        "gmm3_converged": gmm3.converged,
        "gmm_delta_bic_3_vs_1": gmm3.bic - gmm1.bic,
        "gmm_prefers_3": bool(gmm3.bic < gmm1.bic),
        "gmm3_arc_mean_d2": gmm3_match.mean_d2,
        "gmm3_arc_mean_elevation": gmm3_elev,
        "gmm3_arc_mean_scatter": gmm3_scat,
        "gmm3_arc_mean_shape": gmm3_shape,
        "gmm3_arc_mean_tucker": gmm3_match.mean_tucker,
        # Cross-method stability on common sample
        "ari_wk_vs_lpa3": stab_wk_lpa.ari,
        "nmi_wk_vs_lpa3": stab_wk_lpa.nmi,
        "ari_wk_vs_gmm3": stab_wk_gmm.ari,
        "nmi_wk_vs_gmm3": stab_wk_gmm.nmi,
        "ari_lpa3_vs_gmm3": stab_lpa_gmm.ari,
        "nmi_lpa3_vs_gmm3": stab_lpa_gmm.nmi,
        # Gerlach et al. (2018) KDE density enrichment
        "gerlach_n_sig_perm": ev_perm.n_significant_types,
        "gerlach_n_sig_null2": ev_n2.n_significant_types,
        "gerlach_n_sig_null3": ev_n3.n_significant_types,
        "gerlach_max_enrichment_perm": float(np.nanmax(ev_perm.enrichment)),
        "gerlach_max_enrichment_null2": float(np.nanmax(ev_n2.enrichment)),
        "gerlach_max_enrichment_null3": float(np.nanmax(ev_n3.enrichment)),
        "gerlach_min_pval_perm": float(np.nanmin(ev_perm.p_values)),
        "gerlach_min_pval_null2": float(np.nanmin(ev_n2.p_values)),
        "gerlach_min_pval_null3": float(np.nanmin(ev_n3.p_values)),
    }

    if not return_centroids:
        return metrics, None

    wk_props = np.bincount(wk.labels, minlength=3).astype(np.float64) / float(n)
    centroids_summary: dict[str, Any] = {
        "ward_kmeans_k3": _summarize_matched_centroids(wk_match, wk_props),
        "lpa_diag_k3": _summarize_matched_centroids(lpa3_match, lpa3.weights),
        "gmm_full_k3": _summarize_matched_centroids(gmm3_match, gmm3.weights),
        "gerlach_gmm_k4": {
            "centers_neoac": [
                {trait: float(val) for trait, val in zip(TRAIT_ORDER, center)}
                for center in gerlach.centers
            ],
            "weights": gerlach.gmm.weights,
            "empirical_densities": gerlach.empirical_densities,
            "enrichment_by_null": {
                name: ev.enrichment for name, ev in gerlach.evaluations.items()
            },
            "p_values_by_null": {
                name: ev.p_values for name, ev in gerlach.evaluations.items()
            },
            "is_significant_by_null": {
                name: ev.is_significant for name, ev in gerlach.evaluations.items()
            },
        },
    }
    return metrics, centroids_summary


def _run_null_replicate(
    null_model: NullModelName,
    dataset_name: str,
    x_emp: NDArray[np.float64],
    r_emp_pd: NDArray[np.float64],
    null3_latent_corr: NDArray[np.float64],
    rep_idx: int,
    rep_seed_seq: np.random.SeedSequence,
    *,
    gerlach_null_samples: int,
    split_half_splits: int,
    ward_subsample: int,
    max_kde_points: int,
) -> dict[str, Any]:
    """Generate one null replicate and evaluate the full clustering pipeline."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(rep_seed_seq)
    n = int(x_emp.shape[0])

    if null_model == "null1_permutation":
        sim_data = sample_null1_column_permutation(x_emp, rng)
    elif null_model == "null2_mvnormal":
        sim_data = sample_null2_correlated_normal(n=n, corr_matrix=r_emp_pd, rng=rng)
    elif null_model == "null3_gaussian_copula":
        emp_ppfs = empirical_marginal_ppfs(x_emp)
        z = rng.standard_normal((n, x_emp.shape[1]))
        sim_data = _apply_gaussian_copula(z, null3_latent_corr, emp_ppfs)
    else:
        raise ValueError(f"Unknown null_model: {null_model}")

    metrics, _ = _evaluate_sample_pipeline(
        sim_data,
        rng,
        gerlach_null_samples=gerlach_null_samples,
        split_half_splits=split_half_splits,
        ward_subsample=ward_subsample,
        max_kde_points=max_kde_points,
        return_centroids=False,
    )
    elapsed = time.perf_counter() - t0
    return {
        "dataset_name": dataset_name,
        "null_model": null_model,
        "n": n,
        "rep": int(rep_idx),
        **metrics,
        "rep_elapsed_sec": elapsed,
    }


def _summarize_null_comparisons(
    emp_metrics: dict[str, Any],
    df_reps: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute empirical tail probabilities and Gerlach significant-type distributions."""
    comparisons: dict[str, Any] = {}
    gerlach_dist: dict[str, Any] = {}

    for null_model in NULL_MODELS:
        sub = df_reps.loc[df_reps["null_model"] == null_model]
        model_comp: dict[str, Any] = {}

        for metric_name, direction in COMPARISON_METRICS:
            emp_val = float(emp_metrics[metric_name])
            sim_vals = sub[metric_name].to_numpy(dtype=np.float64)
            valid = sim_vals[np.isfinite(sim_vals)]
            n_valid = int(valid.size)

            if n_valid == 0 or not math.isfinite(emp_val):
                model_comp[metric_name] = {
                    "direction": direction,
                    "empirical": emp_val,
                    "n_valid": n_valid,
                    "empirical_prob": None,
                    "mc_p_value": None,
                }
                continue

            if direction == "le":
                hits = int(np.sum(valid <= emp_val))
                prob_label = "P(sim <= emp)"
            else:
                hits = int(np.sum(valid >= emp_val))
                prob_label = "P(sim >= emp)"

            model_comp[metric_name] = {
                "direction": direction,
                "probability_definition": prob_label,
                "empirical": emp_val,
                "null_mean": float(np.mean(valid)),
                "null_sd": float(np.std(valid, ddof=1)) if n_valid > 1 else 0.0,
                "null_median": float(np.median(valid)),
                "null_q05": float(np.quantile(valid, 0.05)),
                "null_q95": float(np.quantile(valid, 0.95)),
                "n_valid": n_valid,
                "n_extreme": hits,
                "empirical_prob": hits / float(n_valid),
                "mc_p_value": (1.0 + hits) / float(n_valid + 1),
            }

        comparisons[null_model] = model_comp

        # Gerlach significant types distribution under each internal null test
        model_gerlach: dict[str, Any] = {}
        for col in GERLACH_SIG_COLS:
            emp_sig = int(emp_metrics[col])
            vals = sub[col].to_numpy(dtype=np.int64)
            n_total = int(vals.size)
            counts = {str(k): int(np.sum(vals == k)) for k in range(5)}
            props = {
                str(k): (counts[str(k)] / float(n_total) if n_total > 0 else 0.0)
                for k in range(5)
            }
            ge_emp = int(np.sum(vals >= emp_sig))
            model_gerlach[col] = {
                "empirical": emp_sig,
                "n_reps": n_total,
                "mean": float(np.mean(vals)) if n_total > 0 else None,
                "median": float(np.median(vals)) if n_total > 0 else None,
                "prob_ge_1": float(np.mean(vals >= 1)) if n_total > 0 else None,
                "prob_ge_empirical": ge_emp / float(n_total) if n_total > 0 else None,
                "mc_p_value_ge_empirical": (
                    (1.0 + ge_emp) / float(n_total + 1) if n_total > 0 else None
                ),
                "counts": counts,
                "proportions": props,
            }
        gerlach_dist[null_model] = model_gerlach

    return comparisons, gerlach_dist


def _run_incremental_validity(
    df_clean: pd.DataFrame,
    x_std: NDArray[np.float64],
    criteria_cols: list[str],
    seed_seq: np.random.SeedSequence,
    *,
    cv_splits: int,
    ward_subsample: int,
) -> dict[str, Any]:
    """Run leak-free k-fold CV incremental predictive validity for each criterion column."""
    results: dict[str, Any] = {}
    col_seeds = seed_seq.spawn(len(criteria_cols))

    for col, col_ss in zip(criteria_cols, col_seeds):
        raw_series = df_clean[col]
        valid_mask = raw_series.notna().to_numpy()
        n_valid = int(np.sum(valid_mask))

        if n_valid < max(20, cv_splits * 2):
            results[col] = {
                "skipped": True,
                "reason": f"Too few non-NA observations (N={n_valid}).",
            }
            continue

        x_sub = x_std[valid_mask]
        y_series = raw_series.loc[valid_mask]

        if pd.api.types.is_bool_dtype(y_series):
            task: Literal["regression", "classification"] = "classification"
            y_arr = y_series.astype(int).to_numpy()
        elif pd.api.types.is_numeric_dtype(y_series):
            y_num = pd.to_numeric(y_series, errors="coerce")
            fin_mask = np.isfinite(y_num.to_numpy(dtype=np.float64))
            x_sub = x_sub[fin_mask]
            y_clean = y_num.to_numpy(dtype=np.float64)[fin_mask]
            unique_vals = np.unique(y_clean)
            if unique_vals.size < 2:
                results[col] = {
                    "skipped": True,
                    "reason": "Criterion has fewer than 2 unique values.",
                }
                continue
            if unique_vals.size == 2:
                task = "classification"
                _, y_arr = np.unique(y_clean, return_inverse=True)
            else:
                task = "regression"
                y_arr = y_clean
        else:
            task = "classification"
            codes, uniques = pd.factorize(y_series, sort=True)
            if len(uniques) < 2:
                results[col] = {
                    "skipped": True,
                    "reason": "Criterion has fewer than 2 unique classes.",
                }
                continue
            y_arr = codes.astype(np.int64)

        effective_splits = cv_splits
        if task == "classification":
            _, class_counts = np.unique(y_arr, return_counts=True)
            min_class = int(np.min(class_counts))
            if min_class < 2:
                results[col] = {
                    "skipped": True,
                    "reason": f"Smallest class has only {min_class} observation(s).",
                }
                continue
            effective_splits = min(cv_splits, min_class)

        n_eff = int(x_sub.shape[0])
        extractors = {
            "ward_kmeans_onehot": make_ward_kmeans_onehot_extractor(
                n_clusters=3,
                ward_subsample=ward_subsample if n_eff > ward_subsample else None,
            ),
            "lpa_diag_posteriors": make_gmm_posterior_extractor(
                n_clusters=3,
                covariance_type="diag",
                n_init=10,
                reg_covar=1e-6,
            ),
            "gmm_full_posteriors": make_gmm_posterior_extractor(
                n_clusters=3,
                covariance_type="full",
                n_init=10,
                reg_covar=1e-6,
            ),
        }

        ext_seeds = col_ss.spawn(len(extractors))
        col_res: dict[str, Any] = {
            "skipped": False,
            "n": n_eff,
            "task": task,
            "n_splits": effective_splits,
            "models": {},
        }
        for (ext_name, extractor), ext_ss in zip(extractors.items(), ext_seeds):
            cv_res = incremental_predictive_validity(
                traits=x_sub,
                target=y_arr,
                cluster_extractor=extractor,
                task=task,
                n_splits=effective_splits,
                alpha=1.0,
                random_state=np.random.default_rng(ext_ss),
            )
            col_res["models"][ext_name] = {
                "metric_name": cv_res.metric_name,
                "mean_baseline": cv_res.mean_baseline,
                "mean_augmented": cv_res.mean_augmented,
                "mean_delta": cv_res.mean_delta,
                "std_delta": cv_res.std_delta,
                "baseline_scores": cv_res.baseline_scores,
                "augmented_scores": cv_res.augmented_scores,
                "delta_scores": cv_res.delta_scores,
            }
        results[col] = col_res

    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare an empirical Big Five dataset against ARC Null 1, Null 2, and Null 3."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the empirical CSV dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Short identifier for output filenames ({dataset_name}_null_reps.parquet, _summary.json).",
    )
    parser.add_argument(
        "--cols-neoac",
        nargs=5,
        default=list(TRAIT_ORDER),
        metavar=("N_COL", "E_COL", "O_COL", "A_COL", "C_COL"),
        help="5 column names in strict (N, E, O, A, C) order with N unreversed (default: N E O A C).",
    )
    parser.add_argument(
        "--criteria-cols",
        nargs="*",
        default=None,
        help="Optional criterion column name(s) for out-of-sample incremental predictive validity CV.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1000,
        help="Number of null replicates per null model (default: 1000).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-2,
        help="Number of parallel worker processes for joblib.Parallel (default: -2).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/empirical"),
        help="Output directory for Parquet and JSON results (default: results/empirical).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help=f"Base SeedSequence entropy (default: {BASE_SEED}).",
    )
    parser.add_argument(
        "--gerlach-null-samples",
        type=int,
        default=200,
        help="Null replicates inside the Gerlach et al. (2018) KDE test (default: 200; must satisfy 1/(B+1) < 0.01).",
    )
    parser.add_argument(
        "--split-half-splits",
        type=int,
        default=5,
        help="Number of split-half splits for Asendorpf's kappa (default: 5).",
    )
    parser.add_argument(
        "--ward-subsample",
        type=int,
        default=2000,
        help="Maximum subsample size for Ward initialization before full k-means (default: 2000).",
    )
    parser.add_argument(
        "--max-kde-points",
        type=int,
        default=5000,
        help="Maximum points used for multivariate KDE evaluation in Gerlach's test (default: 5000).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of CV folds for incremental predictive validity when --criteria-cols is set (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the empirical vs. null-model comparison workflow."""
    args = _parse_args(argv)
    if args.reps < 1:
        raise ValueError(f"--reps must be >= 1, got {args.reps}.")
    if args.gerlach_null_samples < 1:
        raise ValueError(
            f"--gerlach-null-samples must be >= 1, got {args.gerlach_null_samples}."
        )
    if args.split_half_splits < 1:
        raise ValueError(
            f"--split-half-splits must be >= 1, got {args.split_half_splits}."
        )

    criteria_cols: list[str] | None = args.criteria_cols if args.criteria_cols else None
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    total_t0 = time.perf_counter()

    # 1. Load, clean, standardize, and compute empirical diagnostics
    df_clean, x_emp, diagnostics, r_emp_pd = _load_and_prepare_data(
        input_path=args.input,
        cols_neoac=list(args.cols_neoac),
        criteria_cols=criteria_cols,
    )
    n = int(x_emp.shape[0])
    print(
        f"=== Empirical Comparison | dataset={args.dataset_name} | N={n} | "
        f"lambda2/lambda1={diagnostics['lambda2_over_lambda1']:.5f} "
        f"(threshold={NONCOLLINEARITY_THRESHOLD:.5f}) | "
        f"det(R)={diagnostics['det_r_emp']:.5f} ==="
    )

    # Master SeedSequence rooted at --seed (default: 20261121)
    master_ss = np.random.SeedSequence(args.seed)
    emp_ss, cal_ss, n1_ss, n2_ss, n3_ss, cv_ss = master_ss.spawn(6)
    null_seed_map: dict[NullModelName, np.random.SeedSequence] = {
        "null1_permutation": n1_ss,
        "null2_mvnormal": n2_ss,
        "null3_gaussian_copula": n3_ss,
    }

    # 2. Run exact Monte Carlo pipeline on the empirical dataset
    emp_t0 = time.perf_counter()
    emp_metrics, emp_centroids = _evaluate_sample_pipeline(
        x_emp,
        np.random.default_rng(emp_ss),
        gerlach_null_samples=args.gerlach_null_samples,
        split_half_splits=args.split_half_splits,
        ward_subsample=args.ward_subsample,
        max_kde_points=args.max_kde_points,
        return_centroids=True,
    )
    print(
        f"[EMPIRICAL] Done in {_format_duration(time.perf_counter() - emp_t0)} | "
        f"WK mean D2={emp_metrics['wk_arc_mean_d2']:.4f} | "
        f"Tucker={emp_metrics['wk_arc_mean_tucker']:.4f} | "
        f"kappa={emp_metrics['wk_split_half_kappa']:.4f} | "
        f"Gerlach sig (perm/n2/n3)={emp_metrics['gerlach_n_sig_perm']}/"
        f"{emp_metrics['gerlach_n_sig_null2']}/{emp_metrics['gerlach_n_sig_null3']}"
    )

    # Pre-calibrate Null 3 Gaussian copula to R_emp with exact empirical marginals
    cal_t0 = time.perf_counter()
    cal_rng = np.random.default_rng(cal_ss)
    emp_ppfs = empirical_marginal_ppfs(x_emp)
    copula_cal = calibrate_gaussian_copula(
        target_corr=r_emp_pd,
        marginals=emp_ppfs,
        rng=cal_rng,
        n_cal=50_000,
        tol=1e-4,
    )
    n_diag = min(n, 5000)
    diag_sample = _apply_gaussian_copula(
        cal_rng.standard_normal((n_diag, x_emp.shape[1])),
        copula_cal.latent_corr,
        emp_ppfs,
    )
    copula_unimodality = check_joint_unimodality(diag_sample, rng=cal_rng)
    print(
        f"[NULL3 CAL] Done in {_format_duration(time.perf_counter() - cal_t0)} | "
        f"max_abs_err={copula_cal.max_abs_error:.2e} | converged={copula_cal.converged} | "
        f"unimodal={copula_unimodality.is_unimodal}"
    )

    # 3. Run --reps parallel replicates for each of the 3 null models
    all_rows: list[dict[str, Any]] = []
    for null_model in NULL_MODELS:
        block_t0 = time.perf_counter()
        rep_seeds = null_seed_map[null_model].spawn(args.reps)
        rows = Parallel(n_jobs=args.n_jobs)(
            delayed(_run_null_replicate)(
                null_model=null_model,
                dataset_name=args.dataset_name,
                x_emp=x_emp,
                r_emp_pd=r_emp_pd,
                null3_latent_corr=copula_cal.latent_corr,
                rep_idx=rep_idx,
                rep_seed_seq=rep_seeds[rep_idx],
                gerlach_null_samples=args.gerlach_null_samples,
                split_half_splits=args.split_half_splits,
                ward_subsample=args.ward_subsample,
                max_kde_points=args.max_kde_points,
            )
            for rep_idx in range(args.reps)
        )
        all_rows.extend(rows)
        print(
            f"[DONE] {null_model} | {args.reps} reps in "
            f"{_format_duration(time.perf_counter() - block_t0)}"
        )

    df_reps = pd.DataFrame(all_rows)

    # 4. Estimate empirical tail probabilities and Gerlach significant-type distributions
    null_comparisons, gerlach_distribution = _summarize_null_comparisons(
        emp_metrics=emp_metrics,
        df_reps=df_reps,
    )

    # 5. Optional out-of-sample incremental predictive validity
    incremental_validity_summary: dict[str, Any] | None = None
    if criteria_cols:
        cv_t0 = time.perf_counter()
        incremental_validity_summary = _run_incremental_validity(
            df_clean=df_clean,
            x_std=x_emp,
            criteria_cols=criteria_cols,
            seed_seq=cv_ss,
            cv_splits=args.cv_splits,
            ward_subsample=args.ward_subsample,
        )
        print(
            f"[CV] Incremental validity for {len(criteria_cols)} criterion column(s) "
            f"done in {_format_duration(time.perf_counter() - cv_t0)}"
        )

    # 6. Save Parquet of all null replicates and compact JSON summary
    parquet_path = outdir / f"{args.dataset_name}_null_reps.parquet"
    tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
    df_reps.to_parquet(tmp_parquet, index=False)
    tmp_parquet.replace(parquet_path)

    total_elapsed = time.perf_counter() - total_t0
    summary: dict[str, Any] = {
        "metadata": {
            "dataset_name": args.dataset_name,
            "input_path": str(args.input),
            "cols_neoac": list(args.cols_neoac),
            "criteria_cols": criteria_cols,
            "reps_per_null": int(args.reps),
            "n_jobs": int(args.n_jobs),
            "seed": int(args.seed),
            "gerlach_null_samples": int(args.gerlach_null_samples),
            "split_half_splits": int(args.split_half_splits),
            "ward_subsample": int(args.ward_subsample),
            "max_kde_points": int(args.max_kde_points),
            "cv_splits": int(args.cv_splits) if criteria_cols else None,
            "total_elapsed_sec": total_elapsed,
            "null_reps_parquet": str(parquet_path),
        },
        "empirical_diagnostics": diagnostics,
        "null3_copula_calibration": {
            "latent_corr": copula_cal.latent_corr,
            "achieved_corr": copula_cal.achieved_corr,
            "max_abs_error": copula_cal.max_abs_error,
            "iterations": copula_cal.iterations,
            "converged": copula_cal.converged,
            "unimodality": {
                "is_unimodal": copula_unimodality.is_unimodal,
                "marginal_modes": list(copula_unimodality.marginal_modes),
                "pc_modes": list(copula_unimodality.pc_modes),
                "pc12_joint_modes": copula_unimodality.pc12_joint_modes,
            },
        },
        "empirical_metrics": emp_metrics,
        "empirical_centroids": emp_centroids,
        "null_comparisons": null_comparisons,
        "gerlach_significant_types_distribution": gerlach_distribution,
        "incremental_validity": incremental_validity_summary,
    }

    json_path = outdir / f"{args.dataset_name}_summary.json"
    tmp_json = json_path.with_suffix(".json.tmp")
    with tmp_json.open("w", encoding="utf-8") as fh:
        json.dump(_to_json_safe(summary), fh, indent=2, allow_nan=False)
        fh.write("\n")
    tmp_json.replace(json_path)

    print(f"\n=== Saved Outputs ({_format_duration(total_elapsed)}) ===")
    print(f"Null replicates : {parquet_path}")
    print(f"Summary JSON    : {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
