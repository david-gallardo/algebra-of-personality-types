#!/usr/bin/env python3
"""Targeted Gerlach et al. (2018) add-on runner (Monte Carlo + Spanish BFI-2).

Fixes the n_null_samples / alpha bug (using n_null_samples >= 200 so p_min = 1/201 < 0.01)
and evaluates both fixed-bandwidth isotropic Gaussian KDE (h=0.50 and h=0.35) and
Scott covariance-scaled KDE across enrichment thresholds > 1.00 and > 1.25, without
touching the existing 28 Monte Carlo blocks in results/simulations/.
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
from scipy.stats import gaussian_kde

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
for _p in (_SRC_DIR, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from arc_study.clustering import (  # noqa: E402
    fit_gmm_full,
    run_gerlach_procedure,
)
from arc_study.generators import (  # noqa: E402
    _apply_gaussian_copula,
    calibrate_gaussian_copula,
    empirical_marginal_ppfs,
    nearest_pd_correlation,
    sample_null1_column_permutation,
    sample_null2_correlated_normal,
)
from arc_study.matrices import BASE_SEED, TRAIT_ORDER  # noqa: E402
from run_montecarlo import (  # noqa: E402
    CONDITIONS,
    ConditionSpec,
    _format_duration,
    _generate_sample,
    _precalibrate_null3_matrices,
)

Mode = Literal["smoke", "full"]
EmpNullName = Literal["null1_permutation", "null2_mvnormal", "null3_gaussian_copula"]
EMP_NULL_MODELS: tuple[EmpNullName, ...] = (
    "null1_permutation",
    "null2_mvnormal",
    "null3_gaussian_copula",
)
ADDON_N_GRID: tuple[int, ...] = (1000, 5000)


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


def _subsample(
    sample: NDArray[np.float64],
    max_points: int | None,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if max_points is not None and sample.shape[0] > max_points:
        idx = rng.choice(sample.shape[0], size=max_points, replace=False)
        return sample[idx]
    return sample


def _eval_isotropic_kde_exact(
    sub: NDArray[np.float64],
    centers: NDArray[np.float64],
    bandwidth: float,
) -> NDArray[np.float64]:
    """Exact vectorized isotropic Gaussian KDE (identical to sklearn KernelDensity)."""
    n, p = sub.shape
    inv_2h2 = 0.5 / (bandwidth * bandwidth)
    norm_const = (2.0 * math.pi * bandwidth * bandwidth) ** (-0.5 * p) / float(n)
    # ||x_i - c_k||^2 = ||x_i||^2 + ||c_k||^2 - 2 x_i . c_k
    x_sq = np.sum(sub * sub, axis=1, keepdims=True)
    c_sq = np.sum(centers * centers, axis=1, keepdims=True).T
    dists_sq = np.maximum(0.0, x_sq + c_sq - 2.0 * (sub @ centers.T))
    return np.asarray(norm_const * np.sum(np.exp(-dists_sq * inv_2h2), axis=0), dtype=np.float64)


def _eval_scott_kde(
    sub: NDArray[np.float64],
    centers: NDArray[np.float64],
) -> NDArray[np.float64]:
    kde = gaussian_kde(sub.T, bw_method="scott")
    return np.asarray(kde(centers.T), dtype=np.float64)


def _evaluate_gerlach_multi_kde(
    data: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    n_null_samples: int,
    alpha: float,
    max_kde_points: int,
    include_h035: bool,
) -> dict[str, Any]:
    """Run Gerlach GMM (K=4) once and evaluate isotropic (h=0.50, optional h=0.35)
    and Scott KDEs against permutation, Null 2, and Null 3 using shared null draws.
    """
    p_min = 1.0 / float(n_null_samples + 1)
    if p_min >= alpha:
        raise ValueError(
            f"Impossible significance test: 1/({n_null_samples}+1) = {p_min:.5f} >= alpha={alpha}."
        )

    n, p = data.shape
    gmm = fit_gmm_full(data, n_clusters=4, n_init=10, reg_covar=1e-6, random_state=rng)
    centers = gmm.means
    k = centers.shape[0]

    modes: list[str] = ["iso_h050", "scott"]
    if include_h035:
        modes.insert(1, "iso_h035")

    emp_sub = _subsample(data, max_kde_points, rng)
    emp_dens: dict[str, NDArray[np.float64]] = {
        "iso_h050": _eval_isotropic_kde_exact(emp_sub, centers, 0.50),
        "scott": _eval_scott_kde(emp_sub, centers),
    }
    if include_h035:
        emp_dens["iso_h035"] = _eval_isotropic_kde_exact(emp_sub, centers, 0.35)

    # Precompute sample statistics for Null 2 and Null 3 once per replicate
    emp_mean = data.mean(axis=0)
    emp_std = data.std(axis=0, ddof=1)
    emp_std = np.where(emp_std > 0.0, emp_std, 1.0)
    r_emp = nearest_pd_correlation(np.corrcoef(data, rowvar=False))
    ppfs = empirical_marginal_ppfs(data)
    cal = calibrate_gaussian_copula(
        target_corr=r_emp,
        marginals=ppfs,
        rng=rng,
        n_cal=10_000,
        tol=1e-3,
    )

    null_dens: dict[str, dict[str, NDArray[np.float64]]] = {
        nl: {m: np.empty((n_null_samples, k), dtype=np.float64) for m in modes}
        for nl in ("perm", "null2", "null3")
    }

    for b in range(n_null_samples):
        # Null 1: column permutation
        s_perm = _subsample(sample_null1_column_permutation(data, rng), max_kde_points, rng)
        null_dens["perm"]["iso_h050"][b] = _eval_isotropic_kde_exact(s_perm, centers, 0.50)
        if include_h035:
            null_dens["perm"]["iso_h035"][b] = _eval_isotropic_kde_exact(s_perm, centers, 0.35)
        null_dens["perm"]["scott"][b] = _eval_scott_kde(s_perm, centers)

        # Null 2: correlated MVN
        z_n2 = sample_null2_correlated_normal(n, r_emp, rng)
        s_n2 = _subsample(emp_mean + z_n2 * emp_std, max_kde_points, rng)
        null_dens["null2"]["iso_h050"][b] = _eval_isotropic_kde_exact(s_n2, centers, 0.50)
        if include_h035:
            null_dens["null2"]["iso_h035"][b] = _eval_isotropic_kde_exact(s_n2, centers, 0.35)
        null_dens["null2"]["scott"][b] = _eval_scott_kde(s_n2, centers)

        # Null 3: calibrated Gaussian copula
        z_n3 = rng.standard_normal((n, p))
        s_n3 = _subsample(_apply_gaussian_copula(z_n3, cal.latent_corr, ppfs), max_kde_points, rng)
        null_dens["null3"]["iso_h050"][b] = _eval_isotropic_kde_exact(s_n3, centers, 0.50)
        if include_h035:
            null_dens["null3"]["iso_h035"][b] = _eval_isotropic_kde_exact(s_n3, centers, 0.35)
        null_dens["null3"]["scott"][b] = _eval_scott_kde(s_n3, centers)

    out: dict[str, Any] = {}
    for m in modes:
        ed = emp_dens[m]
        for nl in ("perm", "null2", "null3"):
            nd = null_dens[nl][m]
            mean_nd = nd.mean(axis=0)
            enrichment = np.divide(
                ed,
                mean_nd,
                out=np.full_like(ed, np.nan),
                where=mean_nd > 0.0,
            )
            exceed = np.sum(nd >= ed[np.newaxis, :], axis=0)
            pvals = (1.0 + exceed) / float(n_null_samples + 1)

            n_sig_e100 = int(np.sum((enrichment > 1.00) & (pvals < alpha)))
            n_sig_e125 = int(np.sum((enrichment > 1.25) & (pvals < alpha)))
            out[f"{m}_n_sig_e100_{nl}"] = n_sig_e100
            out[f"{m}_n_sig_e125_{nl}"] = n_sig_e125
            out[f"{m}_max_enrichment_{nl}"] = float(np.nanmax(enrichment))
            out[f"{m}_min_pval_{nl}"] = float(np.nanmin(pvals))
    return out


def _run_mc_replicate(
    cond: ConditionSpec,
    n: int,
    rep_idx: int,
    rep_seed_seq: np.random.SeedSequence,
    null3_latent_corrs: dict[str, NDArray[np.float64]],
    *,
    n_null_samples: int,
    alpha: float,
    max_kde_points: int,
    include_h035: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    rng = np.random.default_rng(rep_seed_seq)
    data = _generate_sample(cond, n, rng, null3_latent_corrs)
    metrics = _evaluate_gerlach_multi_kde(
        data,
        rng,
        n_null_samples=n_null_samples,
        alpha=alpha,
        max_kde_points=max_kde_points,
        include_h035=include_h035,
    )
    return {
        "condition": cond.name,
        "null_family": cond.null_family,
        "matrix_name": cond.matrix_name,
        "heterogeneous": cond.heterogeneous,
        "n": int(n),
        "rep": int(rep_idx),
        **metrics,
        "rep_elapsed_sec": time.perf_counter() - t0,
    }


def _run_emp_null_replicate(
    null_model: EmpNullName,
    x_emp: NDArray[np.float64],
    r_emp_pd: NDArray[np.float64],
    null3_latent_corr: NDArray[np.float64],
    rep_idx: int,
    rep_seed_seq: np.random.SeedSequence,
    *,
    n_null_samples: int,
    alpha: float,
    max_kde_points: int,
    include_h035: bool,
) -> dict[str, Any]:
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

    metrics = _evaluate_gerlach_multi_kde(
        sim_data,
        rng,
        n_null_samples=n_null_samples,
        alpha=alpha,
        max_kde_points=max_kde_points,
        include_h035=include_h035,
    )
    return {
        "dataset_name": "bfi2_es_study1",
        "null_model": null_model,
        "n": n,
        "rep": int(rep_idx),
        **metrics,
        "rep_elapsed_sec": time.perf_counter() - t0,
    }


def _load_bfi2_standardized(csv_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load Spanish BFI-2 domains in strict (N, E, O, A, C) order and standardize."""
    df = pd.read_csv(csv_path)
    cols = list(TRAIT_ORDER)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols).reset_index(drop=True)
    x_raw = df[cols].to_numpy(dtype=np.float64)
    x_std = (x_raw - x_raw.mean(axis=0)) / x_raw.std(axis=0, ddof=1)
    r_emp = np.corrcoef(x_std, rowvar=False)
    r_emp = 0.5 * (r_emp + r_emp.T)
    np.fill_diagonal(r_emp, 1.0)
    r_emp_pd = nearest_pd_correlation(r_emp)
    return x_std, r_emp_pd


def _summarize_empirical_gerlach(
    x_emp: NDArray[np.float64],
    emp_ss: np.random.SeedSequence,
    *,
    n_null_samples_emp: int,
    alpha: float,
    max_kde_points: int,
) -> dict[str, Any]:
    """Run run_gerlach_procedure directly on empirical BFI-2 for h=0.50, h=0.35, and Scott."""
    configs: tuple[tuple[str, Literal["isotropic", "scott"], float], ...] = (
        ("iso_h050", "isotropic", 0.50),
        ("iso_h035", "isotropic", 0.35),
        ("scott", "scott", 0.50),
    )
    seeds = emp_ss.spawn(len(configs))
    out: dict[str, Any] = {}
    for (name, kde_mode, bw), ss in zip(configs, seeds):
        res = run_gerlach_procedure(
            x_emp,
            n_clusters=4,
            covariance_type="full",
            n_init=10,
            reg_covar=1e-6,
            n_null_samples=n_null_samples_emp,
            null_models=("permutation", "null2", "null3"),
            alpha=alpha,
            min_enrichment=1.25,
            kde_mode=kde_mode,
            isotropic_bandwidth=bw,
            max_kde_points=max_kde_points if x_emp.shape[0] > max_kde_points else None,
            copula_calibration_n=25_000,
            copula_calibration_tol=1e-3,
            random_state=np.random.default_rng(ss),
        )
        evals_dict: dict[str, Any] = {}
        for nl_name, ev in res.evaluations.items():
            short = "perm" if nl_name == "permutation" else nl_name
            evals_dict[short] = {
                "enrichment": ev.enrichment,
                "p_values": ev.p_values,
                "mean_null_density": ev.mean_null_density,
                "n_sig_e100": ev.count_significant(min_enrichment=1.00, alpha=alpha),
                "n_sig_e125": ev.count_significant(min_enrichment=1.25, alpha=alpha),
                "max_enrichment": float(np.nanmax(ev.enrichment)),
                "min_pval": float(np.nanmin(ev.p_values)),
            }
        out[name] = {
            "kde_mode": kde_mode,
            "isotropic_bandwidth": bw if kde_mode == "isotropic" else None,
            "n_null_samples": n_null_samples_emp,
            "alpha": alpha,
            "centers_neoac": [
                {trait: float(val) for trait, val in zip(TRAIT_ORDER, center)}
                for center in res.centers
            ],
            "weights": res.gmm.weights,
            "empirical_densities": res.empirical_densities,
            "evaluations": evals_dict,
        }
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the targeted Gerlach et al. (2018) add-on (Monte Carlo + BFI-2)."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="Execution mode: 'smoke' (reps=5, emp_reps=5) or 'full' (reps=200, emp_reps=100).",
    )
    parser.add_argument("--reps", type=int, default=None, help="MC reps per block.")
    parser.add_argument("--emp-reps", type=int, default=None, help="BFI-2 null reps per null model.")
    parser.add_argument("--n-null-samples", type=int, default=200, help="Gerlach internal nulls (default: 200).")
    parser.add_argument("--n-null-samples-emp", type=int, default=500, help="Gerlach nulls for empirical BFI-2 (default: 500).")
    parser.add_argument("--alpha", type=float, default=0.01, help="Significance level (default: 0.01).")
    parser.add_argument("--n-jobs", type=int, default=-2, help="Parallel workers (default: -2).")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help=f"Base seed (default: {BASE_SEED}).")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/gerlach_addon"),
        help="Output directory (default: results/gerlach_addon).",
    )
    parser.add_argument(
        "--bfi2-csv",
        type=Path,
        default=Path("data/processed/bfi2_study1_domains.csv"),
        help="Path to Spanish BFI-2 domains CSV.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip already completed output files.")
    parser.add_argument("--no-h035", action="store_true", help="Skip isotropic h=0.35 evaluation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mode: Mode = args.mode
    reps = args.reps if args.reps is not None else (5 if mode == "smoke" else 200)
    emp_reps = args.emp_reps if args.emp_reps is not None else (5 if mode == "smoke" else 100)
    include_h035 = not args.no_h035
    max_kde_points = 5000

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== Gerlach Add-on | mode={mode} | mc_reps={reps} | emp_reps={emp_reps} | "
        f"B={args.n_null_samples} (emp B={args.n_null_samples_emp}) | "
        f"alpha={args.alpha} | h035={include_h035} | n_jobs={args.n_jobs} | seed={args.seed} ==="
    )

    total_t0 = time.perf_counter()
    null3_latent_corrs = _precalibrate_null3_matrices(args.seed)

    # Dedicated master SeedSequence branched cleanly from BASE_SEED
    master_ss = np.random.SeedSequence([args.seed, 2018])
    mc_block_seeds = master_ss.spawn(len(CONDITIONS) * len(ADDON_N_GRID) + 3)
    emp_direct_ss = mc_block_seeds[-3]
    emp_cal_ss = mc_block_seeds[-2]
    emp_null_master_ss = mc_block_seeds[-1]

    # ------------------------------------------------------------------
    # Part 1: Monte Carlo (7 conditions x N in [1000, 5000])
    # ------------------------------------------------------------------
    mc_t0 = time.perf_counter()
    block_idx = 0
    mc_blocks_run = 0
    for cond in CONDITIONS:
        for n_val in ADDON_N_GRID:
            block_ss = mc_block_seeds[block_idx]
            block_idx += 1
            block_path = outdir / f"gerlach_{cond.name}_N{n_val}.parquet"
            if args.resume and block_path.exists() and block_path.stat().st_size > 0:
                print(f"[SKIP] {block_path.name}")
                continue

            b_t0 = time.perf_counter()
            rep_seeds = block_ss.spawn(reps)
            rows = Parallel(n_jobs=args.n_jobs)(
                delayed(_run_mc_replicate)(
                    cond=cond,
                    n=n_val,
                    rep_idx=r,
                    rep_seed_seq=rep_seeds[r],
                    null3_latent_corrs=null3_latent_corrs,
                    n_null_samples=args.n_null_samples,
                    alpha=args.alpha,
                    max_kde_points=max_kde_points,
                    include_h035=include_h035,
                )
                for r in range(reps)
            )
            df_b = pd.DataFrame(rows)
            tmp_p = block_path.with_suffix(".parquet.tmp")
            df_b.to_parquet(tmp_p, index=False)
            tmp_p.replace(block_path)
            mc_blocks_run += 1
            b_elapsed = time.perf_counter() - b_t0
            print(
                f"[DONE] {block_path.name} | {reps} reps in {_format_duration(b_elapsed)} | "
                f"iso_h050 e125 perm={df_b['iso_h050_n_sig_e125_perm'].mean():.2f} "
                f"n2={df_b['iso_h050_n_sig_e125_null2'].mean():.2f} "
                f"n3={df_b['iso_h050_n_sig_e125_null3'].mean():.2f}"
            )

    mc_elapsed = time.perf_counter() - mc_t0
    if mode == "smoke" and mc_blocks_run > 0:
        est_full = mc_elapsed * (200.0 / float(reps))
        print(f"[ESTIMATE] Full MC (200 reps x 14 blocks): ~{_format_duration(est_full)}")

    # ------------------------------------------------------------------
    # Part 2: Empirical Spanish BFI-2 (direct B=500 + 100 null reps/null)
    # ------------------------------------------------------------------
    emp_parquet = outdir / "bfi2_es_study1_gerlach.parquet"
    emp_json = outdir / "bfi2_es_study1_gerlach.json"

    if args.resume and emp_parquet.exists() and emp_json.exists():
        print("[SKIP] Empirical BFI-2 Gerlach files already exist.")
    else:
        emp_t0 = time.perf_counter()
        x_emp, r_emp_pd = _load_bfi2_standardized(args.bfi2_csv)
        emp_direct = _summarize_empirical_gerlach(
            x_emp,
            emp_direct_ss,
            n_null_samples_emp=args.n_null_samples_emp,
            alpha=args.alpha,
            max_kde_points=max_kde_points,
        )
        print(
            f"[EMPIRICAL BFI-2] Direct B={args.n_null_samples_emp} done in "
            f"{_format_duration(time.perf_counter() - emp_t0)}"
        )

        # Calibrate Null 3 copula for BFI-2 once
        cal_rng = np.random.default_rng(emp_cal_ss)
        emp_ppfs = empirical_marginal_ppfs(x_emp)
        emp_cal = calibrate_gaussian_copula(
            target_corr=r_emp_pd,
            marginals=emp_ppfs,
            rng=cal_rng,
            n_cal=50_000,
            tol=1e-4,
        )

        null_seeds = emp_null_master_ss.spawn(len(EMP_NULL_MODELS))
        emp_rows: list[dict[str, Any]] = []
        for nm, nm_ss in zip(EMP_NULL_MODELS, null_seeds):
            nm_t0 = time.perf_counter()
            r_seeds = nm_ss.spawn(emp_reps)
            rows = Parallel(n_jobs=args.n_jobs)(
                delayed(_run_emp_null_replicate)(
                    null_model=nm,
                    x_emp=x_emp,
                    r_emp_pd=r_emp_pd,
                    null3_latent_corr=emp_cal.latent_corr,
                    rep_idx=r,
                    rep_seed_seq=r_seeds[r],
                    n_null_samples=args.n_null_samples,
                    alpha=args.alpha,
                    max_kde_points=max_kde_points,
                    include_h035=include_h035,
                )
                for r in range(emp_reps)
            )
            emp_rows.extend(rows)
            print(
                f"[DONE] BFI-2 {nm} | {emp_reps} reps in "
                f"{_format_duration(time.perf_counter() - nm_t0)}"
            )

        df_emp_reps = pd.DataFrame(emp_rows)
        tmp_p = emp_parquet.with_suffix(".parquet.tmp")
        df_emp_reps.to_parquet(tmp_p, index=False)
        tmp_p.replace(emp_parquet)

        # Summarize null distributions for JSON
        modes_present = ["iso_h050", "iso_h035", "scott"] if include_h035 else ["iso_h050", "scott"]
        null_summary: dict[str, Any] = {}
        for nm in EMP_NULL_MODELS:
            sub = df_emp_reps.loc[df_emp_reps["null_model"] == nm]
            nm_dict: dict[str, Any] = {}
            for m in modes_present:
                m_dict: dict[str, Any] = {}
                for nl in ("perm", "null2", "null3"):
                    c100 = sub[f"{m}_n_sig_e100_{nl}"].to_numpy(dtype=np.float64)
                    c125 = sub[f"{m}_n_sig_e125_{nl}"].to_numpy(dtype=np.float64)
                    emp_e100 = int(emp_direct[m]["evaluations"][nl]["n_sig_e100"])
                    emp_e125 = int(emp_direct[m]["evaluations"][nl]["n_sig_e125"])
                    m_dict[nl] = {
                        "mean_e100": float(np.mean(c100)),
                        "rate_ge1_e100": float(np.mean(c100 >= 1)),
                        "p_emp_ge_e100": float(np.mean(c100 >= emp_e100)),
                        "mean_e125": float(np.mean(c125)),
                        "rate_ge1_e125": float(np.mean(c125 >= 1)),
                        "p_emp_ge_e125": float(np.mean(c125 >= emp_e125)),
                    }
                nm_dict[m] = m_dict
            null_summary[nm] = nm_dict

        payload = {
            "metadata": {
                "dataset_name": "bfi2_es_study1",
                "n": int(x_emp.shape[0]),
                "mode": mode,
                "mc_reps_per_block": reps,
                "emp_reps_per_null": emp_reps,
                "n_null_samples_mc": args.n_null_samples,
                "n_null_samples_emp": args.n_null_samples_emp,
                "alpha": args.alpha,
                "seed": args.seed,
                "include_h035": include_h035,
            },
            "empirical_gerlach": emp_direct,
            "null_replicates_summary": null_summary,
        }
        tmp_j = emp_json.with_suffix(".json.tmp")
        with tmp_j.open("w", encoding="utf-8") as fh:
            json.dump(_to_json_safe(payload), fh, indent=2, allow_nan=False)
            fh.write("\n")
        tmp_j.replace(emp_json)

    print(f"=== Gerlach Add-on Complete in {_format_duration(time.perf_counter() - total_t0)} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
