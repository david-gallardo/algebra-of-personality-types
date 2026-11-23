#!/usr/bin/env python3
"""Monte Carlo factorial simulation runner for the ARC personality types study.

Runs a factorial design across sample sizes N, null DGPs (Null 1 isotropic,
Null 2 correlated normal for R_vdl / R_be with and without SPD heterogeneity,
and Null 3 calibrated Gaussian copula), and clustering/validation procedures
(Ward + k-means, diagonal LPA, full-covariance GMM, and Gerlach et al. 2018
density-enrichment test).

Results are saved incrementally as one Parquet file per (condition, N) block so
that ``--resume`` skips already completed blocks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from joblib import Parallel, delayed

# Allow running directly as `python scripts/run_montecarlo.py` without PYTHONPATH=src
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
    default_skewed_marginals,
    sample_null1_independent_normal,
    sample_null2_correlated_normal,
    sample_study_correlation_matrix,
)
from arc_study.matrices import (  # noqa: E402
    BASE_SEED,
    BUSSERI_ERB_2024,
    IDENTITY_5,
    VAN_DER_LINDEN_2010,
)
from arc_study.metrics import (  # noqa: E402
    ARC_PROTOTYPES,
    assignment_stability,
    match_centroids,
    split_half_replicability,
)

Mode = Literal["smoke", "full"]

SMOKE_N_GRID: tuple[int, ...] = (300, 1000)
FULL_N_GRID: tuple[int, ...] = (300, 1000, 5000, 50000)


@dataclass(frozen=True)
class ConditionSpec:
    """Specification for a single null-model condition in the factorial design."""

    name: str
    null_family: Literal["null1", "null2", "null3"]
    matrix_name: Literal["identity", "vdl", "be"]
    heterogeneous: bool = False
    hetero_df: float = 50.0


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec(name="null1_isotropic", null_family="null1", matrix_name="identity"),
    ConditionSpec(name="null2_vdl", null_family="null2", matrix_name="vdl"),
    ConditionSpec(name="null2_be", null_family="null2", matrix_name="be"),
    ConditionSpec(
        name="null2_vdl_hetero",
        null_family="null2",
        matrix_name="vdl",
        heterogeneous=True,
        hetero_df=50.0,
    ),
    ConditionSpec(
        name="null2_be_hetero",
        null_family="null2",
        matrix_name="be",
        heterogeneous=True,
        hetero_df=50.0,
    ),
    ConditionSpec(name="null3_vdl_copula", null_family="null3", matrix_name="vdl"),
    ConditionSpec(name="null3_be_copula", null_family="null3", matrix_name="be"),
)


def _get_target_matrix(matrix_name: Literal["identity", "vdl", "be"]) -> NDArray[np.float64]:
    """Return the validated correlation matrix for a condition."""
    if matrix_name == "identity":
        return IDENTITY_5
    if matrix_name == "vdl":
        return VAN_DER_LINDEN_2010
    if matrix_name == "be":
        return BUSSERI_ERB_2024
    raise ValueError(f"Unknown matrix_name: {matrix_name}")


def _precalibrate_null3_matrices(base_seed: int) -> dict[str, NDArray[np.float64]]:
    """Calibrate Null 3 Gaussian copula latent correlation matrices once per target R."""
    marginals = default_skewed_marginals()
    calibrated: dict[str, NDArray[np.float64]] = {}
    for offset, matrix_name in enumerate(("vdl", "be")):
        r_target = _get_target_matrix(matrix_name)
        cal_rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, 999, offset])
        )
        cal = calibrate_gaussian_copula(
            target_corr=r_target,
            marginals=marginals,
            rng=cal_rng,
            n_cal=50_000,
            tol=1e-4,
        )
        calibrated[matrix_name] = cal.latent_corr
    return calibrated


def _generate_sample(
    cond: ConditionSpec,
    n: int,
    rng: np.random.Generator,
    null3_latent_corrs: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Generate one synthetic dataset of shape (n, 5) for ``cond``."""
    if cond.null_family == "null1":
        return sample_null1_independent_normal(n=n, rng=rng, dim=5)

    r_base = _get_target_matrix(cond.matrix_name)
    if cond.null_family == "null2":
        if cond.heterogeneous:
            r_study = sample_study_correlation_matrix(
                r_meta=r_base,
                df=cond.hetero_df,
                rng=rng,
                method="invwishart",
            )
        else:
            r_study = r_base
        return sample_null2_correlated_normal(n=n, corr_matrix=r_study, rng=rng)

    if cond.null_family == "null3":
        latent_corr = null3_latent_corrs[cond.matrix_name]
        marginals = default_skewed_marginals()
        z = rng.standard_normal((n, 5))
        return _apply_gaussian_copula(z, latent_corr, marginals)

    raise ValueError(f"Unknown null_family: {cond.null_family}")


def _run_single_replicate(
    cond: ConditionSpec,
    n: int,
    rep_idx: int,
    rep_seed_seq: np.random.SeedSequence,
    null3_latent_corrs: dict[str, NDArray[np.float64]],
    *,
    gerlach_null_samples: int,
    split_half_splits: int,
    ward_subsample: int,
    max_kde_points: int,
) -> dict[str, Any]:
    """Execute one Monte Carlo replicate and return a flat metrics dictionary."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(rep_seed_seq)

    data = _generate_sample(cond, n, rng, null3_latent_corrs)

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

    # 3. Full-covariance GMM (K = 1 vs K = 3)
    gmm1 = fit_gmm_full(data, n_clusters=1, n_init=10, reg_covar=1e-6, random_state=rng)
    gmm3 = fit_gmm_full(data, n_clusters=3, n_init=10, reg_covar=1e-6, random_state=rng)
    gmm3_match = match_centroids(gmm3.means, ARC_PROTOTYPES, metric="d2")

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

    elapsed = time.perf_counter() - t0
    return {
        "condition": cond.name,
        "null_family": cond.null_family,
        "matrix_name": cond.matrix_name,
        "heterogeneous": cond.heterogeneous,
        "n": int(n),
        "rep": int(rep_idx),
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
        # Runtime
        "rep_elapsed_sec": elapsed,
    }


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds into a compact human-readable string."""
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, rem_min = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(rem_min)}m {rem:.0f}s"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the ARC factorial Monte Carlo simulation study."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="Execution mode: 'smoke' uses N=[300, 1000] & reps=10; 'full' uses N=[300, 1000, 5000, 50000].",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Number of replicates per (condition, N) block (default: 10 in smoke, 100 in full).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel worker processes for joblib.Parallel (default: -1).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/simulations"),
        help="Output directory for incremental block Parquet files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (condition, N) blocks whose .parquet file already exists.",
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
        default=None,
        help="Null replicates inside Gerlach et al. (2018) KDE test (default: 100 in smoke, 200 in full; must satisfy 1/(B+1) < 0.01).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run all factorial blocks, saving one Parquet file per (condition, N) block."""
    args = _parse_args(argv)
    mode: Mode = args.mode
    n_grid = SMOKE_N_GRID if mode == "smoke" else FULL_N_GRID
    reps = args.reps if args.reps is not None else (10 if mode == "smoke" else 100)
    if reps < 1:
        raise ValueError(f"--reps must be >= 1, got {reps}.")

    gerlach_null_samples = (
        args.gerlach_null_samples
        if args.gerlach_null_samples is not None
        else (100 if mode == "smoke" else 200)
    )
    split_half_splits = 3 if mode == "smoke" else 5
    ward_subsample = 2000
    max_kde_points = 5000

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== ARC Monte Carlo Simulation | mode={mode} | reps={reps} | "
        f"N={list(n_grid)} | n_jobs={args.n_jobs} | seed={args.seed} ==="
    )

    total_start = time.perf_counter()

    # Pre-calibrate Null 3 Gaussian copula latent correlation matrices once
    null3_latent_corrs = _precalibrate_null3_matrices(args.seed)

    # Master SeedSequence rooted at BASE_SEED (20261121)
    master_ss = np.random.SeedSequence(args.seed)
    all_n_canonical = FULL_N_GRID
    total_canonical_blocks = len(CONDITIONS) * len(all_n_canonical)
    canonical_block_seeds = master_ss.spawn(total_canonical_blocks)

    block_seed_map: dict[tuple[str, int], np.random.SeedSequence] = {}
    idx = 0
    for cond in CONDITIONS:
        for n_val in all_n_canonical:
            block_seed_map[(cond.name, n_val)] = canonical_block_seeds[idx]
            idx += 1

    blocks_run = 0
    blocks_skipped = 0
    executed_elapsed = 0.0

    for cond in CONDITIONS:
        for n in n_grid:
            block_path = outdir / f"block_{cond.name}_N{n}.parquet"
            if args.resume and block_path.exists() and block_path.stat().st_size > 0:
                print(f"[SKIP] {block_path.name} (already exists)")
                blocks_skipped += 1
                continue

            block_t0 = time.perf_counter()
            block_ss = block_seed_map[(cond.name, n)]
            rep_seeds = block_ss.spawn(reps)

            rows = Parallel(n_jobs=args.n_jobs)(
                delayed(_run_single_replicate)(
                    cond=cond,
                    n=n,
                    rep_idx=rep_idx,
                    rep_seed_seq=rep_seeds[rep_idx],
                    null3_latent_corrs=null3_latent_corrs,
                    gerlach_null_samples=gerlach_null_samples,
                    split_half_splits=split_half_splits,
                    ward_subsample=ward_subsample,
                    max_kde_points=max_kde_points,
                )
                for rep_idx in range(reps)
            )

            df_block = pd.DataFrame(rows)
            tmp_path = block_path.with_suffix(".parquet.tmp")
            df_block.to_parquet(tmp_path, index=False)
            tmp_path.replace(block_path)

            block_elapsed = time.perf_counter() - block_t0
            executed_elapsed += block_elapsed
            blocks_run += 1
            print(
                f"[DONE] {block_path.name} | {reps} reps in {_format_duration(block_elapsed)} "
                f"(est. 1000 reps: {_format_duration(block_elapsed * (1000.0 / reps))})"
            )

    total_elapsed = time.perf_counter() - total_start
    print("\n=== Simulation Summary ===")
    print(f"Output directory : {outdir}")
    print(f"Blocks completed : {blocks_run} (skipped: {blocks_skipped})")
    print(f"Total elapsed    : {_format_duration(total_elapsed)}")
    if blocks_run > 0:
        est_1000 = executed_elapsed * (1000.0 / float(reps))
        print(
            f"Estimated 1000 reps (for {blocks_run} executed block(s)): "
            f"{_format_duration(est_1000)} ({est_1000:.1f}s)"
        )
    else:
        print("Estimated 1000 reps: N/A (all blocks were skipped via --resume).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
