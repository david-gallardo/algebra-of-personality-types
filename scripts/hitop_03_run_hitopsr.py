#!/usr/bin/env python3
"""Focused empirical HiTOP-SR validation on KU (N=411) and Czech (N=439).

Tests directly whether the empirical correlation matrix R alone predicts:
  1) Cross-cohort profile congruence (K=3 primary, K=4 sensitivity) vs I0, I1, C0 (MVN R), C1 (copula)
  2) Agreement between empirical centroids and a priori N_5(0, R) centroids
  3) Apparent number of types K selected by diagonal LPA vs full-covariance GMM (K=1..6)
  4) 5D KDE mean-shift topography (mode count and primary basin mass)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hitop_empirical import BASE_SEED  # noqa: E402
from hitop_empirical.clustering import fit_matched_kmeans  # noqa: E402
from hitop_empirical.generators import (  # noqa: E402
    CalibratedCopulaSpec,
    NullCode,
    calibrate_empirical_copula,
    generate_null_sample,
)
from hitop_empirical.geometry import correlation_geometry  # noqa: E402
from hitop_empirical.harmonization import compute_spectrum_composites  # noqa: E402
from hitop_empirical.inference import phipson_smyth_p, wilson_ci  # noqa: E402
from hitop_empirical.matching import match_centroids_hungarian, split_half_stability  # noqa: E402
from hitop_empirical.sources import assert_no_row_level_leakage  # noqa: E402
from hitop_empirical.topography import compute_h_nn, run_mean_shift  # noqa: E402


def _bic_curve(x: np.ndarray, cov_type: str, seed: int, k_max: int = 6) -> dict[str, Any]:
    bics: dict[int, float] = {}
    for k in range(1, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            n_init=10,
            reg_covar=1e-6,
            random_state=seed + k,
        ).fit(x)
        bics[k] = float(gmm.bic(x))
    best_k = min(bics, key=bics.get)
    return {"best_k": int(best_k), "bics": {str(k): v for k, v in bics.items()}}


def _run_null_pair(
    null_code: NullCode,
    x_ku: np.ndarray,
    x_cz: np.ndarray,
    cal_ku: CalibratedCopulaSpec | None,
    cal_cz: CalibratedCopulaSpec | None,
    k: int,
    seed_seq: np.random.SeedSequence,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed_seq)
    s_ku = generate_null_sample(null_code, x_ku, rng, copula_spec=cal_ku, standardize=True)
    s_cz = generate_null_sample(null_code, x_cz, rng, copula_spec=cal_cz, standardize=True)
    f_ku = fit_matched_kmeans(s_ku, n_clusters=k, n_kmeanspp_starts=9, rng=rng)
    f_cz = fit_matched_kmeans(s_cz, n_clusters=k, n_kmeanspp_starts=9, rng=rng)
    m = match_centroids_hungarian(f_ku.centers, f_cz.centers, optimize_on="full_tucker")
    return m.mean_tucker_full, m.mean_tucker_shape


def main() -> int:
    t0 = time.perf_counter()
    b_reps = 2000
    n_jobs = -2
    out_dir = Path("results/empirical/hitop")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_ku = pd.read_csv("data/processed/hitop_empirical/ku_hitopsr_scored.csv")
    df_cz = pd.read_csv("data/raw/hitop_empirical/czech_hitop/dataset.csv")

    res_ku = compute_spectrum_composites(df_ku, cohort="ku", n_spectra=5, standardize_mode="within")
    res_cz = compute_spectrum_composites(df_cz, cohort="cz", n_spectra=5, standardize_mode="within")
    x_ku = res_ku.scores.to_numpy(dtype=np.float64)
    x_cz = res_cz.scores.to_numpy(dtype=np.float64)
    spec_names = list(res_ku.spectrum_order)

    g_ku = correlation_geometry(x_ku)
    g_cz = correlation_geometry(x_cz)

    master_ss = np.random.SeedSequence([BASE_SEED, 2026])
    ss_emp, ss_cal, ss_apriori, ss_topo, ss_k3, ss_k4 = master_ss.spawn(6)
    rng_emp = np.random.default_rng(ss_emp)

    # Calibrate C1 copulas once
    rng_cal = np.random.default_rng(ss_cal)
    cal_ku = calibrate_empirical_copula(x_ku, rng_cal, n_cal=30_000, tol=1e-4)
    cal_cz = calibrate_empirical_copula(x_cz, rng_cal, n_cal=30_000, tol=1e-4)

    # 1. Empirical clustering, split-half stability, and a priori N(0, R) prediction
    rng_ap = np.random.default_rng(ss_apriori)
    chol_ku = np.linalg.cholesky(g_ku.r_matrix)
    chol_cz = np.linalg.cholesky(g_cz.r_matrix)
    x_ap_ku = rng_ap.standard_normal((50_000, 5)) @ chol_ku.T
    x_ap_cz = rng_ap.standard_normal((50_000, 5)) @ chol_cz.T

    emp_k_results: dict[str, Any] = {}
    null_comparisons: dict[str, Any] = {}

    for k, ss_null_k in ((3, ss_k3), (4, ss_k4)):
        f_ku = fit_matched_kmeans(x_ku, n_clusters=k, n_kmeanspp_starts=9, rng=rng_emp)
        f_cz = fit_matched_kmeans(x_cz, n_clusters=k, n_kmeanspp_starts=9, rng=rng_emp)
        m_emp = match_centroids_hungarian(f_ku.centers, f_cz.centers, optimize_on="full_tucker")

        # A priori N_5(0, R) centroids at large N (pure geometry of R)
        ap_ku = fit_matched_kmeans(x_ap_ku, n_clusters=k, n_kmeanspp_starts=9, ward_subsample=2000, rng=rng_ap)
        ap_cz = fit_matched_kmeans(x_ap_cz, n_clusters=k, n_kmeanspp_starts=9, ward_subsample=2000, rng=rng_ap)
        m_ku_vs_r = match_centroids_hungarian(f_ku.centers, ap_ku.centers, optimize_on="full_tucker")
        m_cz_vs_r = match_centroids_hungarian(f_cz.centers, ap_cz.centers, optimize_on="full_tucker")

        stab_ku = split_half_stability(x_ku, n_clusters=k, n_splits=10, rng=rng_emp)
        stab_cz = split_half_stability(x_cz, n_clusters=k, n_splits=10, rng=rng_emp)

        emp_k_results[f"K{k}"] = {
            "cross_cohort_tucker_full": m_emp.mean_tucker_full,
            "cross_cohort_tucker_shape": m_emp.mean_tucker_shape,
            "ku_vs_apriori_R_tucker_full": m_ku_vs_r.mean_tucker_full,
            "ku_vs_apriori_R_tucker_shape": m_ku_vs_r.mean_tucker_shape,
            "cz_vs_apriori_R_tucker_full": m_cz_vs_r.mean_tucker_full,
            "cz_vs_apriori_R_tucker_shape": m_cz_vs_r.mean_tucker_shape,
            "ku_split_half": stab_ku,
            "cz_split_half": stab_cz,
            "ku_centers": f_ku.centers.tolist(),
            "cz_centers": f_cz.centers.tolist(),
        }

        # Parallel null pairs (I0, I1, C0, C1)
        null_seeds = ss_null_k.spawn(4)
        k_null_dict: dict[str, Any] = {}
        for nc, nc_ss in zip(("I0", "I1", "C0", "C1"), null_seeds):
            rep_seeds = nc_ss.spawn(b_reps)
            pairs = Parallel(n_jobs=n_jobs)(
                delayed(_run_null_pair)(
                    nc, x_ku, x_cz, cal_ku if nc == "C1" else None, cal_cz if nc == "C1" else None, k, rep_seeds[i]
                )
                for i in range(b_reps)
            )
            full_arr = np.array([p[0] for p in pairs], dtype=np.float64)
            shape_arr = np.array([p[1] for p in pairs], dtype=np.float64)
            p_full, k_full, b_n = phipson_smyth_p(full_arr, m_emp.mean_tucker_full, direction="ge")
            p_shape, k_shape, _ = phipson_smyth_p(shape_arr, m_emp.mean_tucker_shape, direction="ge")
            k_null_dict[nc] = {
                "full_mean": float(np.mean(full_arr)),
                "full_median": float(np.median(full_arr)),
                "full_q025": float(np.quantile(full_arr, 0.025)),
                "full_q975": float(np.quantile(full_arr, 0.975)),
                "p_full": p_full,
                "k_full": k_full,
                "B": b_n,
                "wilson_ci_full": list(wilson_ci(k_full, b_n)),
                "shape_mean": float(np.mean(shape_arr)),
                "shape_median": float(np.median(shape_arr)),
                "p_shape": p_shape,
            }
        null_comparisons[f"K{k}"] = k_null_dict

    # 2. Apparent K selected by Diagonal LPA vs Full GMM (K=1..6) on KU, CZ, and unimodal N(0, R)
    x_sim_ku = generate_null_sample("C0", x_ku, rng_ap, standardize=True)
    x_sim_cz = generate_null_sample("C0", x_cz, rng_ap, standardize=True)
    bic_selection = {
        "KU_empirical": {
            "diag_LPA": _bic_curve(x_ku, "diag", BASE_SEED),
            "full_GMM": _bic_curve(x_ku, "full", BASE_SEED + 100),
        },
        "CZ_empirical": {
            "diag_LPA": _bic_curve(x_cz, "diag", BASE_SEED + 200),
            "full_GMM": _bic_curve(x_cz, "full", BASE_SEED + 300),
        },
        "KU_unimodal_C0_null": {
            "diag_LPA": _bic_curve(x_sim_ku, "diag", BASE_SEED + 400),
            "full_GMM": _bic_curve(x_sim_ku, "full", BASE_SEED + 500),
        },
        "CZ_unimodal_C0_null": {
            "diag_LPA": _bic_curve(x_sim_cz, "diag", BASE_SEED + 600),
            "full_GMM": _bic_curve(x_sim_cz, "full", BASE_SEED + 700),
        },
    }

    # 3. 5D KDE mean-shift topography on KU and CZ
    rng_topo = np.random.default_rng(ss_topo)
    topography: dict[str, Any] = {}
    for label, x_c in (("KU", x_ku), ("CZ", x_cz)):
        h_nn = compute_h_nn(x_c, rng=rng_topo)
        starts = x_c  # exact full-sample starts (N=411 / 439)
        bw_res: dict[str, Any] = {}
        for mult in (1.00, 1.15, 1.25):
            h = h_nn * mult
            tr = run_mean_shift(x_c, starts, bandwidth=h, merge_radius=0.10)
            bw_res[f"{mult:.2f}_hNN"] = {
                "h": h,
                "n_modes": tr.n_modes,
                "primary_basin_frac": tr.primary_basin_frac,
                "max_secondary_basin_frac": tr.max_secondary_basin_frac,
            }
        topography[label] = {"h_nn": h_nn, "n_starts": int(len(starts)), "bandwidths": bw_res}

    payload = {
        "metadata": {
            "spectra": spec_names,
            "n_ku": int(len(x_ku)),
            "n_cz": int(len(x_cz)),
            "B_null_pairs": b_reps,
            "seed": BASE_SEED,
            "elapsed_sec": time.perf_counter() - t0,
        },
        "copula_calibration": {
            "ku_max_abs_err": cal_ku.max_abs_pearson_error,
            "cz_max_abs_err": cal_cz.max_abs_pearson_error,
        },
        "empirical_and_apriori_R": emp_k_results,
        "null_comparisons": null_comparisons,
        "bic_model_selection_K1_to_6": bic_selection,
        "topography_5d": topography,
    }

    json_path = out_dir / "hitop_empirical_results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # Privacy guard check
    assert_no_row_level_leakage(out_dir)

    # Print crisp executive summary
    print(f"=== HiTOP Empirical Validation Done in {payload['metadata']['elapsed_sec']:.1f}s ===")
    for k in (3, 4):
        ek = emp_k_results[f"K{k}"]
        nc = null_comparisons[f"K{k}"]
        print(
            f"\n--- K = {k} ---\n"
            f"  Empirical KU <-> CZ congruence : phi_full={ek['cross_cohort_tucker_full']:.3f}, phi_shape={ek['cross_cohort_tucker_shape']:.3f}\n"
            f"  Empirical vs a priori N(0, R)  : KU phi={ek['ku_vs_apriori_R_tucker_full']:.3f} | CZ phi={ek['cz_vs_apriori_R_tucker_full']:.3f}\n"
            f"  Split-half kappa               : KU={ek['ku_split_half']['mean_kappa']:.3f} | CZ={ek['cz_split_half']['mean_kappa']:.3f}\n"
            f"  I0 (Indep MVN)  : mean={nc['I0']['full_mean']:.3f} [{nc['I0']['full_q025']:.3f}, {nc['I0']['full_q975']:.3f}], p={nc['I0']['p_full']:.4f}\n"
            f"  I1 (Indep Emp)  : mean={nc['I1']['full_mean']:.3f} [{nc['I1']['full_q025']:.3f}, {nc['I1']['full_q975']:.3f}], p={nc['I1']['p_full']:.4f}\n"
            f"  C0 (Corr MVN R) : mean={nc['C0']['full_mean']:.3f} [{nc['C0']['full_q025']:.3f}, {nc['C0']['full_q975']:.3f}], p={nc['C0']['p_full']:.4f}\n"
            f"  C1 (Copula R+M) : mean={nc['C1']['full_mean']:.3f} [{nc['C1']['full_q025']:.3f}, {nc['C1']['full_q975']:.3f}], p={nc['C1']['p_full']:.4f}"
        )

    print("\n--- Apparent Number of Types K (BIC, K=1..6) ---")
    for name, d in bic_selection.items():
        print(f"  {name:22s} | Diagonal LPA best K = {d['diag_LPA']['best_k']} | Full GMM best K = {d['full_GMM']['best_k']}")

    print("\n--- 5D KDE Mean-Shift Topography (all N starts) ---")
    for name, t in topography.items():
        s100 = t["bandwidths"]["1.00_hNN"]
        s115 = t["bandwidths"]["1.15_hNN"]
        s125 = t["bandwidths"]["1.25_hNN"]
        print(
            f"  {name} (h_NN={t['h_nn']:.3f}): "
            f"1.00h={s100['primary_basin_frac']*100:.1f}% (sec {s100['max_secondary_basin_frac']*100:.1f}%) | "
            f"1.15h={s115['primary_basin_frac']*100:.1f}% (sec {s115['max_secondary_basin_frac']*100:.1f}%) | "
            f"1.25h={s125['primary_basin_frac']*100:.1f}% (sec {s125['max_secondary_basin_frac']*100:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
