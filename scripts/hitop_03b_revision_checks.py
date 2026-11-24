#!/usr/bin/env python3
"""Targeted statistical checks addressing v12 peer-review points:
  1) Exact 2x2 factorial CIs (full & shape-only) at K=3 and K=4 (B=2,000 pairs/null)
  2) Multi-replicate BIC model selection (K=1..6, diag LPA vs full GMM) on C0 and C1 (B=100 reps)
  3) Positive-control topographic power calibration at exact N=411 and N=439 (d=5, K=3,4, 25 reps/cell)
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
    generate_synthetic_mixture,
)
from hitop_empirical.harmonization import compute_spectrum_composites  # noqa: E402
from hitop_empirical.inference import phipson_smyth_p, wilson_ci  # noqa: E402
from hitop_empirical.matching import match_centroids_hungarian  # noqa: E402
from hitop_empirical.sources import assert_no_row_level_leakage  # noqa: E402
from hitop_empirical.topography import compute_h_nn, run_mean_shift  # noqa: E402


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
    m_full = match_centroids_hungarian(f_ku.centers, f_cz.centers, optimize_on="full_tucker")
    m_shape = match_centroids_hungarian(f_ku.centers, f_cz.centers, optimize_on="shape_tucker")
    return m_full.mean_tucker_full, m_shape.mean_tucker_shape


def _factorial_bootstrap_ci(
    i0: np.ndarray,
    i1: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 5000,
) -> dict[str, dict[str, Any]]:
    n = len(i0)
    idx = rng.integers(0, n, size=(n_boot, n))
    m_i0 = i0[idx].mean(axis=1)
    m_i1 = i1[idx].mean(axis=1)
    m_c0 = c0[idx].mean(axis=1)
    m_c1 = c1[idx].mean(axis=1)

    effects = {
        "main_R": 0.5 * ((m_c0 - m_i0) + (m_c1 - m_i1)),
        "main_M": 0.5 * ((m_i1 - m_i0) + (m_c1 - m_c0)),
        "R_x_M": (m_c1 - m_c0) - (m_i1 - m_i0),
        "C1_minus_C0": m_c1 - m_c0,
    }
    point = {
        "main_R": float(0.5 * ((c0.mean() - i0.mean()) + (c1.mean() - i1.mean()))),
        "main_M": float(0.5 * ((i1.mean() - i0.mean()) + (c1.mean() - c0.mean()))),
        "R_x_M": float((c1.mean() - c0.mean()) - (i1.mean() - i0.mean())),
        "C1_minus_C0": float(c1.mean() - c0.mean()),
    }
    return {
        k: {
            "estimate": point[k],
            "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
        }
        for k, v in effects.items()
    }


def _bic_rep(
    null_code: NullCode,
    x_emp: np.ndarray,
    cal: CalibratedCopulaSpec | None,
    seed_seq: np.random.SeedSequence,
) -> tuple[int, int]:
    rng = np.random.default_rng(seed_seq)
    sim = generate_null_sample(null_code, x_emp, rng, copula_spec=cal, standardize=True)
    seed_int = int(rng.integers(0, 2**31 - 1))
    best_diag_k, best_diag_bic = 1, float("inf")
    best_full_k, best_full_bic = 1, float("inf")
    for k in range(1, 7):
        g_d = GaussianMixture(n_components=k, covariance_type="diag", n_init=5, reg_covar=1e-6, random_state=seed_int + k).fit(sim)
        bd = float(g_d.bic(sim))
        if bd < best_diag_bic:
            best_diag_bic, best_diag_k = bd, k
        g_f = GaussianMixture(n_components=k, covariance_type="full", n_init=5, reg_covar=1e-6, random_state=seed_int + 10 + k).fit(sim)
        bf = float(g_f.bic(sim))
        if bf < best_full_bic:
            best_full_bic, best_full_k = bf, k
    return best_diag_k, best_full_k


def _pos_control_rep(
    n: int,
    k: int,
    delta: float,
    seed_seq: np.random.SeedSequence,
) -> bool:
    rng = np.random.default_rng(seed_seq)
    x, _, centers = generate_synthetic_mixture(n=n, d=5, k=k, separation_sd=delta, rng=rng)
    # Standardize per dimension as in empirical pipeline
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    h = compute_h_nn(x, rng=rng)
    # Use full-sample starts (exact same rule as empirical KU/CZ topography)
    tr = run_mean_shift(x, x, bandwidth=h * 1.15, merge_radius=0.15)
    # Count macroscopic modes attracting >= 5% of participants
    macro_modes = int(np.sum((tr.basin_counts / float(n)) >= 0.05))
    return macro_modes == k


def main() -> int:
    t0 = time.perf_counter()
    b_pairs = 2000
    b_bic = 100
    b_pc = 25
    n_jobs = -2
    out_dir = Path("results/empirical/hitop")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_ku = pd.read_csv("data/processed/hitop_empirical/ku_hitopsr_scored.csv")
    df_cz = pd.read_csv("data/raw/hitop_empirical/czech_hitop/dataset.csv")
    x_ku = compute_spectrum_composites(df_ku, cohort="ku", n_spectra=5, standardize_mode="within").scores.to_numpy(dtype=np.float64)
    x_cz = compute_spectrum_composites(df_cz, cohort="cz", n_spectra=5, standardize_mode="within").scores.to_numpy(dtype=np.float64)

    master_ss = np.random.SeedSequence([BASE_SEED, 20261124])
    ss_emp, ss_cal, ss_k3, ss_k4, ss_bic, ss_pc, ss_boot = master_ss.spawn(7)

    rng_cal = np.random.default_rng(ss_cal)
    cal_ku = calibrate_empirical_copula(x_ku, rng_cal, n_cal=30_000, tol=1e-4)
    cal_cz = calibrate_empirical_copula(x_cz, rng_cal, n_cal=30_000, tol=1e-4)

    # 1. Matched Full & Shape-Only Null Comparisons + 2x2 Factorial Bootstrap CIs
    rng_emp = np.random.default_rng(ss_emp)
    rng_boot = np.random.default_rng(ss_boot)
    factorial_out: dict[str, Any] = {}

    for k, ss_k in ((3, ss_k3), (4, ss_k4)):
        f_ku = fit_matched_kmeans(x_ku, n_clusters=k, n_kmeanspp_starts=9, rng=rng_emp)
        f_cz = fit_matched_kmeans(x_cz, n_clusters=k, n_kmeanspp_starts=9, rng=rng_emp)
        emp_full = match_centroids_hungarian(f_ku.centers, f_cz.centers, optimize_on="full_tucker").mean_tucker_full
        emp_shape = match_centroids_hungarian(f_ku.centers, f_cz.centers, optimize_on="shape_tucker").mean_tucker_shape

        null_seeds = ss_k.spawn(4)
        arrays_full: dict[str, np.ndarray] = {}
        arrays_shape: dict[str, np.ndarray] = {}
        null_summary: dict[str, Any] = {}
        for nc, nc_ss in zip(("I0", "I1", "C0", "C1"), null_seeds):
            rep_seeds = nc_ss.spawn(b_pairs)
            res = Parallel(n_jobs=n_jobs)(
                delayed(_run_null_pair)(nc, x_ku, x_cz, cal_ku if nc == "C1" else None, cal_cz if nc == "C1" else None, k, rep_seeds[i])
                for i in range(b_pairs)
            )
            af = np.array([r[0] for r in res], dtype=np.float64)
            as_ = np.array([r[1] for r in res], dtype=np.float64)
            arrays_full[nc], arrays_shape[nc] = af, as_
            pf, kf, bn = phipson_smyth_p(af, emp_full, direction="ge")
            ps, ks, _ = phipson_smyth_p(as_, emp_shape, direction="ge")
            null_summary[nc] = {
                "full_mean": float(af.mean()),
                "full_median": float(np.median(af)),
                "full_ci95": [float(np.quantile(af, 0.025)), float(np.quantile(af, 0.975))],
                "p_full": pf,
                "shape_mean": float(as_.mean()),
                "shape_median": float(np.median(as_)),
                "shape_ci95": [float(np.quantile(as_, 0.025)), float(np.quantile(as_, 0.975))],
                "p_shape": ps,
            }
        factorial_out[f"K{k}"] = {
            "emp_full": emp_full,
            "emp_shape": emp_shape,
            "nulls": null_summary,
            "factorial_effects_full": _factorial_bootstrap_ci(
                arrays_full["I0"], arrays_full["I1"], arrays_full["C0"], arrays_full["C1"], rng_boot
            ),
            "factorial_effects_shape": _factorial_bootstrap_ci(
                arrays_shape["I0"], arrays_shape["I1"], arrays_shape["C0"], arrays_shape["C1"], rng_boot
            ),
        }

    # 2. Multi-replicate BIC model selection on C0 and C1 (B=100 reps each)
    bic_seeds = ss_bic.spawn(4)
    bic_configs = [
        ("KU_C0", "C0", x_ku, None, bic_seeds[0]),
        ("KU_C1", "C1", x_ku, cal_ku, bic_seeds[1]),
        ("CZ_C0", "C0", x_cz, None, bic_seeds[2]),
        ("CZ_C1", "C1", x_cz, cal_cz, bic_seeds[3]),
    ]
    bic_summary: dict[str, Any] = {}
    for name, nc, x_e, cal_e, b_ss in bic_configs:
        r_seeds = b_ss.spawn(b_bic)
        out = Parallel(n_jobs=n_jobs)(
            delayed(_bic_rep)(nc, x_e, cal_e, r_seeds[i]) for i in range(b_bic)
        )
        kd = np.array([o[0] for o in out], dtype=int)
        kf = np.array([o[1] for o in out], dtype=int)
        bic_summary[name] = {
            "B": b_bic,
            "diag_LPA_mean_K": float(kd.mean()),
            "diag_LPA_median_K": float(np.median(kd)),
            "diag_LPA_counts": {str(k): int(np.sum(kd == k)) for k in range(1, 7)},
            "full_GMM_mean_K": float(kf.mean()),
            "full_GMM_median_K": float(np.median(kf)),
            "full_GMM_rate_K_ge_2": float(np.mean(kf >= 2)),
            "full_GMM_counts": {str(k): int(np.sum(kf == k)) for k in range(1, 7)},
        }

    # 3. Positive-control topographic calibration at N=411 and N=439 (d=5, 25 reps/cell)
    pc_grid = [(n, k, d) for n in (411, 439) for k in (3, 4) for d in (2.5, 3.0, 3.5, 4.0)]
    pc_seeds = ss_pc.spawn(len(pc_grid))
    pc_summary: dict[str, Any] = {}
    for (n, k, d), p_ss in zip(pc_grid, pc_seeds):
        r_seeds = p_ss.spawn(b_pc)
        hits = Parallel(n_jobs=n_jobs)(
            delayed(_pos_control_rep)(n, k, d, r_seeds[i]) for i in range(b_pc)
        )
        n_hit = int(sum(hits))
        lo, hi = wilson_ci(n_hit, b_pc)
        pc_summary[f"N{n}_K{k}_d{d:.1f}"] = {
            "n": n,
            "k": k,
            "delta_sd": d,
            "exact_recovery_rate": n_hit / float(b_pc),
            "hits": n_hit,
            "reps": b_pc,
            "wilson_ci95": [lo, hi],
        }

    payload = {
        "elapsed_sec": time.perf_counter() - t0,
        "factorial_full_and_shape": factorial_out,
        "bic_multi_rep_C0_C1": bic_summary,
        "positive_control_N411_N439": pc_summary,
    }
    out_path = out_dir / "hitop_v12_revision_checks.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    assert_no_row_level_leakage(out_dir)

    print(f"=== v12 Revision Checks Done in {payload['elapsed_sec']:.1f}s ===")
    for k in (3, 4):
        fk = factorial_out[f"K{k}"]
        print(f"\n--- K={k} (emp full={fk['emp_full']:.3f}, emp shape={fk['emp_shape']:.3f}) ---")
        for nc in ("I0", "I1", "C0", "C1"):
            d = fk["nulls"][nc]
            print(
                f"  {nc}: full={d['full_mean']:.3f} [{d['full_ci95'][0]:.3f},{d['full_ci95'][1]:.3f}] p={d['p_full']:.4f} | "
                f"shape={d['shape_mean']:.3f} [{d['shape_ci95'][0]:.3f},{d['shape_ci95'][1]:.3f}] p={d['p_shape']:.4f}"
            )
        ef = fk["factorial_effects_full"]
        print(
            f"  Effects (full): MainR={ef['main_R']['estimate']:+.3f} {np.round(ef['main_R']['ci95'],3).tolist()} | "
            f"RxM={ef['R_x_M']['estimate']:+.3f} {np.round(ef['R_x_M']['ci95'],3).tolist()} | "
            f"C1-C0={ef['C1_minus_C0']['estimate']:+.3f} {np.round(ef['C1_minus_C0']['ci95'],3).tolist()}"
        )

    print("\n--- Multi-rep BIC Selection (B=100 reps) ---")
    for name, b in bic_summary.items():
        print(
            f"  {name:6s} | Diag LPA mean K={b['diag_LPA_mean_K']:.2f} (med {b['diag_LPA_median_K']:.0f}) {b['diag_LPA_counts']} | "
            f"Full GMM mean K={b['full_GMM_mean_K']:.2f} (P(K>=2)={b['full_GMM_rate_K_ge_2']*100:.0f}%) {b['full_GMM_counts']}"
        )

    print("\n--- Positive Control at N=411 / 439 (1.15 h_NN, 25 reps/cell) ---")
    for key, p in pc_summary.items():
        print(f"  {key:14s}: {p['exact_recovery_rate']*100:5.1f}% ({p['hits']}/{p['reps']}) CI=[{p['wilson_ci95'][0]*100:.0f}%, {p['wilson_ci95'][1]*100:.0f}%]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
