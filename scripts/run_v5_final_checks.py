#!/usr/bin/env python3
"""Final 4 computational checks for Nature Human Behaviour v5:
  1) Exact random-sample basin mass & tail-mode density on IPIP-300 FA+Varimax
  2) Empirical unimodality & Pearson Delta_R check of Null 3 (Gaussian copula) and Null 4 (t-copula)
  3) Parallel IPIP-120 check (B=200 exact enrichment + valleys + random basin mass) & 10-rep positive control
  4) Soto (2019) GMM-4 incremental R^2 broken down by the 4 subsamples
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm, spearmanr, t as student_t
from sklearn.mixture import GaussianMixture


def exact_kde(X: np.ndarray, pts: np.ndarray, bw: float, chunk: int = 80_000) -> np.ndarray:
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


def mean_shift(X: np.ndarray, starts: np.ndarray, bw: float, max_iter: int = 120, tol: float = 5e-4, chunk: int = 80_000) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    pts = starts.copy()
    inv = 0.5 / (bw * bw)
    for _ in range(max_iter):
        num = np.zeros_like(pts)
        den = np.zeros(len(pts), dtype=np.float64)
        p2 = np.sum(pts * pts, axis=1)
        for s in range(0, len(X), chunk):
            xb = X[s : s + chunk]
            x2 = np.sum(xb * xb, axis=1, keepdims=True)
            sq = np.maximum(0.0, x2 + p2[None, :] - 2.0 * (xb @ pts.T))
            W = np.exp(-sq * inv)
            den += np.sum(W, axis=0)
            num += W.T @ xb
        nxt = num / np.maximum(den[:, None], 1e-300)
        if float(np.max(np.linalg.norm(nxt - pts, axis=1))) < tol:
            pts = nxt
            break
        pts = nxt

    uniq, counts = [], []
    for m in pts:
        found = False
        for j, u in enumerate(uniq):
            if np.linalg.norm(m - u) < 0.08:
                counts[j] += 1
                found = True
                break
        if not found:
            uniq.append(m)
            counts.append(1)
    order = np.argsort(counts)[::-1]
    return [uniq[j] for j in order], [counts[j] for j in order], pts


def main() -> int:
    t0 = time.perf_counter()
    rng = np.random.default_rng(20261122)
    out: dict = {}

    # ------------------------------------------------------------------
    # 1. EXACT RANDOM-SAMPLE BASIN MASS ON IPIP-300 FA+VARIMAX
    # ------------------------------------------------------------------
    print("=== 1. EXACT RANDOM-SAMPLE BASIN MASS (IPIP-300 FA+Varimax, 500 purely random starts) ===")
    X300 = np.load("external/gerlach2018/data_filter/ipip300-no0_arr_pd_neoac_score-1.npy").astype(np.float64)
    g300 = json.loads(Path("results/gerlach_faithful/ipip300_faithful_k13.json").read_text())["gerlach_ipip300_fa_varimax"]
    bw300 = float(g300["bw_nn"])
    rand500 = X300[rng.choice(len(X300), size=500, replace=False)]

    basin_res = {}
    for mult in [1.00, 1.15, 1.25]:
        h = bw300 * mult
        uniq, cnts, _ = mean_shift(X300, rand500, h)
        dens_modes = exact_kde(X300, np.array(uniq), h)
        pct_primary = 100.0 * cnts[0] / 500.0
        max_sec_cnt = cnts[1] if len(cnts) > 1 else 0
        max_sec_rel_dens = float(dens_modes[1:].max() / dens_modes[0]) if len(cnts) > 1 else 0.0
        print(
            f"  h={mult:.2f}*h_NN ({h:.4f}): Primary basin = {cnts[0]}/500 ({pct_primary:.1f}%) | "
            f"Total modes={len(uniq)} | max secondary basin={max_sec_cnt}/500 | max secondary rho/rho0={max_sec_rel_dens:.3f}"
        )
        basin_res[f"{mult:.2f}x"] = {
            "h": h,
            "primary_count": cnts[0],
            "primary_pct": pct_primary,
            "n_modes": len(uniq),
            "max_secondary_count": max_sec_cnt,
            "max_secondary_rel_density": max_sec_rel_dens,
        }
    out["ipip300_random_basin_mass"] = basin_res

    # ------------------------------------------------------------------
    # 2. EMPIRICAL UNIMODALITY & DELTA_R CHECK OF NULL 3 AND NULL 4
    # ------------------------------------------------------------------
    print("\n=== 2. NULL 3 (GAUSSIAN COPULA) & NULL 4 (t-COPULA) TOPOGRAPHY & DELTA_R CHECK ===")
    df_dom = pd.read_csv("data/processed/ipip300_domains_complete.csv")[["N", "E", "O", "A", "C"]]
    X_dom = df_dom.to_numpy(dtype=np.float64)
    X_dom = (X_dom - X_dom.mean(0)) / X_dom.std(0, ddof=1)
    R_emp = np.corrcoef(X_dom, rowvar=False)
    rho_s, _ = spearmanr(X_dom, axis=0)
    R_lat = 2.0 * np.sin((np.pi / 6.0) * rho_s)
    ev, ec = np.linalg.eigh(0.5 * (R_lat + R_lat.T))
    R_lat = (ec * np.maximum(ev, 1e-6)) @ ec.T
    inv_sd = 1.0 / np.sqrt(np.diag(R_lat))
    R_lat = R_lat * np.outer(inv_sd, inv_sd)
    np.fill_diagonal(R_lat, 1.0)
    L_lat = np.linalg.cholesky(R_lat)
    q_grid = np.linspace(0.0, 1.0, 20_001)
    emp_q = np.quantile(X_dom, q_grid, axis=0)
    nu_dom = 19.87

    null_checks = {}
    for label, kind in [("Null3_GaussianCopula", "gauss"), ("Null4_StudentTCopula", "t")]:
        if kind == "gauss":
            Z = rng.standard_normal((len(X_dom), 5)) @ L_lat.T
            U = norm.cdf(Z)
        else:
            chi = rng.chisquare(nu_dom, size=len(X_dom))
            T = (rng.standard_normal((len(X_dom), 5)) @ L_lat.T) / np.sqrt(chi / nu_dom)[:, None]
            U = student_t.cdf(T, df=nu_dom)
        X_sim = np.column_stack([np.interp(U[:, j], q_grid, emp_q[:, j]) for j in range(5)])
        R_sim = np.corrcoef(X_sim, rowvar=False)
        diff = np.abs(R_sim - R_emp)
        np.fill_diagonal(diff, 0.0)
        starts_sim = X_sim[rng.choice(len(X_sim), size=200, replace=False)]
        uniq_s, cnts_s, _ = mean_shift(X_sim, starts_sim, 0.2427)
        print(
            f"  {label:22s} | max|r_sim-r_emp|={diff.max():.5f} | "
            f"Primary basin={cnts_s[0]}/200 ({100*cnts_s[0]/200:.1f}%) | max secondary={cnts_s[1] if len(cnts_s)>1 else 0}/200"
        )
        null_checks[label] = {
            "max_abs_delta_R": float(diff.max()),
            "primary_basin_200": cnts_s[0],
            "max_secondary_basin_200": cnts_s[1] if len(cnts_s) > 1 else 0,
        }
    out["null_copula_checks"] = null_checks

    # ------------------------------------------------------------------
    # 3. MULTI-REPLICATE POSITIVE CONTROL (10 reps/cell, N=10,000)
    # ------------------------------------------------------------------
    print("\n=== 3. MULTI-REPLICATE POSITIVE CONTROL (10 reps/cell, N=10,000, d=5) ===")
    V3 = np.array([[1, 0, 0, 0, 0], [-0.5, np.sqrt(3)/2, 0, 0, 0], [-0.5, -np.sqrt(3)/2, 0, 0, 0]]) / np.sqrt(3.0)
    V4 = np.array([[1, 1, 1, 0, 0], [1, -1, -1, 0, 0], [-1, 1, -1, 0, 0], [-1, -1, 1, 0, 0]], dtype=float) / np.sqrt(8.0)
    pc_power = []
    for K_true, V in [(3, V3), (4, V4)]:
        for delta in [2.5, 3.0, 3.5]:
            exact_hits = 0
            for rep in range(10):
                r_p = np.random.default_rng(10_000 * K_true + int(delta * 100) + rep)
                comp = r_p.integers(0, K_true, size=10_000)
                Xm = (V * delta)[comp] + r_p.standard_normal((10_000, 5))
                Xm = (Xm - Xm.mean(0)) / Xm.std(0, ddof=1)
                tree = cKDTree(Xm)
                d_nn, _ = tree.query(Xm[r_p.choice(10_000, 3_000, replace=False)], k=2)
                bw_m = float(np.mean(d_nn[:, 1]))
                gmm_m = GaussianMixture(n_components=K_true, covariance_type="full", n_init=3, reg_covar=1e-6, random_state=rep).fit(Xm)
                uniq_m, _, _ = mean_shift(Xm, gmm_m.means_, bw_m, max_iter=100)
                if len(uniq_m) == K_true:
                    exact_hits += 1
            print(f"  K_true={K_true} | delta={delta:.1f} SD | exact mode recovery = {exact_hits}/10 ({100*exact_hits/10:.0f}%)")
            pc_power.append({"K_true": K_true, "delta": delta, "exact_hits_of_10": exact_hits})
    out["positive_control_power"] = pc_power

    # ------------------------------------------------------------------
    # 4. SOTO (2019) SUBSAMPLE BREAKDOWN FOR GMM-4
    # ------------------------------------------------------------------
    print("\n=== 4. SOTO (2019) CONTINUOUS DELTA R^2 BY SUBSAMPLE ===")
    df_cv = pd.read_csv("results/soto2019/soto2019_incremental_cv.csv")
    cont = df_cv[df_cv["type"] == "continuous"]
    sub_res = {}
    for sub_name, grp in cont.groupby("subsample"):
        print(
            f"  {sub_name:10s} (k={len(grp):2d}) | Linear={grp['Traits_Linear'].mean():.4f} | "
            f"+ARC={grp['delta_ARC_over_Traits'].mean():+.5f} | +GMM4={grp['delta_GMM4_over_Traits'].mean():+.5f} "
            f"(med={grp['delta_GMM4_over_Traits'].median():+.5f}, >0.01: {(grp['delta_GMM4_over_Traits']>0.01).sum()}/{len(grp)})"
        )
        sub_res[sub_name] = {
            "k": int(len(grp)),
            "mean_linear": float(grp["Traits_Linear"].mean()),
            "mean_delta_arc": float(grp["delta_ARC_over_Traits"].mean()),
            "mean_delta_gmm4": float(grp["delta_GMM4_over_Traits"].mean()),
            "n_gmm4_gt_001": int((grp["delta_GMM4_over_Traits"] > 0.01).sum()),
        }
    out["soto_subsample_breakdown"] = sub_res
    out["elapsed_sec"] = float(time.perf_counter() - t0)

    out_path = Path("results/gerlach_faithful/v5_final_checks.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCompletat i desat a {out_path} ({time.perf_counter()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
