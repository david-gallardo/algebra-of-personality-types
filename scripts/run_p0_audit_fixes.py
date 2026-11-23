#!/usr/bin/env python3
"""P0 Audit Fixes (self-contained):
  1) Diagnose exact cause of 0.911 vs 0.528 D2_ARC discrepancy on IPIP-300 Domains
  2) Multistart Mean-Shift (500 starts) across h in {0.75, 1.0, 1.25} * h_NN
  3) Isserlis-corrected co-kurtosis
  4) Soto (2019): compare Traits+GMM4 against strong continuous nonlinear competitors (Spline-Ridge & HGB)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path("src").resolve()))
from arc_study.metrics import ARC_PROTOTYPES


def arc_d2_phi(centers: np.ndarray) -> tuple[float, float]:
    cost = np.sum((centers[:, None, :] - ARC_PROTOTYPES[None, :, :]) ** 2, axis=2)
    ri, ci = linear_sum_assignment(cost)
    d2s, phis = [], []
    for r, c in zip(ri, ci):
        u, v = centers[r], ARC_PROTOTYPES[c]
        d2s.append(float(np.sum((u - v) ** 2)))
        phis.append(float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)))
    return float(np.mean(d2s)), float(np.mean(phis))


def main() -> int:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. DIAGNOSE 0.911 vs 0.528 DISCREPANCY ON IPIP-300 DOMAINS
    # ------------------------------------------------------------------
    print("=== 1. DIAGNOSING 0.911 vs 0.528 ON IPIP-300 DOMAINS ===")
    emp_sum = json.loads(Path("results/empirical/ipip300_domains_summary.json").read_text())
    c_saved = np.array(
        [[c["centroid_neoac"][t] for t in ["N", "E", "O", "A", "C"]] for c in emp_sum["empirical_centroids"]["ward_kmeans_k3"]],
        dtype=np.float64,
    )
    d2_saved, phi_saved = arc_d2_phi(c_saved)
    print(f"  Table 5 (saved summary.json)          : D2={d2_saved:.4f} (reported {emp_sum['empirical_metrics']['wk_arc_mean_d2']:.4f}), phi={phi_saved:.4f}")
    print("    Centers Table 5:\n", np.round(c_saved, 3))

    df300 = pd.read_csv("data/processed/ipip300_domains_complete.csv")[["N", "E", "O", "A", "C"]]
    X_dom = df300.to_numpy(dtype=np.float64)
    X_dom = (X_dom - X_dom.mean(axis=0)) / X_dom.std(axis=0, ddof=1)

    rng = np.random.default_rng(20261121)
    Z_sub = X_dom[rng.choice(len(X_dom), size=2000, replace=False)]
    w_lbl = fcluster(linkage(Z_sub, method="ward"), t=3, criterion="maxclust")
    init_c = np.vstack([Z_sub[w_lbl == j].mean(axis=0) for j in range(1, 4)])
    km_b = KMeans(n_clusters=3, init=init_c, n_init=1, max_iter=300, random_state=20261121).fit(X_dom)
    d2_b, phi_b = arc_d2_phi(km_b.cluster_centers_)
    print(f"  Table 6 (ward_sub=2000, n_init=1)     : D2={d2_b:.4f}, phi={phi_b:.4f}")
    print("    Centers Table 6:\n", np.round(km_b.cluster_centers_, 3))

    # Compare SSE/inertia of both solutions on the exact same X_dom
    inertia_a = float(np.sum(np.min(np.sum((X_dom[:, None, :] - c_saved[None, :, :]) ** 2, axis=2), axis=1)))
    inertia_b = float(km_b.inertia_)
    print(f"    SSE(Table 5) = {inertia_a:,.1f} vs SSE(Table 6) = {inertia_b:,.1f} (diff = {inertia_b - inertia_a:+,.1f})")

    # ------------------------------------------------------------------
    # 2. MULTISTART MEAN-SHIFT (500 starts) ACROSS BANDWIDTHS ON GERLACH IPIP-300
    # ------------------------------------------------------------------
    print("\n=== 2. MULTISTART MEAN-SHIFT (500 starts) ON GERLACH IPIP-300 (N=145,388) ===")
    X = np.load("external/gerlach2018/data_filter/ipip300-no0_arr_pd_neoac_score-1.npy").astype(np.float64)
    g_res = json.loads(Path("results/gerlach_faithful/ipip300_faithful_k13.json").read_text())["gerlach_ipip300_fa_varimax"]
    centers = np.array(g_res["centers_neoac"], dtype=np.float64)
    bw0 = float(g_res["bw_nn"])

    rng_ms = np.random.default_rng(20261121)
    norms = np.linalg.norm(X, axis=1)
    top_ext_idx = np.argsort(norms)[-500:]
    ext_starts = X[rng_ms.choice(top_ext_idx, size=86, replace=False)]
    rand_starts = X[rng_ms.choice(len(X), size=400, replace=False)]
    starts = np.vstack([centers, np.zeros((1, 5)), rand_starts, ext_starts])

    ms_summary = {}
    for mult in [0.75, 1.00, 1.25]:
        h = bw0 * mult
        pts = starts.copy()
        inv = 0.5 / (h * h)
        for it in range(150):
            num = np.zeros_like(pts)
            den = np.zeros(len(pts), dtype=np.float64)
            p2 = np.sum(pts * pts, axis=1)
            for s in range(0, len(X), 50_000):
                xb = X[s : s + 50_000]
                x2 = np.sum(xb * xb, axis=1, keepdims=True)
                sq = np.maximum(0.0, x2 + p2[None, :] - 2.0 * (xb @ pts.T))
                W = np.exp(-sq * inv)
                den += np.sum(W, axis=0)
                num += W.T @ xb
            nxt = num / np.maximum(den[:, None], 1e-300)
            step = float(np.max(np.linalg.norm(nxt - pts, axis=1)))
            pts = nxt
            if step < 5e-4:
                break
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
        order_m = np.argsort(counts)[::-1]
        uniq = [uniq[j] for j in order_m]
        counts = [counts[j] for j in order_m]
        print(f"  h = {mult:.2f}*h_NN ({h:.4f}): {len(uniq)} distinct mode(s) from 500 starts (iters={it+1}) | basin counts={counts[:5]}")
        print(f"    Primary mode ({counts[0]}/500): " + " ".join(f"{k}={v:+.2f}" for k, v in zip(["N", "E", "O", "A", "C"], uniq[0])))
        ms_summary[f"{mult:.2f}x"] = {
            "h": h,
            "n_modes": len(uniq),
            "basin_counts": counts,
            "primary_mode": uniq[0].tolist(),
        }

    # ------------------------------------------------------------------
    # 3. ISSERLIS-CORRECTED CO-KURTOSIS
    # ------------------------------------------------------------------
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    R = np.corrcoef(Z, rowvar=False)
    ck_isserlis = np.array([[float(np.mean((Z[:, i] ** 2) * (Z[:, j] ** 2)) - (1.0 + 2.0 * (R[i, j] ** 2))) for j in range(5)] for i in range(5)])
    off = ~np.eye(5, dtype=bool)
    print("\n=== 3. ISSERLIS-CORRECTED CO-KURTOSIS E[Z_i^2 Z_j^2] - (1 + 2 r_ij^2) ===")
    print(f"  Off-diagonal range: [{ck_isserlis[off].min():+.4f}, {ck_isserlis[off].max():+.4f}] (Gaussian baseline 2*r_ij^2 max={2 * np.max(R[off] ** 2):.5f})")

    # ------------------------------------------------------------------
    # 4. SOTO (2019): STRONG CONTINUOUS NONLINEAR BENCHMARK (Spline-Ridge & HGB)
    # ------------------------------------------------------------------
    print("\n=== 4. SOTO (2019) STRONG CONTINUOUS NONLINEAR BENCHMARK (Spline-Ridge & HGB) ===")
    soto_dir = Path("data/raw/soto2019/Supporting materials/1 Data and code - Replication attempts")
    df_cv = pd.read_csv("results/soto2019/soto2019_incremental_cv.csv")
    cont = df_cv[df_cv["type"] == "continuous"].copy()
    file_map = {
        "S1_Adults": ("Replication data - Survey 1 - Adults.sav", "S1"),
        "S1_Young": ("Replication data - Survey 1 - Young adults.sav", "S1"),
        "S2_Adults": ("Replication data - Survey 2 - Adults.sav", "S2"),
        "S2_Young": ("Replication data - Survey 2 - Young adults.sav", "S2"),
    }
    loaded = {k: pyreadstat.read_sav(str(soto_dir / fn))[0] for k, (fn, _) in file_map.items()}
    alphas = np.logspace(-2, 4, 25)

    spline_r2s, hgb_r2s = [], []
    for _, row in cont.iterrows():
        sub_name = row["subsample"]
        col = row["outcome"]
        pref = file_map[sub_name][1]
        bf_cols = [
            f"{pref}_PersonalityNegativeEmotionality",
            f"{pref}_PersonalityExtraversion",
            f"{pref}_PersonalityOpenMindedness",
            f"{pref}_PersonalityAgreeableness",
            f"{pref}_PersonalityConscientiousness",
        ]
        sub = loaded[sub_name][bf_cols + [col]].dropna()
        X_s = sub[bf_cols].to_numpy(dtype=np.float64)
        y_s = sub[col].to_numpy(dtype=np.float64)
        y_s = (y_s - y_s.mean()) / max(y_s.std(ddof=1), 1e-12)

        kf = KFold(n_splits=10, shuffle=True, random_state=20261121)
        p_spl = np.empty(len(y_s))
        p_hgb = np.empty(len(y_s))
        for tr, te in kf.split(X_s):
            mu, sd = X_s[tr].mean(0), np.where(X_s[tr].std(0, ddof=1) > 1e-12, X_s[tr].std(0, ddof=1), 1.0)
            Ztr, Zte = (X_s[tr] - mu) / sd, (X_s[te] - mu) / sd
            spl = make_pipeline(SplineTransformer(n_knots=5, degree=3, include_bias=False), RidgeCV(alphas=alphas))
            spl.fit(Ztr, y_s[tr])
            p_spl[te] = spl.predict(Zte)
            hgb = HistGradientBoostingRegressor(max_iter=80, max_depth=3, min_samples_leaf=40, l2_regularization=10.0, random_state=20261121)
            hgb.fit(Ztr, y_s[tr])
            p_hgb[te] = hgb.predict(Zte)
        spline_r2s.append(float(r2_score(y_s, p_spl)))
        hgb_r2s.append(float(r2_score(y_s, p_hgb)))

    cont["Traits_Spline"] = spline_r2s
    cont["Traits_HGB"] = hgb_r2s
    print(f"  Mean R^2 across k={len(cont)} continuous outcomes:")
    print(f"    Traits Linear      : {cont['Traits_Linear'].mean():.4f}")
    print(f"    Traits + ARC Hard  : {cont['Traits+ARC_Hard'].mean():.4f}")
    print(f"    Traits + GMM4 Prob : {cont['Traits+GMM4_Prob'].mean():.4f}")
    print(f"    Traits Spline-GAM  : {cont['Traits_Spline'].mean():.4f} (Delta vs Linear = {(cont['Traits_Spline'] - cont['Traits_Linear']).mean():+.5f})")
    print(f"    Traits HGB (trees) : {cont['Traits_HGB'].mean():.4f} (Delta vs Linear = {(cont['Traits_HGB'] - cont['Traits_Linear']).mean():+.5f})")

    out_path = Path("results/gerlach_faithful/p0_audit_fixes.json")
    out_path.write_text(json.dumps({
        "ipip300_discrepancy": {
            "table5_d2": d2_saved, "table5_phi": phi_saved, "sse_table5": inertia_a,
            "table6_d2": d2_b, "table6_phi": phi_b, "sse_table6": inertia_b,
        },
        "multistart_mean_shift": ms_summary,
        "soto_nonlinear_mean_r2": {
            "Traits_Linear": float(cont["Traits_Linear"].mean()),
            "Traits+GMM4_Prob": float(cont["Traits+GMM4_Prob"].mean()),
            "Traits_Spline": float(cont["Traits_Spline"].mean()),
            "Traits_HGB": float(cont["Traits_HGB"].mean()),
        },
        "elapsed_sec": float(time.perf_counter() - t0),
    }, indent=2), encoding="utf-8")
    print(f"\nDesat a {out_path} ({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
