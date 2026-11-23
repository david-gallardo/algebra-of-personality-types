#!/usr/bin/env python3
"""2x2 Factorial Sufficiency Test (B = 1,000 pairs/condition) with IDENTICAL Ward + k-means pipeline:
   1) Independent MVN (I_5)
   2) Independent Empirical Marginals (Null 1)
   3) Correlated MVN (Null 2: R_BFI vs R_IPIP)
   4) Correlated Empirical-Marginal Gaussian Copula (Null 3: R + marginals)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, spearmanr
from sklearn.cluster import KMeans

NEOAC = ["N", "E", "O", "A", "C"]


def fit_ward_kmeans_identical(X: np.ndarray, seed: int, ward_sub: int = 2000) -> np.ndarray:
    """Identical Ward + k-means (K=3) pipeline for both empirical and simulated datasets."""
    rng = np.random.default_rng(seed)
    if len(X) > ward_sub:
        idx = rng.choice(len(X), size=ward_sub, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    Z_link = linkage(X_sub, method="ward")
    w_lbl = fcluster(Z_link, t=3, criterion="maxclust")
    init_c = np.vstack([X_sub[w_lbl == j].mean(axis=0) for j in range(1, 4)])
    km = KMeans(n_clusters=3, init=init_c, n_init=1, max_iter=300, random_state=seed)
    km.fit(X)
    return km.cluster_centers_


def match_congruence(C1: np.ndarray, C2: np.ndarray, remove_elevation: bool = False) -> tuple[float, float]:
    A, B = C1.copy(), C2.copy()
    if remove_elevation:
        A = A - A.mean(axis=1, keepdims=True)
        B = B - B.mean(axis=1, keepdims=True)
    cost = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
    ri, ci = linear_sum_assignment(cost)
    d2s, phis = [], []
    for r, c in zip(ri, ci):
        u, v = A[r], B[c]
        d2s.append(float(np.sum((u - v) ** 2)))
        phis.append(float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)))
    return float(np.mean(d2s)), float(np.mean(phis))


def prep_dataset_stats(X: np.ndarray) -> dict:
    Xz = (X - X.mean(0)) / X.std(0, ddof=1)
    R_emp = np.corrcoef(Xz, rowvar=False)
    L_emp = np.linalg.cholesky(R_emp)

    rho_s, _ = spearmanr(Xz, axis=0)
    R_lat = 2.0 * np.sin((np.pi / 6.0) * rho_s)
    ev, ec = np.linalg.eigh(0.5 * (R_lat + R_lat.T))
    R_lat = (ec * np.maximum(ev, 1e-6)) @ ec.T
    inv_sd = 1.0 / np.sqrt(np.diag(R_lat))
    R_lat = R_lat * np.outer(inv_sd, inv_sd)
    np.fill_diagonal(R_lat, 1.0)
    L_lat = np.linalg.cholesky(R_lat)

    q_grid = np.linspace(0.0, 1.0, 20_001)
    emp_q = np.quantile(Xz, q_grid, axis=0)
    return {"Xz": Xz, "N": len(Xz), "L_emp": L_emp, "L_lat": L_lat, "q_grid": q_grid, "emp_q": emp_q}


def simulate_one(stats: dict, cond: str, rng: np.random.Generator) -> np.ndarray:
    N = stats["N"]
    if cond == "indep_mvn":
        X = rng.standard_normal((N, 5))
    elif cond == "indep_emp":
        U = rng.random((N, 5))
        X = np.column_stack([np.interp(U[:, j], stats["q_grid"], stats["emp_q"][:, j]) for j in range(5)])
    elif cond == "corr_mvn":
        X = rng.standard_normal((N, 5)) @ stats["L_emp"].T
    elif cond == "corr_copula":
        Z = rng.standard_normal((N, 5)) @ stats["L_lat"].T
        U = norm.cdf(Z)
        X = np.column_stack([np.interp(U[:, j], stats["q_grid"], stats["emp_q"][:, j]) for j in range(5)])
    else:
        raise ValueError(cond)
    return (X - X.mean(0)) / X.std(0, ddof=1)


def run_replicate(b: int, cond: str, s1: dict, s2: dict) -> tuple[float, float, float]:
    rng = np.random.default_rng(100_000 + b * 17)
    X1 = simulate_one(s1, cond, rng)
    X2 = simulate_one(s2, cond, rng)
    c1 = fit_ward_kmeans_identical(X1, seed=200_000 + b)
    c2 = fit_ward_kmeans_identical(X2, seed=300_000 + b)
    d2, ph = match_congruence(c1, c2, remove_elevation=False)
    _, ph_sh = match_congruence(c1, c2, remove_elevation=True)
    return ph, d2, ph_sh


def main() -> int:
    t0 = time.perf_counter()

    # Auto-locate BFI-2 ES and IPIP-300 processed CSVs
    proc_dir = Path("data/processed")
    bfi_candidates = sorted(proc_dir.glob("*bfi*.csv"))
    ipip_candidates = sorted(proc_dir.glob("*ipip300*domain*.csv"))
    if not bfi_candidates or not ipip_candidates:
        print("CSV trobats a data/processed:", [p.name for p in sorted(proc_dir.glob("*.csv"))])
        raise FileNotFoundError("No s'han trobat els CSV de BFI-2 ES o IPIP-300.")

    bfi_path = bfi_candidates[0]
    ipip_path = ipip_candidates[0]
    print(f"Carregant: {bfi_path.name} i {ipip_path.name}")

    X_bfi = pd.read_csv(bfi_path)[NEOAC].dropna().to_numpy(dtype=np.float64)
    X_ipip = pd.read_csv(ipip_path)[NEOAC].dropna().to_numpy(dtype=np.float64)

    s1 = prep_dataset_stats(X_bfi)
    s2 = prep_dataset_stats(X_ipip)

    # Empirical cross-instrument congruence with the EXACT SAME Ward + k-means pipeline
    c_bfi_emp = fit_ward_kmeans_identical(s1["Xz"], seed=20261122)
    c_ipip_emp = fit_ward_kmeans_identical(s2["Xz"], seed=20261122)
    d2_emp, phi_emp = match_congruence(c_bfi_emp, c_ipip_emp, remove_elevation=False)
    d2_emp_sh, phi_emp_sh = match_congruence(c_bfi_emp, c_ipip_emp, remove_elevation=True)

    print(f"\n=== EMPIRICAL (IDENTICAL WARD + K-MEANS: BFI-2 N={s1['N']} <-> IPIP-300 N={s2['N']}) ===")
    print(f"  Full profile : mean_phi = {phi_emp:.4f}, mean_D2 = {d2_emp:.4f}")
    print(f"  Shape only   : mean_phi = {phi_emp_sh:.4f}, mean_D2 = {d2_emp_sh:.4f}")

    B = 1000
    out = {
        "B": B,
        "empirical": {
            "full_phi": phi_emp, "full_d2": d2_emp,
            "shape_phi": phi_emp_sh, "shape_d2": d2_emp_sh,
        },
        "conditions": {},
    }

    for cond, label in [
        ("indep_mvn",   "1. Independent MVN (I_5)         "),
        ("indep_emp",   "2. Independent Empirical (Null 1)"),
        ("corr_mvn",    "3. Correlated MVN (Null 2: R)    "),
        ("corr_copula", "4. Correlated Copula (Null 3: R+M)"),
    ]:
        tc = time.perf_counter()
        res = Parallel(n_jobs=-1)(delayed(run_replicate)(b, cond, s1, s2) for b in range(B))
        arr = np.array(res, dtype=float)
        phis, d2s, phis_sh = arr[:, 0], arr[:, 1], arr[:, 2]
        p_full = float((1 + np.sum(phis >= phi_emp)) / (B + 1))
        p_sh = float((1 + np.sum(phis_sh >= phi_emp_sh)) / (B + 1))

        print(
            f"\n{label} ({time.perf_counter()-tc:.1f}s)\n"
            f"  Full  : mean={phis.mean():.4f} (med={np.median(phis):.4f}, 95% [{np.percentile(phis, 2.5):.4f}, {np.percentile(phis, 97.5):.4f}]) | P(>=emp)={p_full:.4f}\n"
            f"  Shape : mean={phis_sh.mean():.4f} (med={np.median(phis_sh):.4f}, 95% [{np.percentile(phis_sh, 2.5):.4f}, {np.percentile(phis_sh, 97.5):.4f}]) | P(>=emp)={p_sh:.4f}"
        )
        out["conditions"][cond] = {
            "full_phi_mean": float(phis.mean()),
            "full_phi_median": float(np.median(phis)),
            "full_phi_ci95": [float(np.percentile(phis, 2.5)), float(np.percentile(phis, 97.5))],
            "p_full_ge_emp": p_full,
            "d2_mean": float(d2s.mean()),
            "shape_phi_mean": float(phis_sh.mean()),
            "shape_phi_median": float(np.median(phis_sh)),
            "shape_phi_ci95": [float(np.percentile(phis_sh, 2.5)), float(np.percentile(phis_sh, 97.5))],
            "p_shape_ge_emp": p_sh,
        }

    out["elapsed_sec"] = float(time.perf_counter() - t0)
    out_path = Path("results/gerlach_faithful/r_sufficiency_2x2_ward.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCompletat i desat a {out_path} ({time.perf_counter()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
