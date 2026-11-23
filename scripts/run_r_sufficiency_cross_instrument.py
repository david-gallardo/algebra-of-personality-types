#!/usr/bin/env python3
"""Direct causal sufficiency test of R for cross-instrument replicability:
   P(phi_sim >= 0.961 | R_BFI, R_IPIP, MVN) vs. Independent MVN (R = I_5),
   plus elevation-removed (ipsatized) profile shape congruence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

NEOAC = ["N", "E", "O", "A", "C"]


def find_corr_matrix(obj) -> np.ndarray | None:
    """Recursively locate a 5x5 correlation matrix (list-of-lists or NEOAC dict-of-dicts)."""
    if isinstance(obj, dict):
        # Check dict-of-dicts keyed by NEOAC
        if all(t in obj for t in NEOAC) and isinstance(obj["N"], dict) and all(t in obj["N"] for t in NEOAC):
            return np.array([[float(obj[r][c]) for c in NEOAC] for r in NEOAC], dtype=float)
        for k, v in obj.items():
            kl = k.lower()
            if any(tok in kl for tok in ("corr", "r_emp", "r_mat", "matrix")):
                res = find_corr_matrix(v)
                if res is not None:
                    return res
        for v in obj.values():
            res = find_corr_matrix(v)
            if res is not None:
                return res
    elif isinstance(obj, list) and len(obj) == 5 and all(isinstance(row, list) and len(row) == 5 for row in obj):
        try:
            arr = np.array(obj, dtype=float)
            if np.allclose(np.diag(arr), 1.0, atol=1e-3) and np.allclose(arr, arr.T, atol=1e-3):
                return arr
        except Exception:
            pass
    return None


def find_n(d: dict, default_n: int) -> int:
    for k in ("n", "N", "n_complete", "n_obs", "sample_size"):
        if k in d and isinstance(d[k], (int, float)):
            return int(d[k])
    for v in d.values():
        if isinstance(v, dict):
            for k in ("n", "N", "n_complete", "n_obs", "sample_size"):
                if k in v and isinstance(v[k], (int, float)):
                    return int(v[k])
    return default_n


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


def fit_k3(X: np.ndarray, seed: int) -> np.ndarray:
    km = KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=seed)
    km.fit(X)
    return km.cluster_centers_


def main() -> int:
    t0 = time.perf_counter()
    bfi_sum = json.loads(Path("results/empirical/bfi2_es_study1_summary.json").read_text())
    ipip_sum = json.loads(Path("results/empirical/ipip300_domains_summary.json").read_text())

    C_bfi = np.array([[c["centroid_neoac"][t] for t in NEOAC] for c in bfi_sum["empirical_centroids"]["ward_kmeans_k3"]], dtype=float)
    C_ipip = np.array([[c["centroid_neoac"][t] for t in NEOAC] for c in ipip_sum["empirical_centroids"]["ward_kmeans_k3"]], dtype=float)

    R_bfi = find_corr_matrix(bfi_sum)
    R_ipip = find_corr_matrix(ipip_sum)
    if R_bfi is None or R_ipip is None:
        print("No s'ha trobat la matriu R al JSON.")
        print("Claus BFI :", list(bfi_sum.keys()))
        print("Claus IPIP:", list(ipip_sum.keys()))
        return 1

    N_bfi = find_n(bfi_sum, 1146)
    N_ipip = find_n(ipip_sum, 145388)

    d2_emp, phi_emp = match_congruence(C_bfi, C_ipip, remove_elevation=False)
    d2_emp_shape, phi_emp_shape = match_congruence(C_bfi, C_ipip, remove_elevation=True)
    print(f"=== EMPIRICAL CROSS-INSTRUMENT (BFI-2 ES N={N_bfi} <-> IPIP-300 N={N_ipip}) ===")
    print(f"  Full profile  : mean_phi = {phi_emp:.4f}, mean_D2 = {d2_emp:.4f}")
    print(f"  Shape only    : mean_phi = {phi_emp_shape:.4f}, mean_D2 = {d2_emp_shape:.4f}")

    B = 200
    rng = np.random.default_rng(20261122)
    L_bfi = np.linalg.cholesky(R_bfi)
    L_ipip = np.linalg.cholesky(R_ipip)

    results = {}
    for cond, L1, L2 in [
        ("Correlated_MVN (R_BFI vs R_IPIP)", L_bfi, L_ipip),
        ("Independent_MVN (I_5 vs I_5)", np.eye(5), np.eye(5)),
    ]:
        phis, d2s, phis_sh = [], [], []
        for b in range(B):
            X1 = rng.standard_normal((N_bfi, 5)) @ L1.T
            X1 = (X1 - X1.mean(0)) / X1.std(0, ddof=1)
            X2 = rng.standard_normal((N_ipip, 5)) @ L2.T
            X2 = (X2 - X2.mean(0)) / X2.std(0, ddof=1)
            c1 = fit_k3(X1, seed=10_000 + b)
            c2 = fit_k3(X2, seed=20_000 + b)
            d2, ph = match_congruence(c1, c2, remove_elevation=False)
            _, ph_sh = match_congruence(c1, c2, remove_elevation=True)
            phis.append(ph)
            d2s.append(d2)
            phis_sh.append(ph_sh)

        phis_a, d2s_a, phis_sh_a = np.array(phis), np.array(d2s), np.array(phis_sh)
        p_ge = float((1 + np.sum(phis_a >= phi_emp)) / (B + 1))
        p_ge_sh = float((1 + np.sum(phis_sh_a >= phi_emp_shape)) / (B + 1))
        print(f"\n=== {cond} (B={B}) ===")
        print(
            f"  Full profile  : mean_phi = {phis_a.mean():.4f} (SD={phis_a.std():.4f}, "
            f"median={np.median(phis_a):.4f}, 95% [{np.percentile(phis_a, 2.5):.4f}, {np.percentile(phis_a, 97.5):.4f}]) | "
            f"P(phi >= {phi_emp:.3f}) = {p_ge:.4f}"
        )
        print(
            f"  Mean D2       : {d2s_a.mean():.4f} (95% [{np.percentile(d2s_a, 2.5):.4f}, {np.percentile(d2s_a, 97.5):.4f}])"
        )
        print(
            f"  Shape only    : mean_phi = {phis_sh_a.mean():.4f} (95% [{np.percentile(phis_sh_a, 2.5):.4f}, {np.percentile(phis_sh_a, 97.5):.4f}]) | "
            f"P(phi_shape >= {phi_emp_shape:.3f}) = {p_ge_sh:.4f}"
        )
        results[cond] = {
            "B": B,
            "phi_mean": float(phis_a.mean()),
            "phi_sd": float(phis_a.std()),
            "phi_median": float(np.median(phis_a)),
            "phi_ci95": [float(np.percentile(phis_a, 2.5)), float(np.percentile(phis_a, 97.5))],
            "p_phi_ge_emp": p_ge,
            "d2_mean": float(d2s_a.mean()),
            "phi_shape_mean": float(phis_sh_a.mean()),
            "phi_shape_ci95": [float(np.percentile(phis_sh_a, 2.5)), float(np.percentile(phis_sh_a, 97.5))],
            "p_phi_shape_ge_emp": p_ge_sh,
        }

    out_path = Path("results/gerlach_faithful/r_sufficiency_cross_instrument.json")
    out_path.write_text(json.dumps({
        "empirical_full_phi": phi_emp,
        "empirical_full_d2": d2_emp,
        "empirical_shape_phi": phi_emp_shape,
        "empirical_shape_d2": d2_emp_shape,
        "conditions": results,
        "elapsed_sec": float(time.perf_counter() - t0),
    }, indent=2), encoding="utf-8")
    print(f"\nDesat a {out_path} ({time.perf_counter()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
