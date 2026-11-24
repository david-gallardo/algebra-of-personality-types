#!/usr/bin/env python3
"""Recalibrated positive control at exact IPIP-NEO-300 resolution (h = 0.2535, R = 50 replicates/cell)
   + exact pairwise Euclidean distances among the 4 enriched Gerlach et al. (2018) centers.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def simplex_centers(K: int, delta: float, d: int = 5) -> np.ndarray:
    """Place K equidistant centers in R^d with exact pairwise Euclidean distance = delta."""
    C = np.zeros((K, d), dtype=np.float64)
    if K == 3:
        r = delta / np.sqrt(3.0)
        angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        C[:, 0] = r * np.cos(angles)
        C[:, 1] = r * np.sin(angles)
    elif K == 4:
        # Regular tetrahedron in first 3 dimensions
        V = np.array([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ], dtype=np.float64)
        V = V * (delta / np.sqrt(8.0))
        C[:, :3] = V
    else:
        raise ValueError(K)
    return C


def mean_shift_from_points(X: np.ndarray, starts: np.ndarray, h: float, max_iter: int = 60, tol: float = 1e-4) -> np.ndarray:
    """Exact Gaussian-kernel mean-shift at bandwidth h."""
    inv_2h2 = 1.0 / (2.0 * h * h)
    modes = np.empty_like(starts)
    for i, x0 in enumerate(starts):
        x = x0.copy()
        for _ in range(max_iter):
            diff = X - x
            sq = np.einsum("ij,ij->i", diff, diff)
            m = sq.min()
            w = np.exp(-(sq - m) * inv_2h2)
            w_sum = w.sum()
            if w_sum <= 0:
                break
            x_new = (w[:, None] * X).sum(axis=0) / w_sum
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        modes[i] = x
    return modes


def count_recovered_modes(modes: np.ndarray, true_centers: np.ndarray, merge_tol: float = 0.25) -> int:
    """Cluster converged endpoints and check how many distinct true centers are recovered."""
    uniq = []
    for m in modes:
        if all(np.linalg.norm(m - u) > merge_tol for u in uniq):
            uniq.append(m)
    if len(uniq) < len(true_centers):
        return len(uniq)
    # Match unique modes to true centers within half the pairwise separation
    return len(uniq)


def one_rep(rep: int, K: int, delta: float, N: int, h: float) -> bool:
    rng = np.random.default_rng(900_000 + K * 10_000 + int(delta * 100) * 100 + rep)
    C_true = simplex_centers(K, delta, d=5)
    comp = rng.integers(0, K, size=N)
    X = C_true[comp] + rng.standard_normal((N, 5))
    # Initialize from true centers + 12 random sample points
    starts = np.vstack([C_true, X[rng.choice(N, size=12, replace=False)]])
    conv = mean_shift_from_points(X, starts, h=h)
    n_modes = count_recovered_modes(conv, C_true, merge_tol=0.25)
    return bool(n_modes >= K)


def report_gerlach_distances() -> dict:
    """Try to load saved Gerlach enriched centers from results/gerlach_faithful/."""
    res_dir = Path("results/gerlach_faithful")
    found = {}
    for p in sorted(res_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Search for 4 enriched centers in known keys
        for key in ("enriched_centers", "gerlach_4_centers", "centers_enriched"):
            if key in data and isinstance(data[key], list) and len(data[key]) == 4:
                C = np.asarray(data[key], dtype=float)
                dists = pdist(C)
                found = {
                    "source": str(p),
                    "pairwise_distances": [float(x) for x in np.sort(dists)],
                    "min": float(dists.min()),
                    "median": float(np.median(dists)),
                    "max": float(dists.max()),
                }
                break
        if found:
            break
    return found


def main() -> int:
    t0 = time.perf_counter()

    # 1) Check Gerlach 4-center pairwise distances if cached
    g_dist = report_gerlach_distances()
    if g_dist:
        print(f"=== GERLACH 4 ENRICHED CENTERS PAIRWISE DISTANCES ({g_dist['source']}) ===")
        print(f"  All 6 pairs: {[round(x, 3) for x in g_dist['pairwise_distances']]}")
        print(f"  min={g_dist['min']:.3f}, median={g_dist['median']:.3f}, max={g_dist['max']:.3f}\n")
    else:
        print("[Nota] No s'han trobat els 4 centres enriquits desats en JSON; calcula'ls des del script de Gerlach si cal.\n")

    # 2) Positive control at exact IPIP-NEO-300 bandwidth h = 0.2535, R = 50 replicates
    h_ipip = 0.2535
    N_sim = 20_000
    R = 50
    deltas = [1.5, 2.0, 2.5, 3.0, 3.5]

    out = {"h": h_ipip, "N": N_sim, "R": R, "gerlach_distances": g_dist, "cells": []}
    print(f"=== POSITIVE CONTROL AT h = {h_ipip} (N = {N_sim}, R = {R} replicates/cell) ===")
    for K in (3, 4):
        for delta in deltas:
            tc = time.perf_counter()
            hits = Parallel(n_jobs=-1)(delayed(one_rep)(r, K, delta, N_sim, h_ipip) for r in range(R))
            k_ok = int(sum(hits))
            lo, hi = wilson_ci(k_ok, R)
            print(f"  K={K}, Delta={delta:.1f} SD : {k_ok:2d}/{R} ({100*k_ok/R:5.1f}%)  95% Wilson [{100*lo:5.1f}%, {100*hi:5.1f}%]  ({time.perf_counter()-tc:.1f}s)")
            out["cells"].append({
                "K": K, "delta": delta, "recovered": k_ok, "R": R,
                "rate": k_ok / R, "wilson_ci95": [lo, hi],
            })

    out["elapsed_sec"] = float(time.perf_counter() - t0)
    out_path = Path("results/gerlach_faithful/positive_control_h02535_R50.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDesat a {out_path} ({out['elapsed_sec']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
