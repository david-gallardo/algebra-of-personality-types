#!/usr/bin/env python3
"""Exact-resolution positive & negative control at N = 145,388, h = 0.2535
   using cKDTree-accelerated Gaussian mean-shift (exact within 4*h).
   Verifies both specificity (Delta = 0 unimodal) and true-center recovery (dist < delta/3).
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def simplex_centers(K: int, delta: float, d: int = 5) -> np.ndarray:
    C = np.zeros((K, d), dtype=np.float64)
    if K == 1 or delta == 0.0:
        return np.zeros((1, d), dtype=np.float64)
    if K == 3:
        r = delta / np.sqrt(3.0)
        angles = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        C[:, 0] = r * np.cos(angles)
        C[:, 1] = r * np.sin(angles)
    elif K == 4:
        V = np.array([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ], dtype=np.float64) * (delta / np.sqrt(8.0))
        C[:, :3] = V
    return C


def mean_shift_kdtree(X: np.ndarray, starts: np.ndarray, h: float, max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    tree = cKDTree(X)
    inv_2h2 = 1.0 / (2.0 * h * h)
    radius = 4.0 * h
    modes = np.empty_like(starts)
    for i, x0 in enumerate(starts):
        x = x0.copy()
        for _ in range(max_iter):
            idx = tree.query_ball_point(x, r=radius)
            if len(idx) < 5:
                break
            pts = X[idx]
            diff = pts - x
            sq = np.einsum("ij,ij->i", diff, diff)
            w = np.exp(-(sq - sq.min()) * inv_2h2)
            x_new = (w[:, None] * pts).sum(axis=0) / w.sum()
            if np.linalg.norm(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        modes[i] = x
    return modes


def one_rep(rep: int, K: int, delta: float, N: int, h: float) -> bool:
    rng = np.random.default_rng(700_000 + K * 10_000 + int(delta * 100) * 100 + rep)
    C_true = simplex_centers(K, delta, d=5)
    if K == 1 or delta == 0.0:
        X = rng.standard_normal((N, 5))
        starts = X[rng.choice(N, size=20, replace=False)]
        conv = mean_shift_kdtree(X, starts, h=h)
        # Check that >= 90% of random starts converge within 0.5 SD of the origin (unimodal)
        return bool(np.mean(np.linalg.norm(conv, axis=1) < 0.5) >= 0.90)

    comp = rng.integers(0, K, size=N)
    X = C_true[comp] + rng.standard_normal((N, 5))
    # Start from the K true centers (or perturbed starts +/- 0.2 SD)
    starts = C_true + rng.normal(0.0, 0.15, size=C_true.shape)
    conv = mean_shift_kdtree(X, starts, h=h)
    # Match converged endpoints to true centers; all K must be distinct and within delta/3
    cost = np.linalg.norm(conv[:, None, :] - C_true[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    max_err = float(cost[ri, ci].max())
    # Also ensure no two converged endpoints collapsed together
    min_sep = min(np.linalg.norm(conv[a] - conv[b]) for a in range(K) for b in range(a + 1, K))
    return bool(max_err < (delta / 3.0) and min_sep > (delta / 2.0))


def main() -> int:
    t0 = time.perf_counter()
    N_ipip = 145_388
    h_ipip = 0.2535
    R = 30  # 30 replicates at full N=145,388

    out = {"N": N_ipip, "h": h_ipip, "R": R, "cells": []}
    print(f"=== EXACT-N CONTROL AT N = {N_ipip}, h = {h_ipip} (R = {R} replicates/cell) ===")

    # Negative control (Delta = 0, unimodal Gaussian)
    tc = time.perf_counter()
    hits0 = Parallel(n_jobs=-1)(delayed(one_rep)(r, 1, 0.0, N_ipip, h_ipip) for r in range(R))
    k0 = int(sum(hits0))
    lo0, hi0 = wilson_ci(k0, R)
    print(f"  Unimodal (Delta=0.0): {k0:2d}/{R} ({100*k0/R:5.1f}%) unimodal  95% Wilson [{100*lo0:5.1f}%, {100*hi0:5.1f}%] ({time.perf_counter()-tc:.1f}s)")
    out["unimodal_control"] = {"ok": k0, "R": R, "rate": k0 / R, "wilson_ci95": [lo0, hi0]}

    for K in (3, 4):
        for delta in (2.0, 2.5, 3.0, 3.5):
            tc = time.perf_counter()
            hits = Parallel(n_jobs=-1)(delayed(one_rep)(r, K, delta, N_ipip, h_ipip) for r in range(R))
            k_ok = int(sum(hits))
            lo, hi = wilson_ci(k_ok, R)
            print(f"  K={K}, Delta={delta:.1f} SD : {k_ok:2d}/{R} ({100*k_ok/R:5.1f}%)  95% Wilson [{100*lo:5.1f}%, {100*hi:5.1f}%] ({time.perf_counter()-tc:.1f}s)")
            out["cells"].append({"K": K, "delta": delta, "recovered": k_ok, "R": R, "rate": k_ok / R, "wilson_ci95": [lo, hi]})

    out["elapsed_sec"] = float(time.perf_counter() - t0)
    out_path = Path("results/gerlach_faithful/positive_control_exactN_h02535.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDesat a {out_path} ({out['elapsed_sec']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
