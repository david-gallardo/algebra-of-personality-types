#!/usr/bin/env python3
"""Final Via A verification suite:
  1) Faithful Gerlach K=13 + Mean-Shift + valley test on IPIP-120 (N=410,376)
  2) Positive control (power of Mean-Shift & valley test on genuine K=3 and K=4 mixtures)
  3) Representation experiment on IPIP-300: Domains vs Pure Orthogonal Rotation vs ZCA Whitening vs FA-Varimax
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path("src").resolve()))
from arc_study.metrics import ARC_PROTOTYPES


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


def mean_shift_modes(X: np.ndarray, starts: np.ndarray, bw: float, max_iter: int = 120, tol: float = 1e-4, chunk: int = 80_000) -> tuple[list[np.ndarray], np.ndarray]:
    pts = starts.copy()
    inv = 0.5 / (bw * bw)
    n = X.shape[0]
    for _ in range(max_iter):
        num = np.zeros_like(pts)
        den = np.zeros(len(pts), dtype=np.float64)
        p2 = np.sum(pts * pts, axis=1)
        for s in range(0, n, chunk):
            xb = X[s : s + chunk]
            x2 = np.sum(xb * xb, axis=1, keepdims=True)
            sq = np.maximum(0.0, x2 + p2[None, :] - 2.0 * (xb @ pts.T))
            W = np.exp(-sq * inv)
            den += np.sum(W, axis=0)
            num += W.T @ xb
        nxt = num / np.maximum(den[:, None], 1e-300)
        if np.max(np.linalg.norm(nxt - pts, axis=1)) < tol:
            pts = nxt
            break
        pts = nxt

    uniq: list[np.ndarray] = []
    for m in pts:
        if all(np.linalg.norm(m - u) >= 0.08 for u in uniq):
            uniq.append(m)
    return uniq, pts


def avg_nn_bw(X: np.ndarray, n_samples: int = 10_000, seed: int = 20261121) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    tree = cKDTree(X)
    d, _ = tree.query(X[idx], k=2)
    return float(np.mean(d[:, 1]))


def count_valleys(X: np.ndarray, centers: np.ndarray, bw: float) -> tuple[int, int]:
    t_grid = np.linspace(0.0, 1.0, 21)
    n_valleys, n_pairs = 0, 0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            n_pairs += 1
            seg = (1.0 - t_grid)[:, None] * centers[i][None, :] + t_grid[:, None] * centers[j][None, :]
            vals = exact_kde(X, seg, bw)
            if float(np.min(vals[1:-1])) < min(vals[0], vals[-1]):
                n_valleys += 1
    return n_valleys, n_pairs


def ward_kmeans(Z: np.ndarray, k: int = 3, ward_sub: int = 2000, seed: int = 20261121) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Z_sub = Z[rng.choice(len(Z), size=ward_sub, replace=False)] if len(Z) > ward_sub else Z
    w_lbl = fcluster(linkage(Z_sub, method="ward"), t=k, criterion="maxclust")
    init_c = np.vstack([Z_sub[w_lbl == j].mean(axis=0) for j in range(1, k + 1)])
    km = KMeans(n_clusters=k, init=init_c, n_init=1, max_iter=300, random_state=seed).fit(Z)
    return km.cluster_centers_, km.labels_


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
    out: dict = {}

    # ------------------------------------------------------------------
    # 1. IPIP-120 (N=410,376): Faithful Gerlach K=13 + Mean-Shift + Valleys
    # ------------------------------------------------------------------
    print("=== 1. IPIP-120 FAITHFUL GERLACH (FA+Varimax, N=410,376, K=13) ===")
    X120 = np.load("data/processed/ipip120_fa_varimax_gerlach.npy").astype(np.float64)
    # Fit GMM K=13 on full 410k (n_init=3 for speed)
    gmm120 = GaussianMixture(n_components=13, covariance_type="full", n_init=3, max_iter=250, reg_covar=1e-6, random_state=20261121).fit(X120)
    c120 = gmm120.means_
    bw120 = avg_nn_bw(X120, n_samples=10_000, seed=20261121)
    rho_emp = exact_kde(X120, c120, bw120)

    # 100 permutation replicates to identify significant centers (exact Gerlach)
    rng = np.random.default_rng(20261121)
    rho_perm = np.empty((100, 13), dtype=np.float64)
    for b in range(100):
        Xp = np.column_stack([rng.permutation(X120[:, j]) for j in range(5)])
        rho_perm[b] = exact_kde(Xp, c120, bw120)
    enr120 = np.mean(rho_emp[None, :] / np.maximum(rho_perm, 1e-300), axis=0)
    p120 = np.maximum(np.mean(rho_perm >= rho_emp[None, :], axis=0), 1.0 / 100.0)
    sig_mask = (enr120 > 1.25) & (p120 < 0.01)
    sig_c120 = c120[sig_mask]

    # Mean-shift from all 13 centers + origin on exact KDE (h=bw120)
    uniq_modes, _ = mean_shift_modes(X120, np.vstack([c120, np.zeros((1, 5))]), bw=bw120)
    n_val, n_pairs = count_valleys(X120, sig_c120, bw120) if len(sig_c120) >= 2 else (0, 0)

    print(f"  N={len(X120):,} | bw_NN={bw120:.5f} | sig(Perm, e>1.25, p<.01)={int(sig_mask.sum())}/13 (max_e={enr120.max():.3f})")
    print(f"  Mean-Shift distinct modes={len(uniq_modes)} | Pairwise valleys among sig centers={n_val}/{n_pairs}")
    for j, m in enumerate(uniq_modes):
        print(f"    Mode {j}: " + " ".join(f"{k}={v:+.2f}" for k, v in zip(["N","E","O","A","C"], m)))

    out["ipip120_faithful"] = {
        "n": int(len(X120)),
        "bw_nn": bw120,
        "n_sig_perm": int(sig_mask.sum()),
        "max_enrichment_perm": float(enr120.max()),
        "n_distinct_modes": int(len(uniq_modes)),
        "modes": [m.tolist() for m in uniq_modes],
        "n_valleys": int(n_val),
        "n_sig_pairs": int(n_pairs),
    }

    # ------------------------------------------------------------------
    # 2. POSITIVE CONTROL: Genuine K=3 & K=4 mixtures (N=20,000, d=5)
    # ------------------------------------------------------------------
    print("\n=== 2. POSITIVE CONTROL (GENUINE MIXTURES, N=20,000, d=5) ===")
    pos_ctrl = []
    # Regular simplex vertices in 5D (unit pairwise distance before scaling by delta)
    V3 = np.array([
        [ 1.0,  0.0, 0.0, 0.0, 0.0],
        [-0.5,  np.sqrt(3)/2, 0.0, 0.0, 0.0],
        [-0.5, -np.sqrt(3)/2, 0.0, 0.0, 0.0],
    ]) / np.sqrt(3.0)  # pairwise dist = 1.0
    V4 = np.array([
        [ 1,  1,  1, 0, 0],
        [ 1, -1, -1, 0, 0],
        [-1,  1, -1, 0, 0],
        [-1, -1,  1, 0, 0],
    ], dtype=float) / np.sqrt(8.0)  # pairwise dist = 1.0

    for K_true, V in [(3, V3), (4, V4)]:
        for delta in [1.5, 2.5, 3.5, 4.5]:
            rng_p = np.random.default_rng(20261121 + 100 * K_true + int(delta * 10))
            n_sim = 20_000
            comp = rng_p.integers(0, K_true, size=n_sim)
            means = V * delta
            X_mix = means[comp] + rng_p.standard_normal((n_sim, 5))
            X_mix = (X_mix - X_mix.mean(axis=0)) / X_mix.std(axis=0, ddof=1)
            bw_m = avg_nn_bw(X_mix, n_samples=5_000, seed=20261121)
            gmm_m = GaussianMixture(n_components=K_true, covariance_type="full", n_init=5, reg_covar=1e-6, random_state=20261121).fit(X_mix)
            modes_m, _ = mean_shift_modes(X_mix, np.vstack([gmm_m.means_, np.zeros((1, 5))]), bw=bw_m)
            nv, npairs = count_valleys(X_mix, gmm_m.means_, bw_m)
            rec = {"K_true": K_true, "delta_sep": delta, "bw_nn": bw_m, "modes_found": len(modes_m), "valleys": nv, "pairs": npairs}
            pos_ctrl.append(rec)
            print(f"  K_true={K_true} | sep={delta:.1f} SD | bw_NN={bw_m:.3f} | modes_found={len(modes_m)} | valleys={nv}/{npairs}")
    out["positive_control"] = pos_ctrl

    # ------------------------------------------------------------------
    # 3. REPRESENTATION EXPERIMENT ON IPIP-300 (Error 5 fix)
    # ------------------------------------------------------------------
    print("\n=== 3. REPRESENTATION EXPERIMENT ON IPIP-300 (N=145,388) ===")
    df300 = pd.read_csv("data/processed/ipip300_domains_complete.csv")[["N", "E", "O", "A", "C"]]
    X_dom = df300.to_numpy(dtype=np.float64)
    X_dom = (X_dom - X_dom.mean(axis=0)) / X_dom.std(axis=0, ddof=1)

    # (a) Domains
    c_dom, lbl_dom = ward_kmeans(X_dom, k=3, seed=20261121)
    d2_dom, phi_dom = arc_d2_phi(c_dom)

    # (b) Pure orthogonal rotation Y = X Q (invariance control!)
    rng_q = np.random.default_rng(20261121)
    Q, _ = np.linalg.qr(rng_q.standard_normal((5, 5)))
    Y_rot = X_dom @ Q
    c_rot_y, lbl_rot = ward_kmeans(Y_rot, k=3, seed=20261121)
    c_rot_back = c_rot_y @ Q.T  # back to original domain coordinates
    d2_rot_back, phi_rot_back = arc_d2_phi(c_rot_back)

    # (c) Explicit ZCA whitening Z = X R^{-1/2}, evaluated in whitened space AND mapped back to domain space
    R_dom = np.cov(X_dom, rowvar=False, ddof=1)
    evals, evecs = np.linalg.eigh(R_dom)
    W_zca = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    W_inv = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    Z_white = X_dom @ W_zca
    c_w, lbl_w = ward_kmeans(Z_white, k=3, seed=20261121)
    d2_w_native, phi_w_native = arc_d2_phi(c_w)
    # Mapped back to original domain coordinates (exact means of X_dom per whitened cluster)
    c_w_back = np.vstack([X_dom[lbl_w == k].mean(axis=0) for k in range(3)])
    d2_w_back, phi_w_back = arc_d2_phi(c_w_back)

    # (d) Gerlach's FA+Varimax scores, evaluated in native space AND via cluster means in domain space
    X_fa = np.load("external/gerlach2018/data_filter/ipip300-no0_arr_pd_neoac_score-1.npy").astype(np.float64)
    c_fa, lbl_fa = ward_kmeans(X_fa, k=3, seed=20261121)
    d2_fa_native, phi_fa_native = arc_d2_phi(c_fa)
    c_fa_back = np.vstack([X_dom[lbl_fa == k].mean(axis=0) for k in range(3)])
    d2_fa_back, phi_fa_back = arc_d2_phi(c_fa_back)

    rep_rows = [
        ("Domains (standardized)", d2_dom, phi_dom, d2_dom, phi_dom),
        ("Pure orthogonal rotation Y=XQ (control)", float("nan"), float("nan"), d2_rot_back, phi_rot_back),
        ("ZCA Whitening Z=X R^{-1/2}", d2_w_native, phi_w_native, d2_w_back, phi_w_back),
        ("FA + Varimax (Gerlach scores)", d2_fa_native, phi_fa_native, d2_fa_back, phi_fa_back),
    ]
    for label, d2n, phin, d2b, phib in rep_rows:
        print(f"  {label:40s} | Native: D2={d2n:.3f}, phi={phin:.3f} | Domain-coords: D2={d2b:.3f}, phi={phib:.3f}")

    out["representation_experiment_ipip300"] = [
        {"representation": r[0], "d2_native": r[1], "phi_native": r[2], "d2_domain_coords": r[3], "phi_domain_coords": r[4]}
        for r in rep_rows
    ]
    out["elapsed_sec"] = float(time.perf_counter() - t0)

    out_path = Path("results/gerlach_faithful/final_via_a_checks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nTot completat i desat a {out_path} ({time.perf_counter()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
