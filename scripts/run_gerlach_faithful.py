#!/usr/bin/env python3
"""Faithful replication of Gerlach et al. (2018) (K=13, exact NN bandwidth, full-sample KDE, B=1000)
against four progressively stricter unimodal null models (Permutation, MVN, Gaussian copula, Student-t copula)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.special import gammaln
from scipy.stats import norm, rankdata, spearmanr, t as student_t
from sklearn.mixture import GaussianMixture


def exact_gaussian_kde(X: np.ndarray, centers: np.ndarray, bw: float, chunk_size: int = 50_000) -> np.ndarray:
    """Exact isotropic Gaussian KDE evaluated at centers over ALL rows of X (matches sklearn to ~1e-9)."""
    n, d = X.shape
    k = centers.shape[0]
    inv_2h2 = 0.5 / (bw * bw)
    c_norm2 = np.sum(centers * centers, axis=1)
    acc = np.zeros(k, dtype=np.float64)
    for start in range(0, n, chunk_size):
        xb = X[start : start + chunk_size]
        x_norm2 = np.sum(xb * xb, axis=1, keepdims=True)
        sq = np.maximum(0.0, x_norm2 + c_norm2[None, :] - 2.0 * (xb @ centers.T))
        acc += np.sum(np.exp(-sq * inv_2h2), axis=0)
    norm_const = float(n) * ((2.0 * np.pi * bw * bw) ** (0.5 * d))
    return acc / norm_const


def avg_nearest_neighbor_dist(X: np.ndarray, n_samples: int = 10_000, seed: int = 20261121) -> float:
    """Exact equivalent of Gerlach's avg_shortest_dist using cKDTree."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    m = min(n_samples, n)
    idx = rng.choice(n, size=m, replace=False)
    tree = cKDTree(X)
    dists, _ = tree.query(X[idx], k=2)
    return float(np.mean(dists[:, 1]))


def nearest_psd_corr(R: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    R_sym = 0.5 * (R + R.T)
    evals, evecs = np.linalg.eigh(R_sym)
    evals = np.maximum(evals, min_eig)
    R_psd = (evecs * evals) @ evecs.T
    inv_sd = 1.0 / np.sqrt(np.diag(R_psd))
    R_corr = R_psd * np.outer(inv_sd, inv_sd)
    np.fill_diagonal(R_corr, 1.0)
    return R_corr


def fit_copula_params(X: np.ndarray, seed: int = 20261121) -> dict:
    """Estimate latent correlation R_lat (from Spearman) and Student-t copula df nu via MLE."""
    n, d = X.shape
    rho_s, _ = spearmanr(X, axis=0)
    if d == 1:
        rho_s = np.array([[1.0]])
    R_lat = nearest_psd_corr(2.0 * np.sin((np.pi / 6.0) * rho_s))

    rng = np.random.default_rng(seed)
    m = min(25_000, n)
    sub_idx = rng.choice(n, size=m, replace=False)
    U_sub = np.column_stack([(rankdata(X[sub_idx, j]) - 0.5) / float(m) for j in range(d)])

    R_inv = np.linalg.inv(R_lat)
    _, logdet = np.linalg.slogdet(R_lat)

    Z_sub = norm.ppf(U_sub)
    q_gauss = np.einsum("ij,jk,ik->i", Z_sub, R_inv - np.eye(d), Z_sub)
    ll_gauss = float(np.mean(-0.5 * logdet - 0.5 * q_gauss))

    def neg_ll_t(log_nu: float) -> float:
        nu = float(np.exp(log_nu))
        Y = student_t.ppf(U_sub, df=nu)
        q = np.einsum("ij,jk,ik->i", Y, R_inv, Y)
        ll_mv = (
            gammaln(0.5 * (nu + d))
            - gammaln(0.5 * nu)
            - 0.5 * d * np.log(nu * np.pi)
            - 0.5 * logdet
            - 0.5 * (nu + d) * np.log1p(q / nu)
        )
        ll_uni = np.sum(student_t.logpdf(Y, df=nu), axis=1)
        return -float(np.mean(ll_mv - ll_uni))

    res = minimize_scalar(neg_ll_t, bounds=(np.log(2.5), np.log(300.0)), method="bounded")
    nu_hat = float(np.exp(res.x))
    ll_t = float(-res.fun)
    return {
        "R_lat": R_lat,
        "nu_hat": nu_hat,
        "ll_gauss_copula": ll_gauss,
        "ll_t_copula": ll_t,
        "delta_ll_t_vs_gauss": ll_t - ll_gauss,
    }


def _null_batch_worker(
    null_kind: str,
    n_reps_batch: int,
    seed: int,
    X: np.ndarray,
    centers: np.ndarray,
    bw: float,
    mu: np.ndarray,
    L_cov: np.ndarray,
    L_lat: np.ndarray,
    q_grid: np.ndarray,
    emp_q: np.ndarray,
    nu_hat: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    k = centers.shape[0]
    out = np.empty((n_reps_batch, k), dtype=np.float64)

    for b in range(n_reps_batch):
        if null_kind == "permutation":
            X_sim = np.empty_like(X)
            for j in range(d):
                X_sim[:, j] = rng.permutation(X[:, j])
        elif null_kind == "mvnormal":
            X_sim = mu[None, :] + rng.standard_normal((n, d)) @ L_cov.T
        elif null_kind == "gaussian_copula":
            Z = rng.standard_normal((n, d)) @ L_lat.T
            U = norm.cdf(Z)
            X_sim = np.empty_like(Z)
            for j in range(d):
                X_sim[:, j] = np.interp(U[:, j], q_grid, emp_q[:, j])
        elif null_kind == "t_copula":
            chi = rng.chisquare(nu_hat, size=n)
            scale = np.sqrt(chi / nu_hat)[:, None]
            T = (rng.standard_normal((n, d)) @ L_lat.T) / scale
            U = student_t.cdf(T, df=nu_hat)
            X_sim = np.empty_like(T)
            for j in range(d):
                X_sim[:, j] = np.interp(U[:, j], q_grid, emp_q[:, j])
        else:
            raise ValueError(f"Unknown null_kind: {null_kind}")

        out[b, :] = exact_gaussian_kde(X_sim, centers, bw)
    return out


def run_faithful_dataset(
    name: str,
    X: np.ndarray,
    *,
    k_clusters: int = 13,
    n_gmm_reps: int = 10,
    n_null_reps: int = 1000,
    n_jobs: int = -2,
    seed: int = 20261121,
) -> dict:
    t0 = time.perf_counter()
    n, d = X.shape
    print(f"\n==================== {name} (N={n:,}, K={k_clusters}, B={n_null_reps}) ====================")

    best_lb = -np.inf
    best_gmm = None
    for r in range(n_gmm_reps):
        gmm = GaussianMixture(
            n_components=k_clusters,
            covariance_type="full",
            n_init=1,
            max_iter=300,
            reg_covar=1e-6,
            random_state=seed + r,
        )
        gmm.fit(X)
        if gmm.lower_bound_ > best_lb:
            best_lb = float(gmm.lower_bound_)
            best_gmm = gmm

    centers = best_gmm.means_.astype(np.float64)
    weights = best_gmm.weights_.astype(np.float64)
    t_gmm = time.perf_counter()
    print(f"[GMM] Done in {t_gmm - t0:.1f}s | best lower_bound={best_lb:.5f} | BIC={best_gmm.bic(X):,.1f}")

    bw = avg_nearest_neighbor_dist(X, n_samples=10_000, seed=seed)
    rho_emp = exact_gaussian_kde(X, centers, bw)
    print(f"[KDE] NN bandwidth bw={bw:.5f} | min/max emp density={rho_emp.min():.3e} / {rho_emp.max():.3e}")

    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False, ddof=1)
    L_cov = np.linalg.cholesky(cov)
    cop = fit_copula_params(X, seed=seed)
    L_lat = np.linalg.cholesky(cop["R_lat"])
    q_grid = np.linspace(0.0, 1.0, 20_001)
    emp_q = np.quantile(X, q_grid, axis=0)
    print(
        f"[COPULA] nu_hat={cop['nu_hat']:.2f} | ll_gauss={cop['ll_gauss_copula']:.5f} | "
        f"ll_t={cop['ll_t_copula']:.5f} | delta_ll={cop['delta_ll_t_vs_gauss']:+.5f}"
    )

    null_kinds = ["permutation", "mvnormal", "gaussian_copula", "t_copula"]
    n_batches = 20
    base_batch = n_null_reps // n_batches
    batch_sizes = [base_batch + (1 if i < (n_null_reps % n_batches) else 0) for i in range(n_batches)]
    batch_sizes = [b for b in batch_sizes if b > 0]

    evals = {}
    for nk_idx, nk in enumerate(null_kinds):
        tb0 = time.perf_counter()
        seeds = [seed + 10_000 * (nk_idx + 1) + i for i in range(len(batch_sizes))]
        blocks = Parallel(n_jobs=n_jobs)(
            delayed(_null_batch_worker)(
                nk, bs, s, X, centers, bw, mu, L_cov, L_lat, q_grid, emp_q, cop["nu_hat"]
            )
            for bs, s in zip(batch_sizes, seeds)
        )
        rho_null = np.vstack(blocks)

        rho_rel = rho_emp[None, :] / np.maximum(rho_null, 1e-300)
        enr_gerlach = np.mean(rho_rel, axis=0)
        enr_ratio_means = rho_emp / np.mean(rho_null, axis=0)
        pval_gerlach = np.sum(rho_rel < 1.0, axis=0) / float(n_null_reps)
        pval_gerlach = np.where(pval_gerlach == 0.0, 1.0 / float(n_null_reps), pval_gerlach)
        pval_phipson = (1.0 + np.sum(rho_null >= rho_emp[None, :], axis=0)) / float(n_null_reps + 1)

        sig_gerlach = (enr_gerlach > 1.25) & (pval_gerlach < 0.01)
        sig_phipson = (enr_gerlach > 1.25) & (pval_phipson < 0.01)
        evals[nk] = {
            "enrichment_gerlach": enr_gerlach.tolist(),
            "enrichment_ratio_means": enr_ratio_means.tolist(),
            "pval_gerlach": pval_gerlach.tolist(),
            "pval_phipson": pval_phipson.tolist(),
            "n_sig_gerlach": int(np.sum(sig_gerlach)),
            "n_sig_phipson": int(np.sum(sig_phipson)),
        }
        print(
            f"[NULL] {nk:16s} | {n_null_reps} reps in {time.perf_counter() - tb0:.1f}s | "
            f"sig(e>1.25, p<.01)={int(np.sum(sig_gerlach))}/{k_clusters} (max_e={enr_gerlach.max():.3f})"
        )

    order = np.argsort(evals["permutation"]["enrichment_gerlach"])[::-1]
    print("\n  Cl  w      N      E      O      A      C  |  Perm(e,p)      MVN(e,p)       GCop(e,p)      TCop(e,p)")
    print("  " + "-" * 104)
    for rank_i, idx in enumerate(order, start=1):
        c = centers[idx]
        cols = []
        for nk in null_kinds:
            e = evals[nk]["enrichment_gerlach"][idx]
            p = evals[nk]["pval_gerlach"][idx]
            star = "*" if (e > 1.25 and p < 0.01) else " "
            cols.append(f"{e:5.2f}({p:.3f}){star}")
        print(
            f"  {rank_i:2d} {weights[idx]:.3f} {c[0]:+5.2f} {c[1]:+5.2f} {c[2]:+5.2f} {c[3]:+5.2f} {c[4]:+5.2f}  |  "
            + "  ".join(cols)
        )

    return {
        "dataset": name,
        "n": int(n),
        "k_clusters": int(k_clusters),
        "n_null_reps": int(n_null_reps),
        "bw_nn": float(bw),
        "gmm_lower_bound": float(best_lb),
        "gmm_bic": float(best_gmm.bic(X)),
        "copula": {
            "nu_hat": float(cop["nu_hat"]),
            "ll_gauss_copula": float(cop["ll_gauss_copula"]),
            "ll_t_copula": float(cop["ll_t_copula"]),
            "delta_ll_t_vs_gauss": float(cop["delta_ll_t_vs_gauss"]),
            "R_lat": cop["R_lat"].tolist(),
        },
        "centers_neoac": centers.tolist(),
        "weights": weights.tolist(),
        "rho_emp": rho_emp.tolist(),
        "evaluations": evals,
        "elapsed_sec": float(time.perf_counter() - t0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=13)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=-2)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--out", type=Path, default=Path("results/gerlach_faithful/ipip300_faithful_k13.json"))
    args = parser.parse_args()

    results = {}

    X_gerlach = np.load("external/gerlach2018/data_filter/ipip300-no0_arr_pd_neoac_score-1.npy").astype(np.float64)
    results["gerlach_ipip300_fa_varimax"] = run_faithful_dataset(
        "gerlach_ipip300_fa_varimax",
        X_gerlach,
        k_clusters=args.k,
        n_null_reps=args.reps,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    df_dom = pd.read_csv("data/processed/ipip300_domains_complete.csv")[["N", "E", "O", "A", "C"]]
    X_dom = df_dom.to_numpy(dtype=np.float64)
    X_dom = (X_dom - X_dom.mean(axis=0)) / X_dom.std(axis=0, ddof=1)
    results["ipip300_domains_z"] = run_faithful_dataset(
        "ipip300_domains_z",
        X_dom,
        k_clusters=args.k,
        n_null_reps=args.reps,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResultats desats a {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
