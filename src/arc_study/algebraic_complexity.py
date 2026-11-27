"""Algebraic covariance-component complexity, pentad certificates, and 3rd-order diagnostics.

Implements:
  - Constructive weighted-simplex equality K^(2)_diag(R) = 1 + r_+(R)
  - Exact 12-term pentad polynomial P(R) for p = 5
  - Sampling-aware distance upper bound d_K(R; tau) via multistart optimization
  - Joint uncertainty via whole-respondent bootstrap (individual data) or
    Wishart correlation sampling (meta-analytic matrices; SPD-preserving)
  - Homoscedastic minimal-representation 3rd-order cumulant diagnostic ||T||_F^2
    with joint propagation of (D_hat, B_hat, W_hat) estimation uncertainty

Strict trait order for Big Five matrices: ('N', 'E', 'O', 'A', 'C') with N unreversed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

# 12 Hamiltonian 5-cycles of K_5 with alternating signs (Drton, Sturmfels & Sullivant, 2007)
PENTAD_TERMS: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] = (
    (1, ((0, 1), (0, 2), (1, 3), (2, 4), (3, 4))),
    (-1, ((0, 1), (0, 2), (1, 4), (2, 3), (3, 4))),
    (-1, ((0, 1), (0, 3), (1, 2), (2, 4), (3, 4))),
    (1, ((0, 1), (0, 3), (1, 4), (2, 3), (2, 4))),
    (1, ((0, 1), (0, 4), (1, 2), (2, 3), (3, 4))),
    (-1, ((0, 1), (0, 4), (1, 3), (2, 3), (2, 4))),
    (1, ((0, 2), (0, 3), (1, 2), (1, 4), (3, 4))),
    (-1, ((0, 2), (0, 3), (1, 3), (1, 4), (2, 4))),
    (-1, ((0, 2), (0, 4), (1, 2), (1, 3), (3, 4))),
    (1, ((0, 2), (0, 4), (1, 3), (1, 4), (2, 3))),
    (-1, ((0, 3), (0, 4), (1, 2), (1, 4), (2, 3))),
    (1, ((0, 3), (0, 4), (1, 2), (1, 3), (2, 4))),
)


@dataclass(frozen=True)
class SimplexConstruction:
    """Weighted-simplex diagonal Gaussian mixture matching R = D + L L^T."""

    means: NDArray[np.float64]  # shape (K, p)
    weights: NDArray[np.float64]  # shape (K,)
    common_diag: NDArray[np.float64]  # shape (p,)
    between_cov: NDArray[np.float64]  # shape (p, p)
    reconstructed_cov: NDArray[np.float64]  # shape (p, p)


@dataclass(frozen=True)
class RankFitResult:
    """Upper-bound fit to the closed compact family C_{K, tau}."""

    k: int
    rank: int
    tau: float
    distance_upper_bound: float  # ||v(R - S)||_2 (upper bound on true global minimum d_K)
    diag_d: NDArray[np.float64]
    loadings: NDArray[np.float64]
    between_cov: NDArray[np.float64]
    min_positive_eig_b: float


def upper_triangle_vec(mat: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return strict upper-triangle entries v(M) in row-major order."""
    arr = np.asarray(mat, dtype=np.float64)
    idx = np.triu_indices(arr.shape[0], k=1)
    return arr[idx]


def evaluate_pentad(corr: NDArray[np.floating]) -> float:
    """Evaluate the 12-term degree-5 pentad polynomial P(R) for a 5x5 symmetric matrix."""
    arr = np.asarray(corr, dtype=np.float64)
    if arr.shape != (5, 5):
        raise ValueError(f"Pentad requires a 5x5 matrix, got {arr.shape}.")
    total = 0.0
    for sign, edges in PENTAD_TERMS:
        prod = float(sign)
        for i, j in edges:
            prod *= float(arr[i, j])
        total += prod
    return float(total)


def evaluate_tetrads(corr: NDArray[np.floating]) -> list[float]:
    """Evaluate all 3 * C(p, 4) tetrads r_ij*r_kl - r_ik*r_jl for distinct indices."""
    arr = np.asarray(corr, dtype=np.float64)
    p = arr.shape[0]
    vals: list[float] = []
    for i, j, k, l in combinations(range(p), 4):
        vals.append(float(arr[i, j] * arr[k, l] - arr[i, k] * arr[j, l]))
        vals.append(float(arr[i, j] * arr[k, l] - arr[i, l] * arr[j, k]))
        vals.append(float(arr[i, k] * arr[j, l] - arr[i, l] * arr[j, k]))
    return vals


def build_weighted_simplex_mixture(
    loadings: NDArray[np.floating],
    diag_d: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> SimplexConstruction:
    """Construct a K = r + 1 common-diagonal Gaussian mixture with exact Cov(X) = D + L L^T."""
    L = np.asarray(loadings, dtype=np.float64)
    d = np.asarray(diag_d, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError("loadings must be 2D of shape (p, r).")
    p, r = L.shape
    if d.shape != (p,) or np.any(d <= 0.0):
        raise ValueError("diag_d must be strictly positive of shape (p,).")
    k = r + 1
    if weights is None:
        pi = np.full(k, 1.0 / k, dtype=np.float64)
    else:
        pi = np.asarray(weights, dtype=np.float64)
        if pi.shape != (k,) or np.any(pi <= 0.0) or not np.isclose(np.sum(pi), 1.0, atol=1e-12):
            raise ValueError("weights must be strictly positive of length r+1 summing to 1.")

    if r == 0:
        means = np.zeros((1, p), dtype=np.float64)
        B = np.zeros((p, p), dtype=np.float64)
        return SimplexConstruction(means=means, weights=pi, common_diag=d, between_cov=B, reconstructed_cov=np.diag(d))

    s = np.sqrt(pi)
    # Build orthonormal basis Q (r x K) for s^perp via QR on [s, I_K]
    basis = np.column_stack([s, np.eye(k, dtype=np.float64)])
    q_full, _ = np.linalg.qr(basis, mode="reduced")
    # First column is +/- s; remaining r columns span s^perp
    Q = q_full[:, 1 : r + 1].T  # shape (r, K), Q Q^T = I_r, Q s = 0
    Z = Q / s[None, :]  # columns z_k = q_k / sqrt(pi_k), shape (r, K)
    means = (L @ Z).T  # shape (K, p)
    B = (means.T * pi[None, :]) @ means
    R_rec = np.diag(d) + B
    return SimplexConstruction(means=means, weights=pi, common_diag=d, between_cov=B, reconstructed_cov=R_rec)


def theoretical_simplex_third_cumulant_sq(weights: NDArray[np.floating]) -> float:
    """Exact ||T||_F^2 = sum_k pi_k^{-1} - 3*K + 2 for minimal K = r + 1 homoscedastic simplex."""
    pi = np.asarray(weights, dtype=np.float64)
    if np.any(pi <= 0.0) or not np.isclose(np.sum(pi), 1.0, atol=1e-12):
        raise ValueError("weights must be strictly positive and sum to 1.")
    k = pi.shape[0]
    return float(np.sum(1.0 / pi) - 3.0 * k + 2.0)


def fit_covariance_family_upper_bound(
    corr: NDArray[np.floating],
    k: int,
    tau: float = 0.01,
    n_starts: int = 16,
    rng: np.random.Generator | None = None,
) -> RankFitResult:
    """Multistart nonconvex upper bound on d_K(R; tau) = min_{S in C_{K,tau}} ||v(R - S)||_2.

    Note: Because this optimization is nonconvex, the returned distance is a certified
    UPPER BOUND on d_K(R; tau). Failing to reach <= epsilon does NOT prove impossibility.
    """
    R = np.asarray(corr, dtype=np.float64)
    p = R.shape[0]
    if k < 1 or k > p:
        raise ValueError(f"k must be in [1, {p}], got {k}.")
    r = k - 1
    v_target = upper_triangle_vec(R)
    idx_u = np.triu_indices(p, k=1)

    if r == 0:
        dist = float(np.linalg.norm(v_target))
        return RankFitResult(
            k=1,
            rank=0,
            tau=tau,
            distance_upper_bound=dist,
            diag_d=np.ones(p, dtype=np.float64),
            loadings=np.zeros((p, 0), dtype=np.float64),
            between_cov=np.zeros((p, p), dtype=np.float64),
            min_positive_eig_b=0.0,
        )

    max_row_sq = 1.0 - tau
    local_rng = np.random.default_rng(20261121) if rng is None else rng

    def unpack_feasible_L(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        # Smooth row-norm parameterization: row_norm^2 <= 1 - tau via tanh scaling
        raw = theta.reshape(p, r)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        # Map norm in [0, inf) smoothly to [0, sqrt(1 - tau))
        target_norms = np.sqrt(max_row_sq) * np.tanh(norms)
        scale = np.where(norms > 1e-12, target_norms / np.where(norms > 1e-12, norms, 1.0), 0.0)
        return raw * scale

    def objective(theta: NDArray[np.float64]) -> float:
        L = unpack_feasible_L(theta)
        S_off = (L @ L.T)[idx_u]
        diff = S_off - v_target
        return float(np.dot(diff, diff))

    # Spectral initialization + random starts
    eigvals, eigvecs = np.linalg.eigh(0.5 * (R + R.T))
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    l_spec = eigvecs[:, :r] * np.sqrt(np.maximum(eigvals[:r] - tau, 1e-4))[None, :]
    starts = [l_spec.ravel()]
    for _ in range(max(0, n_starts - 1)):
        starts.append(local_rng.normal(0.0, 0.5, size=p * r))

    best_fun = float("inf")
    best_theta = starts[0]
    for s0 in starts:
        res = minimize(objective, s0, method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-14})
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_theta = res.x

    L_best = unpack_feasible_L(best_theta)
    B_best = L_best @ L_best.T
    d_best = np.maximum(1.0 - np.diag(B_best), tau)
    eigs_b = np.sort(np.linalg.eigvalsh(0.5 * (B_best + B_best.T)))[::-1]
    min_pos_eig = float(eigs_b[r - 1]) if r > 0 else 0.0
    return RankFitResult(
        k=k,
        rank=r,
        tau=tau,
        distance_upper_bound=float(np.sqrt(max(best_fun, 0.0))),
        diag_d=d_best,
        loadings=L_best,
        between_cov=B_best,
        min_positive_eig_b=min_pos_eig,
    )


def empirical_b_whitened_third_cumulant_sq(
    x: NDArray[np.floating],
    between_cov: NDArray[np.floating],
    rank: int,
    eig_floor: float = 1e-4,
) -> tuple[float, float]:
    """Compute empirical ||T_hat||_F^2 of W X where W B W^T = I_r.

    Returns (t_norm_sq, min_eig_used). Raises ValueError if B is numerically rank-deficient
    below ``eig_floor``.
    """
    arr_x = np.asarray(x, dtype=np.float64)
    arr_x = arr_x - arr_x.mean(axis=0, keepdims=True)
    B = 0.5 * (np.asarray(between_cov, dtype=np.float64) + np.asarray(between_cov, dtype=np.float64).T)
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    top_eigs = eigvals[:rank]
    min_eig = float(top_eigs[-1])
    if min_eig < eig_floor:
        raise ValueError(f"Ill-conditioned B whitening: smallest rank-{rank} eigenvalue {min_eig:.3e} < {eig_floor}.")
    W = (eigvecs[:, :rank] / np.sqrt(top_eigs)[None, :]).T  # shape (r, p), W B W^T = I_r
    Y = arr_x @ W.T  # shape (N, r), zero-mean
    # Empirical third central moment tensor T_{abc} = mean_n Y_{na} Y_{nb} Y_{nc}
    n = Y.shape[0]
    T = np.einsum("na,nb,nc->abc", Y, Y, Y, optimize=True) / float(n)
    return float(np.sum(T * T)), min_eig


def sweep_feasible_rank3_decompositions(
    x: NDArray[np.floating],
    n_candidates: int = 20,
    tau: float = 0.01,
    fit_tol: float = 1e-4,
    eig_floor: float = 1e-3,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Explore sensitivity of ||T_hat||_F^2 across multiple feasible K=4 (r=3) decompositions R = D + B.

    For p = 5, r = 3, feasible rank-3 decompositions form a 2D manifold. This diagnostic
    collects diverse local fits achieving ||v(R - S)||_2 <= fit_tol and reports the range
    of B-whitened ||T_hat||_F^2 alongside min eigenvalue of B.
    """
    arr = np.asarray(x, dtype=np.float64)
    z = (arr - arr.mean(axis=0)) / arr.std(axis=0, ddof=1)
    R_hat = np.corrcoef(z, rowvar=False)
    rng = np.random.default_rng(seed)
    records: list[dict[str, float]] = []
    for _ in range(n_candidates):
        fit = fit_covariance_family_upper_bound(R_hat, k=4, tau=tau, n_starts=2, rng=rng)
        if fit.distance_upper_bound <= fit_tol:
            try:
                t_sq, min_e = empirical_b_whitened_third_cumulant_sq(z, fit.between_cov, rank=3, eig_floor=eig_floor)
                records.append({
                    "distance_ub": fit.distance_upper_bound,
                    "min_eig_b": min_e,
                    "t_norm_sq": t_sq,
                })
            except ValueError:
                continue
    t_vals = [r["t_norm_sq"] for r in records]
    return {
        "n_candidates_tried": int(n_candidates),
        "n_feasible_well_conditioned": len(records),
        "t_norm_sq_min": float(min(t_vals)) if t_vals else None,
        "t_norm_sq_median": float(np.median(t_vals)) if t_vals else None,
        "t_norm_sq_max": float(max(t_vals)) if t_vals else None,
        "all_below_6": bool(all(v < 6.0 for v in t_vals)) if t_vals else None,
        "records": records,
    }


def bootstrap_individual_algebraic_audit(
    x: NDArray[np.floating],
    n_boot: int = 500,
    tau: float = 0.01,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Joint whole-respondent bootstrap for pentad, R confidence radius, d_K upper bounds, and 3rd cumulant."""
    arr = np.asarray(x, dtype=np.float64)
    n, p = arr.shape
    if p != 5:
        raise ValueError(f"Expected 5 columns, got {p}.")
    # Standardize within sample
    z = (arr - arr.mean(axis=0)) / arr.std(axis=0, ddof=1)
    R_hat = np.corrcoef(z, rowvar=False)
    v_hat = upper_triangle_vec(R_hat)
    pentad_hat = evaluate_pentad(R_hat)

    rng = np.random.default_rng(seed)
    boot_pentads = np.empty(n_boot, dtype=np.float64)
    boot_radii = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        zb = z[idx]
        Rb = np.corrcoef(zb, rowvar=False)
        boot_pentads[b] = evaluate_pentad(Rb)
        boot_radii[b] = float(np.linalg.norm(upper_triangle_vec(Rb) - v_hat))

    # Point upper bounds for K = 1..5
    fits = {k: fit_covariance_family_upper_bound(R_hat, k=k, tau=tau, n_starts=12, rng=rng) for k in range(1, 6)}
    eps95 = float(np.quantile(boot_radii, 0.95))

    # Propagate joint bootstrap uncertainty for K = 4 (r = 3) homoscedastic 3rd-order diagnostic
    t3_boot_vals: list[float] = []
    t3_skipped = 0
    for b in range(min(n_boot, 100)):
        idx = rng.integers(0, n, size=n)
        zb = z[idx]
        Rb = np.corrcoef(zb, rowvar=False)
        fb = fit_covariance_family_upper_bound(Rb, k=4, tau=tau, n_starts=4, rng=rng)
        try:
            t_sq, _ = empirical_b_whitened_third_cumulant_sq(zb, fb.between_cov, rank=3, eig_floor=1e-3)
            t3_boot_vals.append(t_sq)
        except ValueError:
            t3_skipped += 1

    try:
        t3_hat, min_eig_b = empirical_b_whitened_third_cumulant_sq(z, fits[4].between_cov, rank=3, eig_floor=1e-3)
    except ValueError:
        t3_hat, min_eig_b = float("nan"), fits[4].min_positive_eig_b

    return {
        "n": int(n),
        "n_boot": int(n_boot),
        "tau": float(tau),
        "pentad_point": pentad_hat,
        "pentad_boot_ci95": [float(np.quantile(boot_pentads, 0.025)), float(np.quantile(boot_pentads, 0.975))],
        "pentad_ci95_excludes_zero": bool(
            np.quantile(boot_pentads, 0.025) > 0.0 or np.quantile(boot_pentads, 0.975) < 0.0
        ),
        "r_joint_radius_95": eps95,
        "d_k_upper_bounds": {
            str(k): {
                "distance_upper_bound": fits[k].distance_upper_bound,
                "min_positive_eig_b": fits[k].min_positive_eig_b,
                "exceeds_eps95_note": "Upper bound only; cannot certify impossibility without global bound.",
            }
            for k in range(1, 6)
        },
        "homoscedastic_r3_third_cumulant": {
            "theoretical_lower_bound_r3": 6.0,
            "empirical_t_norm_sq_point": t3_hat,
            "min_eig_b_point": min_eig_b,
            "boot_valid_reps": len(t3_boot_vals),
            "boot_ill_conditioned_skipped": t3_skipped,
            "boot_ci95": (
                [float(np.quantile(t3_boot_vals, 0.025)), float(np.quantile(t3_boot_vals, 0.975))]
                if t3_boot_vals
                else None
            ),
        },
    }


def wishart_meta_analytic_audit(
    corr: NDArray[np.floating],
    n_eff: int,
    n_draws: int = 1000,
    tau: float = 0.01,
    seed: int = 20261121,
) -> dict[str, Any]:
    """SPD-preserving Wishart covariance sampling audit for aggregate meta-analytic matrices."""
    R = 0.5 * (np.asarray(corr, dtype=np.float64) + np.asarray(corr, dtype=np.float64).T)
    p = R.shape[0]
    if p != 5:
        raise ValueError(f"Expected 5x5 matrix, got {R.shape}.")
    v_hat = upper_triangle_vec(R)
    pentad_hat = evaluate_pentad(R)
    L_chol = np.linalg.cholesky(R)
    rng = np.random.default_rng(seed)

    # Bartlett decomposition for fast exact Wishart(df = n_eff - 1, R / df)
    df = max(n_eff - 1, p + 1)
    pentads = np.empty(n_draws, dtype=np.float64)
    radii = np.empty(n_draws, dtype=np.float64)
    for b in range(n_draws):
        A = np.zeros((p, p), dtype=np.float64)
        for i in range(p):
            A[i, i] = np.sqrt(rng.chisquare(df - i))
            for j in range(i):
                A[i, j] = rng.standard_normal()
        W_mat = (L_chol @ A) @ (L_chol @ A).T / float(df)
        inv_sd = 1.0 / np.sqrt(np.diag(W_mat))
        Rb = W_mat * np.outer(inv_sd, inv_sd)
        pentads[b] = evaluate_pentad(Rb)
        radii[b] = float(np.linalg.norm(upper_triangle_vec(Rb) - v_hat))

    fits = {k: fit_covariance_family_upper_bound(R, k=k, tau=tau, n_starts=12, rng=rng) for k in range(1, 6)}
    return {
        "n_eff": int(n_eff),
        "n_draws": int(n_draws),
        "tau": float(tau),
        "pentad_point_displayed": pentad_hat,
        "pentad_wishart_ci95": [float(np.quantile(pentads, 0.025)), float(np.quantile(pentads, 0.975))],
        "r_joint_radius_95": float(np.quantile(radii, 0.95)),
        "d_k_upper_bounds": {
            str(k): {
                "distance_upper_bound": fits[k].distance_upper_bound,
                "min_positive_eig_b": fits[k].min_positive_eig_b,
            }
            for k in range(1, 6)
        },
        "caveat": "Wishart sampling approximates joint sampling variability under MVN with effective N; rounded published matrices are not exact population parameters.",
    }
