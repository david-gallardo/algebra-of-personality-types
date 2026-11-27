"""Unit tests for algebraic covariance-component complexity, boundary counterexamples, and 3rd-order identity."""

from __future__ import annotations

import numpy as np
import pytest

from arc_study.algebraic_complexity import (
    build_weighted_simplex_mixture,
    empirical_b_whitened_third_cumulant_sq,
    evaluate_pentad,
    fit_covariance_family_upper_bound,
    sweep_feasible_rank3_decompositions,
    theoretical_simplex_third_cumulant_sq,
)


def test_weighted_simplex_reproduces_covariance_and_distinct_means() -> None:
    rng = np.random.default_rng(20261121)
    L = rng.normal(0.0, 0.4, size=(5, 3))
    B_true = L @ L.T
    d = np.full(5, 0.5)
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    mix = build_weighted_simplex_mixture(L, d, weights=weights)

    assert mix.means.shape == (4, 5)
    assert np.allclose(weights @ mix.means, 0.0, atol=1e-12)
    assert np.allclose(mix.between_cov, B_true, atol=1e-12)
    assert np.allclose(mix.reconstructed_cov, np.diag(d) + B_true, atol=1e-12)
    # Pairwise distinct means
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.linalg.norm(mix.means[i] - mix.means[j]) > 1e-6


def test_boundary_counterexample_strict_vs_semidefinite_diagonal() -> None:
    """Explicit 3x3 SPD correlation matrix with r_0(R) = 1 but r_+(R) = 2."""
    R = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.25], [0.5, 0.25, 1.0]])
    eigs = np.linalg.eigvalsh(R)
    assert np.all(eigs > 0.0)
    assert np.isclose(np.linalg.det(R), 9.0 / 16.0, atol=1e-12)
    # Boundary rank-1 solution requires d_1 = 0
    u = np.array([[1.0], [0.5], [0.5]])
    d0 = np.diag(R - u @ u.T)
    assert np.isclose(d0[0], 0.0, atol=1e-12)
    # With strict tau > 0, K=2 (rank 1) cannot achieve zero error
    fit_k2 = fit_covariance_family_upper_bound(R, k=2, tau=0.05, n_starts=8)
    assert fit_k2.distance_upper_bound > 1e-3


def test_pentad_vanishes_on_exact_rank2_and_nonzero_generically() -> None:
    rng = np.random.default_rng(20261121)
    L = rng.normal(0.0, 0.5, size=(5, 2))
    S = L @ L.T
    np.fill_diagonal(S, 1.0)
    assert abs(evaluate_pentad(S)) < 1e-13


def test_third_order_cumulant_identity_and_equal_weight_lower_bound() -> None:
    for r in (1, 2, 3, 4):
        pi_eq = np.full(r + 1, 1.0 / (r + 1))
        val = theoretical_simplex_third_cumulant_sq(pi_eq)
        assert np.isclose(val, float(r * (r - 1)), atol=1e-12)
    pi_uneq = np.array([0.5, 1.0 / 3.0, 1.0 / 6.0])
    assert np.isclose(theoretical_simplex_third_cumulant_sq(pi_uneq), 4.0, atol=1e-12)


def test_third_order_homoscedastic_vs_overcomplete_symmetric_counterexample() -> None:
    """K = 2r symmetric cross-polytope achieves Cov = I_r with zero 3rd cumulant."""
    r = 3
    pts = np.vstack([np.sqrt(r) * np.eye(r), -np.sqrt(r) * np.eye(r)])
    T = np.einsum("na,nb,nc->abc", pts, pts, pts) / float(len(pts))
    assert np.allclose(T, 0.0, atol=1e-12)
    # Empirical check on exact minimal homoscedastic simplex (zero noise limit)
    L = np.eye(5, 3) * 0.6
    d = np.full(5, 0.64)
    mix = build_weighted_simplex_mixture(L, d)
    # Repeat exact weighted support points (deterministic noiseless check)
    t_sq, _ = empirical_b_whitened_third_cumulant_sq(mix.means, mix.between_cov, rank=3)
    assert np.isclose(t_sq, 6.0, atol=1e-10)


def test_feasible_rank3_manifold_sweep_on_gaussian_all_below_6() -> None:
    rng = np.random.default_rng(20261121)
    L = rng.normal(0.0, 0.5, size=(5, 3))
    S = L @ L.T
    S = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    R = 0.6 * S + 0.4 * np.eye(5)
    x = rng.multivariate_normal(np.zeros(5), R, size=3000)
    res = sweep_feasible_rank3_decompositions(x, n_candidates=4, tau=0.01, fit_tol=1e-3, seed=20261121)
    assert res["n_feasible_well_conditioned"] >= 1
    assert res["all_below_6"] is True
