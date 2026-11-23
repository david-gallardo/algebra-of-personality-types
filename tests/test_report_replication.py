"""Replication tests against the control values in PROJECT_RULES.md."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from arc_study.matrices import (
    BASE_SEED,
    BUSSERI_ERB_2024,
    IDENTITY_5,
    TRAIT_ORDER,
    VAN_DER_LINDEN_2010,
    get_observed_matrices,
    is_positive_definite,
    validate_correlation_matrix,
)
from arc_study.theory import (
    collinear_between_variance,
    compute_cl,
    density_ratio,
    non_collinearity_threshold,
    spectral_analysis,
    triangular_sector_construction,
)

ANALYTIC_TOL = 1e-4


def test_observed_matrices_and_positive_definiteness() -> None:
    assert TRAIT_ORDER == ("N", "E", "O", "A", "C")

    mats = get_observed_matrices()
    assert set(mats.keys()) == {"vdl", "be"}
    for mat in mats.values():
        assert mat.shape == (5, 5)
        assert is_positive_definite(mat)
        validate_correlation_matrix(mat)

    non_pd = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ],
        dtype=np.float64,
    )
    assert not is_positive_definite(non_pd)
    with pytest.raises(ValueError, match="positive definite"):
        validate_correlation_matrix(non_pd)


def test_cl_and_non_collinearity_threshold() -> None:
    c_l = compute_cl()
    threshold = non_collinearity_threshold(c_l)

    assert c_l == pytest.approx(0.809826, abs=ANALYTIC_TOL)
    assert threshold == pytest.approx(0.50764, abs=ANALYTIC_TOL)


def test_van_der_linden_analytic_control_values() -> None:
    spec = spectral_analysis(VAN_DER_LINDEN_2010)
    col = collinear_between_variance(VAN_DER_LINDEN_2010)
    tri = triangular_sector_construction(VAN_DER_LINDEN_2010)
    ratio_origin = density_ratio(np.zeros(5), VAN_DER_LINDEN_2010)

    assert spec.lambda1 == pytest.approx(1.909595, abs=ANALYTIC_TOL)
    assert spec.lambda2 == pytest.approx(1.012154, abs=ANALYTIC_TOL)
    assert spec.ratio == pytest.approx(0.530036, abs=ANALYTIC_TOL)
    assert col == pytest.approx(1.546440, abs=ANALYTIC_TOL)
    assert tri.between_variance == pytest.approx(1.569411, abs=ANALYTIC_TOL)
    assert spec.det == pytest.approx(0.636633, abs=ANALYTIC_TOL)

    assert spec.ratio > non_collinearity_threshold()
    assert tri.between_variance > col

    # Exact identity f_R(0)/f_I(0) = det(R)^(-1/2) and report 3-decimal value 1.253
    assert ratio_origin == pytest.approx(spec.det ** -0.5, abs=ANALYTIC_TOL)
    assert ratio_origin == pytest.approx(1.253, abs=1e-3)


def test_busseri_erb_analytic_control_values() -> None:
    spec = spectral_analysis(BUSSERI_ERB_2024)
    col = collinear_between_variance(BUSSERI_ERB_2024)
    tri = triangular_sector_construction(BUSSERI_ERB_2024)

    assert spec.lambda1 == pytest.approx(1.952222, abs=ANALYTIC_TOL)
    assert spec.lambda2 == pytest.approx(0.947899, abs=ANALYTIC_TOL)
    assert spec.ratio == pytest.approx(0.485549, abs=ANALYTIC_TOL)
    assert col == pytest.approx(1.580960, abs=ANALYTIC_TOL)
    assert tri.between_variance == pytest.approx(1.557794, abs=ANALYTIC_TOL)
    assert spec.det == pytest.approx(0.628258, abs=ANALYTIC_TOL)

    assert spec.ratio < non_collinearity_threshold()
    assert tri.between_variance < col


def test_triangular_construction_geometry_and_phase_invariance() -> None:
    tri_0 = triangular_sector_construction(VAN_DER_LINDEN_2010, phase=0.0)
    tri_rot = triangular_sector_construction(VAN_DER_LINDEN_2010, phase=0.47)

    assert tri_0.centroids.shape == (3, 5)
    assert np.allclose(tri_0.centroids.mean(axis=0), 0.0, atol=1e-12)

    empirical_between_0 = float(np.mean(np.sum(tri_0.centroids**2, axis=1)))
    empirical_between_rot = float(np.mean(np.sum(tri_rot.centroids**2, axis=1)))

    assert empirical_between_0 == pytest.approx(tri_0.between_variance, abs=1e-10)
    assert empirical_between_rot == pytest.approx(tri_0.between_variance, abs=1e-10)


def test_density_ratio_identity_and_batch_consistency() -> None:
    rng = np.random.default_rng(BASE_SEED)
    samples = rng.standard_normal((12, 5))

    ratios_id = density_ratio(samples, IDENTITY_5)
    assert isinstance(ratios_id, np.ndarray)
    assert np.allclose(ratios_id, 1.0, atol=1e-12)

    ratios_batch = density_ratio(samples, VAN_DER_LINDEN_2010)
    assert isinstance(ratios_batch, np.ndarray)
    assert ratios_batch.shape == (12,)
    for i in range(samples.shape[0]):
        single = density_ratio(samples[i], VAN_DER_LINDEN_2010)
        assert single == pytest.approx(ratios_batch[i], rel=1e-12, abs=1e-12)


def _kmeans_between_variance(x: np.ndarray, seed: int) -> float:
    km = KMeans(n_clusters=3, n_init=30, random_state=seed)
    km.fit(x)
    centered = x - x.mean(axis=0, keepdims=True)
    total_var = float(np.mean(np.sum(centered**2, axis=1)))
    within_var = float(km.inertia_ / x.shape[0])
    return total_var - within_var


def _fit_gmm_bics(x: np.ndarray, seed: int) -> dict[str, float]:
    specs = [
        ("1-full", 1, "full"),
        ("1-diag", 1, "diag"),
        ("3-diag", 3, "diag"),
        ("3-full", 3, "full"),
    ]
    bics: dict[str, float] = {}
    for label, n_components, cov_type in specs:
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=cov_type,
            n_init=10,
            reg_covar=1e-6,
            random_state=seed,
        )
        gmm.fit(x)
        bics[label] = float(gmm.bic(x))
    return bics


def test_exploratory_kmeans_and_gmm_replication() -> None:
    n_samples = 10_000
    rng = np.random.default_rng(BASE_SEED)
    z = rng.standard_normal((n_samples, 5))

    x_i = z
    x_vdl = z @ np.linalg.cholesky(VAN_DER_LINDEN_2010).T
    x_be = z @ np.linalg.cholesky(BUSSERI_ERB_2024).T

    var_i = _kmeans_between_variance(x_i, BASE_SEED)
    var_vdl = _kmeans_between_variance(x_vdl, BASE_SEED)
    var_be = _kmeans_between_variance(x_be, BASE_SEED)

    assert var_i == pytest.approx(1.1149, abs=0.05)
    assert var_vdl == pytest.approx(1.6222, abs=0.05)
    assert var_be == pytest.approx(1.6226, abs=0.05)
    assert var_vdl > var_i
    assert var_be > var_i

    bic_vdl = _fit_gmm_bics(x_vdl, BASE_SEED)
    bic_be = _fit_gmm_bics(x_be, BASE_SEED)

    expected_vdl = {
        "1-full": 137531.0,
        "1-diag": 141853.4,
        "3-diag": 138275.7,
        "3-full": 137910.7,
    }
    expected_be = {
        "1-full": 137398.5,
        "1-diag": 141846.4,
        "3-diag": 138096.5,
        "3-full": 137778.5,
    }

    for key, target in expected_vdl.items():
        assert bic_vdl[key] == pytest.approx(target, abs=800.0)
    for key, target in expected_be.items():
        assert bic_be[key] == pytest.approx(target, abs=800.0)

    # Unimodal full Gaussian must beat 3-component and diagonal alternatives
    for bics in (bic_vdl, bic_be):
        assert bics["1-full"] < bics["3-full"] < bics["3-diag"] < bics["1-diag"]

    # Common random numbers preserve the exact log-determinant BIC offset
    expected_diff = n_samples * (
        np.linalg.slogdet(BUSSERI_ERB_2024)[1]
        - np.linalg.slogdet(VAN_DER_LINDEN_2010)[1]
    )
    assert (bic_be["1-full"] - bic_vdl["1-full"]) == pytest.approx(expected_diff, abs=1.0)
