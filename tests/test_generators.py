"""Tests for Null 1-3 generators and between-study SPD correlation heterogeneity."""

from __future__ import annotations

import numpy as np
import pytest

from arc_study.generators import (
    check_joint_unimodality,
    default_skewed_marginals,
    nearest_pd_correlation,
    sample_null1_column_permutation,
    sample_null1_independent_normal,
    sample_null2_correlated_normal,
    sample_null3_gaussian_copula,
    sample_study_correlation_matrices,
    sample_study_correlation_matrix,
)
from arc_study.matrices import (
    BASE_SEED,
    BUSSERI_ERB_2024,
    VAN_DER_LINDEN_2010,
    is_positive_definite,
    validate_correlation_matrix,
)


def test_null1_independent_normal_reproducibility_and_moments() -> None:
    x1 = sample_null1_independent_normal(5000, np.random.default_rng(BASE_SEED))
    x2 = sample_null1_independent_normal(5000, np.random.default_rng(BASE_SEED))

    assert x1.shape == (5000, 5)
    assert np.array_equal(x1, x2)
    assert np.allclose(x1.mean(axis=0), 0.0, atol=0.06)
    assert np.allclose(np.corrcoef(x1, rowvar=False), np.eye(5), atol=0.06)


def test_null1_column_permutation_preserves_quantiles_and_breaks_correlation() -> None:
    rng = np.random.default_rng(BASE_SEED)
    base = sample_null2_correlated_normal(5000, VAN_DER_LINDEN_2010, rng)
    # Apply nonlinear asymmetric distortion to verify non-normal empirical marginals
    base[:, 0] = np.exp(0.3 * base[:, 0])
    base[:, 4] = -np.exp(0.25 * base[:, 4])

    permuted = sample_null1_column_permutation(base, rng)
    assert permuted.shape == base.shape

    probs = np.linspace(0.05, 0.95, 19)
    for j in range(base.shape[1]):
        q_orig = np.quantile(base[:, j], probs)
        q_perm = np.quantile(permuted[:, j], probs)
        assert np.allclose(q_perm, q_orig, atol=1e-12)

    corr_perm = np.corrcoef(permuted, rowvar=False)
    off_diag = corr_perm[~np.eye(5, dtype=bool)]
    assert np.max(np.abs(off_diag)) < 0.06


def test_null2_correlated_normal_recovers_observed_matrices() -> None:
    for r_target in (VAN_DER_LINDEN_2010, BUSSERI_ERB_2024):
        rng = np.random.default_rng(BASE_SEED)
        sample = sample_null2_correlated_normal(20_000, r_target, rng)
        assert sample.shape == (20_000, 5)
        r_emp = np.corrcoef(sample, rowvar=False)
        assert np.max(np.abs(r_emp - r_target)) < 0.025


def test_nearest_pd_correlation_projection() -> None:
    non_pd = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ],
        dtype=np.float64,
    )
    assert not is_positive_definite(non_pd)

    projected = nearest_pd_correlation(non_pd)
    validate_correlation_matrix(projected)
    assert is_positive_definite(projected)

    unchanged = nearest_pd_correlation(VAN_DER_LINDEN_2010)
    assert np.allclose(unchanged, VAN_DER_LINDEN_2010, atol=1e-8)


@pytest.mark.parametrize("r_target", [VAN_DER_LINDEN_2010, BUSSERI_ERB_2024])
def test_null3_copula_pearson_recovery_and_unimodality(r_target: np.ndarray) -> None:
    rng = np.random.default_rng(BASE_SEED)
    result = sample_null3_gaussian_copula(
        n=30_000,
        target_corr=r_target,
        rng=rng,
        marginals=default_skewed_marginals(),
        calibration_n=50_000,
        calibration_tol=1e-4,
        check_unimodality=True,
    )

    # Latent correlation matrix must be strictly SPD
    validate_correlation_matrix(result.calibration.latent_corr)
    assert is_positive_definite(result.calibration.latent_corr)

    # Calibrated copula must recover target Pearson correlations within 0.005
    assert result.calibration.converged
    assert result.calibration.max_abs_error <= 0.005
    assert np.max(np.abs(result.calibration.achieved_corr - r_target)) <= 0.005

    # Out-of-sample realization must also closely match target correlations
    r_sample = np.corrcoef(result.data, rowvar=False)
    assert np.max(np.abs(r_sample - r_target)) < 0.02

    # Joint unimodality diagnostic must confirm unimodality
    assert result.unimodality is not None
    assert result.unimodality.is_unimodal


def test_null3_copula_empirical_marginals_preserve_quantiles_and_correlations() -> None:
    rng = np.random.default_rng(BASE_SEED)
    marginals = default_skewed_marginals()
    u_ref = rng.uniform(1e-6, 1.0 - 1e-6, size=(20_000, 5))
    empirical = np.column_stack([ppf(u_ref[:, j]) for j, ppf in enumerate(marginals)])

    result = sample_null3_gaussian_copula(
        n=30_000,
        target_corr=VAN_DER_LINDEN_2010,
        rng=rng,
        empirical_data=empirical,
        calibration_n=50_000,
        calibration_tol=1e-4,
        check_unimodality=True,
    )

    assert result.calibration.converged
    assert result.calibration.max_abs_error <= 0.005
    assert result.unimodality is not None
    assert result.unimodality.is_unimodal

    probs = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    for j in range(5):
        q_ref = np.quantile(empirical[:, j], probs)
        q_sim = np.quantile(result.data[:, j], probs)
        assert np.allclose(q_sim, q_ref, atol=0.05)


def test_unimodality_diagnostic_detects_bimodal_mixture() -> None:
    rng = np.random.default_rng(BASE_SEED)
    c1 = rng.standard_normal((1000, 5)) - np.array([2.5, 0.0, 0.0, 0.0, 0.0])
    c2 = rng.standard_normal((1000, 5)) + np.array([2.5, 0.0, 0.0, 0.0, 0.0])
    bimodal = np.vstack([c1, c2])

    diag = check_joint_unimodality(bimodal, rng=rng)
    assert not diag.is_unimodal
    assert diag.marginal_modes[0] >= 2 or diag.pc_modes[0] >= 2


@pytest.mark.parametrize("method", ["invwishart", "wishart"])
def test_study_heterogeneity_generates_spd_correlation_matrices(method: str) -> None:
    rng = np.random.default_rng(BASE_SEED)
    corrs = sample_study_correlation_matrices(
        r_meta=VAN_DER_LINDEN_2010,
        n_studies=40,
        df=60.0,
        rng=rng,
        method=method,
    )

    assert corrs.shape == (40, 5, 5)
    for i in range(corrs.shape[0]):
        validate_correlation_matrix(corrs[i])
        assert is_positive_definite(corrs[i])

    mean_corr = corrs.mean(axis=0)
    assert np.allclose(mean_corr, VAN_DER_LINDEN_2010, atol=0.08)

    off_diag_std = corrs.std(axis=0)[~np.eye(5, dtype=bool)]
    assert np.all(off_diag_std > 0.01)

    single = sample_study_correlation_matrix(
        r_meta=BUSSERI_ERB_2024,
        df=60.0,
        rng=rng,
        method=method,
    )
    validate_correlation_matrix(single)


def test_study_heterogeneity_invalid_df_raises() -> None:
    rng = np.random.default_rng(BASE_SEED)
    with pytest.raises(ValueError, match="df > p \\+ 1"):
        sample_study_correlation_matrices(VAN_DER_LINDEN_2010, 5, df=6.0, rng=rng, method="invwishart")
    with pytest.raises(ValueError, match="df > p - 1"):
        sample_study_correlation_matrices(VAN_DER_LINDEN_2010, 5, df=4.0, rng=rng, method="wishart")
