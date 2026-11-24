"""Null data generators (I0, I1, C0, C1) and synthetic mixture positive-control generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from arc_study.generators import (
    _apply_gaussian_copula,
    calibrate_gaussian_copula,
    empirical_marginal_ppfs,
    nearest_pd_correlation,
    sample_null1_column_permutation,
    sample_null2_correlated_normal,
)

NullCode = Literal["I0", "I1", "C0", "C1"]


@dataclass(frozen=True)
class CalibratedCopulaSpec:
    """Pre-calibrated empirical-marginal Gaussian copula (C1) specification."""

    latent_corr: NDArray[np.float64]
    target_corr: NDArray[np.float64]
    achieved_corr: NDArray[np.float64]
    max_abs_pearson_error: float
    converged: bool


def calibrate_empirical_copula(
    x_emp: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    n_cal: int = 40_000,
    tol: float = 1e-4,
) -> CalibratedCopulaSpec:
    """Calibrate Gaussian copula latent R to reproduce empirical Pearson correlations."""
    arr = np.asarray(x_emp, dtype=np.float64)
    r_emp = nearest_pd_correlation(np.corrcoef(arr, rowvar=False))
    ppfs = empirical_marginal_ppfs(arr)
    cal = calibrate_gaussian_copula(
        target_corr=r_emp,
        marginals=ppfs,
        rng=rng,
        n_cal=n_cal,
        tol=tol,
    )
    return CalibratedCopulaSpec(
        latent_corr=cal.latent_corr,
        target_corr=r_emp,
        achieved_corr=cal.achieved_corr,
        max_abs_pearson_error=cal.max_abs_error,
        converged=cal.converged,
    )


def generate_null_sample(
    null_code: NullCode,
    x_emp: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    copula_spec: CalibratedCopulaSpec | None = None,
    standardize: bool = True,
) -> NDArray[np.float64]:
    """Generate one synthetic sample matching ``x_emp`` shape under I0, I1, C0, or C1."""
    arr = np.asarray(x_emp, dtype=np.float64)
    n, p = arr.shape

    if null_code == "I0":
        out = rng.standard_normal((n, p))
    elif null_code == "I1":
        out = sample_null1_column_permutation(arr, rng)
    elif null_code == "C0":
        r_emp = nearest_pd_correlation(np.corrcoef(arr, rowvar=False))
        out = sample_null2_correlated_normal(n=n, corr_matrix=r_emp, rng=rng)
    elif null_code == "C1":
        if copula_spec is None:
            copula_spec = calibrate_empirical_copula(arr, rng)
        ppfs = empirical_marginal_ppfs(arr)
        z = rng.standard_normal((n, p))
        out = _apply_gaussian_copula(z, copula_spec.latent_corr, ppfs)
    else:
        raise ValueError(f"Unknown null_code '{null_code}'. Expected I0, I1, C0, or C1.")

    if standardize:
        mu = out.mean(axis=0, keepdims=True)
        sd = out.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd > 0.0, sd, 1.0)
        out = (out - mu) / sd
    return out


def generate_synthetic_mixture(
    n: int,
    d: int,
    k: int,
    separation_sd: float,
    rng: np.random.Generator,
    *,
    weights: NDArray[np.floating] | None = None,
    within_corr: float = 0.0,
    floor_quantile: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.intp], NDArray[np.float64]]:
    """Generate a genuine K-component Gaussian mixture on a regular simplex (positive control)."""
    if k < 2 or k > d + 1:
        raise ValueError(f"Requires 2 <= k <= d + 1, got k={k}, d={d}.")
    if weights is None:
        w = np.full(k, 1.0 / k, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / np.sum(w)

    # Regular simplex vertices in R^d with exact pairwise Euclidean distance = separation_sd
    verts = np.zeros((k, d), dtype=np.float64)
    for i in range(1, k):
        verts[i, i - 1] = 1.0
    # Regularize pairwise distances via Gram matrix
    gram = np.full((k, k), -1.0 / (k - 1.0), dtype=np.float64)
    np.fill_diagonal(gram, 1.0)
    eigvals, eigvecs = np.linalg.eigh(gram)
    pos = np.argsort(eigvals)[::-1][: k - 1]
    coords = eigvecs[:, pos] * np.sqrt(np.maximum(eigvals[pos], 0.0))
    # Pairwise distance of unit-radius regular simplex is sqrt(2 * k / (k - 1))
    current_dist = float(np.sqrt(2.0 * k / (k - 1.0)))
    centers = np.zeros((k, d), dtype=np.float64)
    centers[:, : k - 1] = coords * (separation_sd / current_dist)

    cov = np.full((d, d), within_corr, dtype=np.float64)
    np.fill_diagonal(cov, 1.0)
    chol = np.linalg.cholesky(cov)

    labels = rng.choice(k, size=n, p=w).astype(np.intp)
    noise = rng.standard_normal((n, d)) @ chol.T
    x = centers[labels] + noise

    if floor_quantile > 0.0:
        thresh = norm.ppf(floor_quantile)
        x = np.maximum(x, thresh)

    return x, labels, centers
