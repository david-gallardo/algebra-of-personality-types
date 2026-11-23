"""Analytic spectral, geometric, and density results for Gaussian 3-means."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import norm

from arc_study.matrices import is_positive_definite

TRIANGULAR_SECTOR_CONSTANT: float = 27.0 / (16.0 * np.pi)


@dataclass(frozen=True)
class SpectralAnalysis:
    """Descending eigendecomposition and key spectral summaries of a covariance/correlation matrix."""

    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    lambda1: float
    lambda2: float
    ratio: float
    det: float


@dataclass(frozen=True)
class TriangularConstruction:
    """Analytic 3-centroid construction from equal 120-degree sectors in the principal plane."""

    centroids: NDArray[np.float64]
    angles: NDArray[np.float64]
    weights: NDArray[np.float64]
    between_variance: float


def spectral_analysis(matrix: NDArray[np.floating]) -> SpectralAnalysis:
    """Compute descending eigenvalues, eigenvectors, lambda2/lambda1 ratio, and determinant."""
    arr = np.asarray(matrix, dtype=np.float64)
    if not is_positive_definite(arr):
        raise ValueError("Matrix must be symmetric and strictly positive definite.")
    if arr.shape[0] < 2:
        raise ValueError("Matrix must be at least 2x2 for spectral ratio analysis.")

    sym = 0.5 * (arr + arr.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    lambda1 = float(eigvals[0])
    lambda2 = float(eigvals[1])
    return SpectralAnalysis(
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        lambda1=lambda1,
        lambda2=lambda2,
        ratio=lambda2 / lambda1,
        det=float(np.prod(eigvals)),
    )


def compute_cl() -> float:
    """Compute the exact 1D 3-means explained variance constant c_L for N(0, 1).

    Solves the symmetric Lloyd fixed-point equation 2 * t * (1 - Phi(t)) = phi(t)
    for the positive boundary t > 0, giving outer centroids +/-2t and
    c_L = 2 * (1 - Phi(t)) * (2t)^2 = 4 * t * phi(t) ~= 0.809826.
    """
    t_star = brentq(
        lambda t: 2.0 * t * norm.sf(t) - norm.pdf(t),
        0.1,
        2.0,
        xtol=1e-14,
        rtol=1e-14,
    )
    return float(4.0 * t_star * norm.pdf(t_star))


def non_collinearity_threshold(c_l: float | None = None) -> float:
    """Return the sufficient threshold on lambda2 / lambda1 for non-collinear 3-means.

    A triangular 120-degree sector partition beats the optimal collinear 1D 3-means
    partition whenever lambda2 / lambda1 > c_L / (27 / (16 * pi)) - 1 ~= 0.50764.
    """
    cl_val = compute_cl() if c_l is None else float(c_l)
    return float(cl_val / TRIANGULAR_SECTOR_CONSTANT - 1.0)


def collinear_between_variance(
    matrix: NDArray[np.floating],
    c_l: float | None = None,
) -> float:
    """Return the optimal collinear 3-means between-cluster variance c_L * lambda1."""
    cl_val = compute_cl() if c_l is None else float(c_l)
    spec = spectral_analysis(matrix)
    return float(cl_val * spec.lambda1)


def triangular_sector_construction(
    matrix: NDArray[np.floating],
    phase: float = 0.0,
) -> TriangularConstruction:
    """Construct the three analytic centroids from equal 120-degree sectors in the (v1, v2) plane.

    In standardized principal-plane coordinates (U1, U2) ~ N(0, I_2), partitioning into
    three 120-degree sectors centered at angles phase + 2*pi*k/3 yields centroids
    mu_k = (3*sqrt(3) / (2*sqrt(2*pi))) * (sqrt(lambda1)*cos(theta_k)*v1 + sqrt(lambda2)*sin(theta_k)*v2)
    with exact between-cluster variance (27 / (16 * pi)) * (lambda1 + lambda2),
    independent of ``phase``.
    """
    spec = spectral_analysis(matrix)
    v1 = spec.eigenvectors[:, 0]
    v2 = spec.eigenvectors[:, 1]

    angles = phase + np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=np.float64)
    radius_factor = 3.0 * np.sqrt(3.0) / (2.0 * np.sqrt(2.0 * np.pi))

    centroids = radius_factor * (
        np.outer(np.sqrt(spec.lambda1) * np.cos(angles), v1)
        + np.outer(np.sqrt(spec.lambda2) * np.sin(angles), v2)
    )
    between_var = float(TRIANGULAR_SECTOR_CONSTANT * (spec.lambda1 + spec.lambda2))
    weights = np.full(3, 1.0 / 3.0, dtype=np.float64)

    return TriangularConstruction(
        centroids=centroids,
        angles=angles,
        weights=weights,
        between_variance=between_var,
    )


def density_ratio(
    x: NDArray[np.floating],
    matrix: NDArray[np.floating],
) -> float | NDArray[np.float64]:
    """Evaluate the Gaussian density ratio f_R(x) / f_I(x) for N(0, R) vs N(0, I).

    Supports a single vector of shape (p,) (returning a float) or a batch of shape
    (n, p) (returning a 1D float64 array of shape (n,)).
    """
    arr_r = np.asarray(matrix, dtype=np.float64)
    if not is_positive_definite(arr_r):
        raise ValueError("Matrix must be symmetric and strictly positive definite.")

    p = arr_r.shape[0]
    arr_x = np.asarray(x, dtype=np.float64)
    sign, logdet = np.linalg.slogdet(arr_r)
    if sign <= 0:
        raise ValueError("Matrix determinant must be strictly positive.")

    if arr_x.ndim == 1:
        if arr_x.shape[0] != p:
            raise ValueError(f"Expected vector of length {p}, got {arr_x.shape[0]}.")
        quad_r = float(arr_x @ np.linalg.solve(arr_r, arr_x))
        quad_i = float(arr_x @ arr_x)
        return float(np.exp(-0.5 * (logdet + quad_r - quad_i)))

    if arr_x.ndim == 2:
        if arr_x.shape[1] != p:
            raise ValueError(f"Expected array with {p} columns, got {arr_x.shape[1]}.")
        r_inv_xt = np.linalg.solve(arr_r, arr_x.T)
        quad_r = np.sum(arr_x.T * r_inv_xt, axis=0)
        quad_i = np.sum(arr_x * arr_x, axis=1)
        return np.exp(-0.5 * (logdet + quad_r - quad_i))

    raise ValueError(f"Expected 1D or 2D input for x, got shape {arr_x.shape}.")
