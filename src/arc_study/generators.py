"""Reproducible data and correlation-matrix generators for the ARC null models.

All generators accept an explicit ``rng: np.random.Generator`` and adhere to the
strict Big Five trait ordering (N, E, O, A, C) with N unreversed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, invwishart, norm, wishart

from arc_study.matrices import is_positive_definite, validate_correlation_matrix

MarginalPPF = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# Default standardized log-normal shape parameters (signed for skewness direction)
# in (N, E, O, A, C) order: N mildly right-skewed; E, O, A, C mildly left-skewed.
DEFAULT_SKEW_PARAMS: tuple[float, ...] = (0.25, -0.20, -0.18, -0.25, -0.22)


@dataclass(frozen=True)
class CopulaCalibrationResult:
    """Result of iterative latent-correlation calibration for a Gaussian copula."""

    latent_corr: NDArray[np.float64]
    achieved_corr: NDArray[np.float64]
    max_abs_error: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class UnimodalityDiagnostic:
    """Diagnostic summary of marginal and joint principal-plane unimodality."""

    is_unimodal: bool
    marginal_modes: tuple[int, ...]
    pc_modes: tuple[int, int]
    pc12_joint_modes: int


@dataclass(frozen=True)
class Null3Sample:
    """Generated sample from Null 3 (Gaussian copula) with calibration and diagnostics."""

    data: NDArray[np.float64]
    calibration: CopulaCalibrationResult
    unimodality: UnimodalityDiagnostic | None


def _cov_to_corr(cov: NDArray[np.floating]) -> NDArray[np.float64]:
    """Scale a symmetric positive-definite covariance matrix to a correlation matrix."""
    arr = np.asarray(cov, dtype=np.float64)
    arr = 0.5 * (arr + arr.T)
    diag = np.diag(arr)
    if np.any(diag <= 0.0):
        raise ValueError("Covariance matrix diagonal must be strictly positive.")
    inv_std = 1.0 / np.sqrt(diag)
    corr = arr * np.outer(inv_std, inv_std)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def nearest_pd_correlation(
    matrix: NDArray[np.floating],
    min_eig: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """Project a symmetric matrix onto the nearest positive-definite correlation matrix.

    Uses Higham's alternating projections with Dykstra's correction, followed by a
    final eigenvalue floor and correlation rescaling to guarantee strict positive
    definiteness and unit diagonal.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty square 2D matrix, got shape {arr.shape}.")

    y = 0.5 * (arr + arr.T)
    ds = np.zeros_like(y)

    for _ in range(max_iter):
        r = y - ds
        eigvals, eigvecs = np.linalg.eigh(0.5 * (r + r.T))
        x = (eigvecs * np.maximum(eigvals, min_eig)) @ eigvecs.T
        x = 0.5 * (x + x.T)
        ds = x - r

        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        denom = max(1.0, float(np.linalg.norm(y, ord="fro")))
        if float(np.linalg.norm(y_new - y, ord="fro")) <= tol * denom:
            y = y_new
            break
        y = y_new

    y = 0.5 * (y + y.T)
    eigvals, eigvecs = np.linalg.eigh(y)
    if eigvals[0] < min_eig:
        y = (eigvecs * np.maximum(eigvals, min_eig)) @ eigvecs.T
        y = _cov_to_corr(y)
    else:
        np.fill_diagonal(y, 1.0)

    return y


# ---------------------------------------------------------------------------
# Null 1: Independence (isotropic normal & column-wise empirical permutation)
# ---------------------------------------------------------------------------


def sample_null1_independent_normal(
    n: int,
    rng: np.random.Generator,
    dim: int = 5,
) -> NDArray[np.float64]:
    """Generate ``n`` samples from the isotropic normal null N_p(0, I_p)."""
    if n <= 0 or dim <= 0:
        raise ValueError("Sample size n and dimension dim must be positive.")
    return rng.standard_normal((n, dim))


def sample_null1_column_permutation(
    data: NDArray[np.floating],
    rng: np.random.Generator,
    n: int | None = None,
) -> NDArray[np.float64]:
    """Generate an independent null by permuting each column of ``data`` independently.

    When ``n`` is None or equals ``data.shape[0]``, each column is an exact permutation
    of the empirical column, preserving all empirical quantiles identically. When
    ``n != data.shape[0]``, each column is resampled independently with replacement.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D array, got shape {arr.shape}.")

    m, p = arr.shape
    target_n = m if n is None else n
    if target_n <= 0:
        raise ValueError("Target sample size n must be positive.")

    out = np.empty((target_n, p), dtype=np.float64)
    if target_n == m:
        for j in range(p):
            out[:, j] = rng.permutation(arr[:, j])
    else:
        for j in range(p):
            idx = rng.integers(0, m, size=target_n)
            out[:, j] = arr[idx, j]
    return out


# ---------------------------------------------------------------------------
# Null 2: Correlated multivariate normal N_p(0, R)
# ---------------------------------------------------------------------------


def sample_null2_correlated_normal(
    n: int,
    corr_matrix: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate ``n`` samples from N_p(0, R) for a validated correlation matrix R."""
    if n <= 0:
        raise ValueError("Sample size n must be positive.")
    r_mat = validate_correlation_matrix(corr_matrix)
    chol = np.linalg.cholesky(r_mat)
    z = rng.standard_normal((n, r_mat.shape[0]))
    return z @ chol.T


# ---------------------------------------------------------------------------
# Null 3: Correlated non-normal continuous via calibrated Gaussian copula
# ---------------------------------------------------------------------------


def _make_standardized_lognormal_ppf(signed_sigma: float) -> MarginalPPF:
    """Return a strictly increasing standardized log-normal PPF (mean 0, var 1).

    Positive ``signed_sigma`` yields right-skewed marginals; negative yields left-skewed;
    zero reduces to the standard normal PPF.
    """
    if signed_sigma == 0.0:
        return lambda u: np.asarray(norm.ppf(u), dtype=np.float64)

    sigma = abs(float(signed_sigma))
    mean = float(np.exp(0.5 * sigma * sigma))
    std = float(np.sqrt((np.exp(sigma * sigma) - 1.0) * np.exp(sigma * sigma)))

    if signed_sigma > 0.0:
        return lambda u: (np.exp(sigma * norm.ppf(u)) - mean) / std
    return lambda u: -(np.exp(sigma * norm.ppf(1.0 - u)) - mean) / std


def default_skewed_marginals(
    skew_params: Sequence[float] = DEFAULT_SKEW_PARAMS,
) -> list[MarginalPPF]:
    """Build unimodal asymmetric reference marginals (standardized log-normal PPFs)."""
    return [_make_standardized_lognormal_ppf(s) for s in skew_params]


def empirical_marginal_ppfs(data: NDArray[np.floating]) -> list[MarginalPPF]:
    """Build linear-interpolated empirical quantile functions for each column of ``data``."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] == 0:
        raise ValueError(f"Expected 2D array with at least 2 rows, got shape {arr.shape}.")

    m, p = arr.shape
    grid = np.linspace(0.0, 1.0, m, dtype=np.float64)
    ppfs: list[MarginalPPF] = []
    for j in range(p):
        sorted_col = np.sort(arr[:, j]).copy()
        ppfs.append(lambda u, col=sorted_col, g=grid: np.interp(u, g, col))
    return ppfs


def _apply_gaussian_copula(
    z_standard: NDArray[np.float64],
    latent_corr: NDArray[np.float64],
    marginals: Sequence[MarginalPPF],
) -> NDArray[np.float64]:
    """Map standard normal draws through latent correlation and marginal PPFs."""
    chol = np.linalg.cholesky(latent_corr)
    g = z_standard @ chol.T
    u = np.clip(norm.cdf(g), 1e-12, 1.0 - 1e-12)
    out = np.empty_like(u)
    for j, ppf in enumerate(marginals):
        out[:, j] = ppf(u[:, j])
    return out


def calibrate_gaussian_copula(
    target_corr: NDArray[np.floating],
    marginals: Sequence[MarginalPPF],
    rng: np.random.Generator,
    n_cal: int = 50_000,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> CopulaCalibrationResult:
    """Iteratively calibrate the Gaussian copula latent correlation matrix.

    Uses whitened common random numbers and nearest-PD correlation projection so the
    transformed non-normal variables match ``target_corr`` in Pearson correlation.
    """
    r_target = validate_correlation_matrix(target_corr)
    p = r_target.shape[0]
    if len(marginals) != p:
        raise ValueError(f"Expected {p} marginal PPFs, got {len(marginals)}.")
    if n_cal < p + 2:
        raise ValueError("n_cal must exceed the number of dimensions.")

    # Whitened common random numbers eliminate Monte Carlo jitter during calibration.
    z_cal = rng.standard_normal((n_cal, p))
    z_cal -= z_cal.mean(axis=0, keepdims=True)
    cov_z = np.cov(z_cal, rowvar=False)
    chol_z = np.linalg.cholesky(cov_z)
    z_cal = np.linalg.solve(chol_z, z_cal.T).T

    r_lat = nearest_pd_correlation(r_target)
    best_lat = r_lat.copy()
    best_obs = r_target.copy()
    best_err = float("inf")
    converged = False
    iters_run = 0

    for it in range(1, max_iter + 1):
        iters_run = it
        x_cal = _apply_gaussian_copula(z_cal, r_lat, marginals)
        r_obs = np.corrcoef(x_cal, rowvar=False)
        diff = r_target - r_obs
        max_err = float(np.max(np.abs(diff)))

        if max_err < best_err:
            best_err = max_err
            best_lat = r_lat.copy()
            best_obs = r_obs.copy()

        if max_err <= tol:
            converged = True
            break

        step = 1.0 if max_err <= best_err + 1e-12 else 0.5
        r_lat = nearest_pd_correlation(r_lat + step * diff)

    return CopulaCalibrationResult(
        latent_corr=best_lat,
        achieved_corr=best_obs,
        max_abs_error=best_err,
        iterations=iters_run,
        converged=converged,
    )


def _count_1d_kde_modes(
    values: NDArray[np.float64],
    grid_size: int = 256,
    min_rel_height: float = 0.05,
    min_rel_prominence: float = 0.05,
) -> int:
    """Count significant interior local maxima of a 1D Gaussian KDE."""
    kde = gaussian_kde(values)
    lo, hi = np.quantile(values, [0.01, 0.99])
    if hi <= lo:
        return 1
    grid = np.linspace(lo, hi, grid_size)
    dens = kde(grid)
    max_dens = float(np.max(dens))
    if max_dens <= 0.0:
        return 1
    peaks, _ = find_peaks(
        dens,
        height=min_rel_height * max_dens,
        prominence=min_rel_prominence * max_dens,
    )
    return max(1, int(len(peaks)))


def _count_2d_kde_modes(
    xy: NDArray[np.float64],
    grid_size: int = 35,
    min_rel_height: float = 0.05,
) -> int:
    """Count strict interior local maxima of a 2D Gaussian KDE on an 8-neighborhood."""
    kde = gaussian_kde(xy.T)
    x_lo, x_hi = np.quantile(xy[:, 0], [0.01, 0.99])
    y_lo, y_hi = np.quantile(xy[:, 1], [0.01, 0.99])
    if x_hi <= x_lo or y_hi <= y_lo:
        return 1

    gx = np.linspace(x_lo, x_hi, grid_size)
    gy = np.linspace(y_lo, y_hi, grid_size)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    coords = np.vstack([xx.ravel(), yy.ravel()])
    dens = kde(coords).reshape(grid_size, grid_size)

    center = dens[1:-1, 1:-1]
    is_peak = center >= min_rel_height * float(np.max(dens))
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor = dens[1 + di : grid_size - 1 + di, 1 + dj : grid_size - 1 + dj]
            is_peak &= center > neighbor

    return max(1, int(np.sum(is_peak)))


def check_joint_unimodality(
    data: NDArray[np.floating],
    rng: np.random.Generator | None = None,
    max_kde_points: int = 2500,
    min_rel_height: float = 0.05,
    min_rel_prominence: float = 0.05,
) -> UnimodalityDiagnostic:
    """Run a KDE-based unimodality diagnostic on marginals and the (PC1, PC2) plane."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 10 or arr.shape[1] < 2:
        raise ValueError(f"Expected 2D array with >= 10 rows and >= 2 cols, got {arr.shape}.")

    n, p = arr.shape
    if n > max_kde_points:
        if rng is None:
            sub = arr[:max_kde_points]
        else:
            idx = rng.choice(n, size=max_kde_points, replace=False)
            sub = arr[idx]
    else:
        sub = arr

    marginal_modes = tuple(
        _count_1d_kde_modes(
            sub[:, j],
            min_rel_height=min_rel_height,
            min_rel_prominence=min_rel_prominence,
        )
        for j in range(p)
    )

    centered = sub - sub.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pcs = centered @ vt[:2].T

    pc1_modes = _count_1d_kde_modes(
        pcs[:, 0],
        min_rel_height=min_rel_height,
        min_rel_prominence=min_rel_prominence,
    )
    pc2_modes = _count_1d_kde_modes(
        pcs[:, 1],
        min_rel_height=min_rel_height,
        min_rel_prominence=min_rel_prominence,
    )
    pc12_modes = _count_2d_kde_modes(pcs, min_rel_height=min_rel_height)

    is_unimodal = (
        all(m == 1 for m in marginal_modes)
        and pc1_modes == 1
        and pc2_modes == 1
        and pc12_modes == 1
    )
    return UnimodalityDiagnostic(
        is_unimodal=is_unimodal,
        marginal_modes=marginal_modes,
        pc_modes=(pc1_modes, pc2_modes),
        pc12_joint_modes=pc12_modes,
    )


def sample_null3_gaussian_copula(
    n: int,
    target_corr: NDArray[np.floating],
    rng: np.random.Generator,
    marginals: Sequence[MarginalPPF] | None = None,
    empirical_data: NDArray[np.floating] | None = None,
    calibration_n: int = 50_000,
    calibration_tol: float = 1e-4,
    check_unimodality: bool = True,
) -> Null3Sample:
    """Generate ``n`` samples from Null 3 (calibrated non-normal Gaussian copula).

    Provide either ``marginals`` (callables mapping (0, 1) -> quantile),
    ``empirical_data`` (used to build empirical marginal PPFs), or neither to use
    ``default_skewed_marginals()``.
    """
    if n <= 0:
        raise ValueError("Sample size n must be positive.")
    if marginals is not None and empirical_data is not None:
        raise ValueError("Provide either 'marginals' or 'empirical_data', not both.")

    r_target = validate_correlation_matrix(target_corr)
    p = r_target.shape[0]

    if empirical_data is not None:
        ppf_list = empirical_marginal_ppfs(empirical_data)
    elif marginals is not None:
        ppf_list = list(marginals)
    else:
        if p != len(DEFAULT_SKEW_PARAMS):
            raise ValueError(
                f"Default skewed marginals require {len(DEFAULT_SKEW_PARAMS)} dimensions, got {p}."
            )
        ppf_list = default_skewed_marginals()

    calibration = calibrate_gaussian_copula(
        target_corr=r_target,
        marginals=ppf_list,
        rng=rng,
        n_cal=calibration_n,
        tol=calibration_tol,
    )

    z = rng.standard_normal((n, p))
    data = _apply_gaussian_copula(z, calibration.latent_corr, ppf_list)
    unimodality = check_joint_unimodality(data, rng=rng) if check_unimodality else None

    return Null3Sample(
        data=data,
        calibration=calibration,
        unimodality=unimodality,
    )


# ---------------------------------------------------------------------------
# Between-study heterogeneity: random SPD correlation matrices around R_meta
# ---------------------------------------------------------------------------


def sample_study_correlation_matrices(
    r_meta: NDArray[np.floating],
    n_studies: int,
    df: float,
    rng: np.random.Generator,
    method: Literal["invwishart", "wishart"] = "invwishart",
) -> NDArray[np.float64]:
    """Draw ``n_studies`` random SPD correlation matrices around ``r_meta``.

    Generates covariance matrices from a Wishart or Inverse-Wishart distribution
    centered on ``r_meta`` and scales each draw to a correlation matrix, guaranteeing
    symmetry, unit diagonal, and strict positive definiteness.
    """
    if n_studies <= 0:
        raise ValueError("n_studies must be positive.")
    r_mat = validate_correlation_matrix(r_meta)
    p = r_mat.shape[0]

    if method == "invwishart":
        if df <= p + 1:
            raise ValueError(f"Inverse-Wishart requires df > p + 1 ({p + 1}), got {df}.")
        scale = (df - p - 1.0) * r_mat
        raw = invwishart.rvs(df=df, scale=scale, size=n_studies, random_state=rng)
    elif method == "wishart":
        if df <= p - 1:
            raise ValueError(f"Wishart requires df > p - 1 ({p - 1}), got {df}.")
        scale = r_mat / df
        raw = wishart.rvs(df=df, scale=scale, size=n_studies, random_state=rng)
    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'invwishart' or 'wishart'.")

    covs = raw[np.newaxis, :, :] if n_studies == 1 else np.asarray(raw, dtype=np.float64)
    corrs = np.empty((n_studies, p, p), dtype=np.float64)
    for idx in range(n_studies):
        corr = _cov_to_corr(covs[idx])
        if not is_positive_definite(corr):
            corr = nearest_pd_correlation(corr)
        corrs[idx] = corr
    return corrs


def sample_study_correlation_matrix(
    r_meta: NDArray[np.floating],
    df: float,
    rng: np.random.Generator,
    method: Literal["invwishart", "wishart"] = "invwishart",
) -> NDArray[np.float64]:
    """Draw a single random SPD correlation matrix around ``r_meta``."""
    return sample_study_correlation_matrices(
        r_meta=r_meta,
        n_studies=1,
        df=df,
        rng=rng,
        method=method,
    )[0]
