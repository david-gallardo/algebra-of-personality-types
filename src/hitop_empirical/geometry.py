"""Spectral geometry diagnostics for correlation matrices (theoretical benchmark only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

NONCOLLINEARITY_THRESHOLD: float = 0.50764047


@dataclass(frozen=True)
class SpectralDiagnostics:
    """Spectral and density-ratio summary of a correlation matrix R."""

    r_matrix: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    lambda1: float
    lambda2: float
    lambda2_over_lambda1: float
    exceeds_noncollinearity_threshold: bool
    det_r: float
    origin_density_ratio: float
    is_pd: bool


def correlation_geometry(data: NDArray[np.floating]) -> SpectralDiagnostics:
    """Compute exact spectral diagnostics of the empirical correlation matrix of ``data``."""
    arr = np.asarray(data, dtype=np.float64)
    r = np.corrcoef(arr, rowvar=False)
    r = 0.5 * (r + r.T)
    np.fill_diagonal(r, 1.0)
    eigvals = np.sort(np.linalg.eigvalsh(r))[::-1]
    is_pd = bool(eigvals[-1] > 1e-10)
    det_r = float(np.prod(eigvals))
    ratio = float(eigvals[1] / eigvals[0])
    origin_ratio = float(det_r ** (-0.5)) if det_r > 0.0 else float("inf")
    return SpectralDiagnostics(
        r_matrix=r,
        eigenvalues=eigvals,
        lambda1=float(eigvals[0]),
        lambda2=float(eigvals[1]),
        lambda2_over_lambda1=ratio,
        exceeds_noncollinearity_threshold=bool(ratio > NONCOLLINEARITY_THRESHOLD),
        det_r=det_r,
        origin_density_ratio=origin_ratio,
        is_pd=is_pd,
    )
