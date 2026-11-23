"""Observed Big Five correlation matrices and positive-definiteness checks.

All matrices follow the strict dimension order (N, E, O, A, C) with N unreversed,
using observed (never disattenuated) meta-analytic correlations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TRAIT_ORDER: tuple[str, ...] = ("N", "E", "O", "A", "C")
BASE_SEED: int = 20261121

VAN_DER_LINDEN_2010: NDArray[np.float64] = np.array(
    [
        [1.00, -0.26, -0.12, -0.26, -0.32],
        [-0.26, 1.00, 0.31, 0.18, 0.21],
        [-0.12, 0.31, 1.00, 0.14, 0.14],
        [-0.26, 0.18, 0.14, 1.00, 0.31],
        [-0.32, 0.21, 0.14, 0.31, 1.00],
    ],
    dtype=np.float64,
)
VAN_DER_LINDEN_2010.flags.writeable = False

BUSSERI_ERB_2024: NDArray[np.float64] = np.array(
    [
        [1.00, -0.27, -0.08, -0.14, -0.22],
        [-0.27, 1.00, 0.31, 0.27, 0.29],
        [-0.08, 0.31, 1.00, 0.27, 0.22],
        [-0.14, 0.27, 0.27, 1.00, 0.27],
        [-0.22, 0.29, 0.22, 0.27, 1.00],
    ],
    dtype=np.float64,
)
BUSSERI_ERB_2024.flags.writeable = False

IDENTITY_5: NDArray[np.float64] = np.eye(5, dtype=np.float64)
IDENTITY_5.flags.writeable = False


def is_positive_definite(
    matrix: NDArray[np.floating],
    tol: float = 1e-10,
) -> bool:
    """Return True if ``matrix`` is 2D, square, symmetric, and strictly positive definite."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if not np.allclose(arr, arr.T, atol=tol, rtol=0.0):
        return False
    eigvals = np.linalg.eigvalsh(0.5 * (arr + arr.T))
    return bool(eigvals[0] > tol)


def validate_correlation_matrix(
    matrix: NDArray[np.floating],
    tol: float = 1e-10,
) -> NDArray[np.float64]:
    """Validate that ``matrix`` is a symmetric positive-definite correlation matrix."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError(f"Expected a non-empty square 2D matrix, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Correlation matrix contains non-finite values.")
    if not np.allclose(arr, arr.T, atol=tol, rtol=0.0):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(arr), 1.0, atol=tol, rtol=0.0):
        raise ValueError("Correlation matrix must have unit diagonal.")
    if not is_positive_definite(arr, tol=tol):
        raise ValueError("Correlation matrix must be strictly positive definite.")
    return arr


def get_observed_matrices() -> dict[str, NDArray[np.float64]]:
    """Return validated copies of the observed meta-analytic correlation matrices."""
    return {
        "vdl": validate_correlation_matrix(VAN_DER_LINDEN_2010).copy(),
        "be": validate_correlation_matrix(BUSSERI_ERB_2024).copy(),
    }
