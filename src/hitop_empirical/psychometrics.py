"""Psychometric reliability, floor/ceiling diagnostics, and cross-cohort comparability."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import kurtosis, skew


def cronbach_alpha(x: NDArray[np.floating]) -> float:
    """Compute Cronbach's alpha on complete rows of a 2D indicator matrix."""
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    k = arr.shape[1]
    if arr.shape[0] < 3 or k < 2:
        return float("nan")
    item_vars = np.var(arr, axis=0, ddof=1)
    total_var = float(np.var(np.sum(arr, axis=1), ddof=1))
    if total_var <= 0.0:
        return float("nan")
    return float((k / (k - 1.0)) * (1.0 - float(np.sum(item_vars)) / total_var))


def mcdonald_omega_total(x: NDArray[np.floating]) -> float:
    """Compute unidimensional McDonald's omega_t from standardized 1-factor loadings."""
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] < 5 or arr.shape[1] < 2:
        return float("nan")
    r = np.corrcoef(arr, rowvar=False)
    r = 0.5 * (r + r.T)
    np.fill_diagonal(r, 1.0)
    # Principal-axis 1-factor extraction (iterative communalities)
    h2 = np.clip(1.0 - 1.0 / np.maximum(np.diag(np.linalg.pinv(r)), 1.0), 0.05, 0.95)
    loadings = np.zeros(r.shape[0], dtype=np.float64)
    for _ in range(30):
        r_red = r.copy()
        np.fill_diagonal(r_red, h2)
        eigvals, eigvecs = np.linalg.eigh(r_red)
        lam1 = max(float(eigvals[-1]), 1e-8)
        v1 = eigvecs[:, -1]
        if np.sum(v1) < 0.0:
            v1 = -v1
        loadings = v1 * np.sqrt(lam1)
        h2 = np.clip(loadings**2, 0.0, 0.99)
    sum_l_sq = float(np.sum(loadings) ** 2)
    uniq = np.maximum(0.0, 1.0 - loadings**2)
    denom = sum_l_sq + float(np.sum(uniq))
    return float(sum_l_sq / denom) if denom > 0.0 else float("nan")


def scale_qc_table(
    df: pd.DataFrame,
    cols: list[str],
    *,
    floor_val: float = 1.0,
    ceiling_val: float = 4.0,
) -> pd.DataFrame:
    """Compute missingness, floor/ceiling percentages, mean, SD, skewness, and kurtosis."""
    rows: list[dict[str, object]] = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        valid = s[np.isfinite(s)]
        n_v = int(valid.size)
        rows.append(
            {
                "column": c,
                "n_valid": n_v,
                "na_pct": float(100.0 * (1.0 - n_v / max(len(s), 1))),
                "mean": float(np.mean(valid)) if n_v > 0 else float("nan"),
                "sd": float(np.std(valid, ddof=1)) if n_v > 1 else float("nan"),
                "floor_pct": float(100.0 * np.mean(valid <= floor_val + 1e-9)) if n_v > 0 else float("nan"),
                "ceiling_pct": float(100.0 * np.mean(valid >= ceiling_val - 1e-9)) if n_v > 0 else float("nan"),
                "skewness": float(skew(valid, bias=False)) if n_v > 2 else float("nan"),
                "kurtosis_excess": float(kurtosis(valid, fisher=True, bias=False)) if n_v > 3 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def loading_tucker_congruence(
    x_a: NDArray[np.floating],
    x_b: NDArray[np.floating],
) -> float:
    """Compute Tucker's phi between 1-factor loadings of two cohorts (configural/metric check)."""
    def _first_loading(arr: NDArray[np.floating]) -> NDArray[np.float64]:
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.all(np.isfinite(a), axis=1)]
        r = np.corrcoef(a, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(0.5 * (r + r.T))
        v = eigvecs[:, -1]
        if np.sum(v) < 0.0:
            v = -v
        return v * np.sqrt(max(float(eigvals[-1]), 1e-12))

    la = _first_loading(x_a)
    lb = _first_loading(x_b)
    denom = float(np.linalg.norm(la) * np.linalg.norm(lb))
    return float(np.clip(np.dot(la, lb) / denom, -1.0, 1.0)) if denom > 0.0 else float("nan")
