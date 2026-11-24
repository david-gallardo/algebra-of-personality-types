"""Exact Monte Carlo inference, Wilson intervals, Holm adjustment, and survey weighting."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def phipson_smyth_p(
    sim_values: NDArray[np.floating],
    observed: float,
    *,
    direction: Literal["ge", "le"] = "ge",
) -> tuple[float, int, int]:
    """Compute exact Phipson-Smyth Monte Carlo p = (1 + k) / (B + 1)."""
    vals = np.asarray(sim_values, dtype=np.float64)
    valid = vals[np.isfinite(vals)]
    b = int(valid.size)
    if b == 0 or not math.isfinite(observed):
        return float("nan"), 0, 0
    k = int(np.sum(valid >= observed)) if direction == "ge" else int(np.sum(valid <= observed))
    return (1.0 + k) / float(b + 1), k, b


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score confidence interval for binomial proportion k / n."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / float(n)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def holm_bonferroni(p_values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Step-down Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(p[idx]) * float(m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def weighted_mean_and_var(
    values: NDArray[np.floating],
    weights: NDArray[np.floating],
) -> tuple[float, float]:
    """Compute exact survey-weighted mean and reliability-weighted variance."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    x, w = x[mask], w[mask]
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return float("nan"), float("nan")
    mu = float(np.sum(w * x) / w_sum)
    var = float(np.sum(w * (x - mu) ** 2) / w_sum)
    return mu, var
