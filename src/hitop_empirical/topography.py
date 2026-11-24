"""Exact isotropic Gaussian KDE mean-shift topography, basin persistence, and valley checks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class TopographyResult:
    """Summary of mean-shift mode discovery at a single bandwidth h."""

    bandwidth: float
    n_modes: int
    mode_locations: NDArray[np.float64]
    mode_densities: NDArray[np.float64]
    basin_counts: NDArray[np.intp]
    primary_basin_frac: float
    max_secondary_basin_frac: float


def compute_h_nn(data: NDArray[np.floating], max_points: int = 10_000, rng: np.random.Generator | None = None) -> float:
    """Compute Gerlach et al. (2018) nearest-neighbour KDE bandwidth h_NN (mean NN distance)."""
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    if n > max_points:
        sub = x[rng.choice(n, size=max_points, replace=False)] if rng is not None else x[:max_points]
    else:
        sub = x
    tree = cKDTree(x)
    dists, _ = tree.query(sub, k=2)
    return float(np.mean(dists[:, 1]))


def eval_isotropic_kde(
    data: NDArray[np.floating],
    points: NDArray[np.floating],
    bandwidth: float,
) -> NDArray[np.float64]:
    """Exact vectorized isotropic Gaussian KDE density evaluated at ``points``."""
    x = np.asarray(data, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    n, d = x.shape
    inv_2h2 = 0.5 / (bandwidth * bandwidth)
    norm_c = (2.0 * math.pi * bandwidth * bandwidth) ** (-0.5 * d) / float(n)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    p_sq = np.sum(pts * pts, axis=1, keepdims=True).T
    d2 = np.maximum(0.0, x_sq + p_sq - 2.0 * (x @ pts.T))
    return np.asarray(norm_c * np.sum(np.exp(-d2 * inv_2h2), axis=0), dtype=np.float64)


def run_mean_shift(
    data: NDArray[np.floating],
    starts: NDArray[np.floating],
    bandwidth: float,
    *,
    tol: float = 5e-4,
    max_iter: int = 300,
    merge_radius: float = 0.10,
) -> TopographyResult:
    """Run exact gradient-ascent mean-shift on isotropic Gaussian KDE from ``starts``."""
    x = np.asarray(data, dtype=np.float64)
    cur = np.asarray(starts, dtype=np.float64).copy()
    inv_2h2 = 0.5 / (bandwidth * bandwidth)
    x_sq = np.sum(x * x, axis=1, keepdims=True)

    for _ in range(max_iter):
        c_sq = np.sum(cur * cur, axis=1, keepdims=True).T
        d2 = np.maximum(0.0, x_sq + c_sq - 2.0 * (x @ cur.T))
        w = np.exp(-d2 * inv_2h2)
        denom = np.sum(w, axis=0, keepdims=True).T
        denom = np.where(denom > 0.0, denom, 1.0)
        nxt = (w.T @ x) / denom
        shift = float(np.max(np.linalg.norm(nxt - cur, axis=1)))
        cur = nxt
        if shift < tol:
            break

    # Greedy merge of converged attractors within merge_radius
    modes: list[NDArray[np.float64]] = []
    counts: list[int] = []
    for pt in cur:
        merged = False
        for idx, m in enumerate(modes):
            if float(np.linalg.norm(pt - m)) <= merge_radius:
                counts[idx] += 1
                merged = True
                break
        if not merged:
            modes.append(pt.copy())
            counts.append(1)

    mode_arr = np.vstack(modes)
    count_arr = np.asarray(counts, dtype=np.intp)
    dens_arr = eval_isotropic_kde(x, mode_arr, bandwidth)
    order = np.argsort(count_arr)[::-1]
    mode_arr = mode_arr[order]
    count_arr = count_arr[order]
    dens_arr = dens_arr[order]

    total = float(len(cur))
    primary = float(count_arr[0]) / total if total > 0 else 0.0
    secondary = float(count_arr[1]) / total if len(count_arr) > 1 and total > 0 else 0.0

    return TopographyResult(
        bandwidth=bandwidth,
        n_modes=int(len(count_arr)),
        mode_locations=mode_arr,
        mode_densities=dens_arr,
        basin_counts=count_arr,
        primary_basin_frac=primary,
        max_secondary_basin_frac=secondary,
    )


def pairwise_valley_depth(
    data: NDArray[np.floating],
    point_a: NDArray[np.floating],
    point_b: NDArray[np.floating],
    bandwidth: float,
    *,
    n_grid: int = 41,
) -> float:
    """Non-negative density valley depth along linear transect between point_a and point_b."""
    a = np.asarray(point_a, dtype=np.float64)
    b = np.asarray(point_b, dtype=np.float64)
    us = np.linspace(0.0, 1.0, n_grid)[:, None]
    transect = (1.0 - us) * a[None, :] + us * b[None, :]
    dens = eval_isotropic_kde(data, transect, bandwidth)
    end_min = min(float(dens[0]), float(dens[-1]))
    if end_min <= 0.0:
        return 0.0
    return float(max(0.0, 1.0 - float(np.min(dens)) / end_min))
