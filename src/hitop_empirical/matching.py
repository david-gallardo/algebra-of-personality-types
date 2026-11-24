"""Hungarian centroid matching, full/shape Tucker congruence, and split-half stability."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, normalized_mutual_info_score

from hitop_empirical.clustering import fit_matched_kmeans


@dataclass(frozen=True)
class CrossCohortMatchResult:
    """Optimal Hungarian alignment between two K x p centroid matrices."""

    row_ind: tuple[int, ...]
    col_ind: tuple[int, ...]
    mean_tucker_full: float
    min_tucker_full: float
    mean_tucker_shape: float
    min_tucker_shape: float
    mean_d2_full: float
    mean_d2_shape: float
    per_cluster_tucker_full: NDArray[np.float64]
    per_cluster_tucker_shape: NDArray[np.float64]


def _tucker(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def match_centroids_hungarian(
    centers_a: NDArray[np.floating],
    centers_b: NDArray[np.floating],
    *,
    optimize_on: str = "full_tucker",
) -> CrossCohortMatchResult:
    """Match two K x p centroid matrices using optimal Hungarian assignment."""
    ca = np.asarray(centers_a, dtype=np.float64)
    cb = np.asarray(centers_b, dtype=np.float64)
    if ca.shape != cb.shape or ca.ndim != 2:
        raise ValueError(f"Expected identical 2D shapes, got {ca.shape} and {cb.shape}.")

    k = ca.shape[0]
    ca_shape = ca - ca.mean(axis=1, keepdims=True)
    cb_shape = cb - cb.mean(axis=1, keepdims=True)

    cost = np.empty((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            if optimize_on == "full_tucker":
                phi = _tucker(ca[i], cb[j])
                cost[i, j] = -phi if np.isfinite(phi) else 1.0
            elif optimize_on == "shape_tucker":
                phi = _tucker(ca_shape[i], cb_shape[j])
                cost[i, j] = -phi if np.isfinite(phi) else 1.0
            elif optimize_on == "d2":
                diff = ca[i] - cb[j]
                cost[i, j] = float(np.dot(diff, diff))
            else:
                raise ValueError(f"Unknown optimize_on='{optimize_on}'.")

    r_ind, c_ind = linear_sum_assignment(cost)
    t_full = np.array([_tucker(ca[i], cb[j]) for i, j in zip(r_ind, c_ind)], dtype=np.float64)
    t_shape = np.array(
        [_tucker(ca_shape[i], cb_shape[j]) for i, j in zip(r_ind, c_ind)], dtype=np.float64
    )
    d2_full = np.array(
        [float(np.sum((ca[i] - cb[j]) ** 2)) for i, j in zip(r_ind, c_ind)], dtype=np.float64
    )
    d2_shape = np.array(
        [float(np.sum((ca_shape[i] - cb_shape[j]) ** 2)) for i, j in zip(r_ind, c_ind)],
        dtype=np.float64,
    )

    return CrossCohortMatchResult(
        row_ind=tuple(int(i) for i in r_ind),
        col_ind=tuple(int(j) for j in c_ind),
        mean_tucker_full=float(np.nanmean(t_full)),
        min_tucker_full=float(np.nanmin(t_full)),
        mean_tucker_shape=float(np.nanmean(t_shape)),
        min_tucker_shape=float(np.nanmin(t_shape)),
        mean_d2_full=float(np.mean(d2_full)),
        mean_d2_shape=float(np.mean(d2_shape)),
        per_cluster_tucker_full=t_full,
        per_cluster_tucker_shape=t_shape,
    )


def split_half_stability(
    data: NDArray[np.floating],
    n_clusters: int = 3,
    *,
    n_splits: int = 10,
    n_kmeanspp_starts: int = 9,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Asendorpf split-half double cross-validation stability (kappa, ARI, NMI)."""
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    half = n // 2
    kappas, aris, nmis = [], [], []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        xa, xb = x[perm[:half]], x[perm[half:]]
        fa = fit_matched_kmeans(xa, n_clusters=n_clusters, n_kmeanspp_starts=n_kmeanspp_starts, rng=rng)
        fb = fit_matched_kmeans(xb, n_clusters=n_clusters, n_kmeanspp_starts=n_kmeanspp_starts, rng=rng)
        m = match_centroids_hungarian(fa.centers, fb.centers, optimize_on="d2")
        remap = {ri: ci for ri, ci in zip(m.row_ind, m.col_ind)}
        aligned_la = np.array([remap[int(v)] for v in fa.labels], dtype=np.intp)
        aligned_ca = np.empty_like(fa.centers)
        for ri, ci in zip(m.row_ind, m.col_ind):
            aligned_ca[ci] = fa.centers[ri]

        pred_b = np.argmin(np.sum((xb[:, None, :] - aligned_ca[None, :, :]) ** 2, axis=2), axis=1)
        pred_a = np.argmin(np.sum((xa[:, None, :] - fb.centers[None, :, :]) ** 2, axis=2), axis=1)
        kappas.append(0.5 * (cohen_kappa_score(fb.labels, pred_b) + cohen_kappa_score(aligned_la, pred_a)))
        aris.append(0.5 * (adjusted_rand_score(fb.labels, pred_b) + adjusted_rand_score(aligned_la, pred_a)))
        nmis.append(0.5 * (normalized_mutual_info_score(fb.labels, pred_b) + normalized_mutual_info_score(aligned_la, pred_a)))

    return {
        "mean_kappa": float(np.mean(kappas)),
        "std_kappa": float(np.std(kappas, ddof=1)) if len(kappas) > 1 else 0.0,
        "mean_ari": float(np.mean(aris)),
        "mean_nmi": float(np.mean(nmis)),
    }
