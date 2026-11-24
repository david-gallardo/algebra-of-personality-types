"""Matched Ward + multistart k-means++ and single-start Ward clustering protocols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans


@dataclass(frozen=True)
class ClusteringSolution:
    """Result of a matched clustering fit with optimizer basin diagnostics."""

    n_clusters: int
    labels: NDArray[np.intp]
    centers: NDArray[np.float64]
    sse: float
    all_start_sses: NDArray[np.float64]
    best_start_index: int  # 0 = Ward initialization; >=1 = k-means++ starts


def _ward_initial_centers(
    x: NDArray[np.float64],
    n_clusters: int,
    rng: np.random.Generator,
    ward_subsample: int | None,
) -> NDArray[np.float64]:
    n, p = x.shape
    if ward_subsample is not None and n > ward_subsample:
        idx = rng.choice(n, size=ward_subsample, replace=False)
        x_w = x[idx]
    else:
        x_w = x
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    w_labels = ward.fit_predict(x_w)
    centers = np.empty((n_clusters, p), dtype=np.float64)
    for k in range(n_clusters):
        centers[k] = x_w[w_labels == k].mean(axis=0)
    return centers


def fit_matched_kmeans(
    data: NDArray[np.floating],
    n_clusters: int = 3,
    *,
    n_kmeanspp_starts: int = 9,
    ward_subsample: int | None = 2000,
    sample_weight: NDArray[np.floating] | None = None,
    rng: np.random.Generator,
) -> ClusteringSolution:
    """Fit Ward initialization + ``n_kmeanspp_starts`` k-means++ starts, retaining min SSE.

    Set ``n_kmeanspp_starts=0`` for the matched single-start Ward + k-means sensitivity protocol.
    """
    x = np.asarray(data, dtype=np.float64)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

    ward_init = _ward_initial_centers(x, n_clusters, rng, ward_subsample)
    seed0 = int(rng.integers(0, 2**31 - 1))
    km0 = KMeans(n_clusters=n_clusters, init=ward_init, n_init=1, random_state=seed0)
    km0.fit(x, sample_weight=sw)

    best_km = km0
    best_idx = 0
    sses = [float(km0.inertia_)]

    for s in range(1, n_kmeanspp_starts + 1):
        seed_s = int(rng.integers(0, 2**31 - 1))
        km_s = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, random_state=seed_s)
        km_s.fit(x, sample_weight=sw)
        sse_s = float(km_s.inertia_)
        sses.append(sse_s)
        if sse_s < float(best_km.inertia_):
            best_km = km_s
            best_idx = s

    return ClusteringSolution(
        n_clusters=n_clusters,
        labels=np.asarray(best_km.labels_, dtype=np.intp),
        centers=np.asarray(best_km.cluster_centers_, dtype=np.float64),
        sse=float(best_km.inertia_),
        all_start_sses=np.asarray(sses, dtype=np.float64),
        best_start_index=best_idx,
    )
