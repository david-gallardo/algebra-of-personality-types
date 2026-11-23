"""Clustering methods and Gerlach et al. (2018) density-enrichment procedure.

All functions assume input data matrices of shape (n_samples, n_features) following
the strict Big Five ordering (N, E, O, A, C) with N unreversed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from arc_study.generators import (
    _apply_gaussian_copula,
    calibrate_gaussian_copula,
    empirical_marginal_ppfs,
    nearest_pd_correlation,
    sample_null1_column_permutation,
    sample_null2_correlated_normal,
)
from arc_study.matrices import BASE_SEED

NullType = Literal["permutation", "null2", "null3"]
KdeMode = Literal["scott", "isotropic"]
RandomStateLike = int | np.random.Generator | None


@dataclass(frozen=True)
class WardKMeansResult:
    """Result of Ward hierarchical initialization followed by k-means refinement."""

    n_clusters: int
    labels: NDArray[np.intp]
    centers: NDArray[np.float64]
    ward_centers: NDArray[np.float64]
    inertia: float
    between_variance: float
    model: KMeans


@dataclass(frozen=True)
class GMMFitResult:
    """Result of a Gaussian Mixture / Latent Profile Analysis fit."""

    n_clusters: int
    covariance_type: Literal["diag", "full"]
    labels: NDArray[np.intp]
    responsibilities: NDArray[np.float64]
    means: NDArray[np.float64]
    weights: NDArray[np.float64]
    covariances: NDArray[np.float64]
    bic: float
    aic: float
    log_likelihood: float
    mean_log_likelihood: float
    converged: bool
    model: GaussianMixture


@dataclass(frozen=True)
class GerlachNullEvaluation:
    """Enrichment and p-value evaluation of GMM centers against a single null model."""

    null_name: str
    null_densities: NDArray[np.float64]
    mean_null_density: NDArray[np.float64]
    enrichment: NDArray[np.float64]
    p_values: NDArray[np.float64]
    is_significant: NDArray[np.bool_]
    n_significant_types: int

    def count_significant(self, *, min_enrichment: float = 1.25, alpha: float = 0.01) -> int:
        """Count centers satisfying enrichment > ``min_enrichment`` and p < ``alpha``."""
        return int(np.sum((self.enrichment > min_enrichment) & (self.p_values < alpha)))


@dataclass(frozen=True)
class GerlachResult:
    """Full output of the Gerlach et al. (2018) KDE-enrichment procedure."""

    gmm: GMMFitResult
    centers: NDArray[np.float64]
    empirical_densities: NDArray[np.float64]
    evaluations: dict[str, GerlachNullEvaluation]
    n_significant_by_null: dict[str, int]


def _validate_data(data: NDArray[np.floating], min_rows: int = 2) -> NDArray[np.float64]:
    """Validate and coerce ``data`` to a finite 2D float64 array."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < min_rows or arr.shape[1] == 0:
        raise ValueError(
            f"Expected a 2D array with at least {min_rows} rows and >= 1 col, got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Input data contains non-finite values.")
    return arr


def _resolve_rng_and_seed(
    random_state: RandomStateLike,
) -> tuple[np.random.Generator, int]:
    """Return a (Generator, int seed) pair from an int, Generator, or None."""
    if isinstance(random_state, np.random.Generator):
        seed = int(random_state.integers(0, 2**31 - 1))
        return random_state, seed
    if random_state is None:
        rng = np.random.default_rng()
        seed = int(rng.integers(0, 2**31 - 1))
        return rng, seed
    seed = int(random_state)
    return np.random.default_rng(seed), seed


# ---------------------------------------------------------------------------
# 1. Ward + k-means (K = 3 default)
# ---------------------------------------------------------------------------


def fit_ward_kmeans(
    data: NDArray[np.floating],
    n_clusters: int = 3,
    *,
    random_state: RandomStateLike = BASE_SEED,
    ward_subsample: int | None = None,
) -> WardKMeansResult:
    """Run Ward agglomerative clustering to initialize centroids, then refine with k-means."""
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}.")
    x = _validate_data(data, min_rows=n_clusters)
    n, p = x.shape
    rng, seed = _resolve_rng_and_seed(random_state)

    if ward_subsample is not None and ward_subsample < n:
        if ward_subsample < n_clusters:
            raise ValueError("ward_subsample must be >= n_clusters.")
        sub_idx = rng.choice(n, size=ward_subsample, replace=False)
        x_ward = x[sub_idx]
    else:
        x_ward = x

    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    ward_labels = ward.fit_predict(x_ward)

    ward_centers = np.empty((n_clusters, p), dtype=np.float64)
    for k in range(n_clusters):
        ward_centers[k] = x_ward[ward_labels == k].mean(axis=0)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init=ward_centers,
        n_init=1,
        random_state=seed,
    )
    labels = np.asarray(kmeans.fit_predict(x), dtype=np.intp)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)

    global_mean = x.mean(axis=0)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    between_var = float(
        np.sum((counts / float(n)) * np.sum((centers - global_mean) ** 2, axis=1))
    )

    return WardKMeansResult(
        n_clusters=n_clusters,
        labels=labels,
        centers=centers,
        ward_centers=ward_centers,
        inertia=float(kmeans.inertia_),
        between_variance=between_var,
        model=kmeans,
    )


# ---------------------------------------------------------------------------
# 2. Diagonal LPA & 3. Full-covariance GMM
# ---------------------------------------------------------------------------


def _fit_gmm(
    data: NDArray[np.floating],
    n_clusters: int,
    covariance_type: Literal["diag", "full"],
    *,
    n_init: int = 10,
    reg_covar: float = 1e-6,
    max_iter: int = 300,
    random_state: RandomStateLike = BASE_SEED,
) -> GMMFitResult:
    """Shared GaussianMixture fitting routine returning BIC, AIC, and log-likelihood."""
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}.")
    x = _validate_data(data, min_rows=n_clusters)
    n = x.shape[0]
    _, seed = _resolve_rng_and_seed(random_state)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        n_init=n_init,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=seed,
    )
    gmm.fit(x)

    responsibilities = np.asarray(gmm.predict_proba(x), dtype=np.float64)
    labels = np.asarray(np.argmax(responsibilities, axis=1), dtype=np.intp)
    mean_ll = float(gmm.score(x))
    total_ll = mean_ll * float(n)

    return GMMFitResult(
        n_clusters=n_clusters,
        covariance_type=covariance_type,
        labels=labels,
        responsibilities=responsibilities,
        means=np.asarray(gmm.means_, dtype=np.float64),
        weights=np.asarray(gmm.weights_, dtype=np.float64),
        covariances=np.asarray(gmm.covariances_, dtype=np.float64),
        bic=float(gmm.bic(x)),
        aic=float(gmm.aic(x)),
        log_likelihood=total_ll,
        mean_log_likelihood=mean_ll,
        converged=bool(gmm.converged_),
        model=gmm,
    )


def fit_lpa_diagonal(
    data: NDArray[np.floating],
    n_clusters: int = 3,
    *,
    n_init: int = 10,
    reg_covar: float = 1e-6,
    max_iter: int = 300,
    random_state: RandomStateLike = BASE_SEED,
) -> GMMFitResult:
    """Fit a diagonal-covariance Latent Profile Analysis (LPA) model for configurable K."""
    return _fit_gmm(
        data=data,
        n_clusters=n_clusters,
        covariance_type="diag",
        n_init=n_init,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=random_state,
    )


def fit_gmm_full(
    data: NDArray[np.floating],
    n_clusters: int = 3,
    *,
    n_init: int = 10,
    reg_covar: float = 1e-6,
    max_iter: int = 300,
    random_state: RandomStateLike = BASE_SEED,
) -> GMMFitResult:
    """Fit a full-covariance Gaussian Mixture Model returning BIC, AIC, and log-likelihood."""
    return _fit_gmm(
        data=data,
        n_clusters=n_clusters,
        covariance_type="full",
        n_init=n_init,
        reg_covar=reg_covar,
        max_iter=max_iter,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# 4. Gerlach et al. (2018) GMM + KDE enrichment against Null 1, Null 2, Null 3
# ---------------------------------------------------------------------------


def _eval_kde_at_centers(
    sample: NDArray[np.float64],
    centers: NDArray[np.float64],
    *,
    kde_mode: KdeMode = "isotropic",
    isotropic_bandwidth: float = 0.50,
    bw_method: str | float | None = "scott",
    rng: np.random.Generator | None = None,
    max_kde_points: int | None = None,
) -> NDArray[np.float64]:
    """Fit a multivariate Gaussian KDE on ``sample`` and evaluate it at ``centers``."""
    n = sample.shape[0]
    if max_kde_points is not None and n > max_kde_points:
        if rng is None:
            sub = sample[:max_kde_points]
        else:
            idx = rng.choice(n, size=max_kde_points, replace=False)
            sub = sample[idx]
    else:
        sub = sample

    if kde_mode == "scott":
        kde = gaussian_kde(sub.T, bw_method=bw_method)
        return np.asarray(kde(centers.T), dtype=np.float64)
    if kde_mode == "isotropic":
        if isotropic_bandwidth <= 0.0:
            raise ValueError(
                f"isotropic_bandwidth must be strictly positive, got {isotropic_bandwidth}."
            )
        kde_iso = KernelDensity(bandwidth=isotropic_bandwidth, kernel="gaussian")
        kde_iso.fit(sub)
        return np.asarray(np.exp(kde_iso.score_samples(centers)), dtype=np.float64)
    raise ValueError(f"Unknown kde_mode '{kde_mode}'. Expected 'scott' or 'isotropic'.")


def _evaluate_gerlach_null(
    data: NDArray[np.float64],
    centers: NDArray[np.float64],
    empirical_densities: NDArray[np.float64],
    null_name: NullType,
    n_null_samples: int,
    rng: np.random.Generator,
    *,
    alpha: float,
    min_enrichment: float,
    kde_mode: KdeMode,
    isotropic_bandwidth: float,
    kde_bw_method: str | float | None,
    max_kde_points: int | None,
    copula_calibration_n: int,
    copula_calibration_tol: float,
) -> GerlachNullEvaluation:
    """Evaluate KDE enrichment and Monte Carlo p-values under a chosen null model."""
    n, p = data.shape
    k = centers.shape[0]
    null_densities = np.empty((n_null_samples, k), dtype=np.float64)

    if null_name == "permutation":
        for b in range(n_null_samples):
            null_sample = sample_null1_column_permutation(data, rng)
            null_densities[b] = _eval_kde_at_centers(
                null_sample,
                centers,
                kde_mode=kde_mode,
                isotropic_bandwidth=isotropic_bandwidth,
                bw_method=kde_bw_method,
                rng=rng,
                max_kde_points=max_kde_points,
            )
    elif null_name == "null2":
        emp_mean = data.mean(axis=0)
        emp_std = data.std(axis=0, ddof=1)
        emp_std = np.where(emp_std > 0.0, emp_std, 1.0)
        r_emp = nearest_pd_correlation(np.corrcoef(data, rowvar=False))
        for b in range(n_null_samples):
            z_null = sample_null2_correlated_normal(n, r_emp, rng)
            null_sample = emp_mean + z_null * emp_std
            null_densities[b] = _eval_kde_at_centers(
                null_sample,
                centers,
                kde_mode=kde_mode,
                isotropic_bandwidth=isotropic_bandwidth,
                bw_method=kde_bw_method,
                rng=rng,
                max_kde_points=max_kde_points,
            )
    elif null_name == "null3":
        r_emp = nearest_pd_correlation(np.corrcoef(data, rowvar=False))
        ppfs = empirical_marginal_ppfs(data)
        calibration = calibrate_gaussian_copula(
            target_corr=r_emp,
            marginals=ppfs,
            rng=rng,
            n_cal=copula_calibration_n,
            tol=copula_calibration_tol,
        )
        for b in range(n_null_samples):
            z = rng.standard_normal((n, p))
            null_sample = _apply_gaussian_copula(z, calibration.latent_corr, ppfs)
            null_densities[b] = _eval_kde_at_centers(
                null_sample,
                centers,
                kde_mode=kde_mode,
                isotropic_bandwidth=isotropic_bandwidth,
                bw_method=kde_bw_method,
                rng=rng,
                max_kde_points=max_kde_points,
            )
    else:
        raise ValueError(
            f"Unknown null_name '{null_name}'. Expected 'permutation', 'null2', or 'null3'."
        )

    mean_null = null_densities.mean(axis=0)
    enrichment = np.divide(
        empirical_densities,
        mean_null,
        out=np.full_like(empirical_densities, np.nan),
        where=mean_null > 0.0,
    )
    exceedances = np.sum(null_densities >= empirical_densities[np.newaxis, :], axis=0)
    p_values = (1.0 + exceedances) / float(n_null_samples + 1)
    is_significant = (enrichment > min_enrichment) & (p_values < alpha)

    return GerlachNullEvaluation(
        null_name=null_name,
        null_densities=null_densities,
        mean_null_density=mean_null,
        enrichment=enrichment,
        p_values=p_values,
        is_significant=is_significant,
        n_significant_types=int(np.sum(is_significant)),
    )


def run_gerlach_procedure(
    data: NDArray[np.floating],
    n_clusters: int = 4,
    *,
    covariance_type: Literal["full", "diag"] = "full",
    n_init: int = 10,
    reg_covar: float = 1e-6,
    n_null_samples: int = 200,
    null_models: Sequence[NullType] = ("permutation", "null2", "null3"),
    alpha: float = 0.01,
    min_enrichment: float = 1.25,
    kde_mode: KdeMode = "isotropic",
    isotropic_bandwidth: float = 0.50,
    kde_bw_method: str | float | None = "scott",
    max_kde_points: int | None = None,
    copula_calibration_n: int = 25_000,
    copula_calibration_tol: float = 1e-3,
    random_state: RandomStateLike = BASE_SEED,
) -> GerlachResult:
    """Run the Gerlach et al. (2018) GMM + KDE enrichment test against multiple nulls.

    Fits a GMM to ``data``, evaluates a multivariate Gaussian KDE at the GMM component
    centers, and computes density enrichment and Monte Carlo p-values against:
      - ``"permutation"``: independent column-wise empirical permutation (Null 1),
      - ``"null2"``: correlated multivariate normal matching empirical mean/SD/R (Null 2),
      - ``"null3"``: calibrated Gaussian copula preserving empirical marginals and R (Null 3).

    Supports fixed-bandwidth isotropic Gaussian KDE (``kde_mode="isotropic"``, closer to
    Gerlach et al., 2018) and covariance-scaled Scott KDE (``kde_mode="scott"``).
    """
    if n_null_samples < 1:
        raise ValueError(f"n_null_samples must be >= 1, got {n_null_samples}.")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
    p_min = 1.0 / float(n_null_samples + 1)
    if p_min >= alpha:
        min_required = int(np.floor(1.0 / alpha))
        raise ValueError(
            f"Impossible significance threshold: with n_null_samples={n_null_samples}, "
            f"the minimum attainable Monte Carlo p-value is 1/(n_null_samples+1) = {p_min:.5f} "
            f">= alpha={alpha}. Set n_null_samples >= {min_required}."
        )
    if not null_models:
        raise ValueError("null_models must contain at least one null model name.")

    x = _validate_data(data, min_rows=max(n_clusters, 10))
    rng, _ = _resolve_rng_and_seed(random_state)

    gmm_fit = _fit_gmm(
        data=x,
        n_clusters=n_clusters,
        covariance_type=covariance_type,
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=rng,
    )
    centers = gmm_fit.means
    emp_densities = _eval_kde_at_centers(
        x,
        centers,
        kde_mode=kde_mode,
        isotropic_bandwidth=isotropic_bandwidth,
        bw_method=kde_bw_method,
        rng=rng,
        max_kde_points=max_kde_points,
    )

    evaluations: dict[str, GerlachNullEvaluation] = {}
    for null_name in null_models:
        evaluations[null_name] = _evaluate_gerlach_null(
            data=x,
            centers=centers,
            empirical_densities=emp_densities,
            null_name=null_name,
            n_null_samples=n_null_samples,
            rng=rng,
            alpha=alpha,
            min_enrichment=min_enrichment,
            kde_mode=kde_mode,
            isotropic_bandwidth=isotropic_bandwidth,
            kde_bw_method=kde_bw_method,
            max_kde_points=max_kde_points,
            copula_calibration_n=copula_calibration_n,
            copula_calibration_tol=copula_calibration_tol,
        )

    n_sig_by_null = {name: ev.n_significant_types for name, ev in evaluations.items()}

    return GerlachResult(
        gmm=gmm_fit,
        centers=centers,
        empirical_densities=emp_densities,
        evaluations=evaluations,
        n_significant_by_null=n_sig_by_null,
    )
