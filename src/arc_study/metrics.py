"""Profile similarity, centroid matching, replicability, and incremental validity metrics.

All profile vectors and centroid matrices follow the strict Big Five ordering
(N, E, O, A, C) with N unreversed.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    cohen_kappa_score,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arc_study.clustering import fit_gmm_full, fit_lpa_diagonal, fit_ward_kmeans
from arc_study.matrices import BASE_SEED

MatchMetric = Literal["d2", "euclidean", "tucker"]
RandomStateLike = int | np.random.Generator | None
ClusterFitFn = Callable[
    [NDArray[np.float64], int],
    tuple[NDArray[np.intp], NDArray[np.float64]],
]
ClusterFeatureExtractor = Callable[
    [NDArray[np.float64], NDArray[np.float64], int],
    tuple[NDArray[np.float64], NDArray[np.float64]],
]

# Stylized theoretical ARC z-score prototypes inspired by the canonical ARC pattern
# (Asendorpf et al., 2001) in strict (N, E, O, A, C) order with N unreversed — not
# literal centroids from a specific published table. Values are frozen to preserve
# exact comparability across all simulation blocks.
ARC_LABELS: tuple[str, ...] = ("Resilient", "Overcontrolled", "Undercontrolled")
ARC_PROTOTYPES: NDArray[np.float64] = np.array(
    [
        [-0.80, 0.60, 0.40, 0.60, 0.70],  # Resilient: low N, high E/O/A/C
        [0.80, -0.70, -0.20, -0.10, 0.10],  # Overcontrolled: high N, low E
        [0.30, 0.10, -0.20, -0.80, -0.80],  # Undercontrolled: low A, low C
    ],
    dtype=np.float64,
)
ARC_PROTOTYPES.flags.writeable = False


@dataclass(frozen=True)
class CronbachGleserResult:
    """Cronbach & Gleser (1953) D^2 decomposition plus Tucker's congruence coefficient."""

    d2: float
    elevation: float
    scatter: float
    shape: float
    tucker_congruence: float


@dataclass(frozen=True)
class CentroidMatchResult:
    """Optimal one-to-one matching between empirical centroids and target prototypes."""

    centroid_indices: tuple[int, ...]
    prototype_indices: tuple[int, ...]
    matched_centroids: NDArray[np.float64]
    cost_matrix: NDArray[np.float64]
    total_cost: float
    mean_d2: float
    mean_tucker: float
    profile_decompositions: tuple[CronbachGleserResult, ...]
    tucker_congruences: NDArray[np.float64]
    method: Literal["exhaustive", "hungarian"]


@dataclass(frozen=True)
class AssignmentStabilityResult:
    """Agreement between two clusterings evaluated on a common sample."""

    ari: float
    nmi: float


@dataclass(frozen=True)
class SplitHalfReplicabilityResult:
    """Split-half double cross-validation replicability (Asendorpf et al., 2001)."""

    n_splits: int
    kappas: NDArray[np.float64]
    aris: NDArray[np.float64]
    nmis: NDArray[np.float64]
    mean_kappa: float
    std_kappa: float
    mean_ari: float
    mean_nmi: float


@dataclass(frozen=True)
class IncrementalValidityResult:
    """Out-of-sample k-fold CV comparison of traits alone vs. traits + cluster features."""

    task: Literal["regression", "classification"]
    metric_name: str
    n_splits: int
    baseline_scores: NDArray[np.float64]
    augmented_scores: NDArray[np.float64]
    delta_scores: NDArray[np.float64]
    mean_baseline: float
    mean_augmented: float
    mean_delta: float
    std_delta: float


def _resolve_rng(random_state: RandomStateLike) -> np.random.Generator:
    """Coerce an int, Generator, or None into a NumPy Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None:
        return np.random.default_rng()
    return np.random.default_rng(int(random_state))


# ---------------------------------------------------------------------------
# 2. Cronbach & Gleser (1953) D^2 decomposition and Tucker's congruence
# ---------------------------------------------------------------------------


def tucker_congruence(
    profile_a: NDArray[np.floating],
    profile_b: NDArray[np.floating],
) -> float:
    """Compute Tucker's congruence coefficient phi between two 1D profiles."""
    a = np.asarray(profile_a, dtype=np.float64)
    b = np.asarray(profile_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError(
            f"Expected two non-empty 1D profiles of identical shape, got {a.shape} and {b.shape}."
        )
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


def cronbach_gleser_decomposition(
    profile_a: NDArray[np.floating],
    profile_b: NDArray[np.floating],
) -> CronbachGleserResult:
    """Decompose squared Euclidean profile distance D^2 into elevation, scatter, and shape.

    For two p-dimensional profiles x and y:
        D^2(x, y) = sum_j (x_j - y_j)^2
                  = p * (mean_x - mean_y)^2
                  + p * (sd_x - sd_y)^2
                  + 2 * p * sd_x * sd_y * (1 - r_xy)
    where sd_x and sd_y are population standard deviations (ddof=0) across traits.
    """
    a = np.asarray(profile_a, dtype=np.float64)
    b = np.asarray(profile_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError(
            f"Expected two non-empty 1D profiles of identical shape, got {a.shape} and {b.shape}."
        )

    p = float(a.size)
    diff = a - b
    d2 = float(np.dot(diff, diff))

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    elevation = p * (mean_a - mean_b) ** 2

    da = a - mean_a
    db = b - mean_b
    sd_a = float(np.sqrt(np.dot(da, da) / p))
    sd_b = float(np.sqrt(np.dot(db, db) / p))
    scatter = p * (sd_a - sd_b) ** 2

    if sd_a > 0.0 and sd_b > 0.0:
        r_ab = float(np.clip(np.dot(da, db) / (p * sd_a * sd_b), -1.0, 1.0))
        shape = max(0.0, 2.0 * p * sd_a * sd_b * (1.0 - r_ab))
    else:
        shape = 0.0

    phi = tucker_congruence(a, b)
    return CronbachGleserResult(
        d2=max(0.0, d2),
        elevation=max(0.0, elevation),
        scatter=max(0.0, scatter),
        shape=shape,
        tucker_congruence=phi,
    )


# ---------------------------------------------------------------------------
# 1. Optimal centroid matching (exhaustive 3! = 6 or Hungarian for arbitrary K)
# ---------------------------------------------------------------------------


def match_centroids(
    centroids: NDArray[np.floating],
    prototypes: NDArray[np.floating] = ARC_PROTOTYPES,
    *,
    metric: MatchMetric = "d2",
    exhaustive_max_k: int = 6,
) -> CentroidMatchResult:
    """Find the optimal one-to-one matching between empirical centroids and prototypes.

    Uses exhaustive search over all K! permutations when K == M <= ``exhaustive_max_k``
    (including the canonical K = 3 ARC case with 3! = 6 permutations), and falls back
    to the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``) for arbitrary K.
    """
    c = np.asarray(centroids, dtype=np.float64)
    p = np.asarray(prototypes, dtype=np.float64)
    if c.ndim != 2 or p.ndim != 2 or c.shape[0] == 0 or p.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D arrays, got {c.shape} and {p.shape}.")
    if c.shape[1] != p.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: centroids have {c.shape[1]}, prototypes have {p.shape[1]}."
        )

    k, m = c.shape[0], p.shape[0]
    cost = np.empty((k, m), dtype=np.float64)
    for i in range(k):
        for j in range(m):
            if metric == "d2":
                diff = c[i] - p[j]
                cost[i, j] = float(np.dot(diff, diff))
            elif metric == "euclidean":
                cost[i, j] = float(np.linalg.norm(c[i] - p[j]))
            elif metric == "tucker":
                phi = tucker_congruence(c[i], p[j])
                cost[i, j] = -phi if np.isfinite(phi) else 1.0
            else:
                raise ValueError(
                    f"Unknown metric '{metric}'. Expected 'd2', 'euclidean', or 'tucker'."
                )

    if k == m and k <= exhaustive_max_k:
        best_perm: tuple[int, ...] = tuple(range(m))
        best_cost = float("inf")
        for perm in itertools.permutations(range(m)):
            perm_cost = float(sum(cost[i, perm[i]] for i in range(k)))
            if perm_cost < best_cost:
                best_cost = perm_cost
                best_perm = perm
        # Sort pairs by prototype index so matched_centroids aligns with prototypes order
        pairs = sorted(enumerate(best_perm), key=lambda pair: pair[1])
        cent_idx = tuple(i for i, _ in pairs)
        proto_idx = tuple(j for _, j in pairs)
        total_cost = best_cost
        method: Literal["exhaustive", "hungarian"] = "exhaustive"
    else:
        row_ind, col_ind = linear_sum_assignment(cost)
        order = np.argsort(col_ind)
        cent_idx = tuple(int(row_ind[idx]) for idx in order)
        proto_idx = tuple(int(col_ind[idx]) for idx in order)
        total_cost = float(sum(cost[i, j] for i, j in zip(cent_idx, proto_idx)))
        method = "hungarian"

    matched_centroids = c[list(cent_idx)]
    decomps = tuple(
        cronbach_gleser_decomposition(c[i], p[j]) for i, j in zip(cent_idx, proto_idx)
    )
    tuckers = np.array([d.tucker_congruence for d in decomps], dtype=np.float64)
    mean_d2 = float(np.mean([d.d2 for d in decomps]))
    mean_tucker = float(np.nanmean(tuckers))

    return CentroidMatchResult(
        centroid_indices=cent_idx,
        prototype_indices=proto_idx,
        matched_centroids=matched_centroids,
        cost_matrix=cost,
        total_cost=total_cost,
        mean_d2=mean_d2,
        mean_tucker=mean_tucker,
        profile_decompositions=decomps,
        tucker_congruences=tuckers,
        method=method,
    )


def assign_nearest_centroids(
    data: NDArray[np.floating],
    centers: NDArray[np.floating],
) -> NDArray[np.intp]:
    """Assign each row of ``data`` to the index of its nearest Euclidean centroid."""
    x = np.asarray(data, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    if x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]:
        raise ValueError(f"Incompatible shapes for data {x.shape} and centers {c.shape}.")
    diffs = x[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    return np.asarray(np.argmin(dists_sq, axis=1), dtype=np.intp)


# ---------------------------------------------------------------------------
# 3. Assignment stability (ARI / NMI) & split-half replicability (Cohen's kappa)
# ---------------------------------------------------------------------------


def assignment_stability(
    labels_a: NDArray[np.integer],
    labels_b: NDArray[np.integer],
) -> AssignmentStabilityResult:
    """Compute Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)."""
    la = np.asarray(labels_a)
    lb = np.asarray(labels_b)
    if la.ndim != 1 or lb.ndim != 1 or la.shape != lb.shape or la.size == 0:
        raise ValueError(
            f"Expected two non-empty 1D label arrays of equal length, got {la.shape} and {lb.shape}."
        )
    ari = float(adjusted_rand_score(la, lb))
    nmi = float(normalized_mutual_info_score(la, lb))
    return AssignmentStabilityResult(ari=ari, nmi=nmi)


def split_half_replicability(
    data: NDArray[np.floating],
    fit_fn: ClusterFitFn | None = None,
    *,
    n_clusters: int = 3,
    n_splits: int = 10,
    random_state: RandomStateLike = BASE_SEED,
) -> SplitHalfReplicabilityResult:
    """Evaluate split-half double cross-validation replicability (Asendorpf et al., 2001).

    In each split, the sample is randomly divided into halves A and B; the clustering
    procedure ``fit_fn`` is run independently on both halves, centroids are optimally
    aligned, and each half is cross-classified by nearest-centroid assignment from the
    other half. Returns Cohen's kappa, ARI, and NMI across all splits.
    """
    x = np.asarray(data, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 * n_clusters:
        raise ValueError(
            f"Expected 2D array with at least {2 * n_clusters} rows, got {x.shape}."
        )
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}.")

    if fit_fn is None:
        def _default_fit(half: NDArray[np.float64], seed: int) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
            res = fit_ward_kmeans(half, n_clusters=n_clusters, random_state=seed)
            return res.labels, res.centers

        fitter: ClusterFitFn = _default_fit
    else:
        fitter = fit_fn

    rng = _resolve_rng(random_state)
    n = x.shape[0]
    half_n = n // 2

    kappas = np.empty(n_splits, dtype=np.float64)
    aris = np.empty(n_splits, dtype=np.float64)
    nmis = np.empty(n_splits, dtype=np.float64)

    for s in range(n_splits):
        perm = rng.permutation(n)
        x_a = x[perm[:half_n]]
        x_b = x[perm[half_n:]]

        seed_a = int(rng.integers(0, 2**31 - 1))
        seed_b = int(rng.integers(0, 2**31 - 1))
        labels_a, centers_a = fitter(x_a, seed_a)
        labels_b, centers_b = fitter(x_b, seed_b)

        # Align solution A's cluster indices to solution B's numbering
        match = match_centroids(centers_a, centers_b, metric="d2")
        remap_a_to_b = {
            c_idx: p_idx
            for c_idx, p_idx in zip(match.centroid_indices, match.prototype_indices)
        }
        aligned_labels_a = np.array([remap_a_to_b[int(lbl)] for lbl in labels_a], dtype=np.intp)
        aligned_centers_a = match.matched_centroids

        pred_b_from_a = assign_nearest_centroids(x_b, aligned_centers_a)
        pred_a_from_b = assign_nearest_centroids(x_a, centers_b)

        kappa_ab = float(cohen_kappa_score(labels_b, pred_b_from_a))
        kappa_ba = float(cohen_kappa_score(aligned_labels_a, pred_a_from_b))
        kappas[s] = 0.5 * (kappa_ab + kappa_ba)

        stab_ab = assignment_stability(labels_b, pred_b_from_a)
        stab_ba = assignment_stability(aligned_labels_a, pred_a_from_b)
        aris[s] = 0.5 * (stab_ab.ari + stab_ba.ari)
        nmis[s] = 0.5 * (stab_ab.nmi + stab_ba.nmi)

    return SplitHalfReplicabilityResult(
        n_splits=n_splits,
        kappas=kappas,
        aris=aris,
        nmis=nmis,
        mean_kappa=float(np.nanmean(kappas)),
        std_kappa=float(np.nanstd(kappas, ddof=0)),
        mean_ari=float(np.mean(aris)),
        mean_nmi=float(np.mean(nmis)),
    )


# ---------------------------------------------------------------------------
# 4. Out-of-sample incremental predictive validity (k-fold CV)
# ---------------------------------------------------------------------------


def make_ward_kmeans_onehot_extractor(
    n_clusters: int = 3,
    *,
    ward_subsample: int | None = None,
) -> ClusterFeatureExtractor:
    """Build a leak-free fold extractor returning one-hot Ward+k-means indicators."""

    def _extract(
        x_train: NDArray[np.float64],
        x_test: NDArray[np.float64],
        fold_seed: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        fit = fit_ward_kmeans(
            x_train,
            n_clusters=n_clusters,
            random_state=fold_seed,
            ward_subsample=ward_subsample,
        )
        test_labels = assign_nearest_centroids(x_test, fit.centers)
        f_train = np.eye(n_clusters, dtype=np.float64)[fit.labels]
        f_test = np.eye(n_clusters, dtype=np.float64)[test_labels]
        return f_train, f_test

    return _extract


def make_gmm_posterior_extractor(
    n_clusters: int = 3,
    *,
    covariance_type: Literal["diag", "full"] = "diag",
    n_init: int = 10,
    reg_covar: float = 1e-6,
) -> ClusterFeatureExtractor:
    """Build a leak-free fold extractor returning GMM/LPA posterior probabilities."""

    def _extract(
        x_train: NDArray[np.float64],
        x_test: NDArray[np.float64],
        fold_seed: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if covariance_type == "diag":
            fit = fit_lpa_diagonal(
                x_train,
                n_clusters=n_clusters,
                n_init=n_init,
                reg_covar=reg_covar,
                random_state=fold_seed,
            )
        else:
            fit = fit_gmm_full(
                x_train,
                n_clusters=n_clusters,
                n_init=n_init,
                reg_covar=reg_covar,
                random_state=fold_seed,
            )
        f_train = fit.responsibilities
        f_test = np.asarray(fit.model.predict_proba(x_test), dtype=np.float64)
        return f_train, f_test

    return _extract


def incremental_predictive_validity(
    traits: NDArray[np.floating],
    target: NDArray[np.floating],
    *,
    cluster_features: NDArray[np.floating] | None = None,
    cluster_extractor: ClusterFeatureExtractor | None = None,
    task: Literal["regression", "classification"] = "regression",
    n_splits: int = 5,
    alpha: float = 1.0,
    random_state: RandomStateLike = BASE_SEED,
) -> IncrementalValidityResult:
    """Compare out-of-sample k-fold CV performance of traits alone vs. traits + profiles.

    Provide either ``cluster_extractor`` (recommended: fits clustering inside each
    training fold to avoid data leakage) or precomputed ``cluster_features`` (e.g.,
    one-hot profile indicators or posterior probabilities of shape (n_samples, K)).
    """
    if (cluster_features is None) == (cluster_extractor is None):
        raise ValueError(
            "Provide exactly one of 'cluster_features' or 'cluster_extractor'."
        )
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}.")

    x = np.asarray(traits, dtype=np.float64)
    y = np.asarray(target)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Incompatible shapes: traits {x.shape}, target {y.shape}."
        )
    if cluster_features is not None:
        cf = np.asarray(cluster_features, dtype=np.float64)
        if cf.ndim == 1:
            cf = cf[:, np.newaxis]
        if cf.ndim != 2 or cf.shape[0] != x.shape[0]:
            raise ValueError(
                f"cluster_features must have {x.shape[0]} rows, got shape {cf.shape}."
            )
    else:
        cf = None

    rng = _resolve_rng(random_state)
    cv_seed = int(rng.integers(0, 2**31 - 1))

    if task == "regression":
        y_arr = y.astype(np.float64)
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
        metric_name = "r2"
        is_binary = False
    elif task == "classification":
        y_arr = y
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)
        is_binary = len(np.unique(y_arr)) == 2
        metric_name = "roc_auc" if is_binary else "accuracy"
    else:
        raise ValueError(f"Unknown task '{task}'. Expected 'regression' or 'classification'.")

    baseline_scores = np.empty(n_splits, dtype=np.float64)
    augmented_scores = np.empty(n_splits, dtype=np.float64)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y_arr)):
        x_tr, x_te = x[train_idx], x[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        if cluster_extractor is not None:
            fold_seed = int(rng.integers(0, 2**31 - 1))
            f_tr, f_te = cluster_extractor(x_tr, x_te, fold_seed)
        else:
            assert cf is not None
            f_tr, f_te = cf[train_idx], cf[test_idx]

        x_tr_aug = np.hstack([x_tr, f_tr])
        x_te_aug = np.hstack([x_te, f_te])

        if task == "regression":
            base_model = Pipeline(
                [("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))]
            )
            aug_model = Pipeline(
                [("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))]
            )
            base_model.fit(x_tr, y_tr)
            aug_model.fit(x_tr_aug, y_tr)
            baseline_scores[fold_idx] = float(r2_score(y_te, base_model.predict(x_te)))
            augmented_scores[fold_idx] = float(r2_score(y_te, aug_model.predict(x_te_aug)))
        else:
            base_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(C=1.0 / alpha, max_iter=1000)),
                ]
            )
            aug_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(C=1.0 / alpha, max_iter=1000)),
                ]
            )
            base_model.fit(x_tr, y_tr)
            aug_model.fit(x_tr_aug, y_tr)
            if is_binary:
                baseline_scores[fold_idx] = float(
                    roc_auc_score(y_te, base_model.predict_proba(x_te)[:, 1])
                )
                augmented_scores[fold_idx] = float(
                    roc_auc_score(y_te, aug_model.predict_proba(x_te_aug)[:, 1])
                )
            else:
                baseline_scores[fold_idx] = float(
                    accuracy_score(y_te, base_model.predict(x_te))
                )
                augmented_scores[fold_idx] = float(
                    accuracy_score(y_te, aug_model.predict(x_te_aug))
                )

    delta_scores = augmented_scores - baseline_scores
    return IncrementalValidityResult(
        task=task,
        metric_name=metric_name,
        n_splits=n_splits,
        baseline_scores=baseline_scores,
        augmented_scores=augmented_scores,
        delta_scores=delta_scores,
        mean_baseline=float(np.mean(baseline_scores)),
        mean_augmented=float(np.mean(augmented_scores)),
        mean_delta=float(np.mean(delta_scores)),
        std_delta=float(np.std(delta_scores, ddof=0)),
    )
