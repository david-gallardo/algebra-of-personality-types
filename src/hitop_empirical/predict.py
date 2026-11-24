"""Leak-free out-of-sample k-fold CV incremental predictive validity benchmarks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

from hitop_empirical.clustering import fit_matched_kmeans


def evaluate_incremental_cv(
    x_traits: NDArray[np.floating],
    y_target: NDArray[np.floating],
    *,
    task: Literal["regression", "classification"] = "regression",
    n_splits: int = 10,
    seed: int = 20261121,
) -> dict[str, dict[str, float]]:
    """Leak-free k-fold CV comparing linear spectra vs hard K=3/4, GMM-4 prob, and nonlinear models."""
    x = np.asarray(x_traits, dtype=np.float64)
    y = np.asarray(y_target, dtype=np.float64 if task == "regression" else np.int64)
    rng = np.random.default_rng(seed)

    splitter = (
        KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if task == "regression"
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )

    model_names = [
        "linear",
        "hard_k3",
        "traits_plus_hard_k3",
        "traits_plus_hard_k4",
        "traits_plus_gmm4_prob",
        "poly2",
    ]
    if task == "regression":
        model_names.extend(["spline_gam", "hgb"])

    fold_scores: dict[str, list[float]] = {m: [] for m in model_names}

    for tr, te in splitter.split(x, y):
        scaler = StandardScaler().fit(x[tr])
        x_tr = scaler.transform(x[tr])
        x_te = scaler.transform(x[te])
        y_tr, y_te = y[tr], y[te]

        # Fit clustering strictly on training fold
        km3 = fit_matched_kmeans(x_tr, n_clusters=3, n_kmeanspp_starts=9, rng=rng)
        km4 = fit_matched_kmeans(x_tr, n_clusters=4, n_kmeanspp_starts=9, rng=rng)
        l3_te = np.argmin(np.sum((x_te[:, None, :] - km3.centers[None, :, :]) ** 2, axis=2), axis=1)
        l4_te = np.argmin(np.sum((x_te[:, None, :] - km4.centers[None, :, :]) ** 2, axis=2), axis=1)
        oh3_tr, oh3_te = np.eye(3)[km3.labels], np.eye(3)[l3_te]
        oh4_tr, oh4_te = np.eye(4)[km4.labels], np.eye(4)[l4_te]

        gmm4 = GaussianMixture(n_components=4, covariance_type="full", n_init=5, random_state=int(rng.integers(0, 2**31 - 1))).fit(x_tr)
        p4_tr, p4_te = gmm4.predict_proba(x_tr), gmm4.predict_proba(x_te)

        poly = PolynomialFeatures(degree=2, include_bias=False).fit(x_tr)
        xp_tr, xp_te = poly.transform(x_tr), poly.transform(x_te)

        feature_sets = {
            "linear": (x_tr, x_te),
            "hard_k3": (oh3_tr, oh3_te),
            "traits_plus_hard_k3": (np.hstack([x_tr, oh3_tr]), np.hstack([x_te, oh3_te])),
            "traits_plus_hard_k4": (np.hstack([x_tr, oh4_tr]), np.hstack([x_te, oh4_te])),
            "traits_plus_gmm4_prob": (np.hstack([x_tr, p4_tr]), np.hstack([x_te, p4_te])),
            "poly2": (xp_tr, xp_te),
        }

        if task == "regression":
            sp = SplineTransformer(n_knots=4, degree=3, include_bias=False).fit(x_tr)
            feature_sets["spline_gam"] = (sp.transform(x_tr), sp.transform(x_te))
            for m, (ftr, fte) in feature_sets.items():
                reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(ftr, y_tr)
                fold_scores[m].append(float(r2_score(y_te, reg.predict(fte))))
            hgb = HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=20, l2_regularization=10.0, random_state=seed).fit(x_tr, y_tr)
            fold_scores["hgb"].append(float(r2_score(y_te, hgb.predict(x_te))))
        else:
            for m, (ftr, fte) in feature_sets.items():
                clf = LogisticRegression(C=1.0, max_iter=1000).fit(ftr, y_tr)
                fold_scores[m].append(float(roc_auc_score(y_te, clf.predict_proba(fte)[:, 1])))

    base = np.asarray(fold_scores["linear"], dtype=np.float64)
    summary: dict[str, dict[str, float]] = {}
    for m, sc in fold_scores.items():
        arr = np.asarray(sc, dtype=np.float64)
        summary[m] = {
            "mean": float(np.mean(arr)),
            "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "mean_delta_vs_linear": float(np.mean(arr - base)),
        }
    return summary
