#!/usr/bin/env python3
"""Bloc 4: Out-of-sample incremental predictive validity on Soto (2019) LOOPR (N=6,126)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import cohen_kappa_score, log_loss, r2_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, str(Path("src").resolve()))
from arc_study.metrics import ARC_PROTOTYPES

GERLACH_4_CENTERS = np.array([
    [-0.53, +0.01, -0.72, -0.13, +0.43],
    [-0.60, +0.62, +0.35, +0.49, +0.73],
    [-0.26, +0.02, -0.63, +0.62, -0.10],
    [-0.25, +0.72, -0.72, -0.60, -0.51],
], dtype=np.float64)

SOTO_FILES = [
    ("S1_Adults", "Replication data - Survey 1 - Adults.sav", "S1"),
    ("S1_Young", "Replication data - Survey 1 - Young adults.sav", "S1"),
    ("S2_Adults", "Replication data - Survey 2 - Adults.sav", "S2"),
    ("S2_Young", "Replication data - Survey 2 - Young adults.sav", "S2"),
]

EXCLUDE_SUBSTRINGS = (
    "CaseID", "Age", "Sex", "ZS1_", "MaritalStatus", "RomanticStatus", "WorkStatus",
    "Immigrant", "HispanicLatino", "Race", "FamilyMother1", "FamilyFather1",
    "FamilyNotUnemployed", "FamilyEmployedVsStudentHomemaker", "FamilyMarried",
    "HeartAgreeablenessSplit", "EatingHeight",
)


def is_outcome_col(col: str, prefix: str) -> bool:
    if col in {
        f"{prefix}_PersonalityNegativeEmotionality",
        f"{prefix}_PersonalityExtraversion",
        f"{prefix}_PersonalityOpenMindedness",
        f"{prefix}_PersonalityAgreeableness",
        f"{prefix}_PersonalityConscientiousness",
    }:
        return False
    return not any(sub in col for sub in EXCLUDE_SUBSTRINGS)


def ward_kmeans(Z: np.ndarray, k: int = 3, ward_sub: int = 2000, seed: int = 20261121) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    Z_sub = Z[rng.choice(n, size=ward_sub, replace=False)] if n > ward_sub else Z
    w_lbl = fcluster(linkage(Z_sub, method="ward"), t=k, criterion="maxclust")
    init_c = np.vstack([Z_sub[w_lbl == j].mean(axis=0) for j in range(1, k + 1)])
    km = KMeans(n_clusters=k, init=init_c, n_init=1, max_iter=300, random_state=seed)
    km.fit(Z)
    return km.cluster_centers_, km.labels_


def split_half_kappa(Z: np.ndarray, k: int = 3, n_splits: int = 5, seed: int = 20261121) -> float:
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    kappas = []
    for s in range(n_splits):
        perm = rng.permutation(n)
        mid = n // 2
        A, B = Z[perm[:mid]], Z[perm[mid:]]
        cA, _ = ward_kmeans(A, k=k, seed=seed + 10 * s)
        _, lB = ward_kmeans(B, k=k, seed=seed + 10 * s + 1)
        pred_A = np.argmin(np.sum((B[:, None, :] - cA[None, :, :]) ** 2, axis=2), axis=1)
        cm = np.zeros((k, k), dtype=int)
        for a_i, b_i in zip(pred_A, lB):
            cm[a_i, b_i] += 1
        ri, ci = linear_sum_assignment(-cm)
        mapping = {ci[j]: ri[j] for j in range(k)}
        lB_al = np.array([mapping[v] for v in lB])
        kappas.append(cohen_kappa_score(pred_A, lB_al))
    return float(np.mean(kappas))


def align_centers_to_arc(centers: np.ndarray) -> tuple[np.ndarray, float, float]:
    cost = np.sum((centers[:, None, :] - ARC_PROTOTYPES[None, :, :]) ** 2, axis=2)
    ri, ci = linear_sum_assignment(cost)
    perm = np.empty(3, dtype=int)
    d2s, phis = [], []
    for r, c in zip(ri, ci):
        perm[c] = r
        u, v = centers[r], ARC_PROTOTYPES[c]
        d2s.append(float(np.sum((u - v) ** 2)))
        phis.append(float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)))
    return perm, float(np.mean(d2s)), float(np.mean(phis))


def fit_fold_features(X_tr: np.ndarray, X_te: np.ndarray, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z_tr = (X_tr - mu) / sd
    Z_te = (X_te - mu) / sd

    c_wk, _ = ward_kmeans(Z_tr, k=3, seed=seed)
    perm, _, _ = align_centers_to_arc(c_wk)
    c_arc = c_wk[perm]
    lbl_tr = np.argmin(np.sum((Z_tr[:, None, :] - c_arc[None, :, :]) ** 2, axis=2), axis=1)
    lbl_te = np.argmin(np.sum((Z_te[:, None, :] - c_arc[None, :, :]) ** 2, axis=2), axis=1)
    arc_hard_tr = np.column_stack([(lbl_tr == 1).astype(float), (lbl_tr == 2).astype(float)])
    arc_hard_te = np.column_stack([(lbl_te == 1).astype(float), (lbl_te == 2).astype(float)])

    gmm3 = GaussianMixture(n_components=3, covariance_type="diag", n_init=5, reg_covar=1e-6, random_state=seed)
    gmm3.fit(Z_tr)
    p3, _, _ = align_centers_to_arc(gmm3.means_)
    lpa3_tr = gmm3.predict_proba(Z_tr)[:, p3[1:]]
    lpa3_te = gmm3.predict_proba(Z_te)[:, p3[1:]]

    gmm4 = GaussianMixture(n_components=4, covariance_type="full", n_init=5, reg_covar=1e-6, random_state=seed)
    gmm4.fit(Z_tr)
    gmm4_tr = gmm4.predict_proba(Z_tr)[:, 1:]
    gmm4_te = gmm4.predict_proba(Z_te)[:, 1:]

    d2_tr = np.sum((Z_tr[:, None, :] - GERLACH_4_CENTERS[None, :, :]) ** 2, axis=2)
    d2_te = np.sum((Z_te[:, None, :] - GERLACH_4_CENTERS[None, :, :]) ** 2, axis=2)
    g_hard_tr = np.column_stack([(np.argmin(d2_tr, axis=1) == k).astype(float) for k in range(1, 4)])
    g_hard_te = np.column_stack([(np.argmin(d2_te, axis=1) == k).astype(float) for k in range(1, 4)])

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_tr = poly.fit_transform(Z_tr)
    poly_te = poly.transform(Z_te)

    return {
        "ARC_Hard": (arc_hard_tr, arc_hard_te),
        "LPA3_Prob": (lpa3_tr, lpa3_te),
        "GMM4_Prob": (gmm4_tr, gmm4_te),
        "Gerlach4_Hard": (g_hard_tr, g_hard_te),
        "Traits_Linear": (Z_tr, Z_te),
        "Traits+ARC_Hard": (np.hstack([Z_tr, arc_hard_tr]), np.hstack([Z_te, arc_hard_te])),
        "Traits+LPA3_Prob": (np.hstack([Z_tr, lpa3_tr]), np.hstack([Z_te, lpa3_te])),
        "Traits+GMM4_Prob": (np.hstack([Z_tr, gmm4_tr]), np.hstack([Z_te, gmm4_te])),
        "Traits+Gerlach4": (np.hstack([Z_tr, g_hard_tr]), np.hstack([Z_te, g_hard_te])),
        "Traits_Poly2": (poly_tr, poly_te),
    }


def evaluate_outcome_cv(X: np.ndarray, y: np.ndarray, *, is_binary: bool, n_splits: int = 10, seed: int = 20261121) -> dict[str, float]:
    alphas = np.logspace(-3, 4, 29)
    if is_binary:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        vals = np.sort(np.unique(y))
        y_int = (y == vals[-1]).astype(int)
        splits = list(splitter.split(X, y_int))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y_std = (y - y.mean()) / max(y.std(ddof=1), 1e-12)
        splits = list(splitter.split(X))

    model_names = [
        "ARC_Hard", "LPA3_Prob", "GMM4_Prob", "Gerlach4_Hard",
        "Traits_Linear", "Traits+ARC_Hard", "Traits+LPA3_Prob",
        "Traits+GMM4_Prob", "Traits+Gerlach4", "Traits_Poly2",
    ]
    oof_preds = {m: np.empty(len(y), dtype=np.float64) for m in model_names}

    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        feats = fit_fold_features(X[tr_idx], X[te_idx], seed=seed + 100 * fold_i)
        for m in model_names:
            F_tr, F_te = feats[m]
            if is_binary:
                clf = LogisticRegression(C=10.0, max_iter=500, solver="lbfgs")
                clf.fit(F_tr, y_int[tr_idx])
                oof_preds[m][te_idx] = clf.predict_proba(F_te)[:, 1]
            else:
                reg = RidgeCV(alphas=alphas)
                reg.fit(F_tr, y_std[tr_idx])
                oof_preds[m][te_idx] = reg.predict(F_te)

    scores: dict[str, float] = {}
    for m in model_names:
        if is_binary:
            scores[m] = float(roc_auc_score(y_int, oof_preds[m]))
            scores[f"{m}_logloss"] = float(log_loss(y_int, np.clip(oof_preds[m], 1e-6, 1 - 1e-6)))
        else:
            scores[m] = float(r2_score(y_std, oof_preds[m]))
    return scores


def mean_shift_mode_count(Z: np.ndarray, starts: np.ndarray, bw: float) -> tuple[int, np.ndarray]:
    pts = starts.copy()
    inv = 0.5 / (bw * bw)
    for _ in range(150):
        p2 = np.sum(pts * pts, axis=1)
        z2 = np.sum(Z * Z, axis=1, keepdims=True)
        sq = np.maximum(0.0, z2 + p2[None, :] - 2.0 * (Z @ pts.T))
        W = np.exp(-sq * inv)
        nxt = (W.T @ Z) / np.maximum(np.sum(W, axis=0)[:, None], 1e-300)
        if np.max(np.linalg.norm(nxt - pts, axis=1)) < 1e-4:
            pts = nxt
            break
        pts = nxt
    uniq = []
    for m in pts:
        if all(np.linalg.norm(m - u) >= 0.05 for u in uniq):
            uniq.append(m)
    return len(uniq), np.array(uniq)


def main() -> int:
    t0 = time.perf_counter()
    soto_dir = Path("data/raw/soto2019/Supporting materials/1 Data and code - Replication attempts")
    pooled_X = []
    rows = []

    for sub_name, fname, pref in SOTO_FILES:
        df, _ = pyreadstat.read_sav(str(soto_dir / fname))
        bf_cols = [
            f"{pref}_PersonalityNegativeEmotionality",
            f"{pref}_PersonalityExtraversion",
            f"{pref}_PersonalityOpenMindedness",
            f"{pref}_PersonalityAgreeableness",
            f"{pref}_PersonalityConscientiousness",
        ]
        pooled_X.append(df[bf_cols].dropna().to_numpy(dtype=np.float64))

        out_cols = [c for c in df.columns if is_outcome_col(c, pref)]
        print(f"Processant {sub_name} (N={len(df):,}, {len(out_cols)} criteris)...")
        for col in out_cols:
            sub = df[bf_cols + [col]].dropna()
            if len(sub) < 100:
                continue
            y = sub[col].to_numpy(dtype=np.float64)
            nunq = int(len(np.unique(y)))
            if nunq < 2:
                continue
            is_bin = (nunq == 2)
            if is_bin:
                _, cnts = np.unique(y, return_counts=True)
                if np.min(cnts) < 20:
                    continue
            X = sub[bf_cols].to_numpy(dtype=np.float64)
            sc = evaluate_outcome_cv(X, y, is_binary=is_bin, n_splits=10, seed=20261121)
            row = {
                "subsample": sub_name,
                "outcome": col,
                "n": int(len(sub)),
                "type": "binary" if is_bin else "continuous",
                "metric": "AUC" if is_bin else "R2",
            }
            row.update(sc)
            row["delta_ARC_over_Traits"] = sc["Traits+ARC_Hard"] - sc["Traits_Linear"]
            row["delta_LPA3_over_Traits"] = sc["Traits+LPA3_Prob"] - sc["Traits_Linear"]
            row["delta_GMM4_over_Traits"] = sc["Traits+GMM4_Prob"] - sc["Traits_Linear"]
            row["delta_Gerlach4_over_Traits"] = sc["Traits+Gerlach4"] - sc["Traits_Linear"]
            row["delta_Poly2_over_Traits"] = sc["Traits_Poly2"] - sc["Traits_Linear"]
            rows.append(row)

    res_df = pd.DataFrame(rows)

    X_all = np.vstack(pooled_X)
    Z_all = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0, ddof=1)
    R_all = np.corrcoef(Z_all, rowvar=False)
    evals = np.linalg.eigvalsh(R_all)[::-1]
    c_wk, _ = ward_kmeans(Z_all, k=3, seed=20261121)
    _, mean_d2, mean_phi = align_centers_to_arc(c_wk)
    mean_kap = split_half_kappa(Z_all, k=3, n_splits=5, seed=20261121)

    tree = cKDTree(Z_all)
    dists, _ = tree.query(Z_all, k=2)
    bw_nn = float(np.mean(dists[:, 1]))
    gmm13 = GaussianMixture(n_components=13, covariance_type="full", n_init=5, reg_covar=1e-6, random_state=20261121).fit(Z_all)
    n_modes, uniq_modes = mean_shift_mode_count(Z_all, np.vstack([gmm13.means_, np.zeros((1, 5))]), bw=bw_nn)

    out_dir = Path("results/soto2019")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "soto2019_incremental_cv.csv", index=False)

    cont = res_df[res_df["type"] == "continuous"]
    bin_ = res_df[res_df["type"] == "binary"]

    summary = {
        "pooled_structural": {
            "n": int(len(Z_all)),
            "corr_neoac": R_all.tolist(),
            "eigenvalues": evals.tolist(),
            "lambda2_over_lambda1": float(evals[1] / evals[0]),
            "det_R": float(np.linalg.det(R_all)),
            "wk_arc_mean_d2": mean_d2,
            "wk_arc_mean_tucker": mean_phi,
            "wk_split_half_kappa": mean_kap,
            "kde_nn_bandwidth": bw_nn,
            "mean_shift_distinct_modes": int(n_modes),
            "mean_shift_modes": uniq_modes.tolist(),
        },
        "continuous_summary": {
            "n_outcomes": int(len(cont)),
            "mean_R2_Traits_Linear": float(cont["Traits_Linear"].mean()),
            "mean_R2_ARC_Hard": float(cont["ARC_Hard"].mean()),
            "mean_R2_LPA3_Prob": float(cont["LPA3_Prob"].mean()),
            "mean_R2_GMM4_Prob": float(cont["GMM4_Prob"].mean()),
            "mean_R2_Gerlach4_Hard": float(cont["Gerlach4_Hard"].mean()),
            "mean_R2_Traits_plus_ARC": float(cont["Traits+ARC_Hard"].mean()),
            "mean_R2_Traits_plus_LPA3": float(cont["Traits+LPA3_Prob"].mean()),
            "mean_R2_Traits_plus_GMM4": float(cont["Traits+GMM4_Prob"].mean()),
            "mean_R2_Traits_plus_Gerlach4": float(cont["Traits+Gerlach4"].mean()),
            "mean_R2_Traits_Poly2": float(cont["Traits_Poly2"].mean()),
            "mean_delta_ARC_over_Traits": float(cont["delta_ARC_over_Traits"].mean()),
            "median_delta_ARC_over_Traits": float(cont["delta_ARC_over_Traits"].median()),
            "max_delta_ARC_over_Traits": float(cont["delta_ARC_over_Traits"].max()),
            "pct_Traits_beats_ARC": float((cont["Traits_Linear"] > cont["ARC_Hard"]).mean() * 100),
            "pct_Traits_beats_Gerlach4": float((cont["Traits_Linear"] > cont["Gerlach4_Hard"]).mean() * 100),
            "pct_delta_ARC_gt_001": float((cont["delta_ARC_over_Traits"] > 0.01).mean() * 100),
            "mean_delta_Poly2_over_Traits": float(cont["delta_Poly2_over_Traits"].mean()),
        },
        "binary_summary": {
            "n_outcomes": int(len(bin_)),
            "mean_AUC_Traits_Linear": float(bin_["Traits_Linear"].mean()),
            "mean_AUC_ARC_Hard": float(bin_["ARC_Hard"].mean()),
            "mean_AUC_LPA3_Prob": float(bin_["LPA3_Prob"].mean()),
            "mean_AUC_Gerlach4_Hard": float(bin_["Gerlach4_Hard"].mean()),
            "mean_AUC_Traits_plus_ARC": float(bin_["Traits+ARC_Hard"].mean()),
            "mean_AUC_Traits_plus_LPA3": float(bin_["Traits+LPA3_Prob"].mean()),
            "mean_AUC_Traits_plus_Gerlach4": float(bin_["Traits+Gerlach4"].mean()),
            "mean_AUC_Traits_Poly2": float(bin_["Traits_Poly2"].mean()),
            "mean_delta_AUC_ARC_over_Traits": float(bin_["delta_ARC_over_Traits"].mean()),
            "pct_Traits_beats_ARC": float((bin_["Traits_Linear"] > bin_["ARC_Hard"]).mean() * 100),
        },
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    (out_dir / "soto2019_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================ RESUM SOTO (2019) LOOPR ================")
    ps = summary["pooled_structural"]
    print(f"Pooled BFI-2 (N={ps['n']:,}): l2/l1={ps['lambda2_over_lambda1']:.4f} | det(R)={ps['det_R']:.4f} | "
          f"D2={ps['wk_arc_mean_d2']:.3f} | phi={ps['wk_arc_mean_tucker']:.3f} | kappa={ps['wk_split_half_kappa']:.3f} | "
          f"KDE modes (h={ps['kde_nn_bandwidth']:.3f})={ps['mean_shift_distinct_modes']}")

    cs = summary["continuous_summary"]
    print(f"\n--- Criteris continus/ordinals (k={cs['n_outcomes']}, 10-fold CV out-of-sample R^2) ---")
    print(f"  Traits (linear, 5) : {cs['mean_R2_Traits_Linear']:.4f}")
    print(f"  ARC Hard (K=3)     : {cs['mean_R2_ARC_Hard']:.4f}  (Traits > ARC en {cs['pct_Traits_beats_ARC']:.1f}% dels criteris)")
    print(f"  LPA-3 Prob (K=3)   : {cs['mean_R2_LPA3_Prob']:.4f}")
    print(f"  GMM-4 Prob (K=4)   : {cs['mean_R2_GMM4_Prob']:.4f}")
    print(f"  Gerlach-4 Hard     : {cs['mean_R2_Gerlach4_Hard']:.4f}  (Traits > Gerlach4 en {cs['pct_Traits_beats_Gerlach4']:.1f}% dels criteris)")
    print(f"  Traits + ARC Hard  : {cs['mean_R2_Traits_plus_ARC']:.4f}  (Delta mean={cs['mean_delta_ARC_over_Traits']:+.5f}, med={cs['median_delta_ARC_over_Traits']:+.5f}, max={cs['max_delta_ARC_over_Traits']:+.5f})")
    print(f"  Traits + LPA3 Prob : {cs['mean_R2_Traits_plus_LPA3']:.4f}  (Delta mean={cont['delta_LPA3_over_Traits'].mean():+.5f})")
    print(f"  Traits + GMM4 Prob : {cs['mean_R2_Traits_plus_GMM4']:.4f}  (Delta mean={cont['delta_GMM4_over_Traits'].mean():+.5f})")
    print(f"  Traits + Gerlach4  : {cs['mean_R2_Traits_plus_Gerlach4']:.4f}  (Delta mean={cont['delta_Gerlach4_over_Traits'].mean():+.5f})")
    print(f"  Traits Poly2 (20)  : {cs['mean_R2_Traits_Poly2']:.4f}  (Delta mean={cs['mean_delta_Poly2_over_Traits']:+.5f})")

    bs = summary["binary_summary"]
    print(f"\n--- Criteris binaris (k={bs['n_outcomes']}, 10-fold CV out-of-sample ROC-AUC) ---")
    print(f"  Traits (linear, 5) : {bs['mean_AUC_Traits_Linear']:.4f}")
    print(f"  ARC Hard (K=3)     : {bs['mean_AUC_ARC_Hard']:.4f}  (Traits > ARC en {bs['pct_Traits_beats_ARC']:.1f}%)")
    print(f"  LPA-3 Prob (K=3)   : {bs['mean_AUC_LPA3_Prob']:.4f}")
    print(f"  Gerlach-4 Hard     : {bs['mean_AUC_Gerlach4_Hard']:.4f}")
    print(f"  Traits + ARC Hard  : {bs['mean_AUC_Traits_plus_ARC']:.4f}  (Delta mean={bs['mean_delta_AUC_ARC_over_Traits']:+.5f})")
    print(f"  Traits + LPA3 Prob : {bs['mean_AUC_Traits_plus_LPA3']:.4f}  (Delta mean={bin_['delta_LPA3_over_Traits'].mean():+.5f})")
    print(f"  Traits + Gerlach4  : {bs['mean_AUC_Traits_plus_Gerlach4']:.4f}  (Delta mean={bin_['delta_Gerlach4_over_Traits'].mean():+.5f})")
    print(f"  Traits Poly2 (20)  : {bs['mean_AUC_Traits_Poly2']:.4f}  (Delta mean={bs['mean_delta_Poly2_over_Traits']:+.5f})")
    print("\nTop 5 criteris continus per Delta R^2 (Traits+ARC - Traits):")
    top5 = cont.sort_values("delta_ARC_over_Traits", ascending=False).head(5)
    for _, r in top5.iterrows():
        print(f"  {r['outcome']:42s} | R2_Traits={r['Traits_Linear']:.4f} | +ARC={r['delta_ARC_over_Traits']:+.4f} | +Poly2={r['delta_Poly2_over_Traits']:+.4f}")
    print(f"\nDesat a {out_dir}/ ({time.perf_counter()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
