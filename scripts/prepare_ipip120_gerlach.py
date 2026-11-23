#!/usr/bin/env python3
"""Prepara IPIP-NEO-120 (Johnson, 2014) en:
  1) data/processed/ipip120_domains_complete.csv
  2) data/processed/ipip120_fa_varimax_gerlach.npy (pipeline exacte FactorAnalysis(5) + Varimax de Gerlach)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.optimize import linear_sum_assignment
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FactorAnalysis

NEOAC = ["N", "E", "O", "A", "C"]
sys.path.insert(0, str(Path("external/gerlach2018/src/external").resolve()))
sys.path.insert(0, str(Path("external/gerlach2018/src/external/factor_rotation").resolve()))
import factor_rotation as fr


def main() -> int:
    in_path = Path("data/raw/ipip120/IPIP120.por")
    print(f"Llegint {in_path}...")
    df, _ = pyreadstat.read_por(str(in_path))
    item_cols = [c for c in df.columns if re.fullmatch(r"I\d+", c, flags=re.IGNORECASE)]
    item_cols.sort(key=lambda c: int(c[1:]))
    if len(item_cols) != 120:
        raise ValueError(f"Esperats 120 ítems, trobats {len(item_cols)}")

    X_raw = df[item_cols].to_numpy(dtype=np.float64)
    valid = np.all((X_raw >= 1.0) & (X_raw <= 5.0) & np.isfinite(X_raw), axis=1)
    X = X_raw[valid]
    n_comp = X.shape[0]
    print(f"Files totals: {len(df):,} | Casos complets (1..5 sense 0/NA): {n_comp:,}")

    domain_idx = {dom: np.arange(i, 120, 5) for i, dom in enumerate(NEOAC)}
    sub = X[: min(30_000, n_comp)]
    neg_items = []
    for dom in NEOAC:
        idxs = domain_idx[dom]
        s_dom = sub[:, idxs].sum(axis=1)
        for j in idxs:
            r = float(np.corrcoef(sub[:, j], s_dom - sub[:, j])[0, 1])
            if r < 0.0:
                neg_items.append((dom, item_cols[j], round(r, 4)))
    if len(neg_items) > 5:
        print(f"[ERROR] {len(neg_items)} ítems amb r < 0: {neg_items[:10]}")
        return 1
    elif neg_items:
        print(f"[AVÍS] Ítems amb r lleugerament < 0: {neg_items}")
    else:
        print("Verificació d'inversió OK: tots 120 ítems correlacionen positivament amb el seu domini.")

    domains = pd.DataFrame({dom: X[:, domain_idx[dom]].mean(axis=1) for dom in NEOAC})
    R_dom = domains.corr().to_numpy()
    evals = np.linalg.eigvalsh(R_dom)[::-1]
    print("\n--- IPIP-120 Dominis ---")
    print(domains.corr().round(3).to_string())
    print(f"Valors propis: {np.round(evals, 4)} | l2/l1={evals[1]/evals[0]:.5f} | det(R)={np.linalg.det(R_dom):.5f}")
    print("Skew:", domains.skew().round(3).to_dict(), "| Kurt:", domains.kurt().round(3).to_dict())

    print("\nCalculant FactorAnalysis(5) + Varimax (pipeline exacte de Gerlach) sobre els 120 ítems...")
    fa = FactorAnalysis(n_components=5, random_state=20261121)
    fa.fit(X)
    arr_pd = fa.transform(X)
    arr_qd = fa.components_.T
    _, T = fr.rotate_factors(arr_qd, "varimax")
    T_inv = np.linalg.inv(T)
    scores_rot = (T_inv @ arr_pd.T).T

    dom_mat = domains.to_numpy()
    dom_z = (dom_mat - dom_mat.mean(axis=0)) / dom_mat.std(axis=0, ddof=1)
    sc_z = (scores_rot - scores_rot.mean(axis=0)) / scores_rot.std(axis=0, ddof=1)
    cc = (dom_z.T @ sc_z) / float(n_comp - 1)
    ri, ci = linear_sum_assignment(-np.abs(cc))
    aligned = np.empty_like(scores_rot)
    print("Alineació FA+Varimax -> Dominis:")
    for r_i, c_i in zip(ri, ci):
        sgn = 1.0 if cc[r_i, c_i] >= 0 else -1.0
        aligned[:, r_i] = sgn * scores_rot[:, c_i]
        print(f"  {NEOAC[r_i]} <- factor_{c_i} (r = {sgn * cc[r_i, c_i]:+.4f})")

    R_fa = np.corrcoef(aligned, rowvar=False)
    print(f"FA+Varimax max|offdiag R|={np.max(np.abs(R_fa - np.eye(5))):.5f} | det(R)={np.linalg.det(R_fa):.5f}")
    print("FA+Varimax Skew:", np.round(skew(aligned, axis=0), 3), "| Kurt:", np.round(kurtosis(aligned, axis=0), 3))

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    domains.to_csv(out_dir / "ipip120_domains_complete.csv", index=False)
    np.save(out_dir / "ipip120_fa_varimax_gerlach.npy", aligned.astype(np.float64))
    print("\nDesats: data/processed/ipip120_domains_complete.csv i data/processed/ipip120_fa_varimax_gerlach.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
