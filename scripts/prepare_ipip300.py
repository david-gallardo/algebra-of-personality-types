#!/usr/bin/env python3
"""Prepara IPIP-NEO-300 (Johnson, 2014) en dos formats per al pipeline ARC:
1) data/processed/ipip300_domains_complete.csv  (mitjanes de 60 ítems per domini, N,E,O,A,C)
2) data/processed/ipip300_varimax_complete.csv  (5 components PCA + rotació Varimax, estil Gerlach 2018)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyreadstat
from scipy.optimize import linear_sum_assignment

NEOAC_ORDER = ["N", "E", "O", "A", "C"]


def varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 100, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Rotació ortogonal Varimax amb normalització de Kaiser."""
    p, k = Phi.shape
    h = np.sqrt(np.sum(Phi ** 2, axis=1, keepdims=True))
    h_safe = np.where(h > 0, h, 1.0)
    Lambda = Phi / h_safe
    R = np.eye(k, dtype=np.float64)
    d = 0.0
    for _ in range(q):
        d_old = d
        Lambda_R = Lambda @ R
        u, s, vt = np.linalg.svd(
            Lambda.T @ (Lambda_R ** 3 - (gamma / p) * Lambda_R @ np.diag(np.sum(Lambda_R ** 2, axis=0)))
        )
        R = u @ vt
        d = float(np.sum(s))
        if d_old != 0.0 and d / d_old < 1.0 + tol:
            break
    return (Lambda @ R) * h_safe, R


def detect_item_columns(columns: list[str]) -> list[str]:
    """Detecta les 300 columnes d'ítems I1..I300."""
    i_cols = [c for c in columns if re.fullmatch(r"I\d+", c, flags=re.IGNORECASE)]
    if len(i_cols) == 300:
        i_cols.sort(key=lambda c: int(c[1:]))
        return i_cols
    raise ValueError(
        f"S'esperaven 300 columnes del tipus I1..I300, però se n'han trobat {len(i_cols)}. "
        f"Columnes inicials: {columns[:25]}"
    )


def inspect_file(input_path: Path) -> None:
    try:
        df, meta = pyreadstat.read_por(str(input_path), row_limit=1000)
    except TypeError:
        df, meta = pyreadstat.read_por(str(input_path), metadataonly=True)
    cols = list(meta.column_names)
    print(f"Columnes totals ({len(cols)}): {cols[:20]} ... {cols[-5:]}")
    print("\nPrimeres 15 etiquetes de variable:")
    for col in cols[:15]:
        print(f"  {col:10s} | label={meta.column_names_to_labels.get(col, '')}")
    item_cols = [c for c in cols if re.fullmatch(r"I\d+", c, flags=re.IGNORECASE)]
    print(f"\nColumnes tipus I#: {len(item_cols)}")
    if item_cols:
        item_cols.sort(key=lambda c: int(c[1:]))
        print("Primers 10 ítems:")
        for c in item_cols[:10]:
            vals = sorted(df[c].dropna().unique().tolist()) if len(df) > 0 else "metadataonly"
            print(f"  {c:6s} | vals={vals} | label={meta.column_names_to_labels.get(c, '')}")


def build_datasets(input_path: Path, outdir: Path) -> None:
    print(f"\nLlegint {input_path} (pot trigar ~15-30s)...")
    df, _ = pyreadstat.read_por(str(input_path))
    item_cols = detect_item_columns(list(df.columns))
    print(f"Files totals en brut: {len(df):,}")

    X_raw = df[item_cols].to_numpy(dtype=np.float64)
    # A Johnson (2014), 0/NA indica resposta perduda; valors vàlids són 1..5
    valid_mask = np.all((X_raw >= 1.0) & (X_raw <= 5.0) & np.isfinite(X_raw), axis=1)
    X = X_raw[valid_mask]
    n_complete = X.shape[0]
    print(f"Casos complets (tots 300 ítems dins 1..5, sense 0/NA): {n_complete:,}")

    # Assignació estàndard IPIP-NEO-300: cicle I1=N, I2=E, I3=O, I4=A, I5=C, I6=N, ...
    domain_idx = {
        "N": np.arange(0, 300, 5),
        "E": np.arange(1, 300, 5),
        "O": np.arange(2, 300, 5),
        "A": np.arange(3, 300, 5),
        "C": np.arange(4, 300, 5),
    }

    # Verificació anti-errors d'inversió d'ítems
    sub = X[: min(20_000, n_complete)]
    neg_items: list[tuple[str, str, float]] = []
    for dom in NEOAC_ORDER:
        idxs = domain_idx[dom]
        dom_sum = sub[:, idxs].sum(axis=1)
        for j in idxs:
            rest = dom_sum - sub[:, j]
            r = float(np.corrcoef(sub[:, j], rest)[0, 1])
            if r < 0.0:
                neg_items.append((dom, item_cols[j], round(r, 4)))

    if len(neg_items) > 10:
        print(f"\n[ERROR] {len(neg_items)} ítems amb correlació negativa amb el seu domini (semblen NO invertits)!")
        print("Exemples:", neg_items[:15])
        sys.exit(1)
    elif neg_items:
        print(f"\n[AVÍS] {len(neg_items)} ítem(s) amb correlació lleugerament negativa: {neg_items}")
    else:
        print("Verificació d'inversió OK: tots 300 ítems correlacionen positivament amb el seu domini.")

    # 1. Puntuacions de domini (mitjana dels 60 ítems en ordre estricte N, E, O, A, C)
    domains = pd.DataFrame(
        {dom: X[:, domain_idx[dom]].mean(axis=1) for dom in NEOAC_ORDER}
    )

    print("\n--- Diagnòstic ràpid: Dominis IPIP-300 ---")
    r_dom = domains.corr().to_numpy()
    evals = np.linalg.eigvalsh(r_dom)[::-1]
    print("Correlacions (N, E, O, A, C):")
    print(domains.corr().round(3).to_string())
    print(f"Valors propis: {np.round(evals, 4)} | lambda2/lambda1 = {evals[1]/evals[0]:.5f}")
    print("Asimetria:", domains.skew().round(3).to_dict())
    print("Curtosi excés:", domains.kurt().round(3).to_dict())

    # 2. Puntuacions factorials ortogonals PCA + Varimax (estil Gerlach et al., 2018)
    print("\nCalculant PCA(5) + rotació Varimax sobre els 300 ítems...")
    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    corr_items = (Xz.T @ Xz) / float(n_complete - 1)
    eigvals, eigvecs = np.linalg.eigh(corr_items)
    top5 = np.argsort(eigvals)[::-1][:5]
    evals5 = eigvals[top5]
    evecs5 = eigvecs[:, top5]

    loadings_unrot = evecs5 * np.sqrt(evals5)
    _, R_rot = varimax(loadings_unrot)
    scores_rot = ((Xz @ evecs5) / np.sqrt(evals5)) @ R_rot

    # Alinear els 5 factors Varimax a l'ordre estricte ['N', 'E', 'O', 'A', 'C'] i signe positiu
    dom_mat = domains.to_numpy()
    dom_z = (dom_mat - dom_mat.mean(axis=0)) / dom_mat.std(axis=0, ddof=1)
    cross_corr = (dom_z.T @ scores_rot) / float(n_complete - 1)
    row_ind, col_ind = linear_sum_assignment(-np.abs(cross_corr))

    aligned_scores = np.empty_like(scores_rot)
    print("\nAlineació Varimax -> Dominis (correlació):")
    for r_i, c_i in zip(row_ind, col_ind):
        sign = 1.0 if cross_corr[r_i, c_i] >= 0 else -1.0
        aligned_scores[:, r_i] = sign * scores_rot[:, c_i]
        print(f"  {NEOAC_ORDER[r_i]} <- factor_{c_i} (r = {sign * cross_corr[r_i, c_i]:+.4f})")

    varimax_df = pd.DataFrame(aligned_scores, columns=NEOAC_ORDER)
    print("Asimetria Varimax:", varimax_df.skew().round(3).to_dict())
    print("Curtosi excés Varimax:", varimax_df.kurt().round(3).to_dict())

    outdir.mkdir(parents=True, exist_ok=True)
    dom_path = outdir / "ipip300_domains_complete.csv"
    var_path = outdir / "ipip300_varimax_complete.csv"
    domains.to_csv(dom_path, index=False)
    varimax_df.to_csv(var_path, index=False)
    print(f"\nFitxers guardats:\n  - {dom_path}\n  - {var_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("data/processed"))
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        inspect_file(args.input)
    if args.build:
        build_datasets(args.input, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
