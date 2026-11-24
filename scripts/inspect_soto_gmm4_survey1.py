#!/usr/bin/env python3
"""Inspect the 4 GMM components in Soto (2019) Survey 1 (Young Adults) to diagnose
   whether the +0.01022 incremental R^2 reflects careless/extreme responding or substantive nonlinearity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# Auto-locate Soto 2019 processed/raw CSVs
candidates = sorted(Path("data").rglob("*soto*")) + sorted(Path("data").rglob("*loopr*")) + sorted(Path("data").rglob("*survey1*"))
csvs = [p for p in candidates if p.suffix.lower() == ".csv"]
print("Fitxers Soto/LOOPR trobats:", [str(p) for p in csvs])

# Pick Survey 1 Young Adults if separate, else filter by survey column
s1_path = next((p for p in csvs if "1" in p.name or "young" in p.name.lower()), csvs[0] if csvs else None)
if s1_path is None:
    raise FileNotFoundError("No s'ha trobat cap CSV de Soto (2019) dins de data/.")

df = pd.read_csv(s1_path)
print(f"Carregat: {s1_path} (shape={df.shape})")

# Identify Big Five columns automatically
bf_sets = [
    ["N", "E", "O", "A", "C"],
    ["bfi_n", "bfi_e", "bfi_o", "bfi_a", "bfi_c"],
    ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"],
    ["n", "e", "o", "a", "c"],
]
cols = next((s for s in bf_sets if all(c in df.columns for c in s)), None)
if cols is None:
    print("Columnes disponibles (primeres 30):", list(df.columns[:30]))
    raise KeyError("No s'han identificat automàticament les 5 columnes Big Five.")

X_raw = df[cols].dropna()
X = (X_raw - X_raw.mean()) / X_raw.std(ddof=1)
X_np = X.to_numpy(dtype=np.float64)

gmm = GaussianMixture(n_components=4, covariance_type="full", n_init=10, random_state=42).fit(X_np)
probs = gmm.predict_proba(X_np)
hard = gmm.predict(X_np)

print(f"\n=== GMM-4 COMPONENTS ON {s1_path.name} (N = {len(X_np)}) ===")
summary = []
for k in range(4):
    mu = gmm.means_[k]
    cov = gmm.covariances_[k]
    evals = np.linalg.eigvalsh(cov)
    w = float(gmm.weights_[k])
    # Mean squared norm (distance from origin) of assigned participants
    sub = X_np[hard == k]
    mean_r2 = float(np.mean(np.sum(sub ** 2, axis=1))) if len(sub) else np.nan
    ext_rate = float(np.mean(np.max(np.abs(sub), axis=1) > 2.5)) if len(sub) else np.nan
    print(
        f"  Comp {k}: weight={w:.3f} (n={len(sub):4d}) | "
        f"mu=[{', '.join(f'{v:+.2f}' for v in mu)}] | "
        f"tr(Cov)={np.trace(cov):.2f}, det(Cov)={np.linalg.det(cov):.3f}, max_eig={evals.max():.2f} | "
        f"mean||z||^2={mean_r2:.2f}, %(|z|>2.5)={100*ext_rate:.1f}%"
    )
    summary.append({
        "comp": k, "weight": w, "n": int(len(sub)),
        "mean": [float(v) for v in mu],
        "trace_cov": float(np.trace(cov)),
        "det_cov": float(np.linalg.det(cov)),
        "max_eig": float(evals.max()),
        "mean_norm2": mean_r2,
        "extreme_rate_gt2_5": ext_rate,
    })

out_path = Path("results/gerlach_faithful/soto_survey1_gmm4_inspection.json")
out_path.write_text(json.dumps({"file": str(s1_path), "N": len(X_np), "cols": cols, "components": summary}, indent=2), encoding="utf-8")
print(f"\nDesat a {out_path}")
