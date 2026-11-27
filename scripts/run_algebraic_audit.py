#!/usr/bin/env python3
"""Run empirical algebraic covariance-complexity & 3rd-order audit across project datasets.

Per PROJECT_RULES.md, this script is written ready-to-run and NOT executed by the assistant.
Run from project root:
    python3 scripts/run_algebraic_audit.py --n-boot 500 --out results/algebraic_audit/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from arc_study.algebraic_complexity import (  # noqa: E402
    bootstrap_individual_algebraic_audit,
    wishart_meta_analytic_audit,
)
from arc_study.matrices import BUSSERI_ERB_2024, VAN_DER_LINDEN_2010  # noqa: E402
from hitop_empirical.harmonization import compute_spectrum_composites  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Empirical algebraic covariance-complexity & 3rd-order audit.")
    parser.add_argument("--n-boot", type=int, default=500, help="Bootstrap/Wishart replicates (default: 500).")
    parser.add_argument("--tau", type=float, default=0.01, help="Minimum diagonal uniqueness bound tau.")
    parser.add_argument("--seed", type=int, default=20261121, help="Base reproducibility seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/algebraic_audit/algebraic_audit_summary.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    results: dict[str, Any] = {
        "seed": args.seed,
        "n_boot": args.n_boot,
        "tau": args.tau,
        "meta_analytic_wishart": {},
        "individual_level_bootstrap": {},
    }

    # 1. Meta-analytic matrices (SPD-preserving Wishart sampling at effective N)
    results["meta_analytic_wishart"]["van_der_linden_2010"] = wishart_meta_analytic_audit(
        VAN_DER_LINDEN_2010, n_eff=144117, n_draws=args.n_boot, tau=args.tau, seed=args.seed + 1
    )
    results["meta_analytic_wishart"]["busseri_erb_2024"] = wishart_meta_analytic_audit(
        BUSSERI_ERB_2024, n_eff=22135, n_draws=args.n_boot, tau=args.tau, seed=args.seed + 2
    )

    # 2. Individual-level datasets (whole-respondent bootstrap; full-precision R)
    individual_specs = [
        ("bfi2_es_study1", Path("data/processed/bfi2_study1_domains.csv"), ["N", "E", "O", "A", "C"]),
        ("ipip300_domains", Path("data/processed/ipip300_domains_complete.csv"), ["N", "E", "O", "A", "C"]),
        ("ipip120_domains", Path("data/processed/ipip120_domains_complete.csv"), ["N", "E", "O", "A", "C"]),
    ]
    for idx, (name, path, cols) in enumerate(individual_specs):
        if path.exists():
            df = pd.read_csv(path, usecols=cols).dropna()
            x = df[cols].to_numpy(dtype=np.float64)
            results["individual_level_bootstrap"][name] = bootstrap_individual_algebraic_audit(
                x, n_boot=args.n_boot, tau=args.tau, seed=args.seed + 10 + idx
            )
        else:
            results["individual_level_bootstrap"][name] = {"status": "missing_file", "path": str(path)}

    # 3. HiTOP-SR individual-level cohorts (5 core spectra)
    ku_path = Path("data/processed/hitop_empirical/ku_hitopsr_scored.csv")
    cz_path = Path("data/raw/hitop_empirical/czech_hitop/dataset.csv")
    for idx, (name, cohort, path) in enumerate([("ku_hitopsr", "ku", ku_path), ("czech_hitop", "cz", cz_path)]):
        if path.exists():
            df = pd.read_csv(path)
            x = compute_spectrum_composites(df, cohort=cohort, n_spectra=5, standardize_mode="within").scores.to_numpy(
                dtype=np.float64
            )
            results["individual_level_bootstrap"][name] = bootstrap_individual_algebraic_audit(
                x, n_boot=args.n_boot, tau=args.tau, seed=args.seed + 20 + idx
            )
        else:
            results["individual_level_bootstrap"][name] = {"status": "missing_file", "path": str(path)}

    results["elapsed_sec"] = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Algebraic audit written to {args.out} in {results['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
