#!/usr/bin/env python3
"""Portable preprocessing script for Spanish BFI-2 Study 1 (OSF kp572).

Reconstructs Big Five domain scores in strict (N, E, O, A, C) order (N unreversed)
directly from the public OSF archive file:
  Data/Study 1/BFI-2 Full data (study 1).csv  (https://osf.io/kp572/, CC-By 4.0)
using the official Soto & John (2017) / Gallardo-Pujol et al. (2022) item-index
scoring key documented in `Spanish BFI-2 Form.pdf` (p. 3).

No copyrighted item wording is stored in this script.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pandas as pd

EXPECTED_RAW_SHA256 = "ec787ff7cca2a27495beee5c1ef6d37d537fc75d1a0c78c1b35818662749b68c"

# Official BFI-2 60-item domain key (Spanish BFI-2 Form.pdf, p. 3; Soto & John, 2017)
# Reverse-keyed items are scored as (6 - x) on the 1..5 Likert scale.
BFI2_DOMAIN_KEY: dict[str, tuple[list[int], set[int]]] = {
    "N": ([4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59], {4, 9, 24, 29, 44, 49}),
    "E": ([1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56], {11, 16, 26, 31, 36, 51}),
    "O": ([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], {5, 25, 30, 45, 50, 55}),
    "A": ([2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57], {12, 17, 22, 37, 42, 47}),
    "C": ([3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58], {3, 8, 23, 28, 48, 58}),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_bfi2_es(input_csv: Path, output_csv: Path, verify_checksum: bool = True) -> pd.DataFrame:
    raw_hash = sha256_file(input_csv)
    if verify_checksum and raw_hash != EXPECTED_RAW_SHA256:
        raise ValueError(
            f"SHA-256 mismatch for {input_csv}: got {raw_hash}, expected {EXPECTED_RAW_SHA256}"
        )

    df = pd.read_csv(input_csv)
    item_cols = [f"BFI{i}" for i in range(1, 61)]
    missing_cols = [c for c in item_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required BFI-2 item columns: {missing_cols[:5]}")

    complete_mask = ~df[item_cols].isna().any(axis=1)
    df_clean = df.loc[complete_mask].copy()

    out = pd.DataFrame(index=df_clean.index)
    for dom in ("N", "E", "O", "A", "C"):
        items, rev_set = BFI2_DOMAIN_KEY[dom]
        cols = [f"BFI{i}" for i in items]
        vals = df_clean[cols].astype(float).copy()
        for idx in rev_set:
            vals[f"BFI{idx}"] = 6.0 - vals[f"BFI{idx}"]
        out[dom] = vals.mean(axis=1)

    for demo_col in ("Age", "Gender"):
        if demo_col in df_clean.columns:
            out[demo_col] = df_clean[demo_col]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Spanish BFI-2 Study 1 raw OSF CSV to N,E,O,A,C domains.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/bfi2_study1_raw.csv"),
        help="Path to raw OSF kp572 Study 1 CSV (`BFI-2 Full data (study 1).csv`).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/bfi2_study1_domains.csv"),
        help="Path to write scored domain CSV (columns N,E,O,A,C,Age,Gender).",
    )
    parser.add_argument("--no-verify-checksum", action="store_true", help="Skip SHA-256 verification.")
    args = parser.parse_args()

    out = prepare_bfi2_es(args.input, args.output, verify_checksum=not args.no_verify_checksum)
    print(f"Prepared {len(out)} complete rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
