"""Source provenance, SHA-256 verification, and row-level data leakage guards."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

FORBIDDEN_ID_COLUMNS: frozenset[str] = frozenset(
    {"participant", "responseid", "subject_id", "caseid", "ursi", "eid"}
)


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> bool:
    """Return True if ``path`` exists and matches ``expected_sha256``."""
    if not path.exists():
        return False
    return sha256_file(path).lower() == expected_sha256.lower()


def assert_no_row_level_leakage(
    output_dir: Path,
    *,
    forbidden_cols: Iterable[str] = FORBIDDEN_ID_COLUMNS,
) -> None:
    """Raise AssertionError if any CSV/Parquet in ``output_dir`` contains participant IDs."""
    forbidden = {c.lower() for c in forbidden_cols}
    if not output_dir.exists():
        return
    for csv_path in output_dir.rglob("*.csv"):
        cols = pd.read_csv(csv_path, nrows=0).columns
        leaked = [c for c in cols if c.lower() in forbidden]
        if leaked:
            raise AssertionError(
                f"Row-level participant identifier(s) {leaked} leaked into {csv_path}!"
            )
    for pq_path in output_dir.rglob("*.parquet"):
        cols = pd.read_parquet(pq_path, columns=[]).columns
        leaked = [c for c in cols if c.lower() in forbidden]
        if leaked:
            raise AssertionError(
                f"Row-level participant identifier(s) {leaked} leaked into {pq_path}!"
            )
