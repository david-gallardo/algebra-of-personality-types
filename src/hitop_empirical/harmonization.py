"""76-scale KU <-> Czech HiTOP-SR crosswalk, spectrum composites, and QC utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import pandas as pd

# Exact 1-to-1 mapping from KU hitop 0.2.0 column names (hsr_*) to Czech HiTOP-CZ column names (HiTOP_*)
KU_TO_CZ_ALIASES: dict[str, str] = {
    "hsr_appearanceFocus": "HiTOP_Body Focus",
    "hsr_nonSuicidalSelfInjury": "HiTOP_NSSI",
}

# Pre-specified, theory-anchored HiTOP-SR spectrum composites (Kotov et al., 2017; Simms et al., 2022).
# Reverse-keyed scales are indicated by sign = -1.0 (currently Well-being under Internalizing).
SPECTRUM_SCALES_6: dict[str, list[tuple[str, float]]] = {
    "INT": [
        ("hsr_distressDysphoria", 1.0),
        ("hsr_socialAnxiety", 1.0),
        ("hsr_panic", 1.0),
        ("hsr_agoraphobia", 1.0),
        ("hsr_traumaReactions", 1.0),
        ("hsr_insomnia", 1.0),
        ("hsr_nightmares", 1.0),
        ("hsr_suicidality", 1.0),
        ("hsr_nonSuicidalSelfInjury", 1.0),
        ("hsr_appetiteLoss", 1.0),
        ("hsr_wellBeing", -1.0),
    ],
    "DET": [
        ("hsr_socialAloofness", 1.0),
        ("hsr_restrictedAffectivity", 1.0),
        ("hsr_romanticDisinterest", 1.0),
        ("hsr_submissiveness", 1.0),
    ],
    "TD": [
        ("hsr_realityDistortion", 1.0),
        ("hsr_eccentricity", 1.0),
        ("hsr_fantasyProneness", 1.0),
        ("hsr_disorganization", 1.0),
        ("hsr_dissociation", 1.0),
        ("hsr_mistrust", 1.0),
    ],
    "A_EXT": [
        ("hsr_callousness", 1.0),
        ("hsr_dishonesty", 1.0),
        ("hsr_domineering", 1.0),
        ("hsr_entitlement", 1.0),
        ("hsr_exhibitionism", 1.0),
        ("hsr_grandiosity", 1.0),
        ("hsr_socialAggression", 1.0),
        ("hsr_oppositionality", 1.0),
    ],
    "D_EXT": [
        ("hsr_antisocialBehavior", 1.0),
        ("hsr_nonPersistence", 1.0),
        ("hsr_nonPlanfulness", 1.0),
        ("hsr_riskTaking", 1.0),
        ("hsr_riskySex", 1.0),
        ("hsr_problematicShopping", 1.0),
        ("hsr_gambling", 1.0),
        ("hsr_gaming", 1.0),
    ],
    "SOM": [
        ("hsr_bodilyDistress", 1.0),
        ("hsr_conversionSymptoms", 1.0),
        ("hsr_diseaseConviction", 1.0),
        ("hsr_healthAnxiety", 1.0),
        ("hsr_somaticPreoccupation", 1.0),
    ],
}

SPECTRUM_ORDER_5: tuple[str, ...] = ("INT", "DET", "TD", "A_EXT", "D_EXT")
SPECTRUM_ORDER_6: tuple[str, ...] = ("INT", "DET", "TD", "A_EXT", "D_EXT", "SOM")


def _normalize_label(name: str) -> str:
    """Normalize a scale label for exact case/punctuation-insensitive matching."""
    cleaned = name
    for prefix in ("hsr_", "HiTOP_"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return "".join(ch.lower() for ch in cleaned if ch.isalnum())


def build_ku_cz_crosswalk(
    ku_cols: list[str],
    cz_cols: list[str],
) -> pd.DataFrame:
    """Build verified 1-to-1 crosswalk between KU (hsr_*) and Czech (HiTOP_*) scale columns."""
    cz_by_norm = {_normalize_label(c): c for c in cz_cols if c.startswith("HiTOP_")}
    scale_to_spec: dict[str, tuple[str, float]] = {}
    for spec, pairs in SPECTRUM_SCALES_6.items():
        for sc, sign in pairs:
            scale_to_spec[sc] = (spec, sign)

    rows: list[dict[str, object]] = []
    for ku_c in sorted(c for c in ku_cols if c.startswith("hsr_")):
        if ku_c in KU_TO_CZ_ALIASES:
            cz_c = KU_TO_CZ_ALIASES[ku_c]
            match_type = "verified_alias"
        else:
            norm = _normalize_label(ku_c)
            cz_c = cz_by_norm.get(norm, "")
            match_type = "exact_normalized" if cz_c else "UNMATCHED"
        spec, sign = scale_to_spec.get(ku_c, ("UNASSIGNED", 1.0))
        rows.append(
            {
                "ku_col": ku_c,
                "cz_col": cz_c,
                "match_type": match_type,
                "assigned_spectrum_6": spec,
                "assigned_spectrum_5": spec if spec in SPECTRUM_ORDER_5 else "UNASSIGNED",
                "sign": sign,
            }
        )
    return pd.DataFrame(rows)


def check_unique_ids(df: pd.DataFrame, id_col: str) -> int:
    """Return number of duplicate IDs in ``id_col`` (raises ValueError if column missing)."""
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found.")
    return int(df[id_col].duplicated().sum())


@dataclass(frozen=True)
class CompositeResult:
    """Computed spectrum composites and sample-flow metadata."""

    scores: pd.DataFrame
    n_input: int
    n_complete: int
    n_dropped: int
    spectrum_order: tuple[str, ...]


def compute_spectrum_composites(
    df: pd.DataFrame,
    *,
    cohort: Literal["ku", "cz"],
    n_spectra: Literal[5, 6] = 5,
    max_missing_frac: float = 0.25,
    standardize_mode: Literal["within", "raw_mean"] = "within",
) -> CompositeResult:
    """Compute pre-specified unit-weighted HiTOP spectrum composites.

    In ``standardize_mode="within"``, each constituent scale is standardized within-cohort
    before signed averaging, then each spectrum composite is z-standardized (ddof=1).
    In ``standardize_mode="raw_mean"``, signed scales (with reverse-keyed 5 - x on 1-4 scale)
    are averaged on the native 1-4 metric to allow pooled-reference standardization.
    """
    spec_order = SPECTRUM_ORDER_5 if n_spectra == 5 else SPECTRUM_ORDER_6
    cz_by_norm = {_normalize_label(c): c for c in df.columns if c.startswith("HiTOP_")}

    def _resolve_col(ku_name: str) -> str:
        if cohort == "ku":
            if ku_name not in df.columns:
                raise KeyError(f"Missing KU scale column: {ku_name}")
            return ku_name
        if ku_name in KU_TO_CZ_ALIASES:
            col = KU_TO_CZ_ALIASES[ku_name]
        else:
            col = cz_by_norm.get(_normalize_label(ku_name), "")
        if col not in df.columns:
            raise KeyError(f"Missing Czech scale column for {ku_name} (resolved: '{col}')")
        return col

    comp_data: dict[str, NDArray[np.float64]] = {}
    for spec in spec_order:
        pairs = SPECTRUM_SCALES_6[spec]
        mat = np.empty((len(df), len(pairs)), dtype=np.float64)
        for j, (ku_name, sign) in enumerate(pairs):
            col = _resolve_col(ku_name)
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            if standardize_mode == "within":
                mu = float(np.nanmean(vals))
                sd = float(np.nanstd(vals, ddof=1))
                vals = ((vals - mu) / sd) * sign if sd > 0.0 else np.zeros_like(vals)
            else:
                vals = (5.0 - vals) if sign < 0.0 else vals
            mat[:, j] = vals

        miss_frac = np.mean(np.isnan(mat), axis=1)
        with np.errstate(all="ignore"):
            spec_mean = np.nanmean(mat, axis=1)
        spec_mean[miss_frac > max_missing_frac] = np.nan
        comp_data[spec] = spec_mean

    out_df = pd.DataFrame(comp_data, index=df.index)
    n_input = int(len(out_df))
    out_clean = out_df.dropna().copy()
    if standardize_mode == "within" and len(out_clean) > 1:
        for spec in spec_order:
            arr = out_clean[spec].to_numpy(dtype=np.float64)
            sd = float(np.std(arr, ddof=1))
            if sd > 0.0:
                out_clean[spec] = (arr - float(np.mean(arr))) / sd

    return CompositeResult(
        scores=out_clean,
        n_input=n_input,
        n_complete=int(len(out_clean)),
        n_dropped=n_input - int(len(out_clean)),
        spectrum_order=spec_order,
    )
