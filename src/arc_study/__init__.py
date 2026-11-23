"""Core package for the Algebra of Personality Types (ARC) study."""

from __future__ import annotations

from arc_study.matrices import (
    BASE_SEED,
    BUSSERI_ERB_2024,
    IDENTITY_5,
    TRAIT_ORDER,
    VAN_DER_LINDEN_2010,
    get_observed_matrices,
    is_positive_definite,
    validate_correlation_matrix,
)
from arc_study.theory import (
    TRIANGULAR_SECTOR_CONSTANT,
    SpectralAnalysis,
    TriangularConstruction,
    collinear_between_variance,
    compute_cl,
    density_ratio,
    non_collinearity_threshold,
    spectral_analysis,
    triangular_sector_construction,
)

__all__ = [
    "BASE_SEED",
    "BUSSERI_ERB_2024",
    "IDENTITY_5",
    "TRAIT_ORDER",
    "TRIANGULAR_SECTOR_CONSTANT",
    "VAN_DER_LINDEN_2010",
    "SpectralAnalysis",
    "TriangularConstruction",
    "collinear_between_variance",
    "compute_cl",
    "density_ratio",
    "get_observed_matrices",
    "is_positive_definite",
    "non_collinearity_threshold",
    "spectral_analysis",
    "triangular_sector_construction",
    "validate_correlation_matrix",
]
