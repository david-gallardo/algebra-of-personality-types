"""Unit tests for Gerlach et al. (2018) density-enrichment clustering procedure."""

from __future__ import annotations

import numpy as np
import pytest

from arc_study.clustering import run_gerlach_procedure
from arc_study.generators import sample_null2_correlated_normal
from arc_study.matrices import BASE_SEED, VAN_DER_LINDEN_2010


def test_gerlach_raises_when_min_pvalue_exceeds_or_equals_alpha() -> None:
    rng = np.random.default_rng(BASE_SEED)
    data = rng.standard_normal((120, 5))

    # 1 / (50 + 1) = 0.0196 >= 0.01 -> must raise ValueError
    with pytest.raises(ValueError, match="minimum attainable Monte Carlo p-value"):
        run_gerlach_procedure(
            data,
            n_clusters=3,
            n_null_samples=50,
            alpha=0.01,
            null_models=("permutation",),
            random_state=BASE_SEED,
        )

    # Boundary case: 1 / (99 + 1) = 0.01 >= 0.01 -> must raise ValueError
    with pytest.raises(ValueError, match="minimum attainable Monte Carlo p-value"):
        run_gerlach_procedure(
            data,
            n_clusters=3,
            n_null_samples=99,
            alpha=0.01,
            null_models=("permutation",),
            random_state=BASE_SEED,
        )


def test_gerlach_supports_isotropic_and_scott_kde_modes() -> None:
    rng = np.random.default_rng(BASE_SEED)
    data = sample_null2_correlated_normal(n=250, corr_matrix=VAN_DER_LINDEN_2010, rng=rng)

    res_iso = run_gerlach_procedure(
        data,
        n_clusters=4,
        n_init=3,
        n_null_samples=25,
        null_models=("permutation", "null2"),
        alpha=0.05,  # 1/26 = 0.0385 < 0.05
        min_enrichment=1.25,
        kde_mode="isotropic",
        isotropic_bandwidth=0.50,
        random_state=BASE_SEED,
    )
    res_scott = run_gerlach_procedure(
        data,
        n_clusters=4,
        n_init=3,
        n_null_samples=25,
        null_models=("permutation", "null2"),
        alpha=0.05,
        min_enrichment=1.25,
        kde_mode="scott",
        random_state=BASE_SEED,
    )

    for res in (res_iso, res_scott):
        assert res.centers.shape == (4, 5)
        assert res.empirical_densities.shape == (4,)
        assert np.all(res.empirical_densities > 0.0)
        for null_name in ("permutation", "null2"):
            ev = res.evaluations[null_name]
            assert ev.enrichment.shape == (4,)
            assert ev.p_values.shape == (4,)
            assert np.all(ev.p_values >= 1.0 / 26.0)
            assert np.all(ev.p_values <= 1.0)
            n_sig_125 = ev.count_significant(min_enrichment=1.25, alpha=0.05)
            n_sig_100 = ev.count_significant(min_enrichment=1.0, alpha=0.05)
            assert n_sig_125 == ev.n_significant_types
            assert n_sig_100 >= n_sig_125
