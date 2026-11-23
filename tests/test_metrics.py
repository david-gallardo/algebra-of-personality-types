"""Unit tests for profile metrics, centroid matching, replicability, and incremental validity."""

from __future__ import annotations

import numpy as np
import pytest

from arc_study.matrices import BASE_SEED
from arc_study.metrics import (
    ARC_PROTOTYPES,
    assignment_stability,
    cronbach_gleser_decomposition,
    incremental_predictive_validity,
    make_gmm_posterior_extractor,
    make_ward_kmeans_onehot_extractor,
    match_centroids,
    split_half_replicability,
    tucker_congruence,
)


def test_cronbach_gleser_identical_profiles_zero_d2() -> None:
    profile = np.array([-0.8, 0.6, 0.4, 0.6, 0.7], dtype=np.float64)
    res = cronbach_gleser_decomposition(profile, profile.copy())

    assert res.d2 == pytest.approx(0.0, abs=1e-12)
    assert res.elevation == pytest.approx(0.0, abs=1e-12)
    assert res.scatter == pytest.approx(0.0, abs=1e-12)
    assert res.shape == pytest.approx(0.0, abs=1e-12)
    assert res.tucker_congruence == pytest.approx(1.0, abs=1e-12)


def test_cronbach_gleser_parallel_profiles_zero_shape() -> None:
    profile_a = np.array([-0.8, 0.6, 0.4, 0.6, 0.7], dtype=np.float64)
    shift = 1.5
    profile_b = profile_a + shift

    res = cronbach_gleser_decomposition(profile_a, profile_b)
    expected_d2 = float(profile_a.size) * shift**2

    assert res.shape == pytest.approx(0.0, abs=1e-12)
    assert res.scatter == pytest.approx(0.0, abs=1e-12)
    assert res.elevation == pytest.approx(expected_d2, abs=1e-12)
    assert res.d2 == pytest.approx(expected_d2, abs=1e-12)


def test_cronbach_gleser_positive_affine_transform_zero_shape() -> None:
    profile_a = np.array([0.8, -0.7, -0.2, -0.1, 0.1], dtype=np.float64)
    profile_b = 2.0 * profile_a - 0.5

    res = cronbach_gleser_decomposition(profile_a, profile_b)

    assert res.shape == pytest.approx(0.0, abs=1e-12)
    assert res.scatter > 0.0
    assert res.elevation + res.scatter + res.shape == pytest.approx(res.d2, abs=1e-12)


def test_cronbach_gleser_exact_decomposition_identity() -> None:
    rng = np.random.default_rng(BASE_SEED)
    for _ in range(25):
        a = rng.standard_normal(5)
        b = rng.standard_normal(5)
        res = cronbach_gleser_decomposition(a, b)

        assert res.d2 >= 0.0
        assert res.elevation >= 0.0
        assert res.scatter >= 0.0
        assert res.shape >= 0.0
        assert res.elevation + res.scatter + res.shape == pytest.approx(res.d2, abs=1e-12)

    # Constant profile edge case (zero variance)
    const = np.full(5, 2.0, dtype=np.float64)
    other = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    res_const = cronbach_gleser_decomposition(const, other)
    assert res_const.shape == pytest.approx(0.0, abs=1e-12)
    assert res_const.elevation + res_const.scatter + res_const.shape == pytest.approx(
        res_const.d2, abs=1e-12
    )


def test_tucker_congruence_properties() -> None:
    v = np.array([-0.8, 0.6, 0.4, 0.6, 0.7], dtype=np.float64)
    assert tucker_congruence(v, v) == pytest.approx(1.0, abs=1e-12)
    assert tucker_congruence(v, 3.5 * v) == pytest.approx(1.0, abs=1e-12)
    assert tucker_congruence(v, -2.0 * v) == pytest.approx(-1.0, abs=1e-12)

    e1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert tucker_congruence(e1, e2) == pytest.approx(0.0, abs=1e-12)


def test_match_centroids_exhaustive_arc_and_hungarian_arbitrary_k() -> None:
    # K = 3 uses exhaustive search over all 3! = 6 permutations
    permuted_arc = ARC_PROTOTYPES[[2, 0, 1]]
    match_3 = match_centroids(permuted_arc, ARC_PROTOTYPES, metric="d2")

    assert match_3.method == "exhaustive"
    assert match_3.centroid_indices == (1, 2, 0)
    assert match_3.prototype_indices == (0, 1, 2)
    assert np.allclose(match_3.matched_centroids, ARC_PROTOTYPES, atol=1e-12)
    assert match_3.total_cost == pytest.approx(0.0, abs=1e-12)
    assert match_3.mean_d2 == pytest.approx(0.0, abs=1e-12)
    assert match_3.mean_tucker == pytest.approx(1.0, abs=1e-12)

    # K = 7 triggers Hungarian algorithm
    rng = np.random.default_rng(BASE_SEED)
    protos_7 = rng.standard_normal((7, 5))
    perm = rng.permutation(7)
    centroids_7 = protos_7[perm]

    match_7 = match_centroids(centroids_7, protos_7, metric="d2")
    assert match_7.method == "hungarian"
    assert np.allclose(match_7.matched_centroids, protos_7, atol=1e-12)
    assert match_7.total_cost == pytest.approx(0.0, abs=1e-12)


def test_assignment_stability_and_split_half_replicability() -> None:
    labels_a = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2], dtype=np.intp)
    labels_b = (labels_a + 1) % 3
    stab = assignment_stability(labels_a, labels_b)

    assert stab.ari == pytest.approx(1.0, abs=1e-12)
    assert stab.nmi == pytest.approx(1.0, abs=1e-12)

    # Well-separated synthetic 3-cluster sample should yield high split-half kappa
    rng = np.random.default_rng(BASE_SEED)
    centers = 3.0 * ARC_PROTOTYPES
    clusters = [
        rng.normal(loc=center, scale=0.25, size=(100, 5)) for center in centers
    ]
    data = np.vstack(clusters)

    rep = split_half_replicability(
        data,
        n_clusters=3,
        n_splits=5,
        random_state=BASE_SEED,
    )
    assert rep.n_splits == 5
    assert rep.kappas.shape == (5,)
    assert rep.mean_kappa > 0.95
    assert rep.mean_ari > 0.95
    assert rep.mean_nmi > 0.95


def test_incremental_predictive_validity_linear_vs_nonlinear_clusters() -> None:
    rng = np.random.default_rng(BASE_SEED)

    # Purely linear DGP: adding cluster indicators/posteriors should not improve out-of-sample R^2
    x_lin = rng.standard_normal((400, 5))
    beta = np.array([-0.6, 0.5, 0.2, 0.4, 0.7], dtype=np.float64)
    y_lin = x_lin @ beta + rng.normal(scale=0.5, size=400)

    res_lin = incremental_predictive_validity(
        traits=x_lin,
        target=y_lin,
        cluster_extractor=make_ward_kmeans_onehot_extractor(n_clusters=3),
        task="regression",
        n_splits=5,
        random_state=BASE_SEED,
    )
    assert res_lin.metric_name == "r2"
    assert res_lin.mean_baseline > 0.6
    assert res_lin.mean_delta < 0.03

    # Nonlinear cluster-specific DGP: posteriors should add clear out-of-sample R^2
    # Three collinear clusters along dimension 0 with non-monotonic outcome (+2, -2, +2)
    c0 = rng.normal(loc=[-3.0, 0.0, 0.0, 0.0, 0.0], scale=0.35, size=(150, 5))
    c1 = rng.normal(loc=[0.0, 0.0, 0.0, 0.0, 0.0], scale=0.35, size=(150, 5))
    c2 = rng.normal(loc=[3.0, 0.0, 0.0, 0.0, 0.0], scale=0.35, size=(150, 5))
    x_nl = np.vstack([c0, c1, c2])
    y_nl = np.concatenate(
        [
            np.full(150, 2.0),
            np.full(150, -2.0),
            np.full(150, 2.0),
        ]
    ) + rng.normal(scale=0.4, size=450)

    res_nl = incremental_predictive_validity(
        traits=x_nl,
        target=y_nl,
        cluster_extractor=make_gmm_posterior_extractor(
            n_clusters=3, covariance_type="diag", n_init=5
        ),
        task="regression",
        n_splits=5,
        random_state=BASE_SEED,
    )
    assert res_nl.mean_delta > 0.5
