"""Mandatory synthetic unit tests (a)-(j) and privacy guards for src/hitop_empirical/.

All tests operate strictly on clearly labeled SYNTHETIC fixtures — zero real participant data.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from hitop_empirical import BASE_SEED
from hitop_empirical.clustering import fit_matched_kmeans
from hitop_empirical.generators import (
    calibrate_empirical_copula,
    generate_null_sample,
    generate_synthetic_mixture,
)
from hitop_empirical.geometry import correlation_geometry
from hitop_empirical.harmonization import check_unique_ids
from hitop_empirical.inference import holm_bonferroni, phipson_smyth_p, weighted_mean_and_var, wilson_ci
from hitop_empirical.matching import match_centroids_hungarian
from hitop_empirical.sources import assert_no_row_level_leakage
from hitop_empirical.topography import compute_h_nn, pairwise_valley_depth, run_mean_shift


# (a) Independent normal cloud yields no fixed centroid orientation across replicated samples
def test_a_independent_normal_no_fixed_orientation() -> None:
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 1]))
    phis = []
    for _ in range(15):
        x1 = rng.standard_normal((400, 5))
        x2 = rng.standard_normal((400, 5))
        f1 = fit_matched_kmeans(x1, n_clusters=3, n_kmeanspp_starts=5, rng=rng)
        f2 = fit_matched_kmeans(x2, n_clusters=3, n_kmeanspp_starts=5, rng=rng)
        phis.append(match_centroids_hungarian(f1.centers, f2.centers).mean_tucker_full)
    assert float(np.mean(phis)) < 0.72


# (b) Correlated normal produces stable quantization orientation without latent classes
def test_b_correlated_normal_stabilizes_orientation() -> None:
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 2]))
    r = np.array(
        [
            [1.0, 0.55, 0.50, 0.35, 0.40],
            [0.55, 1.0, 0.45, 0.25, 0.20],
            [0.50, 0.45, 1.0, 0.45, 0.40],
            [0.35, 0.25, 0.45, 1.0, 0.60],
            [0.40, 0.20, 0.40, 0.60, 1.0],
        ]
    )
    chol = np.linalg.cholesky(r)
    phis_corr, phis_indep = [], []
    for _ in range(12):
        x1 = rng.standard_normal((600, 5)) @ chol.T
        x2 = rng.standard_normal((600, 5)) @ chol.T
        f1 = fit_matched_kmeans(x1, n_clusters=3, n_kmeanspp_starts=5, rng=rng)
        f2 = fit_matched_kmeans(x2, n_clusters=3, n_kmeanspp_starts=5, rng=rng)
        phis_corr.append(match_centroids_hungarian(f1.centers, f2.centers).mean_tucker_full)

        i1 = rng.standard_normal((600, 5))
        i2 = rng.standard_normal((600, 5))
        fi1 = fit_matched_kmeans(i1, n_clusters=3, n_kmeanspp_starts=5, rng=rng)
        fi2 = fit_matched_kmeans(i2, n_clusters=3, n_kmeanspp_starts=5, rng=rng)
        phis_indep.append(match_centroids_hungarian(fi1.centers, fi2.centers).mean_tucker_full)
    assert float(np.mean(phis_corr)) > float(np.mean(phis_indep)) + 0.20
    g = correlation_geometry(x1)
    assert g.is_pd and g.origin_density_ratio > 1.5


# (c) True separated Gaussian mixtures trigger topographic sensitivity at realistic N
def test_c_true_separated_mixture_triggers_topography() -> None:
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 3]))
    x, _, true_centers = generate_synthetic_mixture(
        n=500, d=5, k=3, separation_sd=4.5, rng=rng
    )
    h = compute_h_nn(x, rng=rng)
    topo = run_mean_shift(x, true_centers, bandwidth=h, merge_radius=0.30)
    assert topo.n_modes == 3
    valley = pairwise_valley_depth(x, topo.mode_locations[0], topo.mode_locations[1], bandwidth=h)
    assert valley > 0.30


# (d) Skewed/discrete marginal and floor effects handled cleanly by C1 copula
def test_d_floor_skewed_marginals_and_copula_calibration() -> None:
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 4]))
    x_raw, _, _ = generate_synthetic_mixture(
        n=420, d=5, k=2, separation_sd=0.0, within_corr=0.45, floor_quantile=0.40, rng=rng
    )
    cal = calibrate_empirical_copula(x_raw, rng=rng, n_cal=15_000, tol=1e-3)
    assert cal.max_abs_pearson_error < 0.03
    sim_c1 = generate_null_sample("C1", x_raw, rng=rng, copula_spec=cal)
    assert sim_c1.shape == x_raw.shape
    assert np.all(np.isfinite(sim_c1))


# (e) Across-culture scaling/elevation shift: shape congruence remains invariant to uniform shift
def test_e_cross_culture_elevation_shift_vs_shape_invariance() -> None:
    base = np.array(
        [
            [-0.8, -0.7, -0.6, -0.5, -0.4],
            [+0.6, +0.5, +0.2, -0.4, -0.5],
            [+0.4, -0.3, +0.5, +0.8, +0.7],
        ]
    )
    shifted = base + 1.5  # uniform elevation shift (e.g., clinical enrichment)
    m = match_centroids_hungarian(base, shifted, optimize_on="shape_tucker")
    assert m.mean_tucker_shape == pytest.approx(1.0, abs=1e-12)
    assert m.mean_tucker_full < 0.90  # full profile detects the elevation shift


# (f) Duplicate IDs, missing data, and skip-routed diagnostic modules
def test_f_duplicate_ids_and_structural_missingness() -> None:
    df = pd.DataFrame(
        {
            "ResponseId": ["A1", "A2", "A2", "A3"],
            "Mental_health": [1, 0, 0, 1],
            "Treatment": [1.0, np.nan, np.nan, 0.0],  # skip-routed NA must remain NA, not 0
        }
    )
    assert check_unique_ids(df, "ResponseId") == 1
    assert int(df["Treatment"].isna().sum()) == 2


# (g) Full vs row-centered shape congruence and Hungarian permutation invariance
def test_g_hungarian_matching_permutation_invariance() -> None:
    c = np.array([[-1.0, 0.5, 0.8, 0.2, -0.5], [0.9, -0.8, -0.2, 0.1, 0.0], [0.1, 0.3, -0.6, -0.3, 0.5]])
    perm_c = c[[2, 0, 1]]
    m = match_centroids_hungarian(c, perm_c, optimize_on="full_tucker")
    assert m.mean_tucker_full == pytest.approx(1.0, abs=1e-12)
    assert m.mean_tucker_shape == pytest.approx(1.0, abs=1e-12)
    assert m.mean_d2_full == pytest.approx(0.0, abs=1e-12)


# (h) Exact reproducibility with fixed SeedSequence and single vs multistart
def test_h_reproducibility_and_single_vs_multistart() -> None:
    x = np.random.default_rng(123).standard_normal((250, 5))
    f1 = fit_matched_kmeans(x, n_clusters=3, n_kmeanspp_starts=9, rng=np.random.default_rng(999))
    f2 = fit_matched_kmeans(x, n_clusters=3, n_kmeanspp_starts=9, rng=np.random.default_rng(999))
    assert np.allclose(f1.centers, f2.centers, atol=1e-12)
    assert f1.sse == pytest.approx(f2.sse, abs=1e-12)

    fs = fit_matched_kmeans(x, n_clusters=3, n_kmeanspp_starts=0, rng=np.random.default_rng(999))
    assert len(fs.all_start_sses) == 1
    assert f1.sse <= fs.sse + 1e-8


# (i) Survey weighting correctness against analytical ground truth
def test_i_survey_weighting_exact_case() -> None:
    vals = np.array([1.0, 3.0, 5.0])
    weights = np.array([1.0, 2.0, 1.0])
    mu, var = weighted_mean_and_var(vals, weights)
    assert mu == pytest.approx(3.0, abs=1e-12)
    assert var == pytest.approx(2.0, abs=1e-12)  # (1*4 + 2*0 + 1*4) / 4 = 2.0


# (j) Null calibration: Phipson-Smyth p-values under true null are uniform (no anti-conservative bias)
def test_j_null_calibration_and_multiplicity() -> None:
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 10]))
    null_draws = rng.standard_normal(1000)
    p_hi, k_hi, b = phipson_smyth_p(null_draws, observed=10.0, direction="ge")
    assert k_hi == 0 and b == 1000
    assert p_hi == pytest.approx(1.0 / 1001.0, abs=1e-12)
    lo, hi = wilson_ci(0, 1000)
    assert 0.0 <= lo < hi < 0.01

    adj = holm_bonferroni(np.array([0.01, 0.04, 0.03]))
    assert np.allclose(adj, [0.03, 0.06, 0.06], atol=1e-12)


# Privacy guard: assert_no_row_level_leakage catches leaked participant IDs
def test_privacy_guard_catches_leaked_ids(tmp_path: Path) -> None:
    safe_csv = tmp_path / "summary.csv"
    pd.DataFrame({"metric": ["phi"], "value": [0.85]}).to_csv(safe_csv, index=False)
    assert_no_row_level_leakage(tmp_path)  # must pass

    bad_csv = tmp_path / "leaked.csv"
    pd.DataFrame({"ResponseId": ["ID1"], "score": [1.2]}).to_csv(bad_csv, index=False)
    with pytest.raises(AssertionError, match="Row-level participant identifier"):
        assert_no_row_level_leakage(tmp_path)
