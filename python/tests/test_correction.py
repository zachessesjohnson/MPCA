"""Tests for attenuation correction and naive PCA."""

import numpy as np
import pytest

from mpca import naive_pca, attenuation_correction


def make_data(N=50, K=8, seed=1):
    rng = np.random.default_rng(seed)
    S = rng.uniform(20, 80, (N, K))
    H = rng.uniform(1, 10, (N, K))
    return S, H


class TestNaivePCA:
    def test_output_shapes(self):
        S, _ = make_data()
        N, K = S.shape
        result = naive_pca(S)

        assert result["col_means"].shape == (K,)
        assert result["col_sds"].shape == (K,)
        assert result["S_tilde"].shape == (N, K)
        assert result["R_obs"].shape == (K, K)
        assert result["ell_hat"].shape == (K,)
        assert result["w_hat"].shape == (K,)
        assert result["f_hat"].shape == (N,)

    def test_loadings_sum_positive(self):
        """Sign convention: sum of loadings > 0 (not necessarily all positive)."""
        S, _ = make_data()
        result = naive_pca(S)
        assert result["ell_hat"].sum() > 0

    def test_loadings_all_positive_with_common_factor(self):
        """With a strong positive common factor all loadings should be positive."""
        rng = np.random.default_rng(0)
        N, K = 200, 8
        factor = rng.normal(0, 1, (N, 1))
        # Each sub-index = 50 + 10*factor + small noise → all positively correlated
        S = 50 + 10 * np.tile(factor, (1, K)) + rng.normal(0, 1, (N, K))
        S = np.clip(S, 0, 100)
        result = naive_pca(S)
        assert np.all(result["ell_hat"] > 0)

    def test_var_explained_in_range(self):
        S, _ = make_data()
        result = naive_pca(S)
        assert 0 < result["var_explained"] <= 1

    def test_raises_on_zero_variance_column(self):
        S, _ = make_data()
        S[:, 0] = 50.0
        with pytest.raises(ValueError, match="zero variance"):
            naive_pca(S)


class TestAttenuationCorrection:
    def test_output_shapes(self):
        S, H = make_data()
        N, K = S.shape
        naive = naive_pca(S)
        corr = attenuation_correction(S, H, naive)

        assert corr["psi_hat"].shape == (K,)
        assert corr["reliability"].shape == (K,)
        assert corr["sigma_sig"].shape == (K,)
        assert corr["Sigma_sig"].shape == (K, K)
        assert corr["R_sig"].shape == (K, K)
        assert corr["ell_hat_star"].shape == (K,)
        assert corr["w_hat_star"].shape == (K,)
        assert corr["f_hat_star"].shape == (N,)

    def test_r_sig_is_positive_definite(self):
        S, H = make_data()
        naive = naive_pca(S)
        corr = attenuation_correction(S, H, naive)
        ev = np.linalg.eigvalsh(corr["R_sig"])
        assert np.all(ev > 0)

    def test_signal_variance_le_observed_variance(self):
        S, H = make_data()
        naive = naive_pca(S)
        corr = attenuation_correction(S, H, naive)
        assert np.all(np.diag(corr["Sigma_sig"]) <= np.diag(corr["Sigma_obs"]) + 1e-10)

    def test_reliability_in_valid_range(self):
        S, H = make_data()
        naive = naive_pca(S)
        corr = attenuation_correction(S, H, naive)
        assert np.all(corr["reliability"] > 0)
        assert np.all(corr["reliability"] <= 1 + 1e-10)

    def test_corrected_pc1_variance_share_exceeds_naive(self):
        """Proposition: lambda1_star / K >= lambda1_naive / K."""
        S, H = make_data()
        naive = naive_pca(S)
        corr = attenuation_correction(S, H, naive)
        assert corr["var_explained_star"] >= naive["var_explained"] - 1e-10

    def test_near_singular_correlation_solve_is_stable(self):
        rng = np.random.default_rng(11)
        N, K = 80, 3
        x = rng.normal(size=N)
        S = np.column_stack(
            [
                x,
                x + 1e-8 * rng.normal(size=N),
                rng.normal(size=N),
            ]
        )
        H = np.full((N, K), 0.2)

        naive = naive_pca(S)
        corr = attenuation_correction(S, H, naive)
        assert np.all(np.isfinite(corr["w_hat_star"]))
