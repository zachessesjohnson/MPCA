"""Tests for imputation and Option B filter."""

import numpy as np
import pandas as pd
import pytest

from mpca import option_b_filter, three_pass_imputation


def make_df(**kwargs):
    return pd.DataFrame(kwargs)


class TestOptionBFilter:
    def test_adds_n_obs_and_valid_composite_columns(self):
        df = make_df(
            s1=[50, np.nan, 60],
            s2=[55, 40, np.nan],
            s3=[np.nan, np.nan, 70],
            s4=[60, 55, 80],
            s5=[45, np.nan, 65],
        )
        out = option_b_filter(df, ["s1", "s2", "s3", "s4", "s5"], min_obs=3)
        assert "n_obs" in out.columns
        assert "valid_composite" in out.columns
        assert list(out["n_obs"]) == [4, 2, 4]
        assert list(out["valid_composite"]) == [True, False, True]

    def test_flags_rows_below_min_obs(self):
        scores = np.full((3, 8), 50.0)
        scores[1, :4] = np.nan  # row 1 has only 4 observed
        df = pd.DataFrame(scores, columns=[f"s{k}" for k in range(8)])
        out = option_b_filter(df, [f"s{k}" for k in range(8)], min_obs=5)
        assert list(out["valid_composite"]) == [True, False, True]

    def test_does_not_modify_input(self):
        df = make_df(s1=[50, np.nan], s2=[60, 40])
        _ = option_b_filter(df, ["s1", "s2"], min_obs=1)
        assert "n_obs" not in df.columns


class TestThreePassImputation:
    def _basic_df(self):
        return pd.DataFrame({
            "country": ["A", "A", "A", "B", "B"],
            "year": [2000, 2001, 2002, 2000, 2001],
            "s1": [50.0, np.nan, 60.0, np.nan, 70.0],
            "s1_hw": [5.0, np.nan, 5.0, np.nan, 5.0],
        })

    def test_no_nans_after_imputation(self):
        df = self._basic_df()
        S_hat, H = three_pass_imputation(df, ["s1"], ["s1_hw"])
        assert not np.isnan(S_hat).any()

    def test_h_is_zero_for_imputed_positions(self):
        df = pd.DataFrame({
            "country": ["A", "A", "A"],
            "year": [2000, 2001, 2002],
            "s1": [50.0, np.nan, 60.0],
            "s1_hw": [5.0, np.nan, 5.0],
        })
        S_hat, H = three_pass_imputation(df, ["s1"], ["s1_hw"])
        # After sort: rows are already in order
        # Middle position was NA → H should be 0
        assert H[1, 0] == pytest.approx(0.0)
        assert H[0, 0] == pytest.approx(5.0)
        assert H[2, 0] == pytest.approx(5.0)

    def test_pass1_linear_interpolation(self):
        df = pd.DataFrame({
            "country": ["A", "A", "A"],
            "year": [2000, 2001, 2002],
            "s1": [50.0, np.nan, 60.0],
            "s1_hw": [0.0, np.nan, 0.0],
        })
        S_hat, _ = three_pass_imputation(df, ["s1"], ["s1_hw"])
        assert S_hat[1, 0] == pytest.approx(55.0)

    def test_pass2_year_mean_fallback(self):
        df = pd.DataFrame({
            "country": ["A", "B"],
            "year": [2000, 2000],
            "s1": [np.nan, 80.0],
            "s1_hw": [np.nan, 0.0],
        })
        S_hat, _ = three_pass_imputation(df, ["s1"], ["s1_hw"])
        # Country A has no observations; year 2000 mean = 80
        a_idx = df.sort_values(["country", "year"]).index.tolist()
        sorted_df = df.sort_values(["country", "year"]).reset_index(drop=True)
        a_pos = sorted_df[sorted_df["country"] == "A"].index[0]
        assert S_hat[a_pos, 0] == pytest.approx(80.0)

    def test_pass3_global_mean_fallback(self):
        df = pd.DataFrame({
            "country": ["A", "B"],
            "year": [2000, 2001],
            "s1": [np.nan, 80.0],
            "s1_hw": [np.nan, 0.0],
        })
        S_hat, _ = three_pass_imputation(df, ["s1"], ["s1_hw"])
        # Year 2000 has no observations → global mean = 80
        sorted_df = df.sort_values(["country", "year"]).reset_index(drop=True)
        a_pos = sorted_df[sorted_df["country"] == "A"].index[0]
        assert S_hat[a_pos, 0] == pytest.approx(80.0)
