"""Tests for imputation and Option Filter."""

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
            "group": ["A", "A", "A", "B", "B"],
            "time": [2000, 2001, 2002, 2000, 2001],
            "s1": [50.0, np.nan, 60.0, np.nan, 70.0],
            "s1_hw": [5.0, np.nan, 5.0, np.nan, 5.0],
        })

    def test_no_nans_after_imputation(self):
        df = self._basic_df()
        S_hat, H = three_pass_imputation(
            df, ["s1"], ["s1_hw"], group_col="group", time_col="time"
        )
        assert not np.isnan(S_hat).any()

    def test_h_is_zero_for_imputed_positions(self):
        df = pd.DataFrame({
            "group": ["A", "A", "A"],
            "time": [2000, 2001, 2002],
            "s1": [50.0, np.nan, 60.0],
            "s1_hw": [5.0, np.nan, 5.0],
        })
        S_hat, H = three_pass_imputation(
            df, ["s1"], ["s1_hw"], group_col="group", time_col="time"
        )
        # Middle position was NA → H should be 0
        assert H[1, 0] == pytest.approx(0.0)
        assert H[0, 0] == pytest.approx(5.0)
        assert H[2, 0] == pytest.approx(5.0)

    def test_pass1_linear_interpolation(self):
        df = pd.DataFrame({
            "group": ["A", "A", "A"],
            "time": [2000, 2001, 2002],
            "s1": [50.0, np.nan, 60.0],
            "s1_hw": [0.0, np.nan, 0.0],
        })
        S_hat, _ = three_pass_imputation(
            df, ["s1"], ["s1_hw"], group_col="group", time_col="time"
        )
        assert S_hat[1, 0] == pytest.approx(55.0)

    def test_pass2_time_mean_fallback(self):
        df = pd.DataFrame({
            "group": ["A", "B"],
            "time": [2000, 2000],
            "s1": [np.nan, 80.0],
            "s1_hw": [np.nan, 0.0],
        })
        S_hat, _ = three_pass_imputation(
            df, ["s1"], ["s1_hw"], group_col="group", time_col="time"
        )
        # Group A has no observations; time 2000 mean = 80
        sorted_df = df.sort_values(["group", "time"]).reset_index(drop=True)
        a_pos = sorted_df[sorted_df["group"] == "A"].index[0]
        assert S_hat[a_pos, 0] == pytest.approx(80.0)

    def test_pass3_global_mean_fallback(self):
        df = pd.DataFrame({
            "group": ["A", "B"],
            "time": [2000, 2001],
            "s1": [np.nan, 80.0],
            "s1_hw": [np.nan, 0.0],
        })
        S_hat, _ = three_pass_imputation(
            df, ["s1"], ["s1_hw"], group_col="group", time_col="time"
        )
        # Time 2000 has no observations → global mean = 80
        sorted_df = df.sort_values(["group", "time"]).reset_index(drop=True)
        a_pos = sorted_df[sorted_df["group"] == "A"].index[0]
        assert S_hat[a_pos, 0] == pytest.approx(80.0)

    def test_no_group_col_falls_through_to_global_mean(self):
        df = pd.DataFrame({
            "s1": [np.nan, 80.0, 60.0],
            "s1_hw": [np.nan, 0.0, 0.0],
        })
        S_hat, _ = three_pass_imputation(df, ["s1"], ["s1_hw"])
        # Only Pass 3 applies; global mean = (80+60)/2 = 70
        assert S_hat[0, 0] == pytest.approx(70.0)
