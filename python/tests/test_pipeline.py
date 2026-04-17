"""Integration tests for the full MPCA pipeline."""

import numpy as np
import pandas as pd
import pytest

from mpca import mpca_pipeline, two_stage_bootstrap, postprocess_scores


def make_pipeline_df(N=60, K=8, seed=7):
    rng = np.random.default_rng(seed)
    groups = [f"G{i}" for i in range(1, 21)] * 3
    periods = [2022, 2023, 2024] * 20
    df = pd.DataFrame({"group": groups, "time": periods})

    for k in range(K):
        scores = rng.uniform(20, 80, N).astype(float)
        hw = rng.uniform(1, 10, N)
        # Introduce a few NAs
        na_idx = rng.choice(N, 5, replace=False)
        scores[na_idx] = np.nan
        df[f"s{k}"] = scores
        df[f"s{k}_lo"] = scores - hw
        df[f"s{k}_hi"] = scores + hw
    return df


@pytest.fixture
def pipeline_result():
    df = make_pipeline_df()
    sc = [f"s{k}" for k in range(8)]
    lo = [f"s{k}_lo" for k in range(8)]
    hi = [f"s{k}_hi" for k in range(8)]
    return mpca_pipeline(
        df, sc, lo, hi,
        id_cols=["group", "time"],
        group_col="group",
        time_col="time",
        rankings_value=2024,
        B=20, seed=1,
    )


class TestPipelineOutputStructure:
    def test_returns_three_dataframes(self, pipeline_result):
        assert isinstance(pipeline_result["scores_df"], pd.DataFrame)
        assert isinstance(pipeline_result["contributions_df"], pd.DataFrame)
        assert isinstance(pipeline_result["rankings_df"], pd.DataFrame)

    def test_scores_df_columns(self, pipeline_result):
        expected = {"group", "time", "score", "ci_lower", "ci_upper"}
        assert expected.issubset(pipeline_result["scores_df"].columns)

    def test_contributions_df_has_eight_rows(self, pipeline_result):
        assert len(pipeline_result["contributions_df"]) == 8

    def test_contributions_df_columns(self, pipeline_result):
        expected = {
            "sub_index", "naive_loading", "naive_weight",
            "corrected_loading", "corrected_weight", "reliability",
        }
        assert expected.issubset(pipeline_result["contributions_df"].columns)


class TestScoreRange:
    def test_scores_in_0_100(self, pipeline_result):
        s = pipeline_result["scores_df"]
        assert (s["score"] >= -1e-10).all()
        assert (s["score"] <= 100 + 1e-10).all()

    def test_ci_bounds_in_0_100(self, pipeline_result):
        s = pipeline_result["scores_df"]
        assert (s["ci_lower"] >= -1e-10).all()
        assert (s["ci_lower"] <= 100 + 1e-10).all()
        assert (s["ci_upper"] >= -1e-10).all()
        assert (s["ci_upper"] <= 100 + 1e-10).all()

    def test_ci_lower_le_score_le_ci_upper(self, pipeline_result):
        s = pipeline_result["scores_df"]
        assert (s["ci_lower"] <= s["score"] + 1e-6).all()
        assert (s["score"] <= s["ci_upper"] + 1e-6).all()


class TestRankings:
    def test_rankings_sorted_descending(self, pipeline_result):
        rd = pipeline_result["rankings_df"]
        if len(rd) > 1:
            assert (rd["score"].diff().dropna() <= 1e-10).all()

    def test_rankings_have_rank_column(self, pipeline_result):
        assert "rank" in pipeline_result["rankings_df"].columns

    def test_rankings_start_at_one(self, pipeline_result):
        rd = pipeline_result["rankings_df"]
        if len(rd) > 0:
            assert rd["rank"].iloc[0] == 1

    def test_no_rankings_when_rankings_value_none(self):
        df = make_pipeline_df()
        sc = [f"s{k}" for k in range(8)]
        lo = [f"s{k}_lo" for k in range(8)]
        hi = [f"s{k}_hi" for k in range(8)]
        result = mpca_pipeline(
            df, sc, lo, hi,
            id_cols=["group", "time"],
            group_col="group",
            time_col="time",
            rankings_value=None,
            B=20, seed=1,
        )
        assert len(result["rankings_df"]) == 0

    def test_pipeline_without_id_cols(self):
        df = make_pipeline_df()
        sc = [f"s{k}" for k in range(8)]
        lo = [f"s{k}_lo" for k in range(8)]
        hi = [f"s{k}_hi" for k in range(8)]
        result = mpca_pipeline(df, sc, lo, hi, B=20, seed=1)
        expected = {"score", "ci_lower", "ci_upper"}
        assert expected == set(result["scores_df"].columns)


class TestBootstrap:
    def test_two_stage_wider_than_zero_perturbation(self):
        rng = np.random.default_rng(99)
        N, K = 30, 4
        S = rng.uniform(20, 80, (N, K))
        H = rng.uniform(3, 12, (N, K))
        col_means = S.mean(axis=0)
        sigma_sig = S.std(axis=0, ddof=1)
        w = np.ones(K) / K

        ts = two_stage_bootstrap(S, H, col_means, w, sigma_sig, B=100, seed=5)
        ss = two_stage_bootstrap(S, np.zeros_like(H), col_means, w, sigma_sig,
                                  B=100, seed=5)

        ts_width = np.mean(ts["ci_upper"] - ts["ci_lower"])
        ss_width = np.mean(ss["ci_upper"] - ss["ci_lower"])
        assert ts_width >= ss_width


class TestPostprocess:
    def test_output_in_0_100(self):
        rng = np.random.default_rng(0)
        f = rng.normal(0, 2, 100)
        lo = f - np.abs(rng.normal(0, 0.5, 100))
        hi = f + np.abs(rng.normal(0, 0.5, 100))
        pp = postprocess_scores(f, lo, hi)

        assert np.all(pp["score"] >= -1e-10)
        assert np.all(pp["score"] <= 100 + 1e-10)
        assert np.all(pp["ci_lower"] >= -1e-10)
        assert np.all(pp["ci_upper"] <= 100 + 1e-10)
