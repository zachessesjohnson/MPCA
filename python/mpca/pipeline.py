"""Full MPCA pipeline."""

import numpy as np
import pandas as pd

from .imputation import option_b_filter, three_pass_imputation
from .naive_pca import naive_pca
from .correction import attenuation_correction
from .bootstrap import two_stage_bootstrap
from .postprocess import postprocess_scores


def mpca_pipeline(
    data: pd.DataFrame,
    score_cols: list,
    lower_cols: list,
    upper_cols: list,
    country_col: str = "country",
    iso_col: str = "iso",
    year_col: str = "year",
    B: int = 500,
    seed: int = 42,
    min_obs: int = 5,
    rankings_year: int = 2024,
) -> dict:
    """
    Run the complete Multilevel PCA pipeline.

    Executes the following steps in order:

    1. **Option Filter** — exclude observations with fewer than *min_obs*
       observed sub-indices.
    2. Compute CI **half-widths** :math:`h_{ij} = (U_{ij} - L_{ij})/2`.
    3. **Three-pass imputation** — within-country linear interpolation,
       year-mean fallback, global-mean fallback.
    4. **Naive PCA** — PC1 loadings and regression scoring coefficients on
       the observed correlation matrix.
    5. **Attenuation correction** — disattenuation via signal covariance,
       corrected PCA on :math:`\\hat{\\mathbf{R}}_{\\text{sig}}`.
    6. **Two-stage bootstrap** — propagate measurement error and sampling
       uncertainty; output 95% CI.
    7. **Post-processing** — z-score → Box-Cox (λ = 0.5) → min-max
       rescaling to [0, 100].

    Parameters
    ----------
    data : pd.DataFrame
        Input data with sub-index scores, lower/upper CI bounds, country,
        ISO code, and year columns.
    score_cols : list of str
        Score column names (length K).
    lower_cols : list of str
        Lower CI column names (same order as *score_cols*).
    upper_cols : list of str
        Upper CI column names (same order as *score_cols*).
    country_col : str, optional
        Country identifier column (default ``"country"``).
    iso_col : str, optional
        ISO code column (default ``"iso"``).
    year_col : str, optional
        Year column (default ``"year"``).
    B : int, optional
        Bootstrap replications (default 500).
    seed : int, optional
        Random seed (default 42).
    min_obs : int, optional
        Minimum non-NaN sub-indices for a valid composite (default 5).
    rankings_year : int or None, optional
        Year for which to produce a ranked output table; ``None`` skips
        rankings (default 2024).

    Returns
    -------
    dict with keys:

    ``scores_df`` : pd.DataFrame
        country, iso, year, score, ci_lower, ci_upper for all
        analysis-set observations.
    ``contributions_df`` : pd.DataFrame
        Sub-index names with naive and corrected loadings, scoring weights,
        and reliability estimates.
    ``rankings_df`` : pd.DataFrame
        Observations for *rankings_year* sorted descending by score, with
        a ``rank`` column.
    """
    K = len(score_cols)
    if len(lower_cols) != K or len(upper_cols) != K:
        raise ValueError(
            "score_cols, lower_cols, and upper_cols must all have the same length."
        )

    # ---- Step 1: Option Filter ----
    data = option_b_filter(data, score_cols, min_obs)
    data_valid = data[data["valid_composite"]].copy().reset_index(drop=True)

    # ---- Step 2: CI half-widths ----
    hw_cols = [f"{c}_hw" for c in score_cols]
    for k in range(K):
        lo = data_valid[lower_cols[k]]
        up = data_valid[upper_cols[k]]
        data_valid[hw_cols[k]] = (up - lo) / 2.0

    # ---- Step 3: Three-pass imputation ----
    S_hat, H = three_pass_imputation(
        data_valid, score_cols, hw_cols,
        country_col=country_col,
        year_col=year_col,
    )
    # Reorder data_valid to match sort inside three_pass_imputation
    data_valid = (
        data_valid
        .sort_values([country_col, year_col])
        .reset_index(drop=True)
    )

    # ---- Step 4: Naive PCA ----
    naive = naive_pca(S_hat)

    # ---- Step 5: Attenuation correction ----
    corr = attenuation_correction(S_hat, H, naive)

    # ---- Step 6: Two-stage bootstrap ----
    boot = two_stage_bootstrap(
        S_hat=S_hat,
        H=H,
        col_means=naive["col_means"],
        w_hat_star=corr["w_hat_star"],
        sigma_sig=corr["sigma_sig"],
        B=B,
        seed=seed,
    )

    # ---- Step 7: Post-processing ----
    pp = postprocess_scores(corr["f_hat_star"], boot["ci_lower"], boot["ci_upper"])

    # ---- Assemble scores_df ----
    scores_df = pd.DataFrame({
        "country": data_valid[country_col].values,
        "iso": data_valid[iso_col].values,
        "year": data_valid[year_col].values,
        "score": pp["score"],
        "ci_lower": pp["ci_lower"],
        "ci_upper": pp["ci_upper"],
    })

    # ---- Contributions ----
    contributions_df = pd.DataFrame({
        "sub_index": score_cols,
        "naive_loading": naive["ell_hat"],
        "naive_weight": naive["w_hat"],
        "corrected_loading": corr["ell_hat_star"],
        "corrected_weight": corr["w_hat_star"],
        "reliability": corr["reliability"],
    })

    # ---- Rankings ----
    if rankings_year is not None:
        df_yr = scores_df[scores_df["year"] == rankings_year].copy()
        if len(df_yr) > 0:
            df_yr = df_yr.sort_values("score", ascending=False).reset_index(drop=True)
            df_yr["rank"] = df_yr.index + 1
        rankings_df = df_yr
    else:
        rankings_df = pd.DataFrame()

    return {
        "scores_df": scores_df,
        "contributions_df": contributions_df,
        "rankings_df": rankings_df,
    }
