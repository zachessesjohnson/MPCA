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
    id_cols: list = None,
    group_col: str = None,
    time_col: str = None,
    B: int = 500,
    seed: int = 42,
    min_obs: int = 5,
    rankings_value=None,
) -> dict:
    """
    Run the complete Multilevel PCA pipeline.

    Executes the following steps in order:

    1. **Option Filter** — exclude observations with fewer than *min_obs*
       observed sub-indices.
    2. Compute CI **half-widths** :math:`h_{ij} = (U_{ij} - L_{ij})/2`.
    3. **Three-pass imputation** — within-group linear interpolation
       (*group_col*), time-period-mean fallback (*time_col*),
       global-mean fallback.
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
        Input data with sub-index scores, lower/upper CI bounds, and any
        identifier columns.
    score_cols : list of str
        Score column names (length K).
    lower_cols : list of str
        Lower CI column names (same order as *score_cols*).
    upper_cols : list of str
        Upper CI column names (same order as *score_cols*).
    id_cols : list of str or None, optional
        Identifier columns to carry through to ``scores_df`` (e.g.
        ``["country", "iso", "year"]`` or ``["unit", "time"]``).
        ``None`` produces a ``scores_df`` with only score columns.
    group_col : str or None, optional
        Column that identifies the grouping unit for within-group
        interpolation (Pass 1 of imputation).  ``None`` skips Pass 1.
    time_col : str or None, optional
        Column that identifies the time period for the time-mean fallback
        (Pass 2 of imputation) and for filtering ``rankings_df``.
        ``None`` skips Pass 2 and rankings.
    B : int, optional
        Bootstrap replications (default 500).
    seed : int, optional
        Random seed (default 42).
    min_obs : int, optional
        Minimum non-NaN sub-indices for a valid composite (default 5).
    rankings_value : scalar or None, optional
        Value of *time_col* for which to produce a ranked output table;
        ``None`` skips rankings (default ``None``).

    Returns
    -------
    dict with keys:

    ``scores_df`` : pd.DataFrame
        Identifier columns (*id_cols*) plus score, ci_lower, ci_upper for
        all analysis-set observations.
    ``contributions_df`` : pd.DataFrame
        Sub-index names with naive and corrected loadings, scoring weights,
        and reliability estimates.
    ``rankings_df`` : pd.DataFrame
        Observations for *rankings_value* sorted descending by score, with
        a ``rank`` column.  Empty when *rankings_value* is ``None`` or
        *time_col* is not in *scores_df*.
    """
    K = len(score_cols)
    if K == 0:
        raise ValueError("score_cols must contain at least one column name.")
    if len(lower_cols) != K or len(upper_cols) != K:
        raise ValueError(
            "score_cols, lower_cols, and upper_cols must all have the same length."
        )
    if not isinstance(B, int) or B <= 0:
        raise ValueError("B must be a positive integer.")
    if not isinstance(min_obs, int) or min_obs < 1 or min_obs > K:
        raise ValueError(f"min_obs must be an integer between 1 and {K}.")

    if id_cols is None:
        id_cols = []
    elif not isinstance(id_cols, list):
        raise ValueError("id_cols must be a list of column names or None.")

    required_cols = set(score_cols) | set(lower_cols) | set(upper_cols) | set(id_cols)
    for col in [group_col, time_col]:
        if col is not None:
            required_cols.add(col)
    missing = sorted(col for col in required_cols if col not in data.columns)
    if missing:
        raise ValueError(
            "Input data is missing required columns: " + ", ".join(missing)
        )

    if rankings_value is not None and time_col is None:
        raise ValueError("rankings_value requires time_col to be provided.")

    for k in range(K):
        lo = data[lower_cols[k]]
        up = data[upper_cols[k]]
        invalid = lo.notna() & up.notna() & (up < lo)
        if invalid.any():
            raise ValueError(
                f"Invalid CI bounds in '{lower_cols[k]}'/'{upper_cols[k]}': "
                "upper must be >= lower for all non-missing rows."
            )

    # ---- Step 1: Option Filter ----
    data = option_b_filter(data, score_cols, min_obs)
    data_valid = data[data["valid_composite"]].copy().reset_index(drop=True)
    if data_valid.empty:
        raise ValueError(
            "All rows were filtered out by option_b_filter; no valid observations remain."
        )

    # ---- Step 2: CI half-widths ----
    hw_cols = [f"{c}_hw" for c in score_cols]
    for k in range(K):
        lo = data_valid[lower_cols[k]]
        up = data_valid[upper_cols[k]]
        data_valid[hw_cols[k]] = (up - lo) / 2.0

    # ---- Step 3: Three-pass imputation ----
    S_hat, H = three_pass_imputation(
        data_valid, score_cols, hw_cols,
        group_col=group_col,
        time_col=time_col,
    )
    # Reorder data_valid to match sort inside three_pass_imputation
    sort_by = [c for c in [group_col, time_col] if c is not None]
    if sort_by:
        data_valid = (
            data_valid
            .sort_values(sort_by)
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
    scores_data = {col: data_valid[col].values for col in id_cols}
    scores_data["score"] = pp["score"]
    scores_data["ci_lower"] = pp["ci_lower"]
    scores_data["ci_upper"] = pp["ci_upper"]
    scores_df = pd.DataFrame(scores_data)

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
    if (
        rankings_value is not None
        and time_col is not None
        and time_col in scores_df.columns
    ):
        df_yr = scores_df[scores_df[time_col] == rankings_value].copy()
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
