"""Missing-data imputation and Option Filter coverage filter."""

import numpy as np
import pandas as pd


def option_b_filter(
    data: pd.DataFrame,
    score_cols: list,
    min_obs: int = 5,
) -> pd.DataFrame:
    """
    Flag rows that have insufficient sub-index coverage (Option Filter).

    Observations with fewer than *min_obs* of the K sub-indices observed
    (pre-imputation) receive ``valid_composite = False`` and should be
    excluded from the PCA estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing sub-index score columns.
    score_cols : list of str
        Column names for the K sub-index scores.
    min_obs : int, optional
        Minimum number of non-NaN sub-index scores required (default 5).

    Returns
    -------
    pd.DataFrame
        A copy of *data* with two new columns:

        ``n_obs`` : int
            Number of non-NaN sub-index scores per row.
        ``valid_composite`` : bool
            ``True`` when ``n_obs >= min_obs``.
    """
    data = data.copy()
    data["n_obs"] = data[score_cols].notna().sum(axis=1)
    data["valid_composite"] = data["n_obs"] >= min_obs
    return data


def three_pass_imputation(
    data: pd.DataFrame,
    score_cols: list,
    half_width_cols: list,
    group_col: str = None,
    time_col: str = None,
) -> tuple:
    """
    Fill missing sub-index scores using a three-pass imputation hierarchy.

    Applied column by column (sub-index by sub-index):

    - **Pass 1** *(requires group_col)*: Within-group linear interpolation
      with ``limit_direction='both'`` (equivalent to ``zoo::na.approx``
      with ``rule = 2``).  Interior gaps are linearly interpolated; end
      gaps are filled by carry-forward / carry-back.  Skipped when
      *group_col* is ``None``.
    - **Pass 2** *(requires time_col)*: Time-period-mean fallback for
      groups whose entire series is absent.  Skipped when *time_col* is
      ``None``.
    - **Pass 3**: Global-mean fallback for any positions still missing.

    CI half-widths follow the same column structure.  Positions that are
    imputed receive a half-width of 0 (no measurement-error contribution).

    Parameters
    ----------
    data : pd.DataFrame
        Analysis-set observations (rows passing Option Filter).
    score_cols : list of str
        Score column names (length K).
    half_width_cols : list of str
        CI half-width column names (same order and length as *score_cols*).
    group_col : str or None, optional
        Column that identifies the grouping unit (e.g. subject, entity) for
        within-group interpolation (Pass 1).  ``None`` skips Pass 1.
    time_col : str or None, optional
        Column that identifies the time period for the time-mean fallback
        (Pass 2).  ``None`` skips Pass 2.

    Returns
    -------
    S_hat : np.ndarray, shape (N, K)
        Fully imputed score matrix.
    H : np.ndarray, shape (N, K)
        CI half-width matrix (0 for imputed positions).
    """
    sort_by = [c for c in [group_col, time_col] if c is not None]
    if sort_by:
        data = data.sort_values(sort_by).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    N = len(data)
    K = len(score_cols)

    S_hat = np.full((N, K), np.nan)
    H = np.zeros((N, K))

    for j, (sc, hwc) in enumerate(zip(score_cols, half_width_cols)):
        raw_scores = data[sc].values.astype(float)
        raw_hw = data[hwc].values.astype(float)

        is_observed = ~np.isnan(raw_scores)

        # ---- Pass 1: within-group linear interpolation ----
        if group_col is not None:
            tmp = data[[group_col]].copy()
            tmp["_s"] = raw_scores
            s_pass1 = (
                tmp.groupby(group_col)["_s"]
                .transform(lambda x: x.interpolate(
                    method="linear", limit_direction="both"
                ))
                .values.astype(float)
            )
        else:
            s_pass1 = raw_scores.copy()

        # ---- Pass 2: time-period-mean fallback ----
        s_pass2 = s_pass1.copy()
        if time_col is not None:
            still_na = np.isnan(s_pass2)
            if still_na.any():
                periods = data[time_col].values
                for period in np.unique(periods[still_na]):
                    period_mask = periods == period
                    period_mean = np.nanmean(s_pass1[period_mask])
                    if not np.isnan(period_mean):
                        fill_mask = still_na & period_mask
                        s_pass2[fill_mask] = period_mean

        # ---- Pass 3: global-mean fallback ----
        s_pass3 = s_pass2.copy()
        still_na = np.isnan(s_pass3)
        if still_na.any():
            global_mean = np.nanmean(raw_scores)
            if not np.isnan(global_mean):
                s_pass3[still_na] = global_mean

        S_hat[:, j] = s_pass3

        # Half-widths: 0 for imputed positions
        hw_filled = np.where(is_observed, raw_hw, 0.0)
        hw_filled = np.nan_to_num(hw_filled, nan=0.0)
        H[:, j] = hw_filled

    return S_hat, H
