#' MPCA Full Pipeline
#'
#' Runs the complete Multilevel PCA pipeline on a data.frame of sub-index
#' scores and confidence intervals, returning composite scores, variable
#' contributions, and (optionally) a ranked snapshot for a chosen time period.
#'
#' @details
#' The pipeline executes the following steps in order:
#' \enumerate{
#'   \item **Option Filter** – exclude rows with fewer than `min_obs`
#'     observed sub-indices.
#'   \item Compute CI **half-widths** \eqn{h_{ij} = (U_{ij} - L_{ij})/2}.
#'   \item **Three-pass imputation** – within-group linear interpolation
#'     (`group_col`), time-period-mean fallback (`time_col`), global-mean
#'     fallback.
#'   \item **Naive PCA** – PC1 loadings and regression scoring coefficients
#'     on the observed correlation matrix.
#'   \item **Attenuation correction** – disattenuation of correlations,
#'     corrected PCA on the signal correlation matrix.
#'   \item **Two-stage bootstrap** – propagate measurement error and
#'     sampling uncertainty; output 95% CI.
#'   \item **Post-processing** – z-score, Box-Cox, min-max rescaling.
#' }
#'
#' @param data A data.frame with sub-index scores, lower/upper CI bounds, and
#'   any identifier columns.
#' @param score_cols Character vector of score column names (length K).
#' @param lower_cols Character vector of lower CI column names (same order).
#' @param upper_cols Character vector of upper CI column names (same order).
#' @param id_cols Character vector of identifier columns to carry through to
#'   `scores_df` (e.g. `c("country", "iso", "year")` or `c("unit", "time")`).
#'   `NULL` produces a `scores_df` with only score columns (default `NULL`).
#' @param group_col Name of the grouping-unit column for within-group
#'   interpolation (Pass 1 of imputation).  `NULL` skips Pass 1
#'   (default `NULL`).
#' @param time_col Name of the time-period column for the time-mean fallback
#'   (Pass 2 of imputation) and for filtering `rankings_df`.  `NULL` skips
#'   Pass 2 and rankings (default `NULL`).
#' @param B Integer. Bootstrap replications (default 500).
#' @param seed Integer. Random seed (default 42).
#' @param min_obs Integer. Minimum non-NA sub-indices for a valid composite
#'   (default 5).
#' @param rankings_value Scalar or `NULL`. Value of `time_col` for which to
#'   produce a ranked output table; `NULL` skips rankings (default `NULL`).
#'
#' @return A named list with three data.frames:
#'   \itemize{
#'     \item `scores_df`: identifier columns (`id_cols`) plus score,
#'       ci_lower, ci_upper for all analysis-set observations.
#'     \item `contributions_df`: sub-index names with naive and corrected
#'       loadings and scoring weights.
#'     \item `rankings_df`: observations for `rankings_value` sorted
#'       descending by score, with a `rank` column.  Empty when
#'       `rankings_value` is `NULL` or `time_col` is not in `scores_df`.
#'   }
#' @export
mpca_pipeline <- function(data, score_cols, lower_cols, upper_cols,
                           id_cols        = NULL,
                           group_col      = NULL,
                           time_col       = NULL,
                           B              = 500L,
                           seed           = 42L,
                           min_obs        = 5L,
                           rankings_value = NULL) {

  K <- length(score_cols)
  stopifnot(
    length(lower_cols) == K,
    length(upper_cols) == K
  )

  if (is.null(id_cols)) id_cols <- character(0)

  # ---- Step 1: Option Filter ----
  data       <- option_b_filter(data, score_cols, min_obs)
  data_valid <- data[data$valid_composite, , drop = FALSE]

  # ---- Step 2: CI half-widths ----
  hw_cols <- paste0(score_cols, "_hw")
  for (k in seq_len(K)) {
    lo <- data_valid[[lower_cols[k]]]
    up <- data_valid[[upper_cols[k]]]
    data_valid[[hw_cols[k]]] <- (up - lo) / 2
  }

  # ---- Step 3: Three-pass imputation ----
  imp   <- three_pass_imputation(data_valid, score_cols, hw_cols,
                                  group_col = group_col,
                                  time_col  = time_col)
  S_hat <- imp$S_hat
  H     <- imp$H

  # Reorder data_valid to match imputation sort
  sort_cols <- c(group_col, time_col)
  if (length(sort_cols) > 0) {
    ord        <- do.call(order, lapply(sort_cols, function(col) data_valid[[col]]))
    data_valid <- data_valid[ord, , drop = FALSE]
  }

  # ---- Step 4: Naive PCA ----
  naive <- naive_pca(S_hat)

  # ---- Step 5: Attenuation correction ----
  corr <- attenuation_correction(S_hat, H, naive)

  # ---- Step 6: Two-stage bootstrap ----
  boot <- two_stage_bootstrap(
    S_hat      = S_hat,
    H          = H,
    col_means  = naive$col_means,
    w_hat_star = corr$w_hat_star,
    sigma_sig  = corr$sigma_sig,
    B          = B,
    seed       = seed
  )

  # ---- Step 7: Post-processing ----
  pp <- postprocess_scores(corr$f_hat_star, boot$ci_lower, boot$ci_upper)

  # ---- Assemble scores_df ----
  scores_list <- lapply(id_cols, function(col) data_valid[[col]])
  names(scores_list) <- id_cols
  scores_list$score    <- pp$score
  scores_list$ci_lower <- pp$ci_lower
  scores_list$ci_upper <- pp$ci_upper
  scores_df <- as.data.frame(scores_list, stringsAsFactors = FALSE)

  # ---- Contributions ----
  contributions_df <- data.frame(
    sub_index         = score_cols,
    naive_loading     = naive$ell_hat,
    naive_weight      = naive$w_hat,
    corrected_loading = corr$ell_hat_star,
    corrected_weight  = corr$w_hat_star,
    reliability       = corr$reliability,
    stringsAsFactors  = FALSE
  )

  # ---- Rankings ----
  if (!is.null(rankings_value) && !is.null(time_col) &&
      time_col %in% names(scores_df)) {
    df_yr <- scores_df[scores_df[[time_col]] == rankings_value, , drop = FALSE]
    if (nrow(df_yr) > 0) {
      df_yr      <- df_yr[order(-df_yr$score), , drop = FALSE]
      df_yr$rank <- seq_len(nrow(df_yr))
    }
    rankings_df <- df_yr
  } else {
    rankings_df <- data.frame()
  }

  list(
    scores_df        = scores_df,
    contributions_df = contributions_df,
    rankings_df      = rankings_df
  )
}
