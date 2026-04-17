#' MPCA Full Pipeline
#'
#' Runs the complete Multilevel PCA pipeline on a merged data.frame of
#' sub-index scores and confidence intervals, returning composite scores,
#' variable contributions, and (optionally) year-specific rankings.
#'
#' @details
#' The pipeline executes the following steps in order:
#' \enumerate{
#'   \item **Option Filter** – exclude rows with fewer than `min_obs`
#'     observed sub-indices.
#'   \item Compute CI **half-widths** \eqn{h_{ij} = (U_{ij} - L_{ij})/2}.
#'   \item **Three-pass imputation** – within-country linear interpolation,
#'     year-mean fallback, global-mean fallback.
#'   \item **Naive PCA** – PC1 loadings and regression scoring coefficients
#'     on the observed correlation matrix.
#'   \item **Attenuation correction** – disattenuation of correlations,
#'     corrected PCA on the signal correlation matrix.
#'   \item **Two-stage bootstrap** – propagate measurement error and
#'     sampling uncertainty; output 95% CI.
#'   \item **Post-processing** – z-score, Box-Cox, min-max rescaling.
#' }
#'
#' @param data A data.frame with sub-index scores, lower/upper CI bounds,
#'   a country identifier, an ISO code, and a year column.
#' @param score_cols Character vector of score column names (length K).
#' @param lower_cols Character vector of lower CI column names (same order).
#' @param upper_cols Character vector of upper CI column names (same order).
#' @param country_col Name of the country column (default `"country"`).
#' @param iso_col Name of the ISO code column (default `"iso"`).
#' @param year_col Name of the year column (default `"year"`).
#' @param B Integer. Bootstrap replications (default 500).
#' @param seed Integer. Random seed (default 42).
#' @param min_obs Integer. Minimum non-NA sub-indices for a valid composite
#'   (default 5).
#' @param rankings_year Integer or `NULL`. Year for which to produce a
#'   ranked output table; `NULL` skips rankings (default 2024).
#'
#' @return A named list with three data.frames:
#'   \itemize{
#'     \item `scores_df`: country, iso, year, score, ci_lower, ci_upper
#'       for all analysis-set observations.
#'     \item `contributions_df`: sub-index names with naive and corrected
#'       loadings and scoring weights.
#'     \item `rankings_df`: observations for `rankings_year` sorted
#'       descending by score, with a `rank` column.
#'   }
#' @export
mpca_pipeline <- function(data, score_cols, lower_cols, upper_cols,
                           country_col   = "country",
                           iso_col       = "iso",
                           year_col      = "year",
                           B             = 500L,
                           seed          = 42L,
                           min_obs       = 5L,
                           rankings_year = 2024L) {

  K <- length(score_cols)
  stopifnot(
    length(lower_cols) == K,
    length(upper_cols) == K
  )

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
                                  country_col = country_col,
                                  year_col    = year_col)
  S_hat <- imp$S_hat
  H     <- imp$H

  # Reorder data_valid to match imputation sort (country then year)
  ord        <- order(data_valid[[country_col]], data_valid[[year_col]])
  data_valid <- data_valid[ord, , drop = FALSE]

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
  scores_df <- data.frame(
    country  = data_valid[[country_col]],
    iso      = data_valid[[iso_col]],
    year     = data_valid[[year_col]],
    score    = pp$score,
    ci_lower = pp$ci_lower,
    ci_upper = pp$ci_upper,
    stringsAsFactors = FALSE
  )

  # ---- Contributions ----
  contributions_df <- data.frame(
    sub_index       = score_cols,
    naive_loading   = naive$ell_hat,
    naive_weight    = naive$w_hat,
    corrected_loading = corr$ell_hat_star,
    corrected_weight  = corr$w_hat_star,
    reliability     = corr$reliability,
    stringsAsFactors = FALSE
  )

  # ---- Rankings ----
  if (!is.null(rankings_year)) {
    df_yr <- scores_df[scores_df$year == rankings_year, , drop = FALSE]
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
