#' Option B Minimum Sub-Index Coverage Filter
#'
#' Country-year observations with fewer than `min_obs` of the K sub-indices
#' observed (pre-imputation) are flagged as invalid for the composite score.
#' A new logical column `valid_composite` is added to `data`.
#'
#' @param data A data.frame containing sub-index score columns.
#' @param score_cols Character vector of column names for sub-index scores
#'   (length K).
#' @param min_obs Integer. Minimum number of non-NA sub-indices required for a
#'   valid composite (default 5).
#'
#' @return `data` with an additional logical column `valid_composite` and an
#'   integer column `n_obs` (number of non-NA sub-index scores per row).
#' @export
option_b_filter <- function(data, score_cols, min_obs = 5L) {
  score_mat <- as.matrix(data[, score_cols, drop = FALSE])
  n_obs <- rowSums(!is.na(score_mat))
  data$n_obs <- n_obs
  data$valid_composite <- n_obs >= min_obs
  data
}


#' Linear Interpolation with End-Fill (Equivalent to zoo::na.approx rule=2)
#'
#' @param x Numeric vector possibly containing NAs.
#' @return Numeric vector with interior NAs linearly interpolated and end NAs
#'   filled by nearest observed value (LOCF/NOCB).
#' @keywords internal
.na_approx_rule2 <- function(x) {
  n <- length(x)
  if (n == 0L || all(is.na(x))) return(x)

  obs_idx <- which(!is.na(x))

  # If all observed, return as-is
  if (length(obs_idx) == n) return(x)

  out <- x

  # Linear interpolation between observed points
  for (k in seq_along(obs_idx[-1])) {
    i1 <- obs_idx[k]
    i2 <- obs_idx[k + 1L]
    if (i2 - i1 > 1L) {
      # Fill gap (i1+1):(i2-1)
      gap <- seq_len(i2 - i1 - 1L) + i1
      out[gap] <- x[i1] + (x[i2] - x[i1]) * (gap - i1) / (i2 - i1)
    }
  }

  # NOCB: fill leading NAs with first observed value
  if (obs_idx[1L] > 1L) {
    out[seq_len(obs_idx[1L] - 1L)] <- x[obs_idx[1L]]
  }

  # LOCF: fill trailing NAs with last observed value
  last_obs <- obs_idx[length(obs_idx)]
  if (last_obs < n) {
    out[(last_obs + 1L):n] <- x[last_obs]
  }

  out
}


#' Three-Pass Imputation of Sub-Index Scores
#'
#' Fills missing sub-index scores using a three-pass hierarchy applied
#' column by column:
#'
#' - **Pass 1**: Within-country linear interpolation (interior gaps
#'   interpolated linearly; end gaps filled by nearest-neighbor
#'   carry-forward/carry-back), equivalent to `zoo::na.approx` with
#'   `rule = 2`.
#' - **Pass 2**: Year-mean fallback for countries whose entire series is
#'   absent.
#' - **Pass 3**: Global-mean fallback for years with no observed values.
#'
#' CI half-widths are also assembled; imputed positions receive a
#' half-width of 0 (no measurement-error contribution for imputed cells).
#'
#' @param data A data.frame for the analysis set (rows passing Option B).
#'   Must be sorted by `country_col` then `year_col` prior to calling, or
#'   sorting is applied internally.
#' @param score_cols Character vector of score column names (length K).
#' @param half_width_cols Character vector of half-width column names (same
#'   order and length as `score_cols`).
#' @param country_col Name of the country identifier column (default
#'   `"country"`).
#' @param year_col Name of the year column (default `"year"`).
#'
#' @return A named list with:
#'   \itemize{
#'     \item `S_hat`: N×K numeric matrix of imputed scores.
#'     \item `H`: N×K numeric matrix of CI half-widths (0 for imputed
#'       positions).
#'   }
#' @export
three_pass_imputation <- function(data, score_cols, half_width_cols,
                                   country_col = "country",
                                   year_col = "year") {
  # Sort by country then year
  ord  <- order(data[[country_col]], data[[year_col]])
  data <- data[ord, , drop = FALSE]

  N <- nrow(data)
  K <- length(score_cols)

  S_hat <- matrix(NA_real_, N, K, dimnames = list(NULL, score_cols))
  H     <- matrix(0,        N, K, dimnames = list(NULL, score_cols))

  countries <- data[[country_col]]
  years     <- data[[year_col]]

  for (j in seq_len(K)) {
    raw_scores <- data[[score_cols[j]]]
    raw_hw     <- data[[half_width_cols[j]]]

    is_observed <- !is.na(raw_scores)

    # ---- Pass 1: within-country linear interpolation ----
    s_pass1 <- raw_scores
    for (ctry in unique(countries)) {
      idx <- which(countries == ctry)
      if (length(idx) < 1L) next
      s_pass1[idx] <- .na_approx_rule2(s_pass1[idx])
    }

    # ---- Pass 2: year-mean fallback ----
    s_pass2  <- s_pass1
    still_na <- is.na(s_pass2)
    if (any(still_na)) {
      yr_levels  <- unique(years)
      yr_means   <- vapply(yr_levels, function(yr) {
        mean(s_pass1[years == yr], na.rm = TRUE)
      }, numeric(1L))
      names(yr_means) <- as.character(yr_levels)

      for (i in which(still_na)) {
        yr_chr <- as.character(years[i])
        ym     <- yr_means[[yr_chr]]
        if (!is.na(ym)) s_pass2[i] <- ym
      }
    }

    # ---- Pass 3: global-mean fallback ----
    s_pass3  <- s_pass2
    still_na <- is.na(s_pass3)
    if (any(still_na)) {
      global_mean <- mean(raw_scores, na.rm = TRUE)
      if (!is.na(global_mean)) s_pass3[still_na] <- global_mean
    }

    S_hat[, j] <- s_pass3

    # Half-widths: 0 for imputed positions
    hw_filled           <- ifelse(is_observed, raw_hw, 0)
    hw_filled[is.na(hw_filled)] <- 0
    H[, j]              <- hw_filled
  }

  list(S_hat = S_hat, H = H)
}

