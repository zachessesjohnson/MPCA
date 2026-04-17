make_pipeline_df <- function(N = 60, K = 8, seed = 7) {
  set.seed(seed)
  groups  <- rep(paste0("G", 1:20), each = 3)
  periods <- rep(c(2022, 2023, 2024), times = 20)

  df <- data.frame(grp = groups, time = periods,
                   stringsAsFactors = FALSE)

  for (k in 1:K) {
    scores <- runif(N, 20, 80)
    hw     <- runif(N, 1, 10)
    # Introduce a few NAs
    scores[sample(N, 5)] <- NA
    df[[paste0("s", k)]]        <- scores
    df[[paste0("s", k, "_lo")]] <- scores - hw
    df[[paste0("s", k, "_hi")]] <- scores + hw
  }
  df
}

test_that("mpca_pipeline returns all three output tables", {
  df         <- make_pipeline_df()
  score_cols <- paste0("s", 1:8)
  lower_cols <- paste0("s", 1:8, "_lo")
  upper_cols <- paste0("s", 1:8, "_hi")

  result <- mpca_pipeline(df, score_cols, lower_cols, upper_cols,
                           id_cols   = c("grp", "time"),
                           group_col = "grp",
                           time_col  = "time",
                           B = 20L, seed = 1L)

  expect_true(is.data.frame(result$scores_df))
  expect_true(is.data.frame(result$contributions_df))
  expect_true(is.data.frame(result$rankings_df))
})

test_that("scores_df has expected columns", {
  df     <- make_pipeline_df()
  result <- mpca_pipeline(df, paste0("s", 1:8),
                           paste0("s", 1:8, "_lo"),
                           paste0("s", 1:8, "_hi"),
                           id_cols   = c("grp", "time"),
                           group_col = "grp",
                           time_col  = "time",
                           B = 20L, seed = 1L)

  expect_true(all(c("grp", "time", "score",
                     "ci_lower", "ci_upper") %in% names(result$scores_df)))
})

test_that("scores are in [0, 100]", {
  df     <- make_pipeline_df()
  result <- mpca_pipeline(df, paste0("s", 1:8),
                           paste0("s", 1:8, "_lo"),
                           paste0("s", 1:8, "_hi"),
                           id_cols   = c("grp", "time"),
                           group_col = "grp",
                           time_col  = "time",
                           B = 20L, seed = 1L)

  s <- result$scores_df
  expect_true(all(s$score    >= -1e-10 & s$score    <= 100 + 1e-10))
  expect_true(all(s$ci_lower >= -1e-10 & s$ci_lower <= 100 + 1e-10))
  expect_true(all(s$ci_upper >= -1e-10 & s$ci_upper <= 100 + 1e-10))
})

test_that("ci_lower <= score <= ci_upper for all observations", {
  df     <- make_pipeline_df()
  result <- mpca_pipeline(df, paste0("s", 1:8),
                           paste0("s", 1:8, "_lo"),
                           paste0("s", 1:8, "_hi"),
                           id_cols   = c("grp", "time"),
                           group_col = "grp",
                           time_col  = "time",
                           B = 20L, seed = 1L)

  s <- result$scores_df
  expect_true(all(s$ci_lower <= s$score   + 1e-6))
  expect_true(all(s$score    <= s$ci_upper + 1e-6))
})

test_that("rankings_df is sorted descending and contains rank column", {
  df     <- make_pipeline_df()
  result <- mpca_pipeline(df, paste0("s", 1:8),
                           paste0("s", 1:8, "_lo"),
                           paste0("s", 1:8, "_hi"),
                           id_cols        = c("grp", "time"),
                           group_col      = "grp",
                           time_col       = "time",
                           B              = 20L,
                           seed           = 1L,
                           rankings_value = 2024L)

  rd <- result$rankings_df
  if (nrow(rd) > 1) {
    expect_true(all(diff(rd$score) <= 1e-10))
  }
  expect_true("rank" %in% names(rd))
  if (nrow(rd) > 0) {
    expect_equal(rd$rank[1], 1L)
  }
})

test_that("rankings_df is empty when rankings_value is NULL", {
  df     <- make_pipeline_df()
  result <- mpca_pipeline(df, paste0("s", 1:8),
                           paste0("s", 1:8, "_lo"),
                           paste0("s", 1:8, "_hi"),
                           id_cols   = c("grp", "time"),
                           group_col = "grp",
                           time_col  = "time",
                           B = 20L, seed = 1L,
                           rankings_value = NULL)
  expect_equal(nrow(result$rankings_df), 0L)
})

test_that("contributions_df has all sub-index names and loading/weight columns", {
  df     <- make_pipeline_df()
  result <- mpca_pipeline(df, paste0("s", 1:8),
                           paste0("s", 1:8, "_lo"),
                           paste0("s", 1:8, "_hi"),
                           id_cols   = c("grp", "time"),
                           group_col = "grp",
                           time_col  = "time",
                           B = 20L, seed = 1L)

  cd <- result$contributions_df
  expect_equal(nrow(cd), 8L)
  expect_true(all(c("sub_index", "naive_loading", "corrected_loading",
                     "naive_weight", "corrected_weight",
                     "reliability") %in% names(cd)))
})

test_that("postprocess_scores output is in [0, 100]", {
  set.seed(42)
  f     <- rnorm(100, 0, 2)
  ci_lo <- f - abs(rnorm(100, 0, 0.5))
  ci_up <- f + abs(rnorm(100, 0, 0.5))
  pp    <- postprocess_scores(f, ci_lo, ci_up)

  expect_true(all(pp$score    >= -1e-10 & pp$score    <= 100 + 1e-10))
  expect_true(all(pp$ci_lower >= -1e-10 & pp$ci_lower <= 100 + 1e-10))
  expect_true(all(pp$ci_upper >= -1e-10 & pp$ci_upper <= 100 + 1e-10))
})

test_that("two_stage_bootstrap CI dominance: two-stage CI wider than zero-perturbation", {
  set.seed(99)
  N <- 30; K <- 4
  S_hat     <- matrix(runif(N * K, 20, 80), N, K)
  H         <- matrix(runif(N * K, 3, 12), N, K)
  col_means <- colMeans(S_hat)
  sigma_sig <- apply(S_hat, 2, sd)
  w         <- rep(1 / K, K)

  # Two-stage
  ts <- two_stage_bootstrap(S_hat, H, col_means, w, sigma_sig, B = 100L, seed = 5L)
  # Single-stage (zero perturbation)
  ss <- two_stage_bootstrap(S_hat, H * 0, col_means, w, sigma_sig, B = 100L, seed = 5L)

  ts_width <- mean(ts$ci_upper - ts$ci_lower)
  ss_width <- mean(ss$ci_upper - ss$ci_lower)
  expect_gte(ts_width, ss_width)
})
