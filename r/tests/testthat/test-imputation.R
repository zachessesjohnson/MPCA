test_that("option_b_filter adds n_obs and valid_composite columns", {
  df <- data.frame(
    s1 = c(50, NA, 60),
    s2 = c(55, 40, NA),
    s3 = c(NA, NA, 70),
    s4 = c(60, 55, 80),
    s5 = c(45, NA, 65)
  )
  out <- option_b_filter(df, paste0("s", 1:5), min_obs = 3L)
  expect_true("n_obs" %in% names(out))
  expect_true("valid_composite" %in% names(out))
  expect_equal(out$n_obs, c(4, 2, 4))
  expect_equal(out$valid_composite, c(TRUE, FALSE, TRUE))
})

test_that("option_b_filter with min_obs = 5 and 8 cols flags low-coverage rows", {
  set.seed(1)
  df <- as.data.frame(matrix(c(
    rep(50, 8),           # row 1: all 8 observed
    c(rep(NA, 4), rep(50, 4)),  # row 2: only 4 observed -> invalid
    c(50, NA, 50, NA, 50, NA, 50, 50)  # row 3: 5 observed -> valid
  ), nrow = 3, byrow = TRUE))
  names(df) <- paste0("s", 1:8)

  out <- option_b_filter(df, paste0("s", 1:8), min_obs = 5L)
  expect_equal(out$valid_composite, c(TRUE, FALSE, TRUE))
})

test_that("three_pass_imputation fills all NAs", {
  df <- data.frame(
    country = c("A", "A", "A", "B", "B"),
    year    = c(2000, 2001, 2002, 2000, 2001),
    s1      = c(50, NA, 60, NA, 70),
    s1_hw   = c(5, NA, 5, NA, 5)
  )
  out <- three_pass_imputation(df, "s1", "s1_hw")
  expect_false(any(is.na(out$S_hat)))
})

test_that("three_pass_imputation sets H to 0 for imputed positions", {
  df <- data.frame(
    country = c("A", "A", "A"),
    year    = c(2000, 2001, 2002),
    s1      = c(50, NA, 60),
    s1_hw   = c(5, NA, 5)
  )
  out <- three_pass_imputation(df, "s1", "s1_hw")
  # Middle position was imputed, so H should be 0 there
  expect_equal(out$H[2, 1], 0)
  # Observed positions keep their half-width
  expect_equal(out$H[1, 1], 5)
  expect_equal(out$H[3, 1], 5)
})

test_that("pass 1 interpolates interior gaps linearly", {
  df <- data.frame(
    country = rep("A", 3),
    year    = c(2000, 2001, 2002),
    s1      = c(50, NA, 60),
    s1_hw   = c(0, NA, 0)
  )
  out <- three_pass_imputation(df, "s1", "s1_hw")
  expect_equal(out$S_hat[2, 1], 55)
})

test_that("pass 2 uses year mean for completely missing country-series", {
  df <- data.frame(
    country = c("A", "A", "B"),
    year    = c(2000, 2001, 2000),
    s1      = c(NA, NA, 80),
    s1_hw   = c(NA, NA, 0)
  )
  out <- three_pass_imputation(df, "s1", "s1_hw")
  # Country A has no observed values; year 2000 mean = 80, year 2001 = NA -> global mean = 80
  expect_equal(out$S_hat[which(df$country == "A" & df$year == 2000), 1], 80)
})

test_that("pass 3 uses global mean when entire year missing", {
  df <- data.frame(
    country = c("A", "B"),
    year    = c(2000, 2001),
    s1      = c(NA, 80),
    s1_hw   = c(NA, 0)
  )
  out <- three_pass_imputation(df, "s1", "s1_hw")
  # Year 2000 has no observations at all -> global mean = 80
  expect_equal(out$S_hat[1, 1], 80)
})
