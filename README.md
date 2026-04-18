# MPCA — Multilevel PCA with Measurement Error Correction

> Measurement-error-corrected composite indices with uncertainty propagation.
> Available as both an **R package** and a **Python package**.
> Works with any panel or cross-sectional data.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation and Statistical Background](#motivation-and-statistical-background)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Input Data Requirements](#input-data-requirements)
6. [Pipeline Walkthrough](#pipeline-walkthrough)
   - [Step 1 — Option Filter](#step-1--option-filter)
   - [Step 2 — CI Half-Widths](#step-2--ci-half-widths)
   - [Step 3 — Three-Pass Imputation](#step-3--three-pass-imputation)
   - [Step 4 — Naive PCA](#step-4--naive-pca)
   - [Step 5 — Attenuation Correction](#step-5--attenuation-correction)
   - [Step 6 — Two-Stage Bootstrap](#step-6--two-stage-bootstrap)
   - [Step 7 — Post-Processing](#step-7--post-processing)
7. [API Reference](#api-reference)
   - [mpca_pipeline()](#mpca_pipeline)
   - [option_b_filter()](#option_b_filter)
   - [three_pass_imputation()](#three_pass_imputation)
   - [naive_pca()](#naive_pca)
   - [attenuation_correction()](#attenuation_correction)
   - [two_stage_bootstrap()](#two_stage_bootstrap)
   - [postprocess_scores()](#postprocess_scores)
8. [Output Format](#output-format)
9. [Quick-Start Examples](#quick-start-examples)
10. [Running the Tests](#running-the-tests)
11. [References](#references)

---

## Overview

MPCA builds a composite index from K sub-indices that each come with
bootstrapped 95% confidence intervals (CIs).  It solves two problems that
arise in this setting:

| Problem | MPCA solution |
|---|---|
| Sub-indices are noisy → naive PCA underestimates the common signal | Spearman disattenuation: subtract estimated error variance from the covariance diagonal |
| Final composite lacks uncertainty quantification | Two-stage bootstrap: propagate both measurement error and sampling variability into composite CIs |

The output is a panel of composite scores on a 0–100 scale,
each accompanied by a 95% CI, plus a breakdown of each sub-index's loading
and weight in the naive and corrected solutions.

---

## Motivation and Statistical Background

### The attenuation-bias problem

Suppose you observe K sub-index scores for N observations.  Each
sub-index j is a *noisy* estimate of an unobserved true score:

```
X_ij = T_ij + e_ij,    e_ij ~ N(0, ψ_j)
```

where ψ_j is the per-sub-index measurement error variance.  Because the
observed matrix **X** mixes signal and noise, its sample correlation matrix
**R_obs** satisfies

```
R_obs ≈ R_sig + Ψ / (σ_j σ_k)
```

Running PCA on **R_obs** therefore yields loadings and eigenvalues that are
*attenuated* — the leading eigenvalue is inflated by noise and the
eigenvector is rotated away from the true signal direction (Spearman, 1904).

### MPCA's correction

MPCA estimates ψ_j from the bootstrapped CI half-widths supplied alongside
each sub-index score:

```
ψ̂_j = (1/N) Σ_i (h_ij / 1.96)²
```

where h_ij = (upper_ij − lower_ij) / 2 is the 95% CI half-width for unit i
on sub-index j.  The signal covariance matrix is recovered by subtracting
the error diagonal:

```
Σ̂_sig = Σ̂_obs − diag(ψ̂_1, …, ψ̂_K)
```

PCA on the resulting *corrected* correlation matrix **R̂_sig** produces
loadings and weights that reflect the latent common factor rather than
measurement noise.

### Reliability

For each sub-index the reliability coefficient is

```
r_j = 1 − ψ̂_j / Σ̂_obs,jj
```

A value close to 1 means almost all observed variance is true signal;
a value close to 0 means the sub-index is dominated by noise.

### Two-stage bootstrap uncertainty

Two independent sources of uncertainty enter the composite score:

1. **Stage 1 — measurement error**: each observed score X_ij has an
   associated SE = h_ij / 1.96.
2. **Stage 2 — sampling variability**: the country-year panel is treated as
   a random sample from some population.

These are propagated jointly by:

1. Drawing a noise matrix **E** ~ N(0, (H/1.96)²) element-wise and adding
   it to **Ŝ** (perturb).
2. Resampling rows with replacement (resample).
3. Scoring the perturbed-resampled matrix with the corrected weights **ŵ***.
4. Repeating B times and taking the 2.5th and 97.5th empirical quantiles as
   the 95% CI.

---

## Repository Structure

```
MPCA/
├── README.md                        ← this file
│
├── r/                               R package (mpca)
│   ├── DESCRIPTION
│   ├── NAMESPACE
│   ├── NEWS.md
│   ├── R/
│   │   ├── imputation.R             option_b_filter(), three_pass_imputation()
│   │   ├── naive_pca.R              naive_pca()
│   │   ├── correction.R             attenuation_correction()
│   │   ├── bootstrap.R              two_stage_bootstrap()
│   │   ├── postprocess.R            postprocess_scores()
│   │   └── pipeline.R               mpca_pipeline()
│   └── tests/
│       └── testthat/
│           ├── test-imputation.R
│           ├── test-naive-pca.R
│           ├── test-correction.R
│           ├── test-bootstrap.R
│           ├── test-postprocess.R
│           └── test-pipeline.R
│
└── python/                          Python package (mpca)
    ├── pyproject.toml
    ├── README.md
    ├── CHANGELOG.md
    ├── mpca/
    │   ├── __init__.py
    │   ├── imputation.py            option_b_filter(), three_pass_imputation()
    │   ├── naive_pca.py             naive_pca()
    │   ├── correction.py            attenuation_correction()
    │   ├── bootstrap.py             two_stage_bootstrap()
    │   ├── postprocess.py           postprocess_scores()
    │   └── pipeline.py              mpca_pipeline()
    └── tests/
        ├── test_imputation.py
        ├── test_naive_pca.py
        ├── test_correction.py
        ├── test_bootstrap.py
        ├── test_postprocess.py
        └── test_pipeline.py
```

---

## Installation

### R

Requires R ≥ 4.1 and the `Matrix` package (available on CRAN).

```r
# Install dependencies first (if not already present)
install.packages(c("Matrix", "devtools"))

# Install mpca from the local r/ directory
devtools::install("r/")
```

To verify:

```r
library(mpca)
packageVersion("mpca")  # should print "0.1.0"
```

### Python

Requires Python ≥ 3.9 and the packages `numpy`, `pandas`, and `scipy`
(installed automatically).

```bash
# Editable install (recommended for development)
pip install -e python/

# Or a regular install
pip install python/
```

To verify:

```python
import mpca
print(mpca.__version__)   # "0.1.0"
```

---

## Input Data Requirements

Both implementations expect a flat panel or cross-sectional data structure — one row per observation — with the following columns.

| Column group | Convention | Description |
|---|---|---|
| Sub-index scores | e.g. `s1`, `s2`, …, `sK` | Point estimate for sub-index j for observation i.  Should be on a common scale (e.g. 0–100).  May be `NA`/`NaN` for missing observations. |
| Lower CI bounds | e.g. `s1_lower`, …, `sK_lower` | 95% CI lower bound corresponding to each score. |
| Upper CI bounds | e.g. `s1_upper`, …, `sK_upper` | 95% CI upper bound corresponding to each score. |
| Identifier columns | any names (configurable via `id_cols`) | Any columns you want carried through to the output (e.g. unit ID, time period, group label). |
| Group column | e.g. `group` (configurable via `group_col`) | Optional. Identifies the grouping unit for within-group interpolation (Pass 1 of imputation). |
| Time column | e.g. `time` (configurable via `time_col`) | Optional. Identifies the time period for the time-mean fallback (Pass 2 of imputation) and for filtering the rankings table. |

**Notes**

- The number of sub-indices K must be the same for all three column groups
  (scores, lower, upper) and must be supplied in matching order.
- Rows where *all* sub-indices are missing are automatically excluded by the
  Option Filter (Step 1).
- CI bounds must be on the same scale as the scores.  A row may have a score
  but missing CI bounds; in that case the half-width is treated as 0 (no
  measurement-error contribution for that cell), matching the convention used
  for imputed positions.
- `group_col` and `time_col` are both optional.  Without `group_col`, Pass 1
  is skipped; without `time_col`, Pass 2 is skipped.  At least Pass 3 (global
  mean) always applies.

---

## Pipeline Walkthrough

The complete pipeline runs 7 steps in sequence.  All steps are exposed as
individual functions so you can run or inspect each stage independently.

```
Raw panel data
      │
      ▼
┌─────────────────────────────────┐
│ Step 1 — Option Filter          │  option_b_filter()
│   Drop rows with < min_obs      │
│   observed sub-indices          │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Step 2 — CI half-widths         │  (internal, inside mpca_pipeline)
│   h_ij = (upper - lower) / 2   │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Step 3 — Three-pass imputation  │  three_pass_imputation()
│   Pass 1: within-group interp   │
│   Pass 2: time-mean fallback    │
│   Pass 3: global-mean fallback  │
└─────────────────────────────────┘
      │  produces: Ŝ (N×K), H (N×K)
      ▼
┌─────────────────────────────────┐
│ Step 4 — Naive PCA              │  naive_pca()
│   PC1 of R_obs; loadings ℓ̂    │
│   regression weights ŵ          │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Step 5 — Attenuation correction │  attenuation_correction()
│   ψ̂_j from CI half-widths      │
│   Σ̂_sig = Σ̂_obs − diag(ψ̂)   │
│   PC1 of R̂_sig; corrected ℓ̂* │
│   corrected weights ŵ*          │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Step 6 — Two-stage bootstrap    │  two_stage_bootstrap()
│   Perturb → resample → score    │
│   B replicates → 95% CI         │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│ Step 7 — Post-processing        │  postprocess_scores()
│   z-score → Box-Cox → min-max   │
│   → [0, 100]                    │
└─────────────────────────────────┘
      │
      ▼
  scores_df, contributions_df, rankings_df
```

---

### Step 1 — Option Filter

**Function**: `option_b_filter(data, score_cols, min_obs = 5)`

Observations where fewer than `min_obs` of the K sub-indices are non-missing
are excluded from the PCA estimation.  The function adds two columns to the
input frame:

- `n_obs` — integer count of non-missing sub-index scores per row.
- `valid_composite` — boolean flag; `TRUE` when `n_obs >= min_obs`.

Only rows where `valid_composite` is `TRUE` pass through to subsequent
steps.  The default threshold is 5 observed sub-indices, which ensures that
at least 5 of the K pillars contribute to the composite.

---

### Step 2 — CI Half-Widths

Performed inside `mpca_pipeline` before imputation.  For each sub-index j
and each row i:

```
h_ij = (upper_ij − lower_ij) / 2
```

This converts the 95% CI range into a half-width, which is the key quantity
needed to estimate measurement error variance.  If a row is missing its CI
bounds, the half-width is set to 0 (no error contribution for that cell).

---

### Step 3 — Three-Pass Imputation

**Function**: `three_pass_imputation(data, score_cols, half_width_cols, group_col, time_col)`

Applied independently to each of the K sub-index columns in the order listed
below.  The goal is to fill every cell so that PCA can proceed on a complete
matrix, while preserving as much observed information as possible.

**Pass 1 — Within-group linear interpolation** *(requires group_col)*

For each group, the time series of observed scores is linearly interpolated
across interior gaps, and end gaps are filled by carry-forward / carry-back
(equivalent to `zoo::na.approx(rule = 2)` in R and
`pd.Series.interpolate(limit_direction='both')` in Python).
Skipped when `group_col` is `NULL` / `None`.

*Example*: group A has scores 50, NA, 60 in time periods 1–3 → Pass 1
fills period 2 with 55.

**Pass 2 — Time-period-mean fallback** *(requires time_col)*

Any position still missing after Pass 1 (e.g. a group whose entire series is
absent) is filled with the cross-group mean for that time period, computed
from Pass-1 values.  Skipped when `time_col` is `NULL` / `None`.

**Pass 3 — Global-mean fallback**

Any position still missing after Pass 2 (e.g. a time period where *no* group
has an observed value, or when both `group_col` and `time_col` are omitted)
is filled with the global mean of the raw (pre-imputation) scores.

**Half-width convention**: positions filled by any of the three passes
receive a half-width of 0 in the H matrix.  This means imputed cells
contribute no measurement-error variance to the attenuation correction — a
conservative choice that avoids fabricating uncertainty for values that are
already assumed.

### Step 4 — Naive PCA

**Function**: `naive_pca(S_hat)`

Runs standard PC1 aggregation on the N×K imputed score matrix **Ŝ**.

1. Standardise columns: **S̃** = (Ŝ − col_means) / col_sds.
2. Compute sample correlation matrix: **R_obs** = **S̃**ᵀ**S̃** / (N − 1).
3. Extract the leading eigenpair (λ₁, **ℓ̂**) via spectral decomposition.
4. Enforce sign convention: flip **ℓ̂** if its sum is negative (so all
   loadings are positive when sub-indices share a common factor direction).
5. Compute regression scoring coefficients: **ŵ** = **R_obs**⁻¹ **ℓ̂**.
6. Score observations: **f̂** = **S̃** **ŵ**.

The naive PC1 loadings and regression weights are used in two ways:

- They serve as a baseline for comparing with the corrected solution.
- The column means computed here are reused in the bootstrap (Step 6) to
  ensure comparable standardisation across replicates.

---

### Step 5 — Attenuation Correction

**Function**: `attenuation_correction(S_hat, H, naive_result)`

Removes measurement-error variance from the covariance structure and re-runs
PCA on the resulting signal correlation matrix.

**Error variance estimation**

```
ψ̂_j = (1/N) Σ_i (h_ij / 1.96)²
```

Dividing by 1.96 converts the half-width to a standard error under the
assumption that the 95% CI is symmetric and Gaussian.

**Signal covariance**

```
Σ̂_sig = Σ̂_obs − diag(ψ̂_1, …, ψ̂_K)
```

The diagonal elements are floored at 1% of the corresponding observed
variance to prevent numerical issues when a sub-index has very small
estimated error:

```
Σ̂_sig,jj ← max(Σ̂_sig,jj , 0.01 · Σ̂_obs,jj)
```

**Positive-semidefiniteness**

If the minimum eigenvalue of the resulting correlation matrix **R̂_sig** is
below 10⁻⁸, the matrix is projected to the nearest positive-semidefinite
correlation matrix using the Higham (2002) algorithm (`Matrix::nearPD` in R;
equivalent in Python).

**Corrected PCA**

PC1 of **R̂_sig** yields corrected loadings **ℓ̂*** and, via regression
scoring, corrected weights **ŵ*** = **R̂_sig**⁻¹ **ℓ̂***.

**Reliability**

```
r_j = 1 − ψ̂_j / Σ̂_obs,jj    ∈ [0, 1]
```

**Corrected composite scores**

Observations are standardised using the original column means and the signal
standard deviations σ̂_sig,j = √(Σ̂_sig,jj):

```
S̃_sig = (Ŝ − col_means) / σ̂_sig
f̂* = S̃_sig · ŵ*
```

---

### Step 6 — Two-Stage Bootstrap

**Function**: `two_stage_bootstrap(S_hat, H, col_means, w_hat_star, sigma_sig, B, seed)`

Generates B bootstrap replicates to form a 95% CI for each observation's
composite score.  Each replicate b = 1, …, B proceeds as follows:

1. **Stage 1 — Perturbation** (measurement-error uncertainty)

   Draw an N×K noise matrix **E**⁽ᵇ⁾ with entries drawn independently from
   N(0, (h_ij/1.96)²), then form a perturbed score matrix:

   ```
   Ŝ_pert = clamp(Ŝ + E⁽ᵇ⁾, 0, 100)
   ```

   Clamping ensures perturbed scores stay within the valid score range.

2. **Stage 2 — Row resampling** (sampling uncertainty)

   Draw N row indices with replacement from {1, …, N} to obtain a bootstrap
   sample **Ŝ_boot** from the perturbed matrix.

3. **Scoring**

   Standardise **Ŝ_boot** using the original column means and signal SDs,
   then project:

   ```
   f_boot⁽ᵇ⁾ = ((Ŝ_boot − col_means) / σ̂_sig) · ŵ*
   ```

4. **Row mapping**

   For each original observation i, find its *first* occurrence in the
   bootstrap row indices and record the corresponding score.  Observations
   not sampled in replicate b receive `NA` for that replicate.

After B replicates the 95% CI is:

```
CI_lower_i = quantile({f_boot,i⁽ᵇ⁾ : b = 1,…,B}, 0.025)
CI_upper_i = quantile({f_boot,i⁽ᵇ⁾ : b = 1,…,B}, 0.975)
```

The default number of replicates is B = 500, which provides stable CI
estimates while keeping run times manageable.  For publication-quality
results, B ≥ 1 000 is recommended.

---

### Step 7 — Post-Processing

**Function**: `postprocess_scores(f_hat_star, ci_lower, ci_upper, lambda = 0.5, eps = 1e-6)`

Converts the raw corrected composite scores and their bootstrap CIs to the
final 0–100 scale via three sub-steps.  The *same* affine parameters
estimated from the point estimates are applied to the CI bounds throughout,
so the ordering f̂* ≤ CI_lower is never violated.

**Sub-step 1 — Z-scoring**

```
z = (f̂* − mean(f̂*)) / sd(f̂*)
```

CI bounds receive the same shift and scale.

**Sub-step 2 — Box-Cox (λ = 0.5)**

Shift z to be strictly positive:

```
x = z + |min(z)| + ε
y = (x^λ − 1) / λ
```

The square-root transform (λ = 0.5) compresses the upper tail, reducing the
influence of extreme high scorers on the spread of the distribution.

**Sub-step 3 — Min-max rescaling**

```
score = 100 · (y − min(y)) / (max(y) − min(y))
```

CI bounds are rescaled with the same min/max and then clamped to [0, 100].

---

## API Reference

### `mpca_pipeline()`

The main entry point.  Runs all seven steps end-to-end and returns a
named list / dict.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | data.frame / DataFrame | — | Input data. |
| `score_cols` | character / list of str | — | K sub-index score column names. |
| `lower_cols` | character / list of str | — | K lower CI column names (same order). |
| `upper_cols` | character / list of str | — | K upper CI column names (same order). |
| `id_cols` | character / list of str | `NULL` / `None` | Identifier columns to carry through to `scores_df`. |
| `group_col` | string | `NULL` / `None` | Grouping-unit column for within-group interpolation (Pass 1). |
| `time_col` | string | `NULL` / `None` | Time-period column for time-mean fallback (Pass 2) and rankings filter. |
| `B` | integer | `500` | Bootstrap replicates. |
| `seed` | integer | `42` | Random seed for reproducibility. |
| `min_obs` | integer | `5` | Minimum non-missing sub-indices required. |
| `rankings_value` | scalar / NULL | `NULL` / `None` | Value of `time_col` for ranked output; `NULL` / `None` skips rankings. |

**Returns**: named list / dict with keys `scores_df`, `contributions_df`,
`rankings_df` (see [Output Format](#output-format)).

---

### `option_b_filter()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | data.frame / DataFrame | — | Input data. |
| `score_cols` | character / list of str | — | Sub-index score column names. |
| `min_obs` | integer | `5` | Minimum observed sub-indices for a valid composite. |

**Returns**: a copy of `data` with two additional columns:
`n_obs` (integer) and `valid_composite` (boolean).

---

### `three_pass_imputation()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data` | data.frame / DataFrame | — | Analysis-set rows (valid composites only). |
| `score_cols` | character / list of str | — | Sub-index score column names. |
| `half_width_cols` | character / list of str | — | CI half-width column names (same order). |
| `group_col` | string | `NULL` / `None` | Grouping-unit column; `NULL` skips Pass 1. |
| `time_col` | string | `NULL` / `None` | Time-period column; `NULL` skips Pass 2. |

**Returns**: `(S_hat, H)` — both N×K numeric matrices.  In R these are
returned as a named list `list(S_hat = ..., H = ...)`.

---

### `naive_pca()`

| Parameter | Type | Description |
|---|---|---|
| `S_hat` | N×K matrix / ndarray | Imputed sub-index score matrix. |

**Returns**: named list / dict with keys:

| Key | Shape | Description |
|---|---|---|
| `col_means` | K | Column means of Ŝ. |
| `col_sds` | K | Column standard deviations. |
| `S_tilde` | N×K | Standardised score matrix. |
| `R_obs` | K×K | Sample correlation matrix. |
| `lambda1` | scalar | Leading eigenvalue. |
| `ell_hat` | K | PC1 loadings. |
| `w_hat` | K | Regression scoring coefficients. |
| `f_hat` | N | Naive composite scores. |
| `var_explained` | scalar | Proportion of variance explained by PC1. |

---

### `attenuation_correction()`

| Parameter | Type | Description |
|---|---|---|
| `S_hat` | N×K matrix | Imputed scores. |
| `H` | N×K matrix | CI half-widths (0 for imputed positions). |
| `naive_result` | list / dict | Output of `naive_pca()`. |

**Returns**: named list / dict with keys:

| Key | Shape | Description |
|---|---|---|
| `psi_hat` | K | Per-sub-index error variance estimates. |
| `reliability` | K | Reliability coefficients r_j ∈ [0, 1]. |
| `Sigma_obs` | K×K | Observed covariance matrix. |
| `Sigma_sig` | K×K | Signal covariance matrix. |
| `R_sig` | K×K | Corrected (signal) correlation matrix. |
| `lambda1_star` | scalar | Leading eigenvalue of R̂_sig. |
| `ell_hat_star` | K | Corrected PC1 loadings. |
| `w_hat_star` | K | Corrected regression scoring coefficients. |
| `f_hat_star` | N | Corrected composite scores. |
| `sigma_sig` | K | Signal standard deviations. |
| `var_explained_star` | scalar | PC1 variance share in R̂_sig. |

---

### `two_stage_bootstrap()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `S_hat` | N×K matrix | — | Imputed scores. |
| `H` | N×K matrix | — | CI half-widths. |
| `col_means` | K vector | — | Column means from `naive_pca()`. |
| `w_hat_star` | K vector | — | Corrected scoring coefficients. |
| `sigma_sig` | K vector | — | Signal standard deviations. |
| `B` | integer | `500` | Bootstrap replicates. |
| `seed` | integer | `42` | Random seed. |

**Returns**: named list / dict with keys `ci_lower` (N), `ci_upper` (N),
and `boot_scores` (N×B, `NA`/`NaN` for unsampled cells).

---

### `postprocess_scores()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `f_hat_star` | N vector | — | Corrected composite scores. |
| `ci_lower` | N vector | — | Bootstrap 2.5th-percentile scores. |
| `ci_upper` | N vector | — | Bootstrap 97.5th-percentile scores. |
| `lambda` | numeric | `0.5` | Box-Cox power parameter. |
| `eps` | numeric | `1e-6` | Positivity buffer before Box-Cox. |

**Returns**: named list / dict with keys `score`, `ci_lower`, `ci_upper`
(each length N, rescaled to [0, 100]).

---

## Output Format

`mpca_pipeline()` returns three tables.

### `scores_df`

One row per observation in the analysis set (rows passing the Option Filter).

| Column | Type | Description |
|---|---|---|
| *(id_cols)* | any | Identifier columns passed through from the input (as specified by `id_cols`). |
| `score` | float [0, 100] | Post-processed composite score. |
| `ci_lower` | float [0, 100] | Lower bound of 95% CI. |
| `ci_upper` | float [0, 100] | Upper bound of 95% CI. |

### `contributions_df`

One row per sub-index.

| Column | Type | Description |
|---|---|---|
| `sub_index` | string | Sub-index column name. |
| `naive_loading` | float | PC1 loading from naive PCA (ℓ̂_j). |
| `naive_weight` | float | Regression scoring coefficient from naive PCA (ŵ_j). |
| `corrected_loading` | float | PC1 loading from corrected PCA (ℓ̂*_j). |
| `corrected_weight` | float | Regression scoring coefficient from corrected PCA (ŵ*_j). |
| `reliability` | float [0, 1] | Reliability coefficient r_j = 1 − ψ̂_j / Σ̂_obs,jj. |

### `rankings_df`

Subset of `scores_df` where `time_col == rankings_value`, sorted descending
by `score`, with an additional `rank` column (1 = highest score).  Empty if
`rankings_value` is `NULL` / `None`, `time_col` is not specified, or no rows
match the value.

---

## Quick-Start Examples

### R

```r
library(mpca)

# ---- Minimal synthetic dataset ----
set.seed(1)
n <- 50
data <- data.frame(
  unit = rep(paste0("U", 1:10), each = 5),
  time = rep(1:5, times = 10)
)
for (k in 1:8) {
  data[[paste0("s",  k)]]         <- runif(n, 40, 80)
  data[[paste0("s",  k, "_lower")]] <- data[[paste0("s", k)]] - runif(n, 2, 8)
  data[[paste0("s",  k, "_upper")]] <- data[[paste0("s", k)]] + runif(n, 2, 8)
}

# ---- Run the pipeline ----
result <- mpca_pipeline(
  data           = data,
  score_cols     = paste0("s", 1:8),
  lower_cols     = paste0("s", 1:8, "_lower"),
  upper_cols     = paste0("s", 1:8, "_upper"),
  id_cols        = c("unit", "time"),
  group_col      = "unit",
  time_col       = "time",
  B              = 200L,    # use 500+ in practice
  seed           = 42L,
  min_obs        = 5L,
  rankings_value = 5L
)

# ---- Inspect outputs ----
head(result$scores_df)
result$contributions_df
head(result$rankings_df, 5)

# ---- Reliability: how much signal is in each sub-index? ----
result$contributions_df[, c("sub_index", "reliability")]
```

### Python

```python
import numpy as np
import pandas as pd
from mpca import mpca_pipeline

# ---- Minimal synthetic dataset ----
rng = np.random.default_rng(1)
n_units, n_periods, K = 10, 5, 8
rows = [
    {"unit": f"U{i}", "time": t}
    for i in range(1, n_units + 1)
    for t in range(1, n_periods + 1)
]
data = pd.DataFrame(rows)
for k in range(1, K + 1):
    scores = rng.uniform(40, 80, len(data))
    data[f"s{k}"]         = scores
    data[f"s{k}_lower"]   = scores - rng.uniform(2, 8, len(data))
    data[f"s{k}_upper"]   = scores + rng.uniform(2, 8, len(data))

# ---- Run the pipeline ----
result = mpca_pipeline(
    data=data,
    score_cols=[f"s{k}" for k in range(1, K + 1)],
    lower_cols=[f"s{k}_lower" for k in range(1, K + 1)],
    upper_cols=[f"s{k}_upper" for k in range(1, K + 1)],
    id_cols=["unit", "time"],
    group_col="unit",
    time_col="time",
    B=200,          # use 500+ in practice
    seed=42,
    min_obs=5,
    rankings_value=5,
)

# ---- Inspect outputs ----
print(result["scores_df"].head())
print(result["contributions_df"])
print(result["rankings_df"].head())

# ---- Reliability ----
print(result["contributions_df"][["sub_index", "reliability"]])
```

### Running individual pipeline steps

```python
from mpca import (
    option_b_filter,
    three_pass_imputation,
    naive_pca,
    attenuation_correction,
    two_stage_bootstrap,
    postprocess_scores,
)

# After option_b_filter and half-width computation ...
S_hat, H = three_pass_imputation(
    data_valid, score_cols, hw_cols,
    group_col="unit",
    time_col="time",
)
naive = naive_pca(S_hat)
corr  = attenuation_correction(S_hat, H, naive)

print("Naive PC1 variance explained:    ", naive["var_explained"])
print("Corrected PC1 variance explained:", corr["var_explained_star"])
print("Sub-index reliability:\n", corr["reliability"])
```

---

## Running the Tests

### R

```r
# From the repository root
devtools::test("r/")
```

Or using the standard testthat runner:

```bash
cd r
Rscript -e "testthat::test_dir('tests/testthat')"
```

### Python

```bash
cd python
pytest tests/ -v
```

To run a single test file:

```bash
pytest tests/test_imputation.py -v
```

---

## References

Higham, N. J. (2002). Computing the nearest correlation matrix — a problem
from finance. *IMA Journal of Numerical Analysis*, 22(3), 329–343.

Spearman, C. (1904). The proof and measurement of association between two
things. *American Journal of Psychology*, 15(1), 72–101.
