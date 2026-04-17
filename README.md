# MPCA — Multilevel PCA for Composite Indices

This repository provides both an **R package** and a **Python package** that
implement Multilevel PCA (MPCA): a measurement-error-corrected approach to
building composite indices from sub-indices that carry known uncertainty.

## Motivation

When sub-index scores are noisy estimates (e.g. derived from bootstrapped
factor models), naive PCA on the observed correlation matrix systematically
understates the strength of the common signal—a phenomenon known as
*attenuation bias* (Spearman, 1904).  MPCA corrects this bias by:

1. Estimating per-sub-index measurement error variance from the bootstrap CI
   half-widths.
2. Recovering the *signal* covariance matrix by subtracting the error
   diagonal.
3. Re-running PCA on the disattenuated correlation matrix.
4. Propagating *both* measurement error (Stage 1) and sampling uncertainty
   (Stage 2) into the composite CIs via a two-stage bootstrap.

## Repository Structure

```
MPCA/
├── r/                  R package (mpca)
│   ├── DESCRIPTION
│   ├── NAMESPACE
│   ├── R/
│   │   ├── imputation.R   option_b_filter(), three_pass_imputation()
│   │   ├── naive_pca.R    naive_pca()
│   │   ├── correction.R   attenuation_correction()
│   │   ├── bootstrap.R    two_stage_bootstrap()
│   │   ├── postprocess.R  postprocess_scores()
│   │   └── pipeline.R     mpca_pipeline()
│   └── tests/
└── python/             Python package (mpca)
    ├── pyproject.toml
    ├── mpca/
    │   ├── imputation.py
    │   ├── naive_pca.py
    │   ├── correction.py
    │   ├── bootstrap.py
    │   ├── postprocess.py
    │   └── pipeline.py
    └── tests/
```

## Installation

### R

```r
devtools::install("r/")
```

### Python

```bash
pip install -e python/
```

## Pipeline Overview

| Step | Function | Description |
|---|---|---|
| 1 | `option_b_filter` | Exclude rows with < 5 observed sub-indices |
| 2 | *(internal)* | Compute CI half-widths h = (upper − lower) / 2 |
| 3 | `three_pass_imputation` | Linear interp → year mean → global mean |
| 4 | `naive_pca` | PC1 on observed correlation matrix |
| 5 | `attenuation_correction` | Disattenuation → corrected PCA |
| 6 | `two_stage_bootstrap` | Perturb + resample → 95% CI |
| 7 | `postprocess_scores` | Z-score → Box-Cox → min-max [0, 100] |

## References

Spearman, C. (1904). The proof and measurement of association between two
things. *American Journal of Psychology*, 15(1), 72–101.
