"""Two-stage bootstrap for CI propagation."""

import numpy as np


def two_stage_bootstrap(
    S_hat: np.ndarray,
    H: np.ndarray,
    col_means: np.ndarray,
    w_hat_star: np.ndarray,
    sigma_sig: np.ndarray,
    B: int = 500,
    seed: int = 42,
) -> dict:
    """
    Propagate measurement error and sampling uncertainty via a two-stage bootstrap.

    For each of *B* replications:

    1. **Perturbation** — draw
       :math:`\\mathbf{E}^{(b)} \\sim \\mathcal{N}(\\mathbf{0},
       (\\mathbf{H}/1.96)^{\\circ 2})` element-wise and clamp the
       perturbed scores to :math:`[0, 100]`.
    2. **Resampling** — draw row indices :math:`\\mathcal{R}^{(b)}` with
       replacement.
    3. **Scoring** — standardise using the original column means and signal
       SDs; project with :math:`\\hat{\\mathbf{w}}^*`.
    4. **Row mapping** — for each original observation :math:`i`, record
       the score from its *first* occurrence in :math:`\\mathcal{R}^{(b)}`
       (``NaN`` if unsampled).

    The 95% CI is formed from the 2.5th and 97.5th empirical quantiles
    (``nan``-aware).

    Parameters
    ----------
    S_hat : np.ndarray, shape (N, K)
        Imputed sub-index score matrix.
    H : np.ndarray, shape (N, K)
        CI half-width matrix (0 for imputed positions).
    col_means : np.ndarray, shape (K,)
        Column means from :func:`naive_pca`.
    w_hat_star : np.ndarray, shape (K,)
        Corrected regression scoring coefficients.
    sigma_sig : np.ndarray, shape (K,)
        Signal standard deviations.
    B : int, optional
        Number of bootstrap replications (default 500).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    dict with keys:

    ``ci_lower`` : np.ndarray, shape (N,)
        2.5th-percentile bootstrap scores.
    ``ci_upper`` : np.ndarray, shape (N,)
        97.5th-percentile bootstrap scores.
    ``boot_scores`` : np.ndarray, shape (N, B)
        Full matrix of bootstrap scores (``NaN`` for unsampled rows).
    """
    rng = np.random.default_rng(seed)
    N, K = S_hat.shape
    sigma_e = H / 1.96

    boot_scores = np.full((N, B), np.nan)

    for b in range(B):
        # Stage 1: perturbation
        E = rng.standard_normal((N, K)) * sigma_e
        S_pert = np.clip(S_hat + E, 0.0, 100.0)

        # Stage 2: resample rows
        idx = rng.choice(N, size=N, replace=True)
        S_boot = S_pert[idx]

        # Standardise and score
        S_std = (S_boot - col_means) / sigma_sig
        f_boot = S_std @ w_hat_star

        # Row mapping: first-occurrence convention
        for i in range(N):
            positions = np.where(idx == i)[0]
            if len(positions) > 0:
                boot_scores[i, b] = f_boot[positions[0]]

    ci_lower = np.nanpercentile(boot_scores, 2.5, axis=1)
    ci_upper = np.nanpercentile(boot_scores, 97.5, axis=1)

    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "boot_scores": boot_scores,
    }
