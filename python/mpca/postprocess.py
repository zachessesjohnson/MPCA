"""Post-processing: z-score → Box-Cox → min-max rescaling."""

import numpy as np


def postprocess_scores(
    f_hat_star: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    lam: float = 0.5,
    eps: float = 1e-6,
) -> dict:
    """
    Apply three-step post-processing to composite scores and CI bounds.

    **Step 1 — Z-score**: subtract the mean and divide by the SD of the
    point estimates.  The same shift and scale are applied to CI bounds.

    **Step 2 — Box-Cox** (:math:`\\lambda = 0.5`): shift all values to a
    strictly positive domain (:math:`x \\leftarrow x + |\\min(x)| + \\epsilon`
    using the minimum of the point estimates), then apply
    :math:`y = (x^\\lambda - 1)/\\lambda`.

    **Step 3 — Min-max** to :math:`[0, 100]`: use the min and max of the
    transformed *point estimates*; CI bounds are clamped to
    :math:`[0, 100]`.

    Parameters
    ----------
    f_hat_star : np.ndarray, shape (N,)
        Corrected composite point estimates.
    ci_lower : np.ndarray, shape (N,)
        2.5th-percentile bootstrap scores.
    ci_upper : np.ndarray, shape (N,)
        97.5th-percentile bootstrap scores.
    lam : float, optional
        Box-Cox power parameter (default 0.5).
    eps : float, optional
        Small constant for positivity before Box-Cox (default 1e-6).

    Returns
    -------
    dict with keys:

    ``score`` : np.ndarray, shape (N,)
        Rescaled point estimates in :math:`[0, 100]`.
    ``ci_lower`` : np.ndarray, shape (N,)
        Rescaled lower CI bounds, clamped to :math:`[0, 100]`.
    ``ci_upper`` : np.ndarray, shape (N,)
        Rescaled upper CI bounds, clamped to :math:`[0, 100]`.
    """
    # ---- Step 1: Z-score ----
    mu = np.nanmean(f_hat_star)
    sig = np.nanstd(f_hat_star, ddof=1)

    z_score = (f_hat_star - mu) / sig
    z_lower = (ci_lower - mu) / sig
    z_upper = (ci_upper - mu) / sig

    # ---- Step 2: Box-Cox ----
    shift = abs(np.nanmin(z_score)) + eps
    x_score = z_score + shift
    x_lower = z_lower + shift
    x_upper = z_upper + shift

    def box_cox(x: np.ndarray) -> np.ndarray:
        return (np.maximum(x, 0) ** lam - 1) / lam

    bc_score = box_cox(x_score)
    bc_lower = box_cox(x_lower)
    bc_upper = box_cox(x_upper)

    # ---- Step 3: Min-max to [0, 100] ----
    bc_min = np.nanmin(bc_score)
    bc_max = np.nanmax(bc_score)
    rng = bc_max - bc_min

    def rescale(x: np.ndarray) -> np.ndarray:
        return 100.0 * (x - bc_min) / rng

    score = rescale(bc_score)
    ci_lo = np.clip(rescale(bc_lower), 0.0, 100.0)
    ci_up = np.clip(rescale(bc_upper), 0.0, 100.0)

    return {"score": score, "ci_lower": ci_lo, "ci_upper": ci_up}
