"""Linear algebra helpers for numerically stable deterministic solves."""

import numpy as np


def solve_with_ridge(
    matrix: np.ndarray,
    rhs: np.ndarray,
    base_ridge: float = 1e-10,
    max_attempts: int = 7,
) -> np.ndarray:
    """Solve linear system with deterministic diagonal ridge fallback."""
    eye = np.eye(matrix.shape[0], dtype=matrix.dtype)
    for attempt in range(max_attempts):
        ridge = 0.0 if attempt == 0 else base_ridge * (10 ** (attempt - 1))
        try:
            return np.linalg.solve(matrix + ridge * eye, rhs)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError(
        "Could not solve linear system even after deterministic ridge regularization."
    )
