from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MahalanobisModel:
    """
    Mahalanobis model for a set of feature columns.

    If `by_sector` is True, `params_by_sector` contains (mu, cov_inv) per sector value.
    Otherwise, `params_by_sector` holds a single entry under key `None`.
    """

    feature_cols: tuple[str, ...]
    by_sector: bool
    sector_col: str | None
    params_by_sector: Mapping[Any, tuple[np.ndarray, np.ndarray]]  # sector -> (mu, cov_inv)


def _fit_mu_cov_inv(X: np.ndarray, add_diagonal: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit mu and cov_inv for Mahalanobis distance.

    X must be shaped (n_samples, n_features).
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to estimate covariance.")

    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)

    # Stabilize against singular/near-singular covariance.
    cov = cov + add_diagonal * np.eye(cov.shape[0])
    cov_inv = np.linalg.pinv(cov)
    return mu, cov_inv


def fit_mahalanobis_model(
    df: pd.DataFrame,
    *,
    feature_cols: Iterable[str],
    sector_col: str | None = None,
    per_sector: bool = False,
    add_diagonal: float = 1e-6,
) -> MahalanobisModel:
    """
    Fit a Mahalanobis model on the given dataframe.

    Typical use in your thesis:
    - df is your "relative-to-last" dataset
    - feature_cols are your macro/GPR variables (e.g. config.MACRO_COLS + config.GPR_COLS)
    """
    feature_cols = tuple(feature_cols)
    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")

    if per_sector and not sector_col:
        raise ValueError("If per_sector=True you must provide sector_col.")

    if per_sector:
        params: dict[Any, tuple[np.ndarray, np.ndarray]] = {}
        for sector, df_s in df.groupby(sector_col, sort=True):
            X = df_s.loc[:, feature_cols].to_numpy(dtype=float)
            # Drop rows with any NaN in features.
            X = X[~np.isnan(X).any(axis=1)]
            if X.shape[0] < 2:
                # Skip sectors that cannot support covariance estimation.
                continue
            params[sector] = _fit_mu_cov_inv(X, add_diagonal=add_diagonal)

        return MahalanobisModel(
            feature_cols=feature_cols,
            by_sector=True,
            sector_col=sector_col,
            params_by_sector=params,
        )

    # Pooled: fit on all rows.
    X = df.loc[:, feature_cols].to_numpy(dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    mu, cov_inv = _fit_mu_cov_inv(X, add_diagonal=add_diagonal)

    return MahalanobisModel(
        feature_cols=feature_cols,
        by_sector=False,
        sector_col=None,
        params_by_sector={None: (mu, cov_inv)},
    )


def mahalanobis_d2(
    x: np.ndarray,
    model: MahalanobisModel,
    *,
    sector: Any = None,
) -> float:
    """
    Mahalanobis squared distance d^2(x) under the fitted model.

    x must have shape (n_features,) and be in the same domain as the training features.
    """
    x = np.asarray(x, dtype=float)
    mu, cov_inv = _select_params(model, sector=sector)

    if x.shape != mu.shape:
        raise ValueError(f"x has shape={x.shape}, but expected {mu.shape}")

    diff = x - mu
    # d^2 = diff^T * cov_inv * diff
    return float(diff.T @ cov_inv @ diff)


def mahalanobis_d2_from_df_row(
    row: Mapping[str, Any] | pd.Series,
    model: MahalanobisModel,
    *,
    sector: Any = None,
) -> float:
    """Convenience: compute d^2 from a dict/Series keyed by feature column names."""
    x = np.array([row[c] for c in model.feature_cols], dtype=float)
    return mahalanobis_d2(x, model, sector=sector)


def mahalanobis_d2_batch(
    X: np.ndarray,
    model: MahalanobisModel,
    *,
    sector: Any = None,
) -> np.ndarray:
    """
    Compute d^2 for many samples.

    X must be shaped (n_samples, n_features).
    Returns shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    mu, cov_inv = _select_params(model, sector=sector)

    if X.ndim != 2 or X.shape[1] != mu.shape[0]:
        raise ValueError(f"Expected X shape (n, {mu.shape[0]}), got {X.shape}")

    diff = X - mu[None, :]
    # Vectorized: for each n, diff[n]^T cov_inv diff[n]
    return np.einsum("ni,ij,nj->n", diff, cov_inv, diff).astype(float)


def _select_params(model: MahalanobisModel, *, sector: Any) -> tuple[np.ndarray, np.ndarray]:
    if model.by_sector:
        if sector not in model.params_by_sector:
            raise KeyError(
                f"No Mahalanobis parameters for sector={sector!r}. "
                "Maybe the sector has too few rows after dropping NaNs."
            )
        return model.params_by_sector[sector]

    return model.params_by_sector[None]

