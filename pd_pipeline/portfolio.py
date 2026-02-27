"""Deterministic portfolio loss calculations (single quantile)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_portfolio_loss(
    exposures_csv: str,
    tenor: str = '12_month',
    ead: float = 1_000_000,
    lgd: float = 0.45,
    quantile: float = 0.999,
    verbose: bool = True,
) -> Dict[str, object]:
    """Calculate portfolio loss per exposure using the Basel II formula at a given quantile."""
    df = pd.read_csv(exposures_csv)
    pd_col = tenor
    corr_col = f'{tenor}_correlation'

    df = df.dropna(subset=[pd_col, corr_col]).copy()

    inv_norm_q = norm.ppf(quantile)
    inv_norm_pd = norm.ppf(df[pd_col].values)
    rhos = df[corr_col].values
    conditional_pd = norm.cdf((inv_norm_pd + np.sqrt(rhos) * inv_norm_q) / np.sqrt(1 - rhos))

    df['conditional_pd'] = conditional_pd
    df['individual_loss'] = ead * lgd * conditional_pd

    total_loss = df['individual_loss'].sum()
    mean_loss = df['individual_loss'].mean()
    median_loss = df['individual_loss'].median()
    max_loss = df['individual_loss'].max()

    if verbose:
        print("=" * 70)
        print("PORTFOLIO LOSS CALCULATION")
        print("=" * 70)
        print(f"Exposures: {len(df):,}")
        print(f"Tenor: {tenor}")
        print(f"Quantile: {quantile}")
        print(f"Total loss: {total_loss:,.2f} SEK")

    return {
        'results_df': df,
        'total_loss': total_loss,
        'mean_loss': mean_loss,
        'median_loss': median_loss,
        'max_loss': max_loss,
        'quantile': quantile,
        'ead': ead,
        'lgd': lgd,
    }
