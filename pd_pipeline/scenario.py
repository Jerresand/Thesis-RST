"""Scenario-based portfolio loss simulation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

_DEFAULT_MEAN_VEC = np.array([
    0.169775,
    3.521940,
    6.583915,
    2214.650430,
    78.849186,
    103.002618,
    0.055089,
])

_DEFAULT_COV_MAT = np.array([
    [1.702954, -0.084034, 0.041250, -33.193374, -0.625588, -0.543029, 0.001328],
    [-0.084034, 26.399865, -2.749306, -940.871908, -36.538440, 7.351396, -0.011477],
    [0.041250, -2.749306, 5.631566, 292.639630, 33.600429, -13.845844, 0.002567],
    [-33.193374, -940.871908, 292.639630, 1020321.0, 12367.641324, 6779.128571, 30.526317],
    [-0.625588, -36.538440, 33.600429, 12367.641324, 537.544563, 93.828038, 0.332007],
    [-0.543029, 7.351396, -13.845844, 6779.128571, 93.828038, 2264.893705, 0.972083],
    [0.001328, -0.011477, 0.002567, 30.526317, 0.332007, 0.972083, 0.002642],
])


def calculate_scenario_portfolio_loss(
    exposures_csv: str,
    sensitivities_csv: str,
    macro_vars: List[str],
    gpr_vars: List[str],
    n_scenarios: int = 10000,
    tenor: str = '12_month',
    ead: float = 1_000_000,
    lgd: float = 0.45,
    quantile: float = 0.999,
    seed: int = 42,
    mean_vec: Optional[np.ndarray] = None,
    cov_mat: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Calculate portfolio loss distribution across macroeconomic scenarios."""
    np.random.seed(seed)

    if verbose:
        print("=" * 80)
        print("SCENARIO-BASED PORTFOLIO LOSS CALCULATION")
        print("=" * 80)
        print(f"Scenarios:  {n_scenarios:,}")
        print(f"Tenor:      {tenor}")
        print(f"EAD:        {ead:,.0f} SEK")
        print(f"LGD:        {lgd:.1%}")
        print(f"Quantile:   {quantile}")

    if verbose:
        print("\n[1/5] Loading exposures...")
    df_exp = pd.read_csv(exposures_csv)
    pd_col = tenor
    corr_col = f'{tenor}_correlation'

    df_exp = df_exp.dropna(subset=[pd_col, corr_col, 'Sector']).copy()
    df_exp['PD_zero'] = df_exp[pd_col]

    if verbose:
        print(f"   ✓ {len(df_exp)} exposures loaded")
        print(f"   ✓ PD^0 range: [{df_exp['PD_zero'].min():.6f}, {df_exp['PD_zero'].max():.6f}]")

    if verbose:
        print("\n[2/5] Loading sensitivities...")
    df_sens = pd.read_csv(sensitivities_csv)

    sens_map = {}
    for _, row in df_sens.iterrows():
        sector = row['Sector']
        sens_map[sector] = {
            'beta': {var: row[f'β_{var}'] for var in macro_vars},
            'delta': {var: row[f'δ_{var}'] for var in gpr_vars},
        }

    df_exp = df_exp[df_exp['Sector'].isin(sens_map.keys())].copy()

    if verbose:
        print(f"   ✓ {len(df_sens)} sectors with sensitivities")
        print(f"   ✓ {len(df_exp)} exposures matched to sectors")

    if verbose:
        print("\n[3/5] Generating scenarios...")

    all_vars = macro_vars + gpr_vars
    mean_vec = _DEFAULT_MEAN_VEC if mean_vec is None else np.asarray(mean_vec)
    cov_mat = _DEFAULT_COV_MAT if cov_mat is None else np.asarray(cov_mat)

    scenarios = np.random.multivariate_normal(mean_vec, cov_mat, size=n_scenarios)
    scenarios_df = pd.DataFrame(scenarios, columns=all_vars)

    if verbose:
        print(f"   ✓ Generated {n_scenarios:,} scenarios")

    if verbose:
        print("\n[4/5] Calculating scenario PDs...")

    n_exp = len(df_exp)
    scenario_pds = np.zeros((n_scenarios, n_exp))

    for i, (_, exp) in enumerate(df_exp.iterrows()):
        if verbose and (i + 1) % 100 == 0:
            print(f"   Processing exposure {i+1}/{n_exp}...", end='\r')

        sector = exp['Sector']
        pd_zero = exp['PD_zero']
        sens = sens_map[sector]

        beta_x = sum(sens['beta'][var] * scenarios_df[var].values for var in macro_vars)
        delta_g = sum(sens['delta'][var] * scenarios_df[var].values for var in gpr_vars)
        adjustment = beta_x + delta_g

        exp_adj = np.exp(adjustment)
        pd_adjusted = (pd_zero * exp_adj) / (1 - pd_zero + pd_zero * exp_adj)
        pd_adjusted = np.clip(pd_adjusted, 1e-6, 1 - 1e-6)

        scenario_pds[:, i] = pd_adjusted

    if verbose:
        print(f"\n   ✓ Calculated PDs for {n_exp} exposures × {n_scenarios:,} scenarios")

    if verbose:
        print("\n[5/5] Calculating portfolio losses...")

    portfolio_losses = np.zeros(n_scenarios)
    inv_norm_q = norm.ppf(quantile)
    rhos = df_exp[corr_col].values

    for s in range(n_scenarios):
        if verbose and (s + 1) % 1000 == 0:
            print(f"   Scenario {s+1}/{n_scenarios}...", end='\r')

        pds = scenario_pds[s, :]
        inv_norm_pd = norm.ppf(pds)
        conditional_pd = norm.cdf((inv_norm_pd + np.sqrt(rhos) * inv_norm_q) / np.sqrt(1 - rhos))
        portfolio_losses[s] = np.sum(ead * lgd * conditional_pd)

    if verbose:
        print("\n   ✓ Portfolio losses calculated")

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    var_999 = np.percentile(portfolio_losses, 99.9)
    es_999 = np.mean(portfolio_losses[portfolio_losses >= var_999])

    if verbose:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print("\nPortfolio Loss Distribution:")
        print(f"  Mean:       {np.mean(portfolio_losses):>15,.2f} SEK")
        print(f"  Median:     {np.median(portfolio_losses):>15,.2f} SEK")
        print(f"  Std Dev:    {np.std(portfolio_losses):>15,.2f} SEK")
        print(f"  Min:        {np.min(portfolio_losses):>15,.2f} SEK")
        print(f"  Max:        {np.max(portfolio_losses):>15,.2f} SEK")

        print("\nKey Percentiles:")
        for p in [5, 25, 50, 75, 95, 99, 99.9]:
            print(f"  {p:5.1f}%:    {np.percentile(portfolio_losses, p):>15,.2f} SEK")

        print("\nTail Risk:")
        print(f"  VaR (99.9%):          {var_999:>15,.2f} SEK")
        print(f"  Expected Shortfall:   {es_999:>15,.2f} SEK")
        print(f"  ES / VaR:             {es_999/var_999:>15.2f}x")
        print("=" * 80)

    return {
        'portfolio_losses': portfolio_losses,
        'scenarios': scenarios_df,
        'scenario_pds': scenario_pds,
        'exposures': df_exp,
        'statistics': {
            'mean': np.mean(portfolio_losses),
            'median': np.median(portfolio_losses),
            'std': np.std(portfolio_losses),
            'min': np.min(portfolio_losses),
            'max': np.max(portfolio_losses),
            'var_999': var_999,
            'es_999': es_999,
            'percentiles': {p: np.percentile(portfolio_losses, p) for p in percentiles},
        },
    }
