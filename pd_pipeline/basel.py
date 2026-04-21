"""Basel asset correlation and RWA calculation utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def asset_correlation_formula(pd: float | np.ndarray) -> float | np.ndarray:
    """Simplified Basel asset correlation formula for corporate exposures."""
    return 0.12 + 0.12 * np.exp(-50 * np.asarray(pd))


def append_basel_correlations(
    df_exposures: pd.DataFrame,
    pd_columns: Iterable[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Append Basel correlation columns for PD tenors."""
    df = df_exposures.copy()
    for col in pd_columns:
        if col in df.columns:
            corr_col = f'{col}_correlation'
            df[corr_col] = df[col].apply(
                lambda x: asset_correlation_formula(x) if pd.notna(x) else np.nan
            )

            if verbose:
                non_null = df[col].notna().sum()
                if non_null > 0:
                    print(f"\n{col}:")
                    print(f"  Valid PDs: {non_null}")
                    print(f"  PD range: {df[col].min():.4f} to {df[col].max():.4f}")
                    print(f"  ρ range:  {df[corr_col].min():.4f} to {df[corr_col].max():.4f}")

    if verbose:
        print("\n✓ Correlation columns appended to dataframe")

    return df


def calculate_maturity_adjustment(pd: float | np.ndarray) -> np.ndarray:
    return (0.11852 - 0.05478 * np.log(pd)) ** 2


def calculate_capital_requirement(
    pd: float | np.ndarray,
    lgd: float | np.ndarray,
    correlation: float | np.ndarray,
    maturity: float = 2.5,
    quantile: float = 0.999,
) -> np.ndarray:
    pd = np.asarray(pd)
    lgd = np.asarray(lgd)
    correlation = np.asarray(correlation)

    b = calculate_maturity_adjustment(pd)
    g_pd = norm.ppf(pd)
    g_q = norm.ppf(quantile)

    conditional_pd_term = (
        (g_pd / np.sqrt(1 - correlation))
        + (np.sqrt(correlation / (1 - correlation)) * g_q)
    )
    n_term = norm.cdf(conditional_pd_term)
    base_capital = lgd * (n_term - pd)
    maturity_adjustment = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    return base_capital * maturity_adjustment

def calculate_rwa(
    pd: float | np.ndarray,
    lgd: float | np.ndarray,
    ead: float | np.ndarray,
    correlation: float | np.ndarray,
    maturity: float = 2.5,
    quantile: float = 0.999,
) -> np.ndarray:
    k = calculate_capital_requirement(pd, lgd, correlation, maturity, quantile)
    return k * 12.5 * ead


def calculate_risk_weight(
    pd: float | np.ndarray,
    lgd: float | np.ndarray,
    correlation: float | np.ndarray,
    maturity: float = 2.5,
    quantile: float = 0.999,
) -> np.ndarray:
    k = calculate_capital_requirement(pd, lgd, correlation, maturity, quantile)
    return k * 12.5


def compute_rwa_by_tenor(
    df_portfolio: pd.DataFrame,
    tenors: Iterable[str],
    lgd: float,
    ead: float,
    maturity: float,
    quantile: float = 0.999,
) -> Dict[str, Dict[str, object]]:
    """Compute RWA metrics per tenor and return a results dictionary."""
    results_by_tenor: Dict[str, Dict[str, object]] = {}

    for tenor in tenors:
        pd_col = tenor
        corr_col = f'{tenor}_correlation'
        if pd_col not in df_portfolio.columns or corr_col not in df_portfolio.columns:
            continue

        mask = df_portfolio[pd_col].notna() & df_portfolio[corr_col].notna()
        df_valid = df_portfolio[mask].copy()

        if len(df_valid) == 0:
            continue

        df_valid['K'] = calculate_capital_requirement(
            pd=df_valid[pd_col].values,
            lgd=lgd,
            correlation=df_valid[corr_col].values,
            maturity=maturity,
            quantile=quantile,
        )
        df_valid['RW'] = calculate_risk_weight(
            pd=df_valid[pd_col].values,
            lgd=lgd,
            correlation=df_valid[corr_col].values,
            maturity=maturity,
            quantile=quantile,
        )
        df_valid['RWA'] = calculate_rwa(
            pd=df_valid[pd_col].values,
            lgd=lgd,
            ead=ead,
            correlation=df_valid[corr_col].values,
            maturity=maturity,
            quantile=quantile,
        )

        results_by_tenor[tenor] = {
            'df': df_valid,
            'count': len(df_valid),
            'total_rwa': df_valid['RWA'].sum(),
            'total_ead': ead * len(df_valid),
            'avg_rw': df_valid['RW'].mean(),
            'median_rw': df_valid['RW'].median(),
            'min_rwa': df_valid['RWA'].min(),
            'max_rwa': df_valid['RWA'].max(),
            'avg_pd': df_valid[pd_col].mean(),
            'median_pd': df_valid[pd_col].median(),
        }

    return results_by_tenor


def print_rwa_summary(results_by_tenor: Dict[str, Dict[str, object]]) -> None:
    """Print summary statistics for each tenor's RWA results."""
    print("\n" + "=" * 70)
    print("RWA RESULTS BY TENOR")
    print("=" * 70)

    for tenor, results in results_by_tenor.items():
        print(f"\n{tenor.upper().replace('_', ' ')}:")
        print(f"  Number of exposures: {results['count']:,}")
        print(f"  Total EAD: {results['total_ead']:,.0f} SEK")
        print(f"  Total RWA: {results['total_rwa']:,.0f} SEK")
        print(f"  Average Risk Weight: {results['avg_rw']:.2%}")
        print(f"  Median Risk Weight: {results['median_rw']:.2%}")
        print(f"  Min RWA per exposure: {results['min_rwa']:,.0f} SEK")
        print(f"  Max RWA per exposure: {results['max_rwa']:,.0f} SEK")
        print(f"  Average PD: {results['avg_pd']:.4f} ({results['avg_pd']*100:.2f}%)")
        print(f"  Median PD: {results['median_pd']:.4f} ({results['median_pd']*100:.2f}%)")
        print(f"  Capital Requirement (8% of RWA): {results['total_rwa'] * 0.08:,.0f} SEK")


def print_rwa_detail_12m(results_by_tenor: Dict[str, Dict[str, object]]) -> None:
    """Print detailed 12-month tenor analysis."""
    if '12_month' not in results_by_tenor:
        print("\nError: 12-month tenor results not found. Please run the RWA calculation first.")
        return

    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS - 12 MONTH TENOR")
    print("=" * 70)

    df_12m = results_by_tenor['12_month']['df']
    df_12m_sorted = df_12m.sort_values('RWA', ascending=False)

    print(f"\nTop 10 Exposures by RWA:\n")
    display_cols = ['Company_number', 'Sector', '12_month', '12_month_correlation', 'K', 'RW', 'RWA']
    print(df_12m_sorted[display_cols].head(10).to_string(index=False))

    print(f"\n\nBottom 10 Exposures by RWA:\n")
    print(df_12m_sorted[display_cols].tail(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("RWA BY SECTOR (12 MONTH TENOR)")
    print("=" * 70)

    sector_analysis = df_12m.groupby('Sector').agg({
        'Company_number': 'count',
        '12_month': ['mean', 'median'],
        'RW': ['mean', 'median'],
        'RWA': ['sum', 'mean'],
        'K': 'mean',
    }).round(4)

    sector_analysis.columns = ['Count', 'Avg_PD', 'Med_PD', 'Avg_RW', 'Med_RW', 'Total_RWA', 'Avg_RWA', 'Avg_K']
    sector_analysis = sector_analysis.sort_values('Total_RWA', ascending=False)

    print()
    print(sector_analysis.to_string())

    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY (12 MONTH TENOR)")
    print("=" * 70)

    total_rwa = results_by_tenor['12_month']['total_rwa']
    total_ead = results_by_tenor['12_month']['total_ead']
    avg_rw = results_by_tenor['12_month']['avg_rw']

    print(f"\nTotal Portfolio EAD: {total_ead:,.0f} SEK")
    print(f"Total Portfolio RWA: {total_rwa:,.0f} SEK")
    print(f"Overall Risk Weight: {(total_rwa/total_ead):.2%}")
    print(f"Average Risk Weight: {avg_rw:.2%}")
    print(f"Minimum Capital Required (8%): {total_rwa * 0.08:,.0f} SEK")
    print(f"Minimum Capital as % of EAD: {(total_rwa * 0.08 / total_ead):.2%}")
    print("=" * 70)
