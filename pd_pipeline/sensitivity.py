"""OLS sensitivity analysis for PD changes."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
import statsmodels.api as sm


def calculate_logit(p: np.ndarray | pd.Series) -> np.ndarray:
    """Compute log-odds with clipping to avoid infinities."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def run_sector_ols(
    df: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    pd_col: str,
    pdzero_col: str,
    min_obs: int = 10,
):
    """Fit OLS for a single sector and PD horizon."""
    sector_df = df.copy()
    sector_df['logit_pd'] = calculate_logit(sector_df[pd_col])
    sector_df['logit_pd_zero'] = calculate_logit(sector_df[pdzero_col])
    sector_df['delta_logit'] = sector_df['logit_pd'] - sector_df['logit_pd_zero']

    y = sector_df['delta_logit']
    X = pd.concat([sector_df[macro_cols], sector_df[gpr_cols]], axis=1)
    X = sm.add_constant(X)

    valid_idx = ~(y.isna() | X.isna().any(axis=1))
    y = y[valid_idx]
    X = X[valid_idx]

    if len(y) < min_obs:
        return None, None, None

    model = sm.OLS(y, X).fit()
    conf_int = model.conf_int(alpha=0.05)
    return model, conf_int, len(y)


def run_sensitivity_analysis(
    df_final_cleaned: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    sector_col: str,
    pd_maturity_cols: Iterable[str],
    pdzero_col: str = 'PDzero',
    min_obs: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run OLS sensitivity analysis across sectors and PD horizons."""
    sensitivities_data = []

    for sector in df_final_cleaned[sector_col].unique():
        sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()
        if verbose:
            print(f"\nProcessing sector: {sector} (n={len(sector_df)})")

        for pd_col in pd_maturity_cols:
            try:
                model, conf_int, n_obs = run_sector_ols(
                    sector_df,
                    macro_cols,
                    gpr_cols,
                    pd_col,
                    pdzero_col,
                    min_obs=min_obs,
                )

                if model is None:
                    if verbose:
                        print(f"  Skipping {pd_col}: insufficient data (n={len(sector_df)})")
                    continue

                result = {
                    'Sector': sector,
                    'PD_Horizon': pd_col,
                    'Intercept': model.params['const'],
                    'N_observations': n_obs,
                    'R_squared': model.rsquared,
                }

                for col in macro_cols:
                    result[f'β_{col}'] = model.params[col]
                    result[f'β_{col}_CI_lower'] = conf_int.loc[col, 0]
                    result[f'β_{col}_CI_upper'] = conf_int.loc[col, 1]

                for col in gpr_cols:
                    result[f'δ_{col}'] = model.params[col]
                    result[f'δ_{col}_CI_lower'] = conf_int.loc[col, 0]
                    result[f'δ_{col}_CI_upper'] = conf_int.loc[col, 1]

                sensitivities_data.append(result)
                if verbose:
                    print(f"  ✓ {pd_col}: R²={model.rsquared:.3f}, N={n_obs}")

            except Exception as exc:  # noqa: BLE001 - keep parity with notebook output
                if verbose:
                    print(f"  ✗ Could not fit model for {pd_col}: {exc}")

    return pd.DataFrame(sensitivities_data)


def export_sensitivities(df_sensitivities: pd.DataFrame, output_file: str) -> None:
    """Export sensitivity results to CSV."""
    df_sensitivities.to_csv(output_file, index=False)
    print(f"✓ Sensitivity results with 95% confidence intervals exported to: {output_file}")
    print(f"  Total sectors analyzed: {len(df_sensitivities)}")
    print("\nColumns include:")
    print("  - Point estimates: β_[variable] and δ_[variable]")
    print("  - 95% CI lower bounds: β_[variable]_CI_lower and δ_[variable]_CI_lower")
    print("  - 95% CI upper bounds: β_[variable]_CI_upper and δ_[variable]_CI_upper")


def print_sensitivity_tables(
    df_sensitivities: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
) -> None:
    """Print macro and GPR sensitivity tables for readability."""
    print("=" * 80)
    print("MACRO SENSITIVITIES (β) - Impact of macroeconomic variables on PD")
    print("=" * 80)
    beta_cols = ['Sector', 'PD_Horizon', 'N_observations', 'R_squared']
    for col in macro_cols:
        beta_cols.extend([f'β_{col}', f'β_{col}_CI_lower', f'β_{col}_CI_upper'])
    print(df_sensitivities[beta_cols])

    print("\n" + "=" * 80)
    print("GPR SENSITIVITIES (δ) - Impact of geopolitical risk on PD")
    print("=" * 80)
    delta_cols = ['Sector', 'PD_Horizon', 'N_observations', 'R_squared']
    for col in gpr_cols:
        delta_cols.extend([f'δ_{col}', f'δ_{col}_CI_lower', f'δ_{col}_CI_upper'])
    print(df_sensitivities[delta_cols])


def print_confidence_interval_summary(
    df_sensitivities: pd.DataFrame,
    gpr_cols: List[str],
) -> None:
    """Print confidence interval summary for the first three sectors."""
    print("=" * 80)
    print("SENSITIVITY ESTIMATES WITH 95% CONFIDENCE INTERVALS (First 3 Sectors)")
    print("=" * 80)

    for _, row in df_sensitivities.head(3).iterrows():
        print(f"\n{row['Sector']} - {row['PD_Horizon']} (N={int(row['N_observations'])}, R²={row['R_squared']:.3f})")
        print("\nβ (Macro):")
        for col in ['GDP_Growth', 'Interest_Rate']:
            print(f"  {col}: {row[f'β_{col}']:.4f} [{row[f'β_{col}_CI_lower']:.4f}, {row[f'β_{col}_CI_upper']:.4f}]")
        print("\nδ (GPR):")
        for col in gpr_cols:
            print(f"  {col}: {row[f'δ_{col}']:.4f} [{row[f'δ_{col}_CI_lower']:.4f}, {row[f'δ_{col}_CI_upper']:.4f}]")


def print_sensitivity_details(
    df_sensitivities: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
) -> None:
    """Print detailed sensitivity estimates with confidence intervals."""
    print("\n" + "=" * 80)
    print("SENSITIVITY ESTIMATES WITH 95% CONFIDENCE INTERVALS")
    print("=" * 80)

    for _, row in df_sensitivities.iterrows():
        print(f"\n{'='*80}")
        print(
            f"Sector: {row['Sector']} | PD Horizon: {row['PD_Horizon']} | "
            f"R²={row['R_squared']:.3f} | N={int(row['N_observations'])}"
        )
        print(f"{'='*80}")

        print("\nMACRO SENSITIVITIES (β):")
        print("-" * 80)
        for col in macro_cols:
            beta = row[f'β_{col}']
            ci_lower = row[f'β_{col}_CI_lower']
            ci_upper = row[f'β_{col}_CI_upper']
            ci_width = ci_upper - ci_lower
            print(
                f"  {col:25s}: β = {beta:8.4f}  "
                f"[95% CI: {ci_lower:8.4f}, {ci_upper:8.4f}]  (width: {ci_width:.4f})"
            )

        print("\nGPR SENSITIVITIES (δ):")
        print("-" * 80)
        for col in gpr_cols:
            delta = row[f'δ_{col}']
            ci_lower = row[f'δ_{col}_CI_lower']
            ci_upper = row[f'δ_{col}_CI_upper']
            ci_width = ci_upper - ci_lower
            print(
                f"  {col:25s}: δ = {delta:8.4f}  "
                f"[95% CI: {ci_lower:8.4f}, {ci_upper:8.4f}]  (width: {ci_width:.4f})"
            )

    print("\n" + "=" * 80)
