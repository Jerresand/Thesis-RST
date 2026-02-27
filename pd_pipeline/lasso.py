"""LASSO feature selection utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def run_lasso_feature_selection(
    df_final_cleaned: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    sector_col: str,
    pd_maturity_cols: Iterable[str],
    pdzero_col: str = 'PDzero',
    min_obs: int = 10,
    cv: int = 5,
    random_state: int = 42,
    n_alphas: int = 100,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Run LASSO with cross-validation for each sector and PD horizon."""
    lasso_results = []
    lasso_selected_features: Dict[str, pd.Series] = {}

    if verbose:
        print("=" * 80)
        print("LASSO FEATURE SELECTION - IDENTIFYING IMPORTANT FACTORS")
        print("=" * 80)
        print("\nPerforming LASSO with 5-fold cross-validation to select optimal regularization...")

    for sector in df_final_cleaned[sector_col].unique():
        sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()

        if verbose:
            print(f"\n{'='*80}")
            print(f"Sector: {sector} (n={len(sector_df)})")
            print(f"{'='*80}")

        for pd_col in pd_maturity_cols:
            try:
                sector_df['logit_pd'] = _calculate_logit(sector_df[pd_col])
                sector_df['logit_pd_zero'] = _calculate_logit(sector_df[pdzero_col])
                sector_df['delta_logit'] = sector_df['logit_pd'] - sector_df['logit_pd_zero']

                y = sector_df['delta_logit']
                X = pd.concat([sector_df[macro_cols], sector_df[gpr_cols]], axis=1)

                valid_idx = ~(y.isna() | X.isna().any(axis=1))
                y = y[valid_idx]
                X = X[valid_idx]

                if len(y) < min_obs:
                    if verbose:
                        print(f"  Skipping {pd_col}: insufficient data (n={len(y)})")
                    continue

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                lasso_cv = LassoCV(
                    cv=cv,
                    random_state=random_state,
                    n_alphas=n_alphas,
                    max_iter=10000,
                    tol=0.0001,
                )
                lasso_cv.fit(X_scaled, y)

                optimal_alpha = lasso_cv.alpha_
                lasso = Lasso(alpha=optimal_alpha, max_iter=10000)
                lasso.fit(X_scaled, y)

                coefficients = pd.Series(lasso.coef_, index=X.columns)
                selected = coefficients[coefficients != 0]
                n_selected = len(selected)

                r2 = r2_score(y, lasso.predict(X_scaled))

                result = {
                    'Sector': sector,
                    'PD_Horizon': pd_col,
                    'N_observations': len(y),
                    'Optimal_Alpha': optimal_alpha,
                    'R_squared': r2,
                    'N_features_selected': n_selected,
                    'Intercept': lasso.intercept_,
                }

                for col in macro_cols:
                    result[f'LASSO_β_{col}'] = coefficients[col]
                    result[f'β_selected_{col}'] = 1 if coefficients[col] != 0 else 0

                for col in gpr_cols:
                    result[f'LASSO_δ_{col}'] = coefficients[col]
                    result[f'δ_selected_{col}'] = 1 if coefficients[col] != 0 else 0

                lasso_results.append(result)
                lasso_selected_features[sector] = selected

                if verbose:
                    print(f"\n{pd_col}:")
                    print(f"  Optimal alpha: {optimal_alpha:.6f}")
                    print(f"  R²: {r2:.3f}")
                    print(f"  Features selected: {n_selected}/{len(X.columns)}")

                    if n_selected > 0:
                        print("\n  Selected features (non-zero coefficients):")
                        for feat, coef in selected.items():
                            print(f"    {feat:25s}: {coef:8.4f}")
                    else:
                        print("  No features selected (all coefficients shrunk to zero)")

                    dropped = coefficients[coefficients == 0]
                    if len(dropped) > 0:
                        print("\n  Dropped features (zero coefficients):")
                        for feat in dropped.index:
                            print(f"    {feat:25s}: 0.0000")

            except Exception as exc:  # noqa: BLE001 - parity with notebook output
                if verbose:
                    print(f"  ✗ Could not fit LASSO model for {pd_col}: {exc}")

    return pd.DataFrame(lasso_results), lasso_selected_features


def build_feature_frequency(
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
) -> pd.DataFrame:
    """Create a feature selection frequency table."""
    feature_selection_count = {}
    for col in macro_cols:
        feature_selection_count[f'β_{col}'] = df_lasso[f'β_selected_{col}'].sum()
    for col in gpr_cols:
        feature_selection_count[f'δ_{col}'] = df_lasso[f'δ_selected_{col}'].sum()

    feature_freq_df = pd.DataFrame.from_dict(
        feature_selection_count,
        orient='index',
        columns=['Times Selected'],
    )
    feature_freq_df['Selection Rate'] = (
        feature_freq_df['Times Selected'] / len(df_lasso) * 100
    ).round(1)
    feature_freq_df = feature_freq_df.sort_values('Times Selected', ascending=False)
    return feature_freq_df


def print_lasso_summary(
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
) -> pd.DataFrame:
    """Print LASSO summary statistics and return feature frequency table."""
    print("\n" + "=" * 80)
    print("LASSO RESULTS SUMMARY")
    print("=" * 80)
    print(df_lasso[
        ['Sector', 'PD_Horizon', 'N_observations', 'Optimal_Alpha', 'R_squared', 'N_features_selected']
    ])

    print("\n" + "=" * 80)
    print("FEATURE SELECTION STATISTICS")
    print("=" * 80)
    print(f"\nTotal sectors analyzed: {len(df_lasso)}")
    print(f"Average features selected: {df_lasso['N_features_selected'].mean():.1f} / {len(macro_cols) + len(gpr_cols)}")
    print(f"Min features selected: {df_lasso['N_features_selected'].min()}")
    print(f"Max features selected: {df_lasso['N_features_selected'].max()}")

    feature_freq_df = build_feature_frequency(df_lasso, macro_cols, gpr_cols)
    print("\n" + "=" * 80)
    print("FEATURE SELECTION FREQUENCY (across all sectors)")
    print("=" * 80)
    print(feature_freq_df.to_string())
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("Features selected in >50% of sectors are likely important across the portfolio")
    print("Features rarely selected may be sector-specific or less relevant for PD prediction")
    return feature_freq_df


def compare_ols_lasso(
    df_sensitivities: pd.DataFrame,
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
) -> pd.DataFrame:
    """Print detailed OLS vs LASSO comparison and return merged dataframe."""
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON: OLS vs LASSO COEFFICIENTS")
    print("=" * 80)

    comparison_full = df_sensitivities.merge(df_lasso, on='Sector', suffixes=('_ols', '_lasso'))

    print("\nComparison for selected sectors:")
    print("\nNote: LASSO coefficients are on standardized scale (mean=0, std=1)")
    print("Zero LASSO coefficients indicate features dropped by regularization\n")

    for sector in comparison_full['Sector'].head(5):
        sector_data = comparison_full[comparison_full['Sector'] == sector].iloc[0]

        print(f"\n{'='*80}")
        print(f"Sector: {sector}")
        print(f"R² - OLS: {sector_data['R_squared_ols']:.3f} | LASSO: {sector_data['R_squared_lasso']:.3f}")
        print(f"Features selected by LASSO: {int(sector_data['N_features_selected'])}/{len(macro_cols) + len(gpr_cols)}")
        print(f"{'='*80}")

        print("\nMACRO VARIABLES (β):")
        print(f"{'Variable':<25} {'OLS Coef':>12} {'LASSO Coef':>12} {'Selected':>10}")
        print("-" * 80)
        for col in macro_cols:
            ols_coef = sector_data[f'β_{col}']
            lasso_coef = sector_data[f'LASSO_β_{col}']
            selected = '✓' if sector_data[f'β_selected_{col}'] == 1 else '✗'
            print(f"{col:<25} {ols_coef:>12.4f} {lasso_coef:>12.4f} {selected:>10}")

        print("\nGPR VARIABLES (δ):")
        print(f"{'Variable':<25} {'OLS Coef':>12} {'LASSO Coef':>12} {'Selected':>10}")
        print("-" * 80)
        for col in gpr_cols:
            ols_coef = sector_data[f'δ_{col}']
            lasso_coef = sector_data[f'LASSO_δ_{col}']
            selected = '✓' if sector_data[f'δ_selected_{col}'] == 1 else '✗'
            print(f"{col:<25} {ols_coef:>12.4f} {lasso_coef:>12.4f} {selected:>10}")

    return comparison_full


def print_feature_recommendations(
    feature_freq_df: pd.DataFrame,
    comparison_full: pd.DataFrame,
    selection_threshold_high: float = 0.7,
    selection_threshold_medium: float = 0.4,
) -> None:
    """Print feature selection recommendations based on frequency."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR FEATURE SELECTION")
    print("=" * 80)

    print("\n1. STRONGLY RECOMMENDED FEATURES (selected in >70% of sectors):")
    print("-" * 80)
    strong_features = feature_freq_df[feature_freq_df['Selection Rate'] > selection_threshold_high * 100]
    if len(strong_features) > 0:
        for feat, row in strong_features.iterrows():
            print(f"  {feat:<30} - Selected in {row['Selection Rate']:.0f}% of sectors")
    else:
        print("  None - no features meet the 70% threshold")

    print("\n2. MODERATELY RECOMMENDED FEATURES (selected in 40-70% of sectors):")
    print("-" * 80)
    medium_features = feature_freq_df[
        (feature_freq_df['Selection Rate'] > selection_threshold_medium * 100)
        & (feature_freq_df['Selection Rate'] <= selection_threshold_high * 100)
    ]
    if len(medium_features) > 0:
        for feat, row in medium_features.iterrows():
            print(f"  {feat:<30} - Selected in {row['Selection Rate']:.0f}% of sectors")
    else:
        print("  None")

    print("\n3. WEAK/SECTOR-SPECIFIC FEATURES (selected in <40% of sectors):")
    print("-" * 80)
    weak_features = feature_freq_df[feature_freq_df['Selection Rate'] <= selection_threshold_medium * 100]
    if len(weak_features) > 0:
        for feat, row in weak_features.iterrows():
            print(f"  {feat:<30} - Selected in {row['Selection Rate']:.0f}% of sectors")
            print("     → May be useful for specific sectors only")
    else:
        print("  None")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal features analyzed: {len(feature_freq_df)}")
    print(f"Strongly recommended: {len(strong_features)}")
    print(f"Moderately recommended: {len(medium_features)}")
    print(f"Weak/sector-specific: {len(weak_features)}")

    avg_r2_ols = comparison_full['R_squared_ols'].mean()
    avg_r2_lasso = comparison_full['R_squared_lasso'].mean()
    print(f"\nAverage R² - OLS: {avg_r2_ols:.3f}")
    print(f"Average R² - LASSO: {avg_r2_lasso:.3f}")
    print(f"R² difference: {avg_r2_ols - avg_r2_lasso:.3f}")

    if avg_r2_lasso >= avg_r2_ols * 0.95:
        print("\n✓ LASSO achieves similar performance with fewer features (good for parsimony)")
    elif avg_r2_lasso >= avg_r2_ols * 0.85:
        print("\n⚠ LASSO has moderately lower R² - consider using selected features from LASSO")
        print("  but validate against domain knowledge")
    else:
        print("\n✗ LASSO shows significant performance drop - may be too restrictive")
        print("  Consider using all features or a less aggressive regularization")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review the feature selection frequency table above")
    print("2. Consider using only the 'strongly recommended' features in your sensitivity model")
    print("3. For sector-specific models, review which features are selected per sector")
    print("4. Validate selected features against economic theory and domain expertise")
    print("5. Re-run sensitivity analysis with selected features only (optional)")
    print("=" * 80)


def export_lasso_outputs(
    df_lasso: pd.DataFrame,
    comparison_full: pd.DataFrame,
    lasso_output_file: str = 'lasso_feature_selection_results.csv',
    comparison_output_file: str = 'ols_vs_lasso_comparison.csv',
) -> None:
    """Export LASSO results and comparison to CSV files."""
    df_lasso.to_csv(lasso_output_file, index=False)
    print(f"\n✓ LASSO feature selection results exported to: {lasso_output_file}")
    comparison_full.to_csv(comparison_output_file, index=False)
    print(f"✓ OLS vs LASSO comparison exported to: {comparison_output_file}")


def _calculate_logit(p: np.ndarray | pd.Series) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
