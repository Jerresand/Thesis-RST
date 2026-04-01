"""Elastic-Net feature selection utilities (supersedes pure-LASSO approach)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, enet_path
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# l1_ratio grid: 1.0 = pure LASSO, lower values blend in Ridge to handle correlated lags
_DEFAULT_L1_RATIOS: List[float] = [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]


def run_lasso_feature_selection(
    df_final_cleaned: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    sector_col: str,
    pd_maturity_cols: Iterable[str],
    pdzero_col: str = 'PDzero',
    min_obs: int = 100,
    cv: int = 5,
    random_state: int = 42,
    n_alphas: int = 100,
    l1_ratios: Optional[List[float]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Elastic-Net with cross-validation for each sector and PD horizon.

    Uses ElasticNetCV to jointly search over alpha (overall penalty) and
    l1_ratio (L1 vs L2 mix). l1_ratio=1.0 is pure LASSO; lower values blend
    in Ridge, which handles correlated lag variables more robustly.

    New columns versus pure-LASSO version
    --------------------------------------
    Optimal_L1_Ratio : optimal l1_ratio chosen by CV
    R_squared_cv     : mean cross-validated R² at optimal hyperparameters
    """
    if l1_ratios is None:
        l1_ratios = _DEFAULT_L1_RATIOS

    lasso_results = []
    lasso_selected_features: Dict[str, pd.Series] = {}

    if verbose:
        print("=" * 80)
        print("ELASTIC-NET FEATURE SELECTION")
        print("=" * 80)
        print(f"\nSearching over alpha × l1_ratio grid ({cv}-fold CV).")
        print(f"l1_ratio grid: {l1_ratios}")
        print("l1_ratio=1.0 → pure LASSO  |  <1.0 → Elastic-Net (handles correlated lags)")

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

                enet_cv = ElasticNetCV(
                    l1_ratio=l1_ratios,
                    cv=cv,
                    random_state=random_state,
                    n_alphas=n_alphas,
                    max_iter=1_000_000,
                )
                enet_cv.fit(X_scaled, y)

                optimal_alpha = enet_cv.alpha_
                optimal_l1_ratio = float(enet_cv.l1_ratio_)

                model = ElasticNet(
                    alpha=optimal_alpha,
                    l1_ratio=optimal_l1_ratio,
                    max_iter=100_000,
                )
                model.fit(X_scaled, y)

                # Cross-validated R² at optimal hyperparameters
                cv_scores = cross_val_score(
                    ElasticNet(alpha=optimal_alpha, l1_ratio=optimal_l1_ratio, max_iter=100_000),
                    X_scaled, y, cv=cv, scoring='r2',
                )
                r2_cv = float(cv_scores.mean())

                coefficients = pd.Series(model.coef_, index=X.columns)
                selected = coefficients[coefficients != 0]
                n_selected = len(selected)
                scale_safe = pd.Series(scaler.scale_, index=X.columns).replace(0.0, np.nan)

                r2 = r2_score(y, model.predict(X_scaled))
                n_y = len(y)
                if n_y > n_selected + 1:
                    r2_adj = 1.0 - (1.0 - r2) * (n_y - 1) / (n_y - n_selected - 1)
                else:
                    r2_adj = float('nan')

                result = {
                    'Sector': sector,
                    'PD_Horizon': pd_col,
                    'N_observations': n_y,
                    'Optimal_Alpha': optimal_alpha,
                    'Optimal_L1_Ratio': optimal_l1_ratio,
                    'R_squared': r2,
                    'R_squared_adj': r2_adj,
                    'R_squared_cv': r2_cv,
                    'N_features_selected': n_selected,
                    'Intercept': model.intercept_,
                }

                for col in macro_cols:
                    g = float(coefficients[col])
                    result[f'LASSO_β_{col}'] = g
                    result[f'β_selected_{col}'] = 1 if g != 0.0 else 0
                    sc = scale_safe[col]
                    result[f'LASSO_NATIVE_β_{col}'] = float(g / sc) if pd.notna(sc) else float('nan')

                for col in gpr_cols:
                    g = float(coefficients[col])
                    result[f'LASSO_δ_{col}'] = g
                    result[f'δ_selected_{col}'] = 1 if g != 0.0 else 0
                    sc = scale_safe[col]
                    result[f'LASSO_NATIVE_δ_{col}'] = float(g / sc) if pd.notna(sc) else float('nan')

                lasso_results.append(result)
                lasso_selected_features[sector] = selected

                if verbose:
                    print(f"\n{pd_col}:")
                    print(f"  α={optimal_alpha:.6f}  l1_ratio={optimal_l1_ratio:.2f}")
                    print(f"  R²={r2:.3f}  R²_adj={r2_adj:.3f}  R²_cv={r2_cv:.3f}")
                    print(f"  Features selected: {n_selected}/{len(X.columns)}")
                    if n_selected > 0:
                        print("  Selected features:")
                        for feat, coef in selected.items():
                            print(f"    {feat:30s}: {coef:8.4f}")
                    else:
                        print("  No features selected (all shrunk to zero)")

            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  Could not fit model for {pd_col}: {exc}")

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
    print("ELASTIC-NET RESULTS SUMMARY")
    print("=" * 80)
    summary_cols = ['Sector', 'PD_Horizon', 'N_observations',
                    'Optimal_Alpha', 'Optimal_L1_Ratio',
                    'R_squared', 'R_squared_adj', 'R_squared_cv',
                    'N_features_selected']
    print(df_lasso[[c for c in summary_cols if c in df_lasso.columns]])

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

    merge_keys = (
        ['Sector', 'PD_Horizon']
        if 'PD_Horizon' in df_sensitivities.columns and 'PD_Horizon' in df_lasso.columns
        else ['Sector']
    )
    comparison_full = df_sensitivities.merge(df_lasso, on=merge_keys, suffixes=('_ols', '_lasso'))

    print("\nComparison for selected sectors:")
    print("\nNote: LASSO_β_* / LASSO_δ_* are on standardized X (mean=0, std=1).")
    print("LASSO_NATIVE_* columns match OLS units (Δ logit PD per one unit of X).")
    print("Zero LASSO coefficients indicate features dropped by regularization.\n")

    for sector in comparison_full['Sector'].head(5):
        sector_data = comparison_full[comparison_full['Sector'] == sector].iloc[0]

        print(f"\n{'='*80}")
        print(f"Sector: {sector}")
        ols_r2 = sector_data['R_squared_ols']
        la_r2 = sector_data['R_squared_lasso']
        line = f"R² — OLS: {ols_r2:.3f} | EN: {la_r2:.3f}"
        if 'R_squared_adj_ols' in sector_data.index and 'R_squared_adj_lasso' in sector_data.index:
            line += (
                f"  |  R²_adj — OLS: {sector_data['R_squared_adj_ols']:.3f} | "
                f"EN: {sector_data['R_squared_adj_lasso']:.3f}"
            )
        if 'R_squared_cv_lasso' in sector_data.index:
            line += f"  |  R²_cv (EN): {sector_data['R_squared_cv_lasso']:.3f}"
        print(line)
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


def get_lasso_selected_cols(
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
) -> Tuple[List[str], List[str]]:
    """Return the union of LASSO-selected features across all sectors.

    A feature is included if LASSO selected it (non-zero coefficient) in at
    least one sector.  The returned lists preserve the original column order.

    Returns:
        (macro_selected, gpr_selected)
    """
    macro_selected = [
        col for col in macro_cols
        if f'β_selected_{col}' in df_lasso.columns and df_lasso[f'β_selected_{col}'].sum() > 0
    ]
    gpr_selected = [
        col for col in gpr_cols
        if f'δ_selected_{col}' in df_lasso.columns and df_lasso[f'δ_selected_{col}'].sum() > 0
    ]
    return macro_selected, gpr_selected


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


def run_bootstrap_stability(
    df_final_cleaned: pd.DataFrame,
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    sector_col: str,
    pd_maturity_cols: Iterable[str],
    pdzero_col: str = 'PDzero',
    n_bootstrap: int = 200,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Bootstrap selection stability for each feature × sector.

    Refits ElasticNet (with the optimal alpha and l1_ratio already stored in
    *df_lasso*) on *n_bootstrap* bootstrap samples. The stability score for a
    feature in a sector is the fraction of bootstrap samples in which that
    feature received a non-zero coefficient.

    Adds columns
    ------------
    β_stability_{col}  and  δ_stability_{col}  ∈ [0, 1]
        1.0 = selected in every bootstrap sample (maximally stable)
        0.0 = never selected

    Returns the augmented df_lasso (original is not mutated).
    """
    rng = np.random.default_rng(random_state)
    df_out = df_lasso.copy()
    pd_maturity_cols = list(pd_maturity_cols)
    all_feature_cols = list(macro_cols) + list(gpr_cols)
    prefixes = ['β_'] * len(macro_cols) + ['δ_'] * len(gpr_cols)

    for col, prefix in zip(all_feature_cols, prefixes):
        df_out[f'{prefix}stability_{col}'] = float('nan')

    if verbose:
        print("=" * 80)
        print(f"BOOTSTRAP STABILITY  (B={n_bootstrap} per sector)")
        print("=" * 80)

    for row_idx, lasso_row in df_out.iterrows():
        sector = lasso_row['Sector']
        pd_col = lasso_row.get('PD_Horizon', pd_maturity_cols[0])
        alpha = lasso_row['Optimal_Alpha']
        l1_ratio = float(lasso_row.get('Optimal_L1_Ratio', 1.0))

        sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()
        try:
            sector_df['logit_pd'] = _calculate_logit(sector_df[pd_col])
            sector_df['logit_pd_zero'] = _calculate_logit(sector_df[pdzero_col])
            sector_df['delta_logit'] = sector_df['logit_pd'] - sector_df['logit_pd_zero']

            y_s = sector_df['delta_logit']
            X_s = pd.concat([sector_df[macro_cols], sector_df[gpr_cols]], axis=1)
            valid = ~(y_s.isna() | X_s.isna().any(axis=1))
            y_s = y_s[valid].values
            X_s = X_s[valid].values

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_s)
            n = len(y_s)

            counts = np.zeros(len(all_feature_cols))
            for _ in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100_000)
                try:
                    m.fit(X_sc[idx], y_s[idx])
                    counts += (m.coef_ != 0).astype(float)
                except Exception:
                    pass

            stability = counts / n_bootstrap
            for j, (col, prefix) in enumerate(zip(all_feature_cols, prefixes)):
                df_out.loc[row_idx, f'{prefix}stability_{col}'] = stability[j]

            if verbose:
                stable = [all_feature_cols[j] for j in range(len(all_feature_cols)) if stability[j] >= 0.5]
                print(f"  {sector}: {len(stable)}/{len(all_feature_cols)} features stable (≥50 %)")

        except Exception as exc:
            if verbose:
                print(f"  {sector}: bootstrap failed — {exc}")

    return df_out


def compute_regularization_paths(
    df_final_cleaned: pd.DataFrame,
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    sector_col: str,
    pd_maturity_cols: Iterable[str],
    pdzero_col: str = 'PDzero',
    n_alphas_path: int = 60,
    verbose: bool = True,
) -> dict:
    """Compute ElasticNet regularization paths for every sector.

    Returns
    -------
    dict  keyed by sector name, each value::

        {
          'alphas'       : 1-D array of alpha values (decreasing),
          'coefs'        : 2-D array (n_features, n_alphas),
          'feature_names': list of str,
          'optimal_alpha': float,
          'l1_ratio'     : float,
        }
    """
    pd_maturity_cols = list(pd_maturity_cols)
    paths: dict = {}

    for _, lasso_row in df_lasso.iterrows():
        sector = lasso_row['Sector']
        pd_col = lasso_row.get('PD_Horizon', pd_maturity_cols[0])
        l1_ratio = float(lasso_row.get('Optimal_L1_Ratio', 1.0))
        opt_alpha = lasso_row['Optimal_Alpha']

        sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()
        try:
            sector_df['logit_pd'] = _calculate_logit(sector_df[pd_col])
            sector_df['logit_pd_zero'] = _calculate_logit(sector_df[pdzero_col])
            sector_df['delta_logit'] = sector_df['logit_pd'] - sector_df['logit_pd_zero']

            y_s = sector_df['delta_logit']
            X_s = pd.concat([sector_df[macro_cols], sector_df[gpr_cols]], axis=1)
            valid = ~(y_s.isna() | X_s.isna().any(axis=1))
            y_s = y_s[valid]
            X_s = X_s[valid]

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X_s)

            alphas, coefs, _ = enet_path(
                X_sc, y_s.values,
                l1_ratio=l1_ratio,
                n_alphas=n_alphas_path,
                fit_intercept=True,
            )
            paths[sector] = {
                'alphas': alphas,
                'coefs': coefs,
                'feature_names': list(X_s.columns),
                'optimal_alpha': opt_alpha,
                'l1_ratio': l1_ratio,
            }
        except Exception as exc:
            if verbose:
                print(f"  Path failed for {sector}: {exc}")

    if verbose:
        print(f"Regularization paths computed for {len(paths)} sectors.")
    return paths


def _calculate_logit(p: np.ndarray | pd.Series) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
