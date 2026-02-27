"""Plotting utilities for the PD analysis pipeline."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_lasso_summary(
    df_lasso: pd.DataFrame,
    feature_freq_df: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    df_sensitivities: pd.DataFrame,
) -> None:
    """Visualize LASSO feature selection and compare with OLS."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('LASSO Feature Selection Analysis', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    colors = ['steelblue' if 'β' in idx else 'coral' for idx in feature_freq_df.index]
    bars = ax1.barh(feature_freq_df.index, feature_freq_df['Times Selected'], color=colors, edgecolor='black')
    ax1.axvline(len(df_lasso) / 2, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax1.set_xlabel('Number of Sectors', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Selection Frequency Across Sectors', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    for i, (idx, row) in enumerate(feature_freq_df.iterrows()):
        ax1.text(row['Times Selected'] + 0.2, i, f"{row['Selection Rate']:.0f}%", va='center', fontsize=9)

    ax2 = axes[0, 1]
    sector_names = [s[:20] for s in df_lasso['Sector']]
    colors_sector = plt.cm.viridis(np.linspace(0, 1, len(df_lasso)))
    ax2.barh(sector_names, df_lasso['N_features_selected'], color=colors_sector, edgecolor='black')
    ax2.axvline(len(macro_cols) + len(gpr_cols), color='red', linestyle='--', linewidth=2, label='All features')
    ax2.set_xlabel('Number of Features Selected', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sector', fontsize=12, fontweight='bold')
    ax2.set_title('Features Selected by Sector', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    ax3 = axes[1, 0]
    selection_matrix = []
    feature_names = [f'β_{col}' for col in macro_cols] + [f'δ_{col}' for col in gpr_cols]

    for _, row in df_lasso.iterrows():
        sector_selection = [row[f'β_selected_{col}'] for col in macro_cols]
        sector_selection += [row[f'δ_selected_{col}'] for col in gpr_cols]
        selection_matrix.append(sector_selection)

    selection_df = pd.DataFrame(
        selection_matrix,
        index=[s[:20] for s in df_lasso['Sector']],
        columns=feature_names,
    )

    sns.heatmap(
        selection_df,
        cmap='RdYlGn',
        cbar_kws={'label': 'Selected'},
        annot=False,
        linewidths=0.5,
        ax=ax3,
        vmin=0,
        vmax=1,
    )
    ax3.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sector', fontsize=12, fontweight='bold')
    ax3.set_title('Feature Selection Heatmap (Green = Selected, Red = Dropped)', fontsize=13, fontweight='bold')
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    ax4 = axes[1, 1]
    comparison_df = df_lasso.merge(
        df_sensitivities[['Sector', 'R_squared']],
        on='Sector',
        suffixes=('_lasso', '_ols'),
    )
    ax4.scatter(
        comparison_df['R_squared_ols'],
        comparison_df['R_squared_lasso'],
        s=100,
        alpha=0.7,
        edgecolors='black',
        c=comparison_df['N_features_selected'],
        cmap='viridis',
    )
    max_r2 = max(comparison_df['R_squared_ols'].max(), comparison_df['R_squared_lasso'].max())
    ax4.plot([0, max_r2], [0, max_r2], 'r--', linewidth=2, label='y=x (same R²)')
    ax4.set_xlabel('OLS R²', fontsize=12, fontweight='bold')
    ax4.set_ylabel('LASSO R²', fontsize=12, fontweight='bold')
    ax4.set_title('Model Performance: LASSO vs OLS', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Features Selected', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_scenario_loss(results: dict) -> None:
    """Plot scenario-based portfolio loss distribution and percentiles."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scenario-Based Portfolio Loss Analysis', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    ax1.hist(results['portfolio_losses'], bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(results['statistics']['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean: {results['statistics']['mean']:,.0f} SEK")
    ax1.axvline(results['statistics']['median'], color='green', linestyle='--', linewidth=2,
               label=f"Median: {results['statistics']['median']:,.0f} SEK")
    ax1.axvline(results['statistics']['var_999'], color='orange', linestyle='--', linewidth=2,
               label=f"VaR 99.9%: {results['statistics']['var_999']:,.0f} SEK")
    ax1.set_xlabel('Portfolio Loss (SEK)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Portfolio Losses Across Scenarios', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    sorted_losses = np.sort(results['portfolio_losses'])
    cumulative = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses) * 100
    ax2.plot(sorted_losses, cumulative, linewidth=2, color='darkblue')
    ax2.axhline(99.9, color='orange', linestyle='--', linewidth=2, label='99.9th Percentile')
    ax2.axvline(results['statistics']['var_999'], color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('Portfolio Loss (SEK)', fontsize=12)
    ax2.set_ylabel('Cumulative Probability (%)', fontsize=12)
    ax2.set_title('Cumulative Distribution of Portfolio Losses', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    scenario_data = [
        results['scenarios']['GDP_Growth'],
        results['scenarios']['Interest_Rate'],
        results['scenarios']['Unemployment_Rate'],
    ]
    ax3.boxplot(scenario_data, labels=['GDP Growth', 'Interest Rate', 'Unemployment'])
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Distribution of Key Macro Variables Across Scenarios', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = axes[1, 1]
    percentile_labels = ['5%', '25%', '50%', '75%', '95%', '99%', '99.9%']
    percentile_values = [results['statistics']['percentiles'][p] for p in [5, 25, 50, 75, 95, 99, 99.9]]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(percentile_values)))
    bars = ax4.bar(percentile_labels, percentile_values, color=colors, edgecolor='black')
    ax4.set_xlabel('Percentile', fontsize=12)
    ax4.set_ylabel('Portfolio Loss (SEK)', fontsize=12)
    ax4.set_title('Portfolio Loss by Percentile', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=45)

    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height/1e6:.1f}M',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_portfolio_loss_breakdown(results: dict) -> None:
    """Plot per-exposure loss distributions and concentration metrics."""
    sns.set_style("whitegrid")

    df_results = results['results_df']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax1 = axes[0, 0]
    ax1.hist(df_results['individual_loss'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(results['mean_loss'], color='red', linestyle='--', linewidth=2,
                label=f"Mean: {results['mean_loss']:,.0f} SEK")
    ax1.axvline(results['median_loss'], color='orange', linestyle='--', linewidth=2,
                label=f"Median: {results['median_loss']:,.0f} SEK")
    ax1.set_xlabel('Individual Loss (SEK)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Individual Losses', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.hist(df_results['conditional_pd'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(df_results['conditional_pd'].mean(), color='red', linestyle='--', linewidth=2,
                label=f"Mean: {df_results['conditional_pd'].mean():.4f}")
    ax2.set_xlabel('Conditional PD', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Conditional Probabilities of Default', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        df_results['12_month'],
        df_results['individual_loss'],
        c=df_results['12_month_correlation'],
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5,
    )
    ax3.set_xlabel('Original PD (12-month)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Individual Loss (SEK)', fontsize=12, fontweight='bold')
    ax3.set_title('PD vs Individual Loss (colored by correlation ρ)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Correlation (ρ)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    sorted_losses = df_results['individual_loss'].sort_values(ascending=False).reset_index(drop=True)
    cumulative_loss = sorted_losses.cumsum()
    cumulative_pct = (cumulative_loss / cumulative_loss.iloc[-1]) * 100
    ax4.plot(range(len(cumulative_pct)), cumulative_pct, linewidth=2.5, color='darkgreen')
    ax4.axhline(80, color='red', linestyle='--', linewidth=2, label='80% of total loss')
    ax4.axhline(95, color='orange', linestyle='--', linewidth=2, label='95% of total loss')
    ax4.set_xlabel('Number of Exposures (ranked by loss)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Loss (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Cumulative Loss Distribution', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_sector_regressions(
    df_sensitivities: pd.DataFrame,
    df_final_cleaned: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    sector_col: str,
) -> None:
    """Plot regression lines for each sector and variable."""
    sectors_to_plot = df_sensitivities['Sector'].unique()

    for sector in sectors_to_plot:
        sector_sens = df_sensitivities[df_sensitivities['Sector'] == sector].iloc[0]
        sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()
        sector_df['logit_pd'] = _calculate_logit(sector_df['12_month'])
        sector_df['logit_pd_zero'] = _calculate_logit(sector_df['PDzero'])
        sector_df['delta_logit'] = sector_df['logit_pd'] - sector_df['logit_pd_zero']

        n_plots = len(macro_cols) + len(gpr_cols)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        fig.suptitle(
            f'Sensitivity Analysis for {sector} (N={len(sector_df)}, R²={sector_sens["R_squared"]:.3f})',
            fontsize=14,
            fontweight='bold',
        )

        for idx, macro_var in enumerate(macro_cols):
            ax = axes[idx]
            X_data = sector_df[macro_var].values
            y_data = sector_df['delta_logit'].values
            valid_mask = ~(np.isnan(X_data) | np.isnan(y_data))
            X_data = X_data[valid_mask]
            y_data = y_data[valid_mask]

            ax.scatter(X_data, y_data, alpha=0.3, s=10, label='Data points')

            intercept = sector_sens['Intercept']
            coef = sector_sens[f'β_{macro_var}']

            other_vars_contribution = 0
            for other_var in macro_cols:
                if other_var != macro_var:
                    mean_val = sector_df[other_var].mean()
                    other_vars_contribution += sector_sens[f'β_{other_var}'] * mean_val
            for gpr_var in gpr_cols:
                mean_val = sector_df[gpr_var].mean()
                other_vars_contribution += sector_sens[f'δ_{gpr_var}'] * mean_val

            X_range = np.linspace(X_data.min(), X_data.max(), 100)
            y_pred = intercept + coef * X_range + other_vars_contribution

            ax.plot(X_range, y_pred, 'r-', linewidth=2, label=f'β={coef:.4f}')
            ax.set_xlabel(macro_var, fontsize=11)
            ax.set_ylabel('Δ logit(PD)', fontsize=11)
            ax.set_title(f'{macro_var} (β)', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        for idx, gpr_var in enumerate(gpr_cols):
            ax = axes[len(macro_cols) + idx]

            X_data = sector_df[gpr_var].values
            y_data = sector_df['delta_logit'].values
            valid_mask = ~(np.isnan(X_data) | np.isnan(y_data))
            X_data = X_data[valid_mask]
            y_data = y_data[valid_mask]

            ax.scatter(X_data, y_data, alpha=0.3, s=10, label='Data points', color='green')

            intercept = sector_sens['Intercept']
            coef = sector_sens[f'δ_{gpr_var}']

            other_vars_contribution = 0
            for macro_var in macro_cols:
                mean_val = sector_df[macro_var].mean()
                other_vars_contribution += sector_sens[f'β_{macro_var}'] * mean_val
            for other_gpr_var in gpr_cols:
                if other_gpr_var != gpr_var:
                    mean_val = sector_df[other_gpr_var].mean()
                    other_vars_contribution += sector_sens[f'δ_{other_gpr_var}'] * mean_val

            X_range = np.linspace(X_data.min(), X_data.max(), 100)
            y_pred = intercept + coef * X_range + other_vars_contribution

            ax.plot(X_range, y_pred, 'darkgreen', linewidth=2, label=f'δ={coef:.4f}')
            ax.set_xlabel(gpr_var, fontsize=11)
            ax.set_ylabel('Δ logit(PD)', fontsize=11)
            ax.set_title(f'{gpr_var} (δ)', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        print(f"\nCompleted plot for {sector}")


def plot_sector_comparison(
    df_sensitivities: pd.DataFrame,
    df_final_cleaned: pd.DataFrame,
    macro_cols: List[str],
    sector_col: str,
    title: str = 'Regression Lines for All Sectors - Macro Variables',
    ylabel: str = 'Δ logit(PD)',
) -> None:
    """Plot regression lines for all sectors on shared macro variable panels."""
    sectors_to_plot = df_sensitivities['Sector'].unique()

    fig, axes = plt.subplots(1, len(macro_cols), figsize=(6 * len(macro_cols), 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    colors = plt.cm.tab20(np.linspace(0, 1, len(sectors_to_plot)))

    for idx, macro_var in enumerate(macro_cols):
        ax = axes[idx]
        for sector_idx, sector in enumerate(sectors_to_plot):
            sector_sens = df_sensitivities[df_sensitivities['Sector'] == sector].iloc[0]
            sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()
            sector_df['logit_pd'] = _calculate_logit(sector_df['12_month'])

            X_data = sector_df[macro_var].values
            valid_mask = ~np.isnan(X_data)
            X_data = X_data[valid_mask]

            if len(X_data) == 0:
                continue

            intercept = sector_sens['Intercept']
            coef = sector_sens[f'β_{macro_var}']

            other_vars_contribution = 0
            for other_var in macro_cols:
                if other_var != macro_var:
                    mean_val = sector_df[other_var].mean()
                    other_vars_contribution += sector_sens[f'β_{other_var}'] * mean_val

            X_range = np.linspace(X_data.min(), X_data.max(), 100)
            y_pred = intercept + coef * X_range + other_vars_contribution

            ax.plot(X_range, y_pred, linewidth=2, color=colors[sector_idx],
                    label=f'{sector[:20]} (β={coef:.3f})', alpha=0.7)

        ax.set_xlabel(macro_var, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{macro_var}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    print("\nCompleted comparison plot for all sectors")


def _calculate_logit(p: np.ndarray | pd.Series) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
