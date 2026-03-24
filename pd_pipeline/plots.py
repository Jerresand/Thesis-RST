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
    min_sector_obs: int = 1000,
    top_n_sectors: int = 20,
) -> None:
    """Visualize LASSO feature selection and compare with OLS."""
    lasso_sector_obs = df_lasso.groupby('Sector')['N_observations'].max()
    sens_sector_obs = df_sensitivities.groupby('Sector')['N_observations'].max()
    eligible_sectors = lasso_sector_obs[lasso_sector_obs >= min_sector_obs].index
    eligible_sectors = eligible_sectors.intersection(
        sens_sector_obs[sens_sector_obs >= min_sector_obs].index
    )
    if eligible_sectors.empty:
        print(f"No sectors with N_observations >= {min_sector_obs}.")
        return

    # Cap to top_n_sectors by observation count to keep plots readable
    top_sectors = (
        lasso_sector_obs[lasso_sector_obs.index.isin(eligible_sectors)]
        .nlargest(top_n_sectors)
        .index
    )
    df_lasso = df_lasso[df_lasso['Sector'].isin(top_sectors)].copy()
    df_sensitivities = df_sensitivities[df_sensitivities['Sector'].isin(top_sectors)].copy()
    if df_lasso.empty:
        print(f"No LASSO results with N_observations >= {min_sector_obs}.")
        return

    n_sectors = len(df_lasso)
    print(f"Plotting {n_sectors} sectors (top {top_n_sectors} by observation count)")

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

    # Dynamically size rows based on sector count
    row_height = max(0.35 * n_sectors, 6)
    fig_height = max(row_height + 6, 14)
    fig = plt.figure(figsize=(18, fig_height))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    fig.suptitle('LASSO Feature Selection Analysis', fontsize=16, fontweight='bold')

    colors = ['steelblue' if 'β' in idx else 'coral' for idx in feature_freq_df.index]
    ax1.barh(feature_freq_df.index, feature_freq_df['Times Selected'], color=colors, edgecolor='black')
    ax1.axvline(len(df_lasso) / 2, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax1.set_xlabel('Number of Sectors', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Selection Frequency Across Sectors', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')

    for i, (idx, row) in enumerate(feature_freq_df.iterrows()):
        ax1.text(row['Times Selected'] + 0.2, i, f"{row['Selection Rate']:.0f}%", va='center', fontsize=9)

    sector_names = [s[:25] for s in df_lasso['Sector']]
    colors_sector = plt.cm.viridis(np.linspace(0, 1, n_sectors))
    ax2.barh(sector_names, df_lasso['N_features_selected'], color=colors_sector, edgecolor='black')
    ax2.axvline(len(macro_cols) + len(gpr_cols), color='red', linestyle='--', linewidth=2, label='All features')
    ax2.set_xlabel('Number of Features Selected', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sector', fontsize=10, fontweight='bold')
    ax2.set_title('Features Selected by Sector', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    feature_names = [f'β_{col}' for col in macro_cols] + [f'δ_{col}' for col in gpr_cols]
    selection_matrix = []
    for _, row in df_lasso.iterrows():
        sector_selection = [row[f'β_selected_{col}'] for col in macro_cols]
        sector_selection += [row[f'δ_selected_{col}'] for col in gpr_cols]
        selection_matrix.append(sector_selection)

    selection_df = pd.DataFrame(
        selection_matrix,
        index=[s[:25] for s in df_lasso['Sector']],
        columns=feature_names,
    )

    sns.heatmap(
        selection_df,
        cmap='RdYlGn',
        cbar_kws={'label': 'Selected'},
        annot=True,
        fmt='d',
        linewidths=0.5,
        ax=ax3,
        vmin=0,
        vmax=1,
    )
    ax3.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sector', fontsize=10, fontweight='bold')
    ax3.set_title('Feature Selection Heatmap (Green = Selected, Red = Dropped)', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='y', labelsize=8)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    comparison_df = df_lasso.merge(
        df_sensitivities[['Sector', 'R_squared']],
        on='Sector',
        suffixes=('_lasso', '_ols'),
    )
    sc = ax4.scatter(
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

    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label('Features Selected', fontsize=10, fontweight='bold')

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


def _extract_all_betas(sector_sens: pd.Series) -> dict:
    """Extract all β_* and δ_* coefficients from a sensitivity result row."""
    betas = {}
    for col in sector_sens.index:
        if col.startswith('β_') and not col.endswith(('_CI_lower', '_CI_upper')):
            betas[col[2:]] = sector_sens[col]
        elif col.startswith('δ_') and not col.endswith(('_CI_lower', '_CI_upper')):
            betas[col[2:]] = sector_sens[col]
    return betas


def _compute_cpr(
    sector_df: pd.DataFrame,
    sector_sens: pd.Series,
    plot_var: str,
) -> tuple:
    """Return (X_k, cpr, coef) for a component-plus-residual plot.

    CPR residuals = OLS residuals + β_k * X_k.  Plotting CPR vs X_k and
    drawing the line  y = mean(CPR) + β_k * (X - mean(X))  makes the
    scatter and the fitted line directly comparable regardless of how many
    other regressors (including lags) are in the model.
    """
    betas = _extract_all_betas(sector_sens)
    intercept = sector_sens['Intercept']

    # Full OLS prediction using every variable that is both in betas and in df
    y_hat = pd.Series(float(intercept), index=sector_df.index)
    for var, coef in betas.items():
        if var in sector_df.columns:
            y_hat = y_hat + coef * sector_df[var]

    y = sector_df['delta_logit']
    residuals = y - y_hat

    X_k = sector_df[plot_var]
    coef_k = betas.get(plot_var, 0.0)
    cpr = residuals + coef_k * X_k

    valid = ~(cpr.isna() | X_k.isna())
    return X_k[valid].values, cpr[valid].values, coef_k


def plot_sector_regressions(
    df_sensitivities: pd.DataFrame,
    df_final_cleaned: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    sector_col: str,
    min_sector_obs: int = 1000,
) -> None:
    """Plot component-plus-residual (CPR) plots for each sector and variable.

    Uses partial-regression / CPR plots so the scatter and the fitted line
    are always on the same scale, even when the underlying OLS regression
    contains many additional regressors (e.g. lagged macro variables).
    """
    sector_obs = df_sensitivities.groupby('Sector')['N_observations'].max()
    sectors_to_plot = sector_obs[sector_obs >= min_sector_obs].index.tolist()
    if not sectors_to_plot:
        print(f"No sectors with N_observations >= {min_sector_obs}.")
        return

    for sector in sectors_to_plot:
        sector_sens = (
            df_sensitivities[df_sensitivities['Sector'] == sector]
            .sort_values('N_observations', ascending=False)
            .iloc[0]
        )
        sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()
        sector_df['logit_pd'] = _calculate_logit(sector_df['12_month'])
        sector_df['logit_pd_zero'] = _calculate_logit(sector_df['PDzero'])
        sector_df['delta_logit'] = sector_df['logit_pd'] - sector_df['logit_pd_zero']

        n_plots = len(macro_cols) + len(gpr_cols)
        n_cols = 4 if n_plots > 4 else n_plots
        n_rows = int(np.ceil(n_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
        axes_flat = np.ravel(axes)
        fig.suptitle(
            f'Sensitivity Analysis – {sector}  (N={len(sector_df)}, R²={sector_sens["R_squared"]:.3f})',
            fontsize=14,
            fontweight='bold',
        )

        for idx, macro_var in enumerate(macro_cols):
            ax = axes_flat[idx]
            X_k, cpr, coef = _compute_cpr(sector_df, sector_sens, macro_var)
            if len(X_k) == 0:
                ax.set_visible(False)
                continue

            ax.scatter(X_k, cpr, alpha=0.3, s=10, label='Partial residuals')

            X_range = np.linspace(X_k.min(), X_k.max(), 100)
            y_line = cpr.mean() + coef * (X_range - X_k.mean())

            ax.plot(X_range, y_line, 'r-', linewidth=2, label=f'β={coef:.4f}')
            ax.set_xlabel(macro_var, fontsize=11)
            ax.set_ylabel('Partial Δ logit(PD)', fontsize=11)
            ax.set_title(f'{macro_var} (β={coef:.4f})', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_box_aspect(1)

        for idx, gpr_var in enumerate(gpr_cols):
            ax = axes_flat[len(macro_cols) + idx]
            X_k, cpr, coef = _compute_cpr(sector_df, sector_sens, gpr_var)
            if len(X_k) == 0:
                ax.set_visible(False)
                continue

            ax.scatter(X_k, cpr, alpha=0.3, s=10, label='Partial residuals', color='green')

            X_range = np.linspace(X_k.min(), X_k.max(), 100)
            y_line = cpr.mean() + coef * (X_range - X_k.mean())

            ax.plot(X_range, y_line, 'darkgreen', linewidth=2, label=f'δ={coef:.4f}')
            ax.set_xlabel(gpr_var, fontsize=11)
            ax.set_ylabel('Partial Δ logit(PD)', fontsize=11)
            ax.set_title(f'{gpr_var} (δ={coef:.4f})', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_box_aspect(1)

        for ax in axes_flat[n_plots:]:
            ax.set_visible(False)

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
    min_sector_obs: int = 1000,
    top_n_sectors: int = 15,
) -> None:
    """Plot regression lines for the top N sectors on shared macro variable panels."""
    sector_obs = df_sensitivities.groupby('Sector')['N_observations'].max()
    eligible = sector_obs[sector_obs >= min_sector_obs]
    if eligible.empty:
        print(f"No sectors with N_observations >= {min_sector_obs}.")
        return

    # Limit to top_n_sectors by observation count so the legend stays readable
    sectors_to_plot = eligible.nlargest(top_n_sectors).index.tolist()
    print(f"Plotting top {len(sectors_to_plot)} sectors by observation count")

    n_plots = len(macro_cols)
    n_cols = 4 if n_plots > 4 else n_plots
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.5 * n_rows))
    axes_flat = np.ravel(axes)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Use a colormap that supports more than 20 distinct colours
    cmap = plt.cm.get_cmap('tab20' if len(sectors_to_plot) <= 20 else 'nipy_spectral')
    colors = cmap(np.linspace(0, 1, len(sectors_to_plot)))

    for idx, macro_var in enumerate(macro_cols):
        ax = axes_flat[idx]
        for sector_idx, sector in enumerate(sectors_to_plot):
            sector_sens = (
                df_sensitivities[df_sensitivities['Sector'] == sector]
                .sort_values('N_observations', ascending=False)
                .iloc[0]
            )
            sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()
            sector_df['logit_pd'] = _calculate_logit(sector_df['12_month'])
            sector_df['logit_pd_zero'] = _calculate_logit(sector_df['PDzero'])
            sector_df['delta_logit'] = sector_df['logit_pd'] - sector_df['logit_pd_zero']

            X_k, cpr, coef = _compute_cpr(sector_df, sector_sens, macro_var)
            if len(X_k) == 0:
                continue

            X_range = np.linspace(X_k.min(), X_k.max(), 100)
            y_pred = cpr.mean() + coef * (X_range - X_k.mean())

            ax.plot(X_range, y_pred, linewidth=2, color=colors[sector_idx],
                    label=f'{sector[:25]} (β={coef:.3f})', alpha=0.8)

        ax.set_xlabel(macro_var, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{macro_var}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(1)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()
    print(f"\nCompleted comparison plot for top {len(sectors_to_plot)} sectors")


def _calculate_logit(p: np.ndarray | pd.Series) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
