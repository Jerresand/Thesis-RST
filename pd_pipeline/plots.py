"""Plotting utilities for the PD analysis pipeline."""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

# ── Publication-quality defaults ──────────────────────────────────────────────
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.75,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.55,
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})


def _add_row_bands(ax: "plt.Axes", n: int, color: str = '#F5F5F5') -> None:
    """Add subtle alternating horizontal bands behind forest plot rows."""
    for k in range(0, n, 2):
        ax.axhspan(k - 0.5, k + 0.5, color=color, alpha=0.55, zorder=0, linewidth=0)


_PRETTY_LABELS = {
    'GDP_Growth':    'GDP Growth',
    'Interest_Rate': 'Interest Rate',
    'Brent_Oil':     'Brent Oil',
    'Fuel_Index':    'Fuel Index',
    'CPI':           'CPI',
    'GPR_Global':    'GPR Global',
}

# Confidence levels to draw: 50 %, 90 %, 95 %, 99 %
_CONTOUR_LEVELS = [0.50, 0.90, 0.95, 0.99]


def plot_normal_contours_pairwise(
    cov_matrix: pd.DataFrame,
    mean_vector: pd.Series,
    cols: List[str],
    title: str = 'Bivariate normal contours — macro & geopolitical variables',
    figsize: Optional[tuple] = None,
) -> None:
    """Pairwise bivariate-normal contour grid based on the covariance matrix.

    Each panel shows the 50 %, 90 %, 95 %, and 99 % probability contour
    ellipses of the fitted bivariate normal for that pair of variables.
    Nothing else is drawn — no scatter, no annotations.

    Parameters
    ----------
    cov_matrix : DataFrame
        Full covariance matrix (rows and columns = variable names).
    mean_vector : Series
        Mean of each variable (index = variable names).
    cols : list of str
        Variables to include (subset of cov_matrix columns).
    """
    sns.set_theme(style='white', context='notebook')
    from scipy.stats import chi2 as _chi2

    n = len(cols)
    labels = [_PRETTY_LABELS.get(c, c.replace('_', ' ')) for c in cols]

    # Lower-triangular pairs only  (i > j)
    pairs = [(i, j) for i in range(n) for j in range(n) if i > j]
    n_pairs = len(pairs)

    # Arrange in a grid — as square as possible
    n_cols_grid = int(np.ceil(np.sqrt(n_pairs)))
    n_rows_grid = int(np.ceil(n_pairs / n_cols_grid))

    if figsize is None:
        figsize = (3.2 * n_cols_grid, 3.0 * n_rows_grid)

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=figsize, squeeze=False)
    axes_flat = list(axes.flat)
    fig.patch.set_facecolor('white')

    # Radii for each confidence level from chi²(2) distribution
    radii = [np.sqrt(_chi2.ppf(p, df=2)) for p in _CONTOUR_LEVELS]
    line_styles = [':', '--', '-', '-']
    line_widths = [0.9, 1.0, 1.2, 1.5]
    alphas      = [0.6, 0.7, 0.85, 1.0]

    theta = np.linspace(0, 2 * np.pi, 300)
    unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])

    for idx, (i, j) in enumerate(pairs):
        ax = axes_flat[idx]

        mu = np.array([mean_vector[cols[j]], mean_vector[cols[i]]])
        sigma = cov_matrix.loc[[cols[j], cols[i]], [cols[j], cols[i]]].values

        # Cholesky decomposition to map unit circle → ellipse
        try:
            L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            ax.set_visible(False)
            continue

        for r, ls, lw, alpha in zip(radii, line_styles, line_widths, alphas):
            ellipse = (L @ (r * unit_circle).T).T + mu
            ax.plot(ellipse[:, 0], ellipse[:, 1],
                    color='#2166AC', linestyle=ls, linewidth=lw, alpha=alpha)

        ax.set_xlabel(labels[j], fontsize=8, labelpad=3)
        ax.set_ylabel(labels[i], fontsize=8, labelpad=3)
        ax.tick_params(labelsize=7)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    # Hide unused panels
    for idx in range(n_pairs, n_rows_grid * n_cols_grid):
        axes_flat[idx].set_visible(False)

    # Legend for confidence levels
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='#2166AC', linestyle=ls, linewidth=lw, alpha=alpha,
               label=f'{int(p*100)} %')
        for p, ls, lw, alpha in zip(_CONTOUR_LEVELS, line_styles, line_widths, alphas)
    ]
    fig.legend(handles=legend_handles, title='Confidence level',
               loc='lower right', fontsize=8, title_fontsize=8,
               framealpha=0.9, edgecolor='#cccccc')

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


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

    colors = ['#4C72B0' if 'β' in idx else '#DD8452' for idx in feature_freq_df.index]
    ax1.barh(feature_freq_df.index, feature_freq_df['Times Selected'], color=colors, edgecolor='none')
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
    ax2.barh(sector_names, df_lasso['N_features_selected'], color=colors_sector, edgecolor='none')
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

    sens_cols = ['Sector', 'R_squared']
    if 'R_squared_adj' in df_sensitivities.columns:
        sens_cols.append('R_squared_adj')
    comparison_df = df_lasso.merge(
        df_sensitivities[sens_cols],
        on='Sector',
        suffixes=('_lasso', '_ols'),
    )
    use_adj = (
        'R_squared_adj_ols' in comparison_df.columns
        and 'R_squared_adj_lasso' in comparison_df.columns
        and comparison_df[['R_squared_adj_ols', 'R_squared_adj_lasso']].notna().all(axis=None)
    )
    if use_adj:
        xcol, ycol = 'R_squared_adj_ols', 'R_squared_adj_lasso'
        xlabel, ylabel = 'OLS adjusted R²', 'LASSO adjusted R²'
        diag_label = 'y = x (same adjusted R²)'
        plot_title = 'Model fit: LASSO vs OLS (adjusted R²)'
    else:
        xcol, ycol = 'R_squared_ols', 'R_squared_lasso'
        xlabel, ylabel = 'OLS R²', 'LASSO R²'
        diag_label = 'y = x (same R²)'
        plot_title = 'Model fit: LASSO vs OLS (R²)'
    sc = ax4.scatter(
        comparison_df[xcol],
        comparison_df[ycol],
        s=80,
        alpha=0.82,
        edgecolors='white',
        linewidths=0.7,
        c=comparison_df['N_features_selected'],
        cmap='viridis',
    )
    max_r2 = max(comparison_df[xcol].max(), comparison_df[ycol].max())
    ax4.plot([0, max_r2], [0, max_r2], 'r--', linewidth=2, label=diag_label)
    ax4.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax4.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax4.set_title(plot_title, fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax4)
    cbar.set_label('Features Selected', fontsize=10, fontweight='bold')

    plt.show()


def _filter_sensitivity_horizon(df: pd.DataFrame, pd_horizon: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if pd_horizon is not None and 'PD_Horizon' in out.columns:
        out = out[out['PD_Horizon'] == pd_horizon]
    return out.sort_values('Sector').reset_index(drop=True)


def _short_sector(name: str, max_len: int = 34) -> str:
    name = str(name)
    return name if len(name) <= max_len else name[: max_len - 1] + '…'


def _short_predictor_label(name: str) -> str:
    s = name.replace('GPR_Global', 'GPR')
    s = s.replace('_lag1', ' (−1Q)').replace('_lag2', ' (−2Q)').replace('_lag3', ' (−3Q)').replace('_lag4', ' (−4Q)')
    return s.replace('_', ' ')


def plot_sensitivity_model_fit(
    df_sensitivities: pd.DataFrame,
    pd_horizon: Optional[str] = None,
    figsize: tuple[float, float] = (12, 7),
) -> None:
    """Bar comparison of R² vs adjusted R² by sector, plus sample size vs fit scatter."""
    sns.set_theme(style='whitegrid', context='notebook')
    df = _filter_sensitivity_horizon(df_sensitivities, pd_horizon)
    if df.empty:
        print('No sensitivity rows to plot (check PD horizon filter).')
        return
    if 'R_squared_adj' not in df.columns:
        print('Re-run sensitivity analysis to populate R_squared_adj.')
        return

    df = df.sort_values('R_squared_adj', ascending=True)
    y = np.arange(len(df))
    h = 0.35
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    _add_row_bands(ax, len(df))
    ax.barh(y - h / 2, df['R_squared'], height=h, label='R²', color='#4C72B0', edgecolor='none')
    ax.barh(
        y + h / 2,
        df['R_squared_adj'],
        height=h,
        label='Adjusted R²',
        color='#DD8452',
        edgecolor='none',
    )
    ax.set_yticks(y)
    ax.set_yticklabels([_short_sector(s) for s in df['Sector']], fontsize=9)
    ax.set_xlabel('Share of variance explained', fontsize=11, fontweight='bold')
    ax.set_title('In-sample fit by sector (OLS)', fontsize=12, fontweight='bold')
    xmax = float(max(df['R_squared'].max(), df['R_squared_adj'].max(), 0.01)) * 1.12
    ax.set_xlim(0, min(1.0, xmax))
    ax.legend(loc='lower right', frameon=True)
    ax.grid(True, axis='x', alpha=0.3)

    ax2 = axes[1]
    sc = ax2.scatter(
        df['N_observations'],
        df['R_squared_adj'],
        s=70,
        c=df['R_squared'],
        cmap='viridis',
        edgecolors='white',
        linewidths=0.7,
        alpha=0.88,
    )
    ax2.set_xlabel('Regression sample size (N)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Adjusted R²', fontsize=11, fontweight='bold')
    ax2.set_title('Fit vs sample size (color = R²)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax2, label='R²')

    sub = (
        f'PD horizon: {pd_horizon}' if pd_horizon else 'All horizons in frame'
    )
    fig.suptitle(f'OLS sensitivity — model fit overview\n({sub})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cumulative_coefficient_forest(
    df_sensitivities: pd.DataFrame,
    base_cols: List[str],
    all_cols: List[str],
    kind: str = 'macro',
    pd_horizon: Optional[str] = None,
    n_lags: int = 4,
    fig_width: float = 7.5,
) -> None:
    """One thesis-ready forest plot per base variable showing the cumulative (summed) OLS coefficient.

    For each base variable the contemporaneous term and its *n_lags* quarterly lags are summed:
        β_total = β + β_lag1 + β_lag2 + … + β_lag{n_lags}

    The 95 % CI is propagated in quadrature (independent-lags approximation):
        SE_total = sqrt( SE² + SE_lag1² + … )   where SE_i = (CI_upper_i - CI_lower_i) / (2 × 1.96)

    One matplotlib figure is produced per base variable — paste each directly into a thesis chapter.

    Parameters
    ----------
    base_cols : list of str
        Variable names *without* lag suffixes (e.g. config.MACRO_COLS or config.GPR_COLS).
    all_cols : list of str
        Full list including lag-suffixed names used during estimation (e.g. config.ALL_MACRO_COLS).
    kind : 'macro' | 'gpr'
        Selects column prefix (β_ or δ_) and axis label.
    n_lags : int
        Number of quarterly lags that were included in the regression (default 4).
    fig_width : float
        Figure width in inches. Height is set automatically from the number of sectors.
    """
    sns.set_theme(style='whitegrid', context='notebook')
    df = _filter_sensitivity_horizon(df_sensitivities, pd_horizon)
    if df.empty:
        print('No sensitivity rows to plot (check PD horizon filter).')
        return

    prefix = 'β_' if kind == 'macro' else 'δ_'
    letter = 'β' if kind == 'macro' else 'δ'
    y_pos = np.arange(len(df))
    sectors_short = [_short_sector(s, 34) for s in df['Sector']]
    fig_height = max(0.45 * len(df) + 1.8, 4.0)

    for base in base_cols:
        # Collect all lag-variant column names that belong to this base variable
        lag_cols: List[str] = [base] + [f'{base}_lag{k}' for k in range(1, n_lags + 1)]
        lag_cols = [c for c in lag_cols if c in all_cols]  # keep only what was actually regressed

        coef_cols = [f'{prefix}{c}' for c in lag_cols]
        lo_cols   = [f'{prefix}{c}_CI_lower' for c in lag_cols]
        hi_cols   = [f'{prefix}{c}_CI_upper' for c in lag_cols]

        # Skip if none of the expected columns exist
        if not any(c in df.columns for c in coef_cols):
            print(f'Skipping {base}: no regression columns found.')
            continue

        # Sum point estimates; propagate SE in quadrature
        beta_sum = np.zeros(len(df))
        var_sum  = np.zeros(len(df))
        for cc, lc, hc in zip(coef_cols, lo_cols, hi_cols):
            if cc not in df.columns:
                continue
            b  = df[cc].values.astype(float)
            lo = df[lc].values.astype(float) if lc in df.columns else b
            hi = df[hc].values.astype(float) if hc in df.columns else b
            se = (hi - lo) / (2 * 1.96)
            beta_sum += np.nan_to_num(b)
            var_sum  += np.nan_to_num(se ** 2)

        se_sum = np.sqrt(var_sum)
        ci_lo  = beta_sum - 1.96 * se_sum
        ci_hi  = beta_sum + 1.96 * se_sum
        sig    = (ci_lo > 0) | (ci_hi < 0)
        colors = np.where(sig, '#2166AC', '#9E9E9E')
        xerr   = np.row_stack([beta_sum - ci_lo, ci_hi - beta_sum])

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        _add_row_bands(ax, len(df))

        ax.errorbar(
            beta_sum, y_pos,
            xerr=xerr,
            fmt='none',
            ecolor='#555555',
            capsize=3,
            linewidth=0.9,
            alpha=0.7,
            zorder=2,
        )
        ax.scatter(beta_sum, y_pos, c=colors, s=50, zorder=3, edgecolors='white', linewidths=0.6)
        ax.axvline(0.0, color='#444444', linestyle='--', linewidth=0.85, alpha=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sectors_short, fontsize=8)
        ax.set_ylim(y_pos.min() - 0.6, y_pos.max() + 0.6)

        n_terms = len(lag_cols)
        lag_range = f'lag 0–{n_terms - 1}' if n_terms > 1 else 'lag 0'
        ax.set_xlabel(
            f'Cumulative {letter} ({lag_range}, 95 % CI)  —  Δ logit PD per unit {_short_predictor_label(base)}',
            fontsize=9,
        )

        sig_patch = plt.scatter([], [], c='#2166AC', s=40, edgecolors='white', linewidths=0.5,
                                label='Significant (95 % CI)')
        ns_patch  = plt.scatter([], [], c='#9E9E9E', s=40, edgecolors='white', linewidths=0.5,
                                label='Not significant')
        ax.legend(handles=[sig_patch, ns_patch], fontsize=8, loc='lower right',
                  framealpha=0.9, edgecolor='#cccccc')
        ax.grid(True, axis='x', alpha=0.2)

        horizon_str = f' · horizon {pd_horizon}' if pd_horizon else ''
        fig.suptitle(
            f'Cumulative effect: {_short_predictor_label(base)}{horizon_str}',
            fontsize=11,
            fontweight='bold',
            y=1.01,
        )
        plt.tight_layout()
        plt.show()


def _forest_grid_n_cols(n_predictors: int, n_cols: Optional[int], max_cols: int = 5) -> int:
    if n_predictors <= 0:
        return 1
    if n_cols is not None:
        return max(1, min(int(n_cols), n_predictors))
    if n_predictors <= 4:
        return min(3, n_predictors)
    return min(max_cols, n_predictors)


def plot_sensitivity_coefficient_forest(
    df_sensitivities: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: Optional[List[str]] = None,
    pd_horizon: Optional[str] = None,
    figsize: tuple[float, float] | None = None,
    n_cols: Optional[int] = None,
    max_cols: int = 5,
    # legacy keyword kept for backward compatibility — ignored
    kind: str = 'macro',
    predictor_cols: Optional[List[str]] = None,
) -> None:
    """Forest plots: one panel per regressor (macro + GPR combined), sectors on the y-axis, 95% CIs.

    Macro variables use the β_ column prefix; GPR variables use the δ_ column prefix in the
    underlying DataFrame, but all coefficients are labelled β in the plot.
    """
    # Backward-compat: old callers pass a positional predictor_cols as second arg
    if predictor_cols is not None:
        all_cols = list(predictor_cols)
        prefix_map = {c: ('β_' if kind == 'macro' else 'δ_') for c in predictor_cols}
    else:
        _gpr = list(gpr_cols) if gpr_cols is not None else []
        all_cols = list(macro_cols) + _gpr
        prefix_map = {c: 'β_' for c in macro_cols}
        prefix_map.update({c: 'δ_' for c in _gpr})

    sns.set_theme(style='whitegrid', context='notebook')
    df = _filter_sensitivity_horizon(df_sensitivities, pd_horizon)
    if df.empty:
        print('No sensitivity rows to plot.')
        return

    n_p = len(all_cols)
    n_cols = _forest_grid_n_cols(n_p, n_cols, max_cols=max_cols)
    n_rows = int(np.ceil(n_p / n_cols))
    if figsize is None:
        figsize = (5.2 * n_cols, max(2.8 * n_rows, 0.45 * len(df) * n_rows / n_cols + 2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(
        f'OLS coefficients with 95% CI — β\n'
        f'(Δ logit PD; horizon: {pd_horizon or "all rows"})',
        fontsize=13,
        fontweight='bold',
        y=1.01,
    )

    y_pos = np.arange(len(df))
    sectors_short = [_short_sector(s, 28) for s in df['Sector']]

    for i, col in enumerate(all_cols):
        ax = axes.flat[i]
        cname = f'{prefix_map[col]}{col}'
        lo, hi = f'{cname}_CI_lower', f'{cname}_CI_upper'
        if cname not in df.columns:
            ax.set_visible(False)
            continue
        _add_row_bands(ax, len(df))
        coefs = df[cname].values
        xerr = np.row_stack([coefs - df[lo].values, df[hi].values - coefs])
        colors = np.where((df[lo].values > 0) | (df[hi].values < 0), '#2166AC', '#9E9E9E')
        ax.scatter(coefs, y_pos, c=colors, s=42, zorder=3, edgecolors='white', linewidths=0.5)
        ax.errorbar(
            coefs,
            y_pos,
            xerr=xerr,
            fmt='none',
            ecolor='#555555',
            capsize=2.5,
            linewidth=0.9,
            alpha=0.75,
        )
        ax.axvline(0.0, color='#444444', linestyle='--', linewidth=0.85, alpha=0.5)
        ax.set_yticks(y_pos)
        if i % n_cols == 0:
            ax.set_yticklabels(sectors_short, fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_title(_short_predictor_label(col), fontsize=10, fontweight='bold')
        ax.set_xlabel('β (95% CI)', fontsize=9)
        ax.grid(True, axis='x', alpha=0.2)

    for j in range(len(all_cols), len(axes.flat)):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_sensitivity_forests_all_predictors(
    df_sensitivities: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    pd_horizon: Optional[str] = None,
    n_cols_macro: Optional[int] = None,
    n_cols_gpr: Optional[int] = None,
    max_cols: int = 5,
) -> None:
    """OLS forests for every regressor (macro + GPR) in a single combined figure."""
    n_cols = n_cols_macro or n_cols_gpr
    plot_sensitivity_coefficient_forest(
        df_sensitivities,
        macro_cols=macro_cols,
        gpr_cols=gpr_cols,
        pd_horizon=pd_horizon,
        n_cols=n_cols,
        max_cols=max_cols,
    )


def plot_lasso_coefficient_forest(
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: Optional[List[str]] = None,
    pd_horizon: Optional[str] = None,
    figsize: tuple[float, float] | None = None,
    n_cols: Optional[int] = None,
    max_cols: int = 5,
    # legacy keyword kept for backward compatibility — ignored
    kind: str = 'macro',
    predictor_cols: Optional[List[str]] = None,
) -> None:
    """Forest of LASSO slopes in native X units (same interpretation as OLS β); no CIs.

    Macro and GPR variables are combined in a single figure; all coefficients labelled β.
    Non-selected features are shown as a small gray marker at zero.
    """
    # Backward-compat: old callers pass a positional predictor_cols as second arg
    if predictor_cols is not None:
        all_cols = list(predictor_cols)
        native_map = {c: ('LASSO_NATIVE_β_' if kind == 'macro' else 'LASSO_NATIVE_δ_') for c in predictor_cols}
        sel_map = {c: ('β_selected_' if kind == 'macro' else 'δ_selected_') for c in predictor_cols}
    else:
        _gpr = list(gpr_cols) if gpr_cols is not None else []
        all_cols = list(macro_cols) + _gpr
        native_map = {c: 'LASSO_NATIVE_β_' for c in macro_cols}
        native_map.update({c: 'LASSO_NATIVE_δ_' for c in _gpr})
        sel_map = {c: 'β_selected_' for c in macro_cols}
        sel_map.update({c: 'δ_selected_' for c in _gpr})

    sns.set_theme(style='whitegrid', context='notebook')
    df = _filter_sensitivity_horizon(df_lasso, pd_horizon)
    if df.empty:
        print('No LASSO rows to plot.')
        return

    if all_cols:
        test_col = f'{native_map[all_cols[0]]}{all_cols[0]}'
        if test_col not in df.columns:
            print(
                'LASSO_NATIVE_* columns missing — re-run lasso.run_lasso_feature_selection '
                'to export native-scale coefficients for comparison with OLS.'
            )
            return

    n_p = len(all_cols)
    n_cols = _forest_grid_n_cols(n_p, n_cols, max_cols=max_cols)
    n_rows = int(np.ceil(n_p / n_cols))
    if figsize is None:
        figsize = (5.2 * n_cols, max(2.8 * n_rows, 0.45 * len(df) * n_rows / n_cols + 2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(
        f'LASSO coefficients (native units, same as OLS) — β\n'
        f'Orange diamond = selected; gray dot = shrunk to zero. Horizon: {pd_horizon or "all rows"}',
        fontsize=12,
        fontweight='bold',
        y=1.02,
    )

    y_pos = np.arange(len(df))
    sectors_short = [_short_sector(s, 28) for s in df['Sector']]

    for i, col in enumerate(all_cols):
        ax = axes.flat[i]
        nc = f'{native_map[col]}{col}'
        if nc not in df.columns:
            ax.set_visible(False)
            continue
        _add_row_bands(ax, len(df))
        native = np.nan_to_num(df[nc].values.astype(float), nan=0.0)
        selected = df[f'{sel_map[col]}{col}'].values.astype(int) == 1
        ax.axvline(0.0, color='#444444', linestyle='--', linewidth=0.85, alpha=0.5)
        ax.scatter(
            native[selected],
            y_pos[selected],
            c='#DD8452',
            s=44,
            marker='D',
            zorder=3,
            edgecolors='white',
            linewidths=0.5,
        )
        ax.scatter(
            native[~selected],
            y_pos[~selected],
            c='#cccccc',
            s=24,
            marker='o',
            zorder=2,
            edgecolors='none',
            linewidths=0,
        )
        ax.set_yticks(y_pos)
        if i % n_cols == 0:
            ax.set_yticklabels(sectors_short, fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_title(_short_predictor_label(col), fontsize=10, fontweight='bold')
        ax.set_xlabel('β (native units)', fontsize=9)
        ax.grid(True, axis='x', alpha=0.2)

    for j in range(len(all_cols), len(axes.flat)):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_lasso_forests_all_predictors(
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    pd_horizon: Optional[str] = None,
    n_cols_macro: Optional[int] = None,
    n_cols_gpr: Optional[int] = None,
    max_cols: int = 5,
) -> None:
    """LASSO forests: macro + GPR combined in a single figure."""
    n_cols = n_cols_macro or n_cols_gpr
    plot_lasso_coefficient_forest(
        df_lasso,
        macro_cols=macro_cols,
        gpr_cols=gpr_cols,
        pd_horizon=pd_horizon,
        n_cols=n_cols,
        max_cols=max_cols,
    )


def _merge_sens_lasso(
    df_sensitivities: pd.DataFrame,
    df_lasso: pd.DataFrame,
) -> pd.DataFrame:
    keys = (
        ['Sector', 'PD_Horizon']
        if 'PD_Horizon' in df_sensitivities.columns and 'PD_Horizon' in df_lasso.columns
        else ['Sector']
    )
    return df_sensitivities.merge(df_lasso, on=keys, suffixes=('_ols', '_lasso'))


def plot_ols_lasso_forest_comparison(
    df_sensitivities: pd.DataFrame,
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: Optional[List[str]] = None,
    pd_horizon: Optional[str] = None,
    figsize: tuple[float, float] | None = None,
    n_cols: Optional[int] = None,
    max_cols: int = 5,
    # legacy keyword kept for backward compatibility — ignored
    kind: str = 'macro',
    predictor_cols: Optional[List[str]] = None,
) -> None:
    """Per regressor: OLS point ±95% CI (lower y) vs LASSO native coefficient (upper y).

    Macro and GPR variables are combined in a single figure; all coefficients labelled β.
    LASSO has no standard inferential CI here; orange diamonds = selected, gray = not selected.
    """
    # Backward-compat: old callers pass a positional predictor_cols as second arg
    if predictor_cols is not None:
        all_cols = list(predictor_cols)
        native_map = {c: ('LASSO_NATIVE_β_' if kind == 'macro' else 'LASSO_NATIVE_δ_') for c in predictor_cols}
        sel_map = {c: ('β_selected_' if kind == 'macro' else 'δ_selected_') for c in predictor_cols}
        ols_map = {c: ('β_' if kind == 'macro' else 'δ_') for c in predictor_cols}
    else:
        _gpr = list(gpr_cols) if gpr_cols is not None else []
        all_cols = list(macro_cols) + _gpr
        native_map = {c: 'LASSO_NATIVE_β_' for c in macro_cols}
        native_map.update({c: 'LASSO_NATIVE_δ_' for c in _gpr})
        sel_map = {c: 'β_selected_' for c in macro_cols}
        sel_map.update({c: 'δ_selected_' for c in _gpr})
        ols_map = {c: 'β_' for c in macro_cols}
        ols_map.update({c: 'δ_' for c in _gpr})

    sns.set_theme(style='whitegrid', context='notebook')
    comp = _merge_sens_lasso(df_sensitivities, df_lasso)
    if pd_horizon is not None and 'PD_Horizon' in comp.columns:
        comp = comp[comp['PD_Horizon'] == pd_horizon]
    comp = comp.sort_values('Sector').reset_index(drop=True)
    if comp.empty:
        print('No merged OLS/LASSO rows to plot.')
        return

    if all_cols and f'{native_map[all_cols[0]]}{all_cols[0]}' not in comp.columns:
        print('LASSO_NATIVE_* columns missing — re-run LASSO feature selection.')
        return

    n_p = len(all_cols)
    n_cols = _forest_grid_n_cols(n_p, n_cols, max_cols=max_cols)
    n_rows = int(np.ceil(n_p / n_cols))
    if figsize is None:
        figsize = (5.4 * n_cols, max(3.0 * n_rows, 0.5 * len(comp) * n_rows / n_cols + 2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(
        f'OLS vs LASSO — β\n'
        f'Lower marker: OLS ±95% CI. Upper: LASSO native (Δ logit per unit X). '
        f'Horizon: {pd_horizon or "all rows"}',
        fontsize=12,
        fontweight='bold',
        y=1.02,
    )

    y_base = np.arange(len(comp), dtype=float)
    dy = 0.2
    y_lo = y_base - dy
    y_hi = y_base + dy
    sectors_short = [_short_sector(s, 28) for s in comp['Sector']]

    for i, col in enumerate(all_cols):
        ax = axes.flat[i]
        cname = f'{ols_map[col]}{col}'
        lo_n, hi_n = f'{cname}_CI_lower', f'{cname}_CI_upper'
        nn = f'{native_map[col]}{col}'
        if cname not in comp.columns or nn not in comp.columns:
            ax.set_visible(False)
            continue

        _add_row_bands(ax, len(comp))
        ols_c = comp[cname].values.astype(float)
        lo = comp[lo_n].values.astype(float)
        hi = comp[hi_n].values.astype(float)
        xerr = np.row_stack([ols_c - lo, hi - ols_c])
        lasso_v = np.nan_to_num(comp[nn].values.astype(float), nan=0.0)
        selected = comp[f'{sel_map[col]}{col}'].values.astype(int) == 1
        ols_sig = (lo > 0) | (hi < 0)
        ols_colors = np.where(ols_sig, '#2166AC', '#9E9E9E')

        lbl = 'OLS ±95% CI' if i == 0 else None
        ax.errorbar(
            ols_c,
            y_lo,
            xerr=xerr,
            fmt='none',
            ecolor='#555555',
            capsize=2.5,
            linewidth=0.85,
            label=lbl,
            zorder=2,
            alpha=0.75,
        )
        ax.scatter(ols_c, y_lo, c=ols_colors, s=36, zorder=3, edgecolors='white', linewidths=0.5)

        sel_mask = selected
        if i == 0:
            ax.scatter(
                lasso_v[sel_mask],
                y_hi[sel_mask],
                c='#DD8452',
                s=40,
                marker='D',
                zorder=3,
                edgecolors='white',
                linewidths=0.5,
                label='LASSO selected',
            )
            ax.scatter(
                lasso_v[~sel_mask],
                y_hi[~sel_mask],
                c='#cccccc',
                s=32,
                marker='D',
                zorder=3,
                edgecolors='none',
                linewidths=0,
                label='LASSO not selected',
            )
        else:
            ax.scatter(
                lasso_v[sel_mask],
                y_hi[sel_mask],
                c='#DD8452',
                s=40,
                marker='D',
                zorder=3,
                edgecolors='white',
                linewidths=0.5,
            )
            ax.scatter(
                lasso_v[~sel_mask],
                y_hi[~sel_mask],
                c='#cccccc',
                s=32,
                marker='D',
                zorder=3,
                edgecolors='none',
                linewidths=0,
            )

        ax.axvline(0.0, color='#444444', linestyle='--', linewidth=0.85, alpha=0.5)
        ax.set_yticks(y_base)
        if i % n_cols == 0:
            ax.set_yticklabels(sectors_short, fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(y_base.min() - 0.55, y_base.max() + 0.55)
        ax.set_title(_short_predictor_label(col), fontsize=10, fontweight='bold')
        ax.set_xlabel('β (native units)', fontsize=9)
        ax.grid(True, axis='x', alpha=0.2)
        if i == 0:
            ax.legend(loc='lower right', fontsize=7, framealpha=0.9, edgecolor='#cccccc')

    for j in range(len(all_cols), len(axes.flat)):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_ols_lasso_forest_comparison_all(
    df_sensitivities: pd.DataFrame,
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    pd_horizon: Optional[str] = None,
    n_cols_macro: Optional[int] = None,
    n_cols_gpr: Optional[int] = None,
    max_cols: int = 5,
) -> None:
    """OLS vs LASSO comparison forests for macro + GPR combined in a single figure."""
    n_cols = n_cols_macro or n_cols_gpr
    plot_ols_lasso_forest_comparison(
        df_sensitivities,
        df_lasso,
        macro_cols=macro_cols,
        gpr_cols=gpr_cols,
        pd_horizon=pd_horizon,
        n_cols=n_cols,
        max_cols=max_cols,
    )


def plot_sensitivity_significance_heatmap(
    df_sensitivities: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    pd_horizon: Optional[str] = None,
    figsize: tuple[float, float] = (16, 8),
) -> None:
    """Heatmap of OLS coefficients; green/red outline marks 95% CI excluding zero."""
    sns.set_theme(style='white', context='notebook')
    df = _filter_sensitivity_horizon(df_sensitivities, pd_horizon)
    if df.empty:
        print('No sensitivity rows to plot.')
        return

    col_names: List[str] = [f'β_{c}' for c in macro_cols] + [f'δ_{c}' for c in gpr_cols]
    xlabels = ['β ' + _short_predictor_label(c) for c in macro_cols] + ['β ' + _short_predictor_label(c) for c in gpr_cols]

    mat = np.zeros((len(df), len(col_names)))
    sig = np.zeros_like(mat, dtype=bool)
    for j, cname in enumerate(col_names):
        lo_n = f'{cname}_CI_lower'
        hi_n = f'{cname}_CI_upper'
        mat[:, j] = df[cname].values
        sig[:, j] = (df[lo_n].values > 0) | (df[hi_n].values < 0)

    vmax = np.nanmax(np.abs(mat)) or 1.0
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([_short_sector(s) for s in df['Sector']], fontsize=9)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=55, ha='right', fontsize=7)
    ax.set_title(
        'Sensitivity coefficients (OLS on Δ logit PD)\n'
        'Green border: 95% CI excludes 0; gray: not significant at 5%',
        fontsize=12,
        fontweight='bold',
        pad=12,
    )

    if mat.shape[1] <= 18:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                t = f'{val:.2f}' if abs(val) < 100 else f'{val:.1f}'
                ax.text(
                    j,
                    i,
                    t,
                    ha='center',
                    va='center',
                    fontsize=5,
                    color='white' if abs(val) > 0.55 * vmax else 'black',
                )

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not sig[i, j]:
                continue
            color = '#1B5E20' if mat[i, j] > 0 else '#B71C1C'
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.2,
                )
            )

    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='Coefficient')
    plt.tight_layout()
    plt.show()


def plot_bootstrap_stability(
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    pd_horizon: Optional[str] = None,
    stability_threshold: float = 0.5,
    figsize: Optional[tuple] = None,
) -> None:
    """Heatmap of bootstrap selection stability for each feature × sector.

    Colour encodes the fraction of bootstrap samples in which each feature
    received a non-zero coefficient (0 = never selected, 1 = always selected).
    A dashed contour marks the *stability_threshold* boundary.
    """
    sns.set_theme(style='white', context='notebook')
    df = _filter_sensitivity_horizon(df_lasso, pd_horizon)
    if df.empty:
        print('No rows to plot.')
        return

    all_cols = list(macro_cols) + list(gpr_cols)
    prefixes = ['β_'] * len(macro_cols) + ['δ_'] * len(gpr_cols)
    stab_cols = [f'{p}stability_{c}' for p, c in zip(prefixes, all_cols)]

    if not any(c in df.columns for c in stab_cols):
        print('No stability columns found — run lasso.run_bootstrap_stability() first.')
        return

    mat = np.zeros((len(df), len(all_cols)))
    for j, sc in enumerate(stab_cols):
        if sc in df.columns:
            mat[:, j] = np.nan_to_num(df[sc].values.astype(float))

    sectors_short = [_short_sector(s) for s in df['Sector']]
    xlabels = [_short_predictor_label(c) for c in all_cols]

    if figsize is None:
        figsize = (max(0.55 * len(all_cols) + 2, 10), max(0.38 * len(df) + 2, 5))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(sectors_short, fontsize=8)
    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels(xlabels, rotation=50, ha='right', fontsize=8)
    ax.tick_params(length=0)

    # Annotate each cell with the stability %
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                    fontsize=6,
                    color='white' if (v > 0.75 or v < 0.25) else 'black')

    # Threshold contour
    ax.contour(mat, levels=[stability_threshold - 0.001],
               colors='#1A237E', linewidths=1.2, linestyles='--')

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label='Selection frequency')
    ax.set_title(
        f'Bootstrap selection stability  (dashed = {int(stability_threshold*100)} % threshold)',
        fontsize=12, fontweight='bold', pad=10,
    )
    plt.tight_layout()
    plt.show()


def plot_regularization_paths(
    path_data: dict,
    macro_cols: List[str],
    gpr_cols: List[str],
    top_n: int = 4,
    figsize: Optional[tuple] = None,
) -> None:
    """Elastic-Net regularization path for the top *top_n* sectors.

    Each panel shows how each feature's coefficient evolves as the penalty
    alpha decreases (left = heavily regularised / sparse; right = full model).
    A vertical dashed line marks the CV-optimal alpha.
    """
    sns.set_theme(style='whitegrid', context='notebook')
    if not path_data:
        print('path_data is empty — run lasso.compute_regularization_paths() first.')
        return

    sectors = list(path_data.keys())[:top_n]
    n_panels = len(sectors)
    if figsize is None:
        figsize = (6.5 * n_panels, 5)

    # Build a consistent color + style map across all features
    all_cols = list(macro_cols) + list(gpr_cols)
    base_vars = list(dict.fromkeys(
        c.split('_lag')[0] for c in all_cols
    ))
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(base_vars)))
    base_color = {b: palette[i] for i, b in enumerate(base_vars)}

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)

    for panel_idx, sector in enumerate(sectors):
        ax = axes[0, panel_idx]
        info = path_data[sector]
        alphas = info['alphas']
        coefs  = info['coefs']          # (n_features, n_alphas)
        names  = info['feature_names']
        opt_a  = info['optimal_alpha']

        log_alphas = np.log10(alphas + 1e-12)

        for j, name in enumerate(names):
            base = name.split('_lag')[0]
            lag  = int(name.split('_lag')[1]) if '_lag' in name else 0
            color = base_color.get(base, 'gray')
            lw = 1.6 - 0.25 * lag  # contemporaneous thicker, lags thinner
            ls = '-' if lag == 0 else '--' if lag <= 2 else ':'
            ax.plot(log_alphas, coefs[j], color=color, linewidth=lw,
                    linestyle=ls, alpha=0.85)

        ax.axvline(np.log10(opt_a + 1e-12), color='#B71C1C',
                   linestyle='--', linewidth=1.2, label=f'CV α={opt_a:.4f}')
        ax.axhline(0, color='#444444', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('log₁₀(α)  ←  more regularised', fontsize=9)
        if panel_idx == 0:
            ax.set_ylabel('Standardised coefficient', fontsize=9)
        ax.set_title(_short_sector(sector, 30), fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85, edgecolor='#cccccc')
        ax.grid(True, axis='both', alpha=0.2)

    # Shared legend for base variables
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=base_color[b], linewidth=1.5, label=_short_predictor_label(b))
        for b in base_vars if b in base_color
    ]
    lag_handles = [
        Line2D([0], [0], color='gray', linewidth=1.5, linestyle='-',  label='lag 0'),
        Line2D([0], [0], color='gray', linewidth=1.2, linestyle='--', label='lag 1–2'),
        Line2D([0], [0], color='gray', linewidth=1.0, linestyle=':',  label='lag 3–4'),
    ]
    fig.legend(handles=legend_handles + lag_handles,
               loc='lower center', ncol=len(base_vars) + 3,
               fontsize=8, framealpha=0.9, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle('Elastic-Net regularization paths', fontsize=12, fontweight='bold', y=1.01)
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
        fit_txt = f'R²={sector_sens["R_squared"]:.3f}'
        if 'R_squared_adj' in sector_sens.index and pd.notna(sector_sens['R_squared_adj']):
            fit_txt += f', R²_adj={sector_sens["R_squared_adj"]:.3f}'
        fig.suptitle(
            f'Sensitivity Analysis – {sector}  (N={len(sector_df)}, {fit_txt})',
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


def plot_lasso_beta_heatmap(
    df_lasso: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    pd_horizon: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> None:
    """Heatmap of LASSO betas (standardised) for every variable × sector.

    Macro and GPR variables are shown in a single combined heatmap; all
    coefficients are labelled β.  Cells where LASSO shrunk the coefficient to
    zero are shown as light grey (not selected); non-zero cells are coloured on
    a diverging scale so positive / negative sensitivities are immediately
    visible.

    Parameters
    ----------
    df_lasso     : output of ``lasso.run_lasso_feature_selection``
    macro_cols   : e.g. ``config.ALL_MACRO_COLS``
    gpr_cols     : e.g. ``config.ALL_GPR_COLS``
    pd_horizon   : filter to a single PD_Horizon row if the column exists
    figsize      : override automatic figure size
    """
    df = df_lasso.copy()
    if pd_horizon is not None and 'PD_Horizon' in df.columns:
        df = df[df['PD_Horizon'] == pd_horizon]
    if df.empty:
        print("plot_lasso_beta_heatmap: no rows to plot.")
        return

    # ── build combined beta matrix (macro first, then GPR) ────────────────────
    sectors = list(df['Sector'])
    n_sectors = len(sectors)
    short_sectors = [s[:30] for s in sectors]

    macro_data = {col: df[f'LASSO_β_{col}'].values for col in macro_cols}
    gpr_data = {col: df[f'LASSO_δ_{col}'].values for col in gpr_cols}
    all_data = {**macro_data, **gpr_data}

    betas = pd.DataFrame(all_data, index=short_sectors)

    vmax = np.nanmax(np.abs(betas.values))
    if vmax == 0:
        vmax = 1.0

    zero_mask = betas == 0

    n_cols_total = len(betas.columns)
    if figsize is None:
        col_w = max(0.55 * n_cols_total, 10)
        row_h = max(0.38 * n_sectors, 4)
        figsize = (col_w, row_h)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # blue–white–red

    # coloured cells for selected (non-zero) variables
    sns.heatmap(
        betas,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-vmax,
        vmax=vmax,
        mask=zero_mask,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 7},
        linewidths=0.4,
        linecolor='#dddddd',
        cbar=True,
        cbar_kws={'label': 'Standardised LASSO β', 'shrink': 0.6},
    )
    # grey cells for not-selected (shrunk to zero)
    sns.heatmap(
        betas,
        ax=ax,
        cmap=sns.color_palette(['#e8e8e8'], as_cmap=True),
        mask=~zero_mask,
        annot=False,
        linewidths=0.4,
        linecolor='#dddddd',
        cbar=False,
    )
    ax.set_title('β', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel('Sector', fontsize=10)
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    horizon_label = f' — {pd_horizon}' if pd_horizon else ''
    fig.suptitle(
        f'LASSO Feature Selection by Sector{horizon_label}\n'
        '(coloured = selected, grey = shrunk to zero)',
        fontsize=13,
        fontweight='bold',
        y=1.02,
    )
    fig.tight_layout()
    plt.show()


def _calculate_logit(p: np.ndarray | pd.Series) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))
