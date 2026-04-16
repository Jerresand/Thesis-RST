import sys, pathlib
_here = pathlib.Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import pdModelling  # kör dataladdning och definierar cov_matrix, corr_matrix, plots m.m.

from pd_pipeline import plots
from pd_pipeline import config
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

_PRETTY_LABELS = {
    'GDP_Growth':    'GDP',
    'Interest_Rate': 'IR',
    'Brent_Oil':     'Oil',
    'Fuel_Index':    'CIX',
    'CPI':           'CPI',
    'GPR_Global':    'GPR',
}


def _pretty_label(col: str) -> str:
    if "_lag" in col:
        base, lag = col.rsplit("_lag", 1)
        base_label = _PRETTY_LABELS.get(base, base.replace("_", " "))
        return f"{base_label} (t-{lag})"
    return _PRETTY_LABELS.get(col, col.replace("_", " "))

#### Correlation and Covaraiance Matrix Heatmap
sns.set_theme(style='white', context='paper')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('white')


cov_matrix  = pdModelling.cov_matrix
corr_matrix = pdModelling.corr_matrix  

for ax, matrix, title in zip(
    axes,
    [cov_matrix, corr_matrix],
    ['Covariance Matrix', 'Correlation Matrix'],
):
    labels = [plots._PRETTY_LABELS.get(c, c.replace('_', ' ')) for c in matrix.columns.tolist()]
    n = len(labels)
    vals = matrix.values

    vmax = 1.0 if 'Correlation' in title else None
    vmin = -1.0 if 'Correlation' in title else None
    cmap = 'RdBu_r' if 'Correlation' in title else 'YlOrBr'

    im = ax.imshow(vals, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=12)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=13)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            v = vals[i, j]
            txt = f"{v:.2f}" if 'Correlation' in title else f"{v:.1f}"
            color = 'white' if abs(v) > (0.6 if 'Correlation' in title else 0.7 * vals.max()) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=13, color=color)

plt.suptitle('Macro & GPR Variable Matrices (Unlagged)', fontsize=16, y=0.98, fontweight='bold')
plt.tight_layout()
plt.show(block=False)
 
#Bivariate normal contour plots 

mean_vector = pdModelling.mean_vector

# Confidence levels to draw: 50 %, 90 %, 95 %, 99 %
_CONTOUR_LEVELS = [0.50, 0.90, 0.95, 0.99]


cols = config.MACRO_COLS + config.GPR_COLS
title = 'Bivariate normal contours — macro & geopolitical variables'
figsize = None

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

    ax.set_xlabel(labels[j], fontsize=13, labelpad=3)
    ax.set_ylabel(labels[i], fontsize=13, labelpad=3)
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
            loc='lower right', fontsize=13, title_fontsize=13,
            framealpha=0.9, edgecolor='#cccccc')

fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout()

# --- Plot: logit_pd vs each regression input (grid) ---
# Creates one grid figure per sector. Each grid has one subplot per X variable.
macro_base_cols = config.MACRO_COLS + config.GPR_COLS  # same X columns as pdModelling.py
lagged_feature_cols = config.ALL_PREDICTOR_COLS_WITH_LAGS

df_rel = pdModelling.df_sector_macro_relative
df_coef = pdModelling.df_per_sector
df_beta_ci = pdModelling.df_beta_ci
df_coef_lagged = pdModelling.df_per_sector_lagged
df_beta_ci_lagged = pdModelling.df_beta_ci_lagged

sectors = sorted(df_coef[config.SECTOR_COL].dropna().unique().tolist())

# If you only want one sector, set to e.g. ["Communications"].
SECTORS_TO_PLOT = sectors

def plot_per_sector_grids(
    feature_cols: list[str],
    df_coef_model,
    model_label: str,
) -> None:
    n_vars = len(feature_cols)
    n_cols_grid = int(np.ceil(np.sqrt(n_vars)))
    n_rows_grid = int(np.ceil(n_vars / n_cols_grid))

    for plot_sector in SECTORS_TO_PLOT:
        df_s = (
            df_rel[df_rel[config.SECTOR_COL] == plot_sector]
            .dropna(subset=feature_cols + ["logit_pd"])
            .copy()
        )
        if df_s.empty:
            continue

        coef_row = df_coef_model[df_coef_model[config.SECTOR_COL] == plot_sector].iloc[0]
        means = df_s[feature_cols].mean(numeric_only=True)

        fig_reg, axes_reg = plt.subplots(
            n_rows_grid, n_cols_grid,
            figsize=(3.2 * n_cols_grid, 3.0 * n_rows_grid),
            squeeze=False
        )
        axes_flat = list(axes_reg.flat)
        fig_reg.patch.set_facecolor("white")

        for idx, x_col in enumerate(feature_cols):
            ax = axes_flat[idx]

            x = df_s[x_col].to_numpy()
            y = df_s["logit_pd"].to_numpy()
            x_min = float(np.min(x))
            x_max = float(np.max(x))
            x_grid = np.linspace(x_min, x_max, 100)

            # Multivariate line: vary only x_col, keep others at sector means.
            base = float(coef_row["intercept"])
            for c in feature_cols:
                if c == x_col:
                    continue
                base += float(coef_row[c]) * float(means[c])
            y_line = base + float(coef_row[x_col]) * x_grid

            ax.scatter(x, y, s=10, alpha=0.35)
            ax.plot(x_grid, y_line, "r-", linewidth=1.6)
            ax.set_title(_pretty_label(x_col), fontsize=10)
            ax.set_xlabel(_pretty_label(x_col), fontsize=9)
            ax.set_ylabel("logit_pd" if (idx % n_cols_grid == 0) else "")
            ax.grid(True, alpha=0.2)

        for j in range(n_vars, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig_reg.suptitle(
            f"logit_pd vs regression inputs ({model_label}, sector: {plot_sector})",
            fontsize=12,
            fontweight="bold",
        )
        fig_reg.tight_layout()


# --- Heatmap: exact visual copy of plot_lasso_beta_heatmap ---
def build_heatmap_frame(
    df_coef_model: pd.DataFrame,
    df_beta_ci_model: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    df_heat = df_coef_model[[config.SECTOR_COL]].copy()

    for feature in feature_cols:
        prefix = "β_" if feature.split("_lag")[0] in config.MACRO_COLS else "δ_"
        df_heat[f"{prefix}{feature}"] = df_coef_model[feature].values

        ci_feature = (
            df_beta_ci_model[df_beta_ci_model["variable"] == feature]
            [[config.SECTOR_COL, "ci_lower_999", "ci_upper_999"]]
            .rename(columns={
                "ci_lower_999": f"{prefix}{feature}_CI_lower",
                "ci_upper_999": f"{prefix}{feature}_CI_upper",
            })
        )
        df_heat = df_heat.merge(ci_feature, on=config.SECTOR_COL, how="left")

    return df_heat


df_heatmap_lagged = build_heatmap_frame(
    df_coef_model=df_coef_lagged,
    df_beta_ci_model=df_beta_ci_lagged,
    feature_cols=lagged_feature_cols,
)

plots.plot_lasso_beta_heatmap(
    df_lasso=df_heatmap_lagged,
    macro_cols=config.ALL_MACRO_COLS,
    gpr_cols=config.ALL_GPR_COLS,
    title="OLS by Sector with current and lagged betas\n(coloured = significant, grey = insignificant)",
    mode="ols",
)
heatmap_out_path = pdModelling.DATA_DIR / "analysis" / "plots" / "per_sector_beta_heatmap_with_lags.png"
plt.gcf().savefig(heatmap_out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {heatmap_out_path}")

plot_per_sector_grids(
    feature_cols=macro_base_cols,
    df_coef_model=df_coef,
    model_label="current-period model",
)
plot_per_sector_grids(
    feature_cols=lagged_feature_cols,
    df_coef_model=df_coef_lagged,
    model_label="lagged model",
)

# --- Forest plot: one panel per macro variable, sectors on y-axis ---
def plot_forest(
    df_beta_ci_model,
    variables_ci: list[str],
    title: str,
    out_name: str,
) -> None:
    plot_ci = df_beta_ci_model.copy()
    plot_ci["pretty_variable"] = plot_ci["variable"].map(_pretty_label).fillna(plot_ci["variable"])

    n_vars_ci = len(variables_ci)
    n_cols_ci = 3
    n_rows_ci = int(np.ceil(n_vars_ci / n_cols_ci))

    fig_ci, axes_ci = plt.subplots(
        n_rows_ci,
        n_cols_ci,
        figsize=(5.4 * n_cols_ci, 3.8 * n_rows_ci),
        squeeze=False,
    )
    axes_ci_flat = list(axes_ci.flat)
    fig_ci.patch.set_facecolor("white")

    for idx, variable in enumerate(variables_ci):
        ax = axes_ci_flat[idx]
        df_v_ci = (
            plot_ci[plot_ci["variable"] == variable]
            .sort_values(config.SECTOR_COL)
            .reset_index()
        )

        y_pos = np.arange(len(df_v_ci))
        beta = df_v_ci["beta"].to_numpy()
        ci_low = df_v_ci["ci_lower_999"].to_numpy()
        ci_high = df_v_ci["ci_upper_999"].to_numpy()
        xerr = np.vstack([beta - ci_low, ci_high - beta])

        ax.errorbar(beta, y_pos, xerr=xerr, fmt="o", color="#2166AC", ecolor="#2166AC", capsize=3)
        ax.axvline(0, color="black", linewidth=1, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_v_ci[config.SECTOR_COL], fontsize=8)
        ax.set_title(_pretty_label(variable), fontsize=10)
        ax.grid(True, axis="x", alpha=0.2)
        ax.invert_yaxis()

    for idx in range(n_vars_ci, len(axes_ci_flat)):
        axes_ci_flat[idx].set_visible(False)

    fig_ci.suptitle(title, fontsize=12, fontweight="bold")
    fig_ci.tight_layout()

    forest_out_path = pdModelling.DATA_DIR / "analysis" / "plots" / out_name
    fig_ci.savefig(forest_out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {forest_out_path}")


plot_forest(
    df_beta_ci_model=df_beta_ci,
    variables_ci=macro_base_cols,
    title="Forest plot of betas with 99.9% CI",
    out_name="per_variable_beta_forest_plot.png",
)
plot_forest(
    df_beta_ci_model=df_beta_ci_lagged,
    variables_ci=lagged_feature_cols,
    title="Forest plot of lagged-model betas with 99.9% CI",
    out_name="per_variable_beta_forest_plot_with_lags.png",
)

# Keep the figures open on Windows until the user closes them.
# (Using block=False makes the script exit immediately, and the windows disappear.)
plt.show()
# %%
