from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def plot_per_sector_r2(
    df_per_sector: pd.DataFrame,
    *,
    sector_col: str,
    plots_dir: Path,
    output_filename: str = "per_sector_r2_logit_pd_vs_macro_relative.png",
) -> Path:
    """
    Plot per-sector R^2 for logit(PD) vs macro variables (relative to last date).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df_per_sector[sector_col], df_per_sector["r2"], color="#4C72B0")
    ax.set_ylabel("R² (per sector)")
    ax.set_xlabel("Sector")
    ax.set_title("Per-sector R²: logit(PD) vs macro variables (relative to last date)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=45)

    out_path = plots_dir / output_filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_pred_vs_actual_per_sector(
    df_sector_macro_relative: pd.DataFrame,
    *,
    macro_base_cols: Iterable[str],
    sector_col: str,
    plots_dir: Path,
    output_filename: str = "per_sector_pred_vs_actual_logit_pd_vs_macro_relative.png",
    n_cols_grid: int = 4,
) -> Path:
    """
    For each sector, fit a per-sector linear regression on the relative-to-last dataset,
    then show predicted vs actual logit(PD) scatter plots (with y=x dashed reference line).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")

    sectors_sorted = sorted(df_sector_macro_relative[sector_col].unique().tolist())
    n_sectors = len(sectors_sorted)
    n_rows_grid = int(np.ceil(n_sectors / n_cols_grid))

    fig2, axes2 = plt.subplots(
        n_rows_grid,
        n_cols_grid,
        figsize=(4.2 * n_cols_grid, 3.8 * n_rows_grid),
    )
    axes2_flat = list(np.array(axes2).flat)

    macro_base_cols = list(macro_base_cols)

    for ax, sector in zip(axes2_flat, sectors_sorted):
        df_s = df_sector_macro_relative[df_sector_macro_relative[sector_col] == sector].copy()
        df_s = df_s.dropna(subset=macro_base_cols + ["logit_pd"])
        if df_s.empty:
            ax.set_visible(False)
            continue

        X_s = df_s[macro_base_cols].to_numpy()
        y_s = df_s["logit_pd"].to_numpy()

        m_s = LinearRegression().fit(X_s, y_s)
        y_hat = m_s.predict(X_s)

        ax.scatter(y_s, y_hat, s=18, alpha=0.45, color="#4C72B0")

        # y = x reference line
        lo = float(np.nanmin([y_s.min(), y_hat.min()]))
        hi = float(np.nanmax([y_s.max(), y_hat.max()]))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--", alpha=0.8)

        ax.set_title(str(sector))
        ax.set_xlabel("Actual logit_pd")
        ax.set_ylabel("Predicted logit_pd")
        ax.grid(True, alpha=0.25)

    # Hide unused panels
    for j in range(n_sectors, len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    fig2.tight_layout()
    out_path = plots_dir / output_filename
    fig2.savefig(out_path, dpi=200)
    plt.close(fig2)
    return out_path


if __name__ == "__main__":
    # When run as a script, load the CSV outputs produced by `01_pd_analysis/pdModelling.py`
    # and generate the two per-sector PNG plots.
    from pd_pipeline import config

    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_DIR = PROJECT_ROOT / "data"

    analysis_dir = DATA_DIR / "analysis"
    plots_dir = analysis_dir / "plots"

    df_per_sector_path = plots_dir / "per_sector_regression_logit_pd_vs_macro_relative.csv"
    df_sector_macro_relative_path = analysis_dir / "df_sector_macro_relative_to_last.csv"

    df_per_sector = pd.read_csv(df_per_sector_path)
    df_sector_macro_relative = pd.read_csv(df_sector_macro_relative_path)

    macro_base_cols = config.MACRO_COLS + config.GPR_COLS

    r2_plot_path = plot_per_sector_r2(
        df_per_sector,
        sector_col=config.SECTOR_COL,
        plots_dir=plots_dir,
    )
    print(f"Per-sector R² plot saved to: {r2_plot_path}")

    pred_plot_path = plot_pred_vs_actual_per_sector(
        df_sector_macro_relative,
        macro_base_cols=macro_base_cols,
        sector_col=config.SECTOR_COL,
        plots_dir=plots_dir,
    )
    print(f"Per-sector predicted-vs-actual plot saved to: {pred_plot_path}")

