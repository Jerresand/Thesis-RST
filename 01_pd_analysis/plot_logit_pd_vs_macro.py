"""
Plot logit(PD) vs macro variables using the relative-to-last dataset.

Input:
  data/analysis/df_sector_macro_relative_to_last.csv

Output:
  data/analysis/plots/logit_pd_vs_macro_relative.png
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import sys


def main() -> None:
    project_root = pathlib.Path.cwd().resolve()
    if not (project_root / "pd_pipeline").exists():
        project_root = next(
            (candidate for candidate in [project_root, *project_root.parents] if (candidate / "pd_pipeline").exists()),
            project_root,
        )

    # Make sure `pd_pipeline` is importable when running from a different CWD.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pd_pipeline import config

    data_path = project_root / "data" / "analysis" / "df_sector_macro_relative_to_last.csv"
    out_dir = project_root / "data" / "analysis" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if "logit_pd" not in df.columns:
        raise KeyError("Expected column `logit_pd` in the relative dataset.")

    macro_cols = config.MACRO_COLS + config.GPR_COLS

    missing = [c for c in macro_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing macro columns in dataset: {missing}")

    sns.set_theme(style="whitegrid", context="paper")

    n = len(macro_cols)
    n_cols_grid = 3
    n_rows = int((n + n_cols_grid - 1) / n_cols_grid)

    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(5.2 * n_cols_grid, 4.2 * n_rows))
    axes_flat = list(axes.flat)

    for i, col in enumerate(macro_cols):
        ax = axes_flat[i]
        sns.scatterplot(
            data=df,
            x=col,
            y="logit_pd",
            hue=config.SECTOR_COL,
            ax=ax,
            s=18,
            alpha=0.5,
            legend=False,
        )
        ax.axhline(0, color="black", linewidth=1, alpha=0.6)
        ax.axvline(0, color="black", linewidth=1, alpha=0.6)
        ax.set_title(f"logit_pd vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("logit_pd")

    # Hide any unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Single legend (based on first subplot)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", title=config.SECTOR_COL)

    fig.tight_layout()
    out_path = out_dir / "logit_pd_vs_macro_relative.png"
    fig.savefig(out_path, dpi=200)
    size = out_path.stat().st_size if out_path.exists() else None
    print(f"Saved plot: {out_path} (exists={out_path.exists()}, size={size})")

    # In some remote/headless setups, matplotlib uses the `Agg` backend and
    # `show()` will not display anything. We still save the PNG, so the plot
    # is always accessible.
    try:
        from matplotlib import get_backend

        backend = get_backend()
        if backend.lower() != "agg":
            print(f"Matplotlib backend: {backend} -> showing plot window...")
            plt.show()
        else:
            print(f"Matplotlib backend: {backend} -> not showing window; open the PNG instead.")
    except Exception:
        print("Could not detect matplotlib backend; open the PNG instead.")

    plt.close(fig)


if __name__ == "__main__":
    main()

