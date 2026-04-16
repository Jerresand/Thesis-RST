"""
Run per-sector pooled OLS (linear regression) of:
  y = logit_pd
  X = macro_base_cols = GDP_Growth, Interest_Rate, Brent_Oil, Fuel_Index, CPI, GPR_Global

Uses:
  data/analysis/df_sector_macro_relative_to_last.csv
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def main() -> None:
    project_root = pathlib.Path.cwd().resolve()
    if not (project_root / "pd_pipeline").exists():
        project_root = next(
            (candidate for candidate in [project_root, *project_root.parents] if (candidate / "pd_pipeline").exists()),
            project_root,
        )
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pd_pipeline import config

    data_path = project_root / "data" / "analysis" / "df_sector_macro_relative_to_last.csv"
    out_dir = project_root / "data" / "analysis" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    macro_cols = config.MACRO_COLS + config.GPR_COLS
    required = ["logit_pd", config.SECTOR_COL] + macro_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in dataset: {missing}")

    rows: list[dict[str, object]] = []
    for sector, df_s in df.groupby(config.SECTOR_COL, sort=True):
        df_s = df_s.dropna(subset=macro_cols + ["logit_pd"])
        if df_s.empty:
            continue

        X = df_s[macro_cols].to_numpy()
        y = df_s["logit_pd"].to_numpy()

        model = LinearRegression().fit(X, y)

        row: dict[str, object] = {
            config.SECTOR_COL: sector,
            "n_obs": int(len(df_s)),
            "r2": float(model.score(X, y)),
            "intercept": float(model.intercept_),
        }
        for col, coef in zip(macro_cols, model.coef_):
            row[col] = float(coef)
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(config.SECTOR_COL).reset_index(drop=True)
    out_path = out_dir / "per_sector_regression_logit_pd_vs_macro_relative.csv"
    out_df.to_csv(out_path, index=False)

    # Print a compact summary
    print(f"Per-sector regression saved to: {out_path}")
    cols_to_print = [config.SECTOR_COL, "n_obs", "r2", "intercept"] + macro_cols
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(out_df[cols_to_print].to_string(index=False))


if __name__ == "__main__":
    main()

