"""
Create a dataset for the last common date where:
  - all macro variables (GDP_Growth, Interest_Rate, Brent_Oil, Fuel_Index, CPI, GPR_Global)
    are available, and
  - logit(PD) is available on sector-date level.

Output:
  - data/analysis/last_common_date_dataset.csv
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd


def main() -> None:
    # Setup paths (mirrors the approach in `pdModelling.py`)
    project_root = pathlib.Path.cwd().resolve()
    if not (project_root / "pd_pipeline").exists():
        project_root = next(
            (candidate for candidate in [project_root, *project_root.parents] if (candidate / "pd_pipeline").exists()),
            project_root,
        )

    # Make sure `pd_pipeline` is importable when running from a different CWD.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from pd_pipeline import config, data

    data_dir = project_root / "data"

    # Load + merge macro + GPR (same logic as in `pdModelling.py`)
    macro_frames = data.load_macro_data(
        gdp_path=str(data_dir / "macro" / "GDPREALGLOBAL_monthly.csv"),
        interest_path=str(data_dir / "macro" / "intrest FRED.csv"),
        brent_path=str(data_dir / "macro" / "brent_oil_monthly.csv"),
        fuel_path=str(data_dir / "macro" / "fuel_index_monthly.csv"),
        cpi_path=str(data_dir / "macro" / "global_cpi_monthly.csv"),
        verbose=False,
    )

    df_gpr = data.load_gpr_data(str(data_dir / "geopolitical" / "data_gpr_Data_GPR.csv"), verbose=False)
    df_merged = data.merge_macro_data(macro_frames, df_gpr)

    # Replace monthly GDP_Growth with quarterly GDP series (same as `pdModelling.py`)
    df_gdp_quarterly = data.load_gdprealglobal_quarterly(str(data_dir / "macro" / "GDPREALGLOBAL.csv"))
    df_merged = df_merged.drop(columns=["GDP_Growth"]).merge(df_gdp_quarterly, on="Date", how="inner")

    macro_base_cols = config.MACRO_COLS + config.GPR_COLS  # without lags

    # Load PDs, expand to monthly panel, and merge with macro
    df_pds_raw = data.load_pds_data(
        str(data_dir / "PDs" / "fitch_pds_20260301_sic_div2_dedup.csv"),
        verbose=False,
    )
    df_pds = data.expand_pds_to_monthly_panel(df_pds_raw, verbose=False)
    df_final = data.merge_pds_macro(df_pds, df_merged, verbose=False)

    # Exclude sectors (same as `pdModelling.py`)
    df_final = df_final[~df_final[config.SECTOR_COL].isin(config.EXCLUDED_SECTORS)].copy()

    # Sector-date mean PD (mirrors `pdModelling.py`)
    df_sector_pd = (
        df_final.groupby([config.SECTOR_COL, "Date"], as_index=False)["12_month"]
        .mean()
        .sort_values([config.SECTOR_COL, "Date"])
        .reset_index(drop=True)
    )

    # Merge sector-date PD with macro data
    df_sector_macro = data.merge_pds_macro(df_sector_pd, df_merged, verbose=False)
    df_sector_macro = df_sector_macro.sort_values([config.SECTOR_COL, "Date"]).reset_index(drop=True)

    # logit(PD) on sector-date level
    pd_col = "12_month"
    eps = 1e-9
    pd_clipped = df_sector_macro[pd_col].clip(lower=eps, upper=1 - eps)
    df_sector_macro["logit_pd"] = np.log(pd_clipped / (1 - pd_clipped))

    # Find last common date where ALL macro vars AND logit_pd are non-missing
    required_cols = macro_base_cols + ["logit_pd"]
    df_common = df_sector_macro.dropna(subset=required_cols).copy()
    if df_common.empty:
        raise RuntimeError("No rows found with complete macro variables and logit_pd.")

    last_date = df_common["Date"].max()
    df_dataset = df_common[df_common["Date"] == last_date].copy()

    # Keep only the columns the user asked for (plus Sector/Date for indexing)
    keep_cols = [config.SECTOR_COL, "Date", "logit_pd"] + macro_base_cols
    df_dataset = df_dataset[keep_cols].sort_values([config.SECTOR_COL]).reset_index(drop=True)

    out_dir = data_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "last_common_date_dataset.csv"
    df_dataset.to_csv(out_path, index=False)

    print(f"Last common date: {last_date.strftime('%Y-%m-%d')}")
    print(f"Dataset shape: {df_dataset.shape} (rows x columns)")
    print(f"Saved to: {out_path}")
    print("Sectors:", sorted(df_dataset[config.SECTOR_COL].unique().tolist()))


if __name__ == "__main__":
    main()

