# Setup paths
import sys, pathlib

PROJECT_ROOT = pathlib.Path.cwd().resolve()
if not (PROJECT_ROOT / 'pd_pipeline').exists():
    PROJECT_ROOT = next(
        (candidate for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents] if (candidate / 'pd_pipeline').exists()),
        PROJECT_ROOT,
    )
NOTEBOOK_DIR = PROJECT_ROOT / '01_pd_analysis'
DATA_DIR = PROJECT_ROOT / 'data'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
from pd_pipeline import basel, capital, config, data, lasso, plots, portfolio, scenario, sensitivity

# Load and merge macro + GPR data
# Regression pipeline uses the interpolated monthly GDP series.
macro_frames = data.load_macro_data(
    gdp_path      = str(DATA_DIR / 'macro' / 'GDPREALGLOBAL_monthly.csv'),
    interest_path = str(DATA_DIR / 'macro' / 'intrest FRED.csv'),
    brent_path    = str(DATA_DIR / 'macro' / 'brent_oil_monthly.csv'),
    fuel_path     = str(DATA_DIR / 'macro' / 'fuel_index_monthly.csv'),
    cpi_path      = str(DATA_DIR / 'macro' / 'global_cpi_monthly.csv'),
    verbose       = False,
)

df_gpr = data.load_gpr_data(str(DATA_DIR / 'geopolitical' / 'data_gpr_Data_GPR.csv'), verbose=False)
df_merged = data.merge_macro_data(macro_frames, df_gpr)

df_gdp_quarterly = data.load_gdprealglobal_quarterly(str(DATA_DIR / 'macro' / 'GDPREALGLOBAL.csv'))
df_summary = (
    df_merged.drop(columns=['GDP_Growth'])
    .merge(df_gdp_quarterly, on='Date', how='inner')
)

cov_matrix, corr_matrix, mean_vector = data.summarize_macro_data(
    df_summary,
    config.ALL_PREDICTOR_COLS,
    verbose=True,
)


# Add t-1 … t-4 lags for all macro + GPR variables
df_merged = data.add_macro_lags(df_merged, config.MACRO_COLS + config.GPR_COLS, n_lags=4)
print(f"df_merged now has {df_merged.shape[1]} columns ({config.N_LAGS} lags added per variable)")



# (Re-run this cell whenever the source Fitch data or sector mapping changes)
import runpy
runpy.run_path(str(PROJECT_ROOT / 'build_pd_data.py'))

# Load PDs and merge with macro data
df_pds = data.load_pds_data(str(DATA_DIR / 'PDs' / 'fitch_pds_20260301_sic_div2_dedup.csv'), verbose=False)
df_final = data.merge_pds_macro(df_pds, df_merged, verbose=False)

# Drop rows where any current or lagged predictor is missing (removes first ~12 months per series)
df_final_cleaned = data.prepare_model_data(
    df_final,
    config.ALL_PREDICTOR_COLS_WITH_LAGS,
    sector_col=config.SECTOR_COL,
    verbose=False,
)

# Remove excluded sectors 
before_excl = len(df_final_cleaned)
df_final_cleaned = df_final_cleaned[
    ~df_final_cleaned[config.SECTOR_COL].isin(config.EXCLUDED_SECTORS)
].copy()
removed = before_excl - len(df_final_cleaned)
print(
    f"\nExcluded sectors {config.EXCLUDED_SECTORS}:\n"
    f"  Removed {removed:,} rows  →  {len(df_final_cleaned):,} rows remaining"
)
print("Remaining sectors:", sorted(df_final_cleaned[config.SECTOR_COL].unique()))