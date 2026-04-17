# Setup paths
import sys, pathlib
import numpy as np

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

# Load PDs, expand to monthly panel, and merge with macro data
df_pds_raw = data.load_pds_data(str(DATA_DIR / 'PDs' / 'fitch_pds_20260301_sic_div2_dedup.csv'), verbose=False)
df_pds = data.expand_pds_to_monthly_panel(df_pds_raw, verbose=True)
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



df_final_cleaned.info()

# Mean PD per sector and date
df_sector_pd = (
    df_final_cleaned
    .groupby([config.SECTOR_COL, 'Date'], as_index=False)['12_month']
    .mean()
    .sort_values([config.SECTOR_COL, 'Date'])
    .reset_index(drop=True)
)
print(f"\nSector-date mean PD: {len(df_sector_pd):,} rows | "
      f"{df_sector_pd[config.SECTOR_COL].nunique()} sectors | "
      f"{df_sector_pd['Date'].nunique()} unique dates")
print(df_sector_pd.head(10).to_string())

# Merge sector-date PD with macro data
df_sector_macro = data.merge_pds_macro(df_sector_pd, df_merged, verbose=False)
df_sector_macro = df_sector_macro.sort_values([config.SECTOR_COL, 'Date']).reset_index(drop=True)
print(f"\nSector-macro merged: {len(df_sector_macro):,} rows | shape: {df_sector_macro.shape}")
print(df_sector_macro.head(10).to_string())


# --- Delta dataframe: ΔLogit(PD) and Δ for every macro/GPR variable ---
macro_base_cols = config.MACRO_COLS + config.GPR_COLS  # current-period only, no lags

df_sector_macro = df_sector_macro.sort_values([config.SECTOR_COL, 'Date'])

# Logit PD (needed before differencing)
df_sector_macro['logit_pd'] = np.log(
    df_sector_macro['12_month'] / (1 - df_sector_macro['12_month'])
)

# Delta logit PD: diff within each sector
df_sector_macro['delta_logit_pd'] = (
    df_sector_macro.groupby(config.SECTOR_COL)['logit_pd'].diff()
)

# Delta macro: identical across sectors so diff on unique dates, then merge back
macro_unique = (
    df_sector_macro[['Date'] + macro_base_cols]
    .drop_duplicates('Date')
    .sort_values('Date')
    .copy()
)
delta_macro_cols = [f'delta_{c}' for c in macro_base_cols]
macro_unique[delta_macro_cols] = macro_unique[macro_base_cols].diff()

df_sector_macro = df_sector_macro.merge(
    macro_unique[['Date'] + delta_macro_cols], on='Date', how='left'
) 

# Assemble final delta df
df_delta = (
    df_sector_macro[[config.SECTOR_COL, 'Date', 'delta_logit_pd'] + delta_macro_cols]
    .dropna()
    .sort_values([config.SECTOR_COL, 'Date'])
    .reset_index(drop=True)
)

print(f"\nDelta df: {len(df_delta):,} rows | {df_delta[config.SECTOR_COL].nunique()} sectors | "
      f"{df_delta['Date'].nunique()} unique dates")
print(f"Columns: {df_delta.columns.tolist()}")
print(df_delta.head(10).to_string())


# regression analysis for each sector
from sklearn.linear_model import LinearRegression
import pandas as pd

X = df_delta[delta_macro_cols].values
y = df_delta['delta_logit_pd'].values

model = LinearRegression().fit(X, y)
coefs = pd.Series(model.coef_, index=delta_macro_cols)
print(f"\nR² (pooled): {model.score(X, y):.4f}")
print("Coefficients:")
print(coefs.to_string())


### Regression analysis


