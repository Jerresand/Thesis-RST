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
    cpi_path      = str(DATA_DIR / 'macro' / 'global_cpi_mom_growth.csv'),
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
    f"  Removed {removed:,} rows  ->  {len(df_final_cleaned):,} rows remaining"
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


# --- Dataframe: Logit(PD) and macro/GPR variables ---
macro_base_cols = config.MACRO_COLS + config.GPR_COLS  # current-period only, no lags

df_sector_macro = df_sector_macro.sort_values([config.SECTOR_COL, 'Date'])

# Logit PD
df_sector_macro['logit_pd'] = np.log(
    df_sector_macro['12_month'] / (1 - df_sector_macro['12_month'])
)

# --- Export: last common date dataset (macro vars + logit(PD)) ---
# Defined as the latest `Date` where (all macro base columns are present)
# and `logit(PD)` can be computed on sector-date level.
out_dir = DATA_DIR / "analysis"
out_dir.mkdir(parents=True, exist_ok=True)

# Avoid -inf/inf in case PD hits 0/1 exactly
eps = 1e-9
pd_clipped = df_sector_macro["12_month"].clip(lower=eps, upper=1 - eps)
df_sector_macro["logit_pd_safe"] = np.log(pd_clipped / (1 - pd_clipped))

required_cols = macro_base_cols + ["logit_pd_safe"]
df_common = df_sector_macro.dropna(subset=required_cols).copy()
last_date = df_common["Date"].max()

df_dataset = (
    df_common[df_common["Date"] == last_date]
    [[config.SECTOR_COL, "Date", "logit_pd_safe"] + macro_base_cols]
    .rename(columns={"logit_pd_safe": "logit_pd"})
    .sort_values([config.SECTOR_COL])
    .reset_index(drop=True)
)

out_path = out_dir / "last_common_date_dataset.csv"
df_dataset.to_csv(out_path, index=False)
print(f"Last common date: {last_date.strftime('%Y-%m-%d')} | dataset rows: {len(df_dataset)}")

# --- Build dataframe relative to last date ---
# For each Sector, compute deltas versus the value at `last_date`, then remove
# the baseline rows (Date == last_date). `logit_pd` is kept in levels while the
# macro/other numeric columns are expressed relative to the last common date.
df_sector_macro_relative = df_sector_macro[df_sector_macro["Date"] != last_date].copy()

relative_cols = [
    c for c in df_sector_macro.columns
    if c not in [config.SECTOR_COL, "Date", "logit_pd", "logit_pd_safe"]
]

# Baseline values at last_date (indexed by sector)
baseline = (
    df_sector_macro[df_sector_macro["Date"] == last_date][[config.SECTOR_COL] + relative_cols]
    .set_index(config.SECTOR_COL)
)

for col in relative_cols:
    df_sector_macro_relative[col] = df_sector_macro_relative[col] - df_sector_macro_relative[config.SECTOR_COL].map(baseline[col])

out_rel_path = out_dir / "df_sector_macro_relative_to_last.csv"
df_sector_macro_relative.to_csv(out_rel_path, index=False)
print(
    f"Relative-to-last dataset: {len(df_sector_macro_relative):,} rows | "
    f"saved to {out_rel_path}"
)

# --- Per-sector regression on relative-to-last macro data ---
from sklearn.linear_model import LinearRegression

print("\nPer-sector regression (one model per sector):")
print("  y = logit_pd")
print("  X = [GDP_Growth, Interest_Rate, Brent_Oil, Fuel_Index, CPI, GPR_Global]")

plots_dir = DATA_DIR / "analysis" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
out_per_sector_path = plots_dir / "per_sector_regression_logit_pd_vs_macro_relative.csv"

rows: list[dict[str, object]] = []
for sector, df_s in df_sector_macro_relative.groupby(config.SECTOR_COL, sort=True):
    df_s = df_s.dropna(subset=macro_base_cols + ["logit_pd"])
    if df_s.empty:
        continue

    X_s = df_s[macro_base_cols].to_numpy()
    y_s = df_s["logit_pd"].to_numpy()

    m_s = LinearRegression().fit(X_s, y_s)

    row: dict[str, object] = {
        config.SECTOR_COL: sector,
        "n_obs": int(len(df_s)),
        "r2": float(m_s.score(X_s, y_s)),
        "intercept": float(m_s.intercept_),
    }
    for col, beta in zip(macro_base_cols, m_s.coef_):
        row[col] = float(beta)
    rows.append(row)

df_per_sector = pd.DataFrame(rows).sort_values(config.SECTOR_COL).reset_index(drop=True)
df_per_sector.to_csv(out_per_sector_path, index=False)
print(f"Saved: {out_per_sector_path}")
print(df_per_sector.to_string(index=False))
