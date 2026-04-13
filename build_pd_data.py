"""
Build fitch_pds_20260301_sic_div2.csv from the new Fitch Ratings Corporate file.

Steps:
  1. Load '20260301 Fitch Ratings Corporate.csv' (new Fitch data)
  2. Filter to Long Term Rating + issued_paid == true
  3. Look up SIC codes via IsinCusiptoSic.csv (ISIN first, CUSIP fallback)
  4. Drop obligors with no SIC code
  5. Convert letter ratings to numeric 12-month PD values
  6. Map SIC codes to div-2 sectors
  7. Save in the format expected by the pd_pipeline
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from pd_pipeline.data import sic_to_div2_sector

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FITCH_PATH    = PROJECT_ROOT / 'data/PDs/20260301 Fitch Ratings Corporate.csv'
ISIN_SIC_PATH = PROJECT_ROOT / 'data/PDs/IsinCusiptoSic.csv'
OUTPUT_PATH   = PROJECT_ROOT / 'data/PDs/fitch_pds_20260301_sic_div2.csv'

# ---------------------------------------------------------------------------
# Rating → 12-month PD mapping  (sourced from Fitch data averages)
# ---------------------------------------------------------------------------
PD_MAP = {
    'AAA':  0.0011,   'AA+':  0.000967, 'AA':   0.000833, 'AA-':  0.0007,
    'A+':   0.00065,  'A':    0.0006,   'A-':   0.0006,
    'BBB+': 0.0008,   'BBB':  0.0007,   'BBB-': 0.0020,
    'BB+':  0.0024,   'BB':   0.0050,   'BB-':  0.0103,
    'B+':   0.0137,   'B':    0.0193,   'B-':   0.0310,
    'CCC+': 0.2387,   'CCC':  0.2387,   'CCC-': 0.2387,
    'CC':   0.2387,   'C':    0.2387,   'D':    1.0000,
}

# ---------------------------------------------------------------------------
# 1. Build ISIN→SIC and CUSIP→SIC lookup tables from IsinCusiptoSic.csv
#    isin column  format: "I_<code>"   → strip "I_"
#    cusip column format: "CSP_<code>" → strip "CSP_"
# ---------------------------------------------------------------------------
print("Loading IsinCusiptoSic.csv …")
df_sic = pd.read_csv(ISIN_SIC_PATH, sep=';', dtype=str)
df_sic.columns = df_sic.columns.str.replace('\ufeff', '', regex=False)

isin_mask  = df_sic['isin'].str.startswith('I_', na=False)
cusip_mask = df_sic['cusip'].str.startswith('CSP_', na=False)

isin_to_sic = (
    df_sic.loc[isin_mask, ['isin', 'SIC']]
    .assign(isin_clean=lambda d: d['isin'].str[2:])
    .dropna(subset=['SIC'])
    .query("SIC != '' and SIC != 'nan'")
    .drop_duplicates(subset=['isin_clean'])
    .set_index('isin_clean')['SIC']
)

cusip_to_sic = (
    df_sic.loc[cusip_mask, ['cusip', 'SIC']]
    .assign(cusip_clean=lambda d: d['cusip'].str[4:])
    .dropna(subset=['SIC'])
    .query("SIC != '' and SIC != 'nan'")
    .drop_duplicates(subset=['cusip_clean'])
    .set_index('cusip_clean')['SIC']
)

print(f"  ISIN lookup : {len(isin_to_sic):,} entries")
print(f"  CUSIP lookup: {len(cusip_to_sic):,} entries")

# ---------------------------------------------------------------------------
# 2. Load new Fitch data and apply filters
# ---------------------------------------------------------------------------
print("\nLoading Fitch Ratings Corporate file …")
df = pd.read_csv(FITCH_PATH, dtype=str, low_memory=False)
df.columns = df.columns.str.replace('\ufeff', '', regex=False)
print(f"  Raw rows: {len(df):,}")

# Keep Long Term Rating only
df = df[df['rating_type'] == 'Long Term Rating'].copy()
print(f"  After Long Term Rating filter: {len(df):,}")

# Keep issued_paid == true only
df = df[df['issued_paid'].str.lower() == 'true'].copy()
print(f"  After issued_paid filter: {len(df):,}")

# ---------------------------------------------------------------------------
# 3. SIC lookup: ISIN first, then CUSIP fallback
# ---------------------------------------------------------------------------
df['SIC'] = pd.NA

# ISIN match
isin_rows = df['instrument_identifier_schema'] == 'ISIN'
df.loc[isin_rows, 'SIC'] = (
    df.loc[isin_rows, 'instrument_identifier'].map(isin_to_sic)
)

# CUSIP fallback for rows still missing SIC
cusip_rows = df['SIC'].isna() & df['CUSIP_number'].notna() & (df['CUSIP_number'] != '')
df.loc[cusip_rows, 'SIC'] = (
    df.loc[cusip_rows, 'CUSIP_number'].map(cusip_to_sic)
)

print(f"  Rows with SIC found: {df['SIC'].notna().sum():,}")

# ---------------------------------------------------------------------------
# 4. Build company-level SIC (first non-null SIC per company)
#    and drop obligors that have NO SIC at all
# ---------------------------------------------------------------------------
df['Company_number'] = pd.to_numeric(df['issuer_identifier'], errors='coerce')

company_sic = (
    df[['Company_number', 'SIC']]
    .dropna(subset=['Company_number', 'SIC'])
    .query("SIC != '' and SIC != 'nan'")
    .drop_duplicates(subset=['Company_number'])
    .set_index('Company_number')['SIC']
)

print(f"  Unique companies with SIC: {len(company_sic):,}")

df['CompanySIC'] = df['Company_number'].map(company_sic)
before = len(df)
df = df.dropna(subset=['CompanySIC'])
print(f"  Dropped {before - len(df):,} rows (obligors with no SIC)")
print(f"  Remaining rows: {len(df):,}")

# ---------------------------------------------------------------------------
# 5. Convert rating → 12-month PD
# ---------------------------------------------------------------------------
df['12_month'] = df['rating'].map(PD_MAP)

# ---------------------------------------------------------------------------
# 6. Map SIC → div-2 sector name
# ---------------------------------------------------------------------------
df['Sector'] = df['CompanySIC'].apply(sic_to_div2_sector)

# ---------------------------------------------------------------------------
# 7. Format Date as YYYY-MM
# ---------------------------------------------------------------------------
df['Date'] = pd.to_datetime(df['rating_action_date'], errors='coerce').dt.strftime('%Y-%m')

# ---------------------------------------------------------------------------
# 8. Build output dataframe matching target column order
# ---------------------------------------------------------------------------
df_out = df[['Company_number', 'Date', '12_month', 'Sector']].copy()
for col in ['1_month', '3_month', '6_month', '24_month', '36_month', '60_month']:
    df_out[col] = None

df_out = df_out[[
    'Company_number', 'Date',
    '1_month', '3_month', '6_month', '12_month',
    '24_month', '36_month', '60_month',
    'Sector',
]]

df_out = df_out.sort_values(['Company_number', 'Date']).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Saved {len(df_out):,} rows to {OUTPUT_PATH}")
print("\nSector distribution:")
print(df_out['Sector'].value_counts().to_string())
print("\nSample output:")
print(df_out.head(10).to_string())
