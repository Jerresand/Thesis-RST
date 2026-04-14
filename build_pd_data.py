"""
Build the two kept PD outputs from the Fitch Corporate extract.

Outputs:
  1. fitch_long_term_pds_with_sic.csv
     Detailed long-term ratings with mapped 12-month PDs and SIC metadata.
  2. fitch_pds_20260301_sic_div2_dedup.csv
     Deduplicated company-month PD file in the format expected by pd_pipeline.

Filtering:
  - Keep long-term ratings only
  - Drop WD and NR rows
  - Drop any long-term ratings not present in the PD map
  - Drop rows without a mapped SIC
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
LONG_OUTPUT_PATH = PROJECT_ROOT / 'data/PDs/fitch_long_term_pds_with_sic.csv'
CODE_OUTPUT_PATH = PROJECT_ROOT / 'data/PDs/fitch_pds_20260301_sic_div2_dedup.csv'

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

# Keep long-term ratings only
long_term_types = {
    'Long Term Rating',
    'Long Term Issuer Default Rating',
    'Local Currency Long Term Issuer Default Rating',
    'Unenhanced Long Term Rating',
}
df = df[df['rating_type'].isin(long_term_types)].copy()
print(f"  After long-term filter: {len(df):,}")

# Remove withdrawn / not rated rows early
df = df[~df['rating'].isin({'WD', 'NR'})].copy()
print(f"  After WD/NR filter: {len(df):,}")

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
before = len(df)
df = df.dropna(subset=['12_month']).copy()
print(f"  Dropped {before - len(df):,} rows (unmapped long-term ratings)")

# ---------------------------------------------------------------------------
# 6. Map SIC → div-2 sector name
# ---------------------------------------------------------------------------
df['Sector'] = df['CompanySIC'].apply(sic_to_div2_sector)

# ---------------------------------------------------------------------------
# 7. Format Date as YYYY-MM
# ---------------------------------------------------------------------------
df['Date'] = pd.to_datetime(df['rating_action_date'], errors='coerce').dt.strftime('%Y-%m')

# ---------------------------------------------------------------------------
# 8. Build the detailed long-term output
# ---------------------------------------------------------------------------
df['Div2_range'] = df['CompanySIC'].apply(
    lambda sic: (
        '1000-1999' if 1000 <= int(sic) <= 1999 else
        '2000-2999' if 2000 <= int(sic) <= 2999 else
        '3000-3999' if 3000 <= int(sic) <= 3999 else
        '4000-4799' if 4000 <= int(sic) <= 4799 else
        '4800-4899' if 4800 <= int(sic) <= 4899 else
        '4900-4999' if 4900 <= int(sic) <= 4999 else
        '5000-5999' if 5000 <= int(sic) <= 5999 else
        '6000-6999' if 6000 <= int(sic) <= 6999 else
        '7000-7999' if 7000 <= int(sic) <= 7999 else
        '8000-8999' if 8000 <= int(sic) <= 8999 else
        '9000-9999' if 9000 <= int(sic) <= 9999 else
        '—'
    )
)

df_long = df[[
    'Company_number', 'issuer_name', 'Date', 'rating', 'rating_type',
    'rating_action_class', 'object_type_rated', 'instrument_name',
    'CUSIP_number', 'instrument_identifier', 'instrument_identifier_schema',
    '12_month', 'CompanySIC', 'Div2_range', 'Sector',
]].copy()
df_long = df_long.rename(columns={'issuer_name': 'Company_name', 'CompanySIC': 'SIC'})
df_long = df_long.sort_values(['Company_number', 'Date', 'instrument_name', 'rating']).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 9. Build the code-ready deduplicated company-month output
# ---------------------------------------------------------------------------
df_code = (
    df_long.groupby(['Company_number', 'Date', 'Sector'], as_index=False)['12_month']
    .mean()
)
for col in ['1_month', '3_month', '6_month', '24_month', '36_month', '60_month']:
    df_code[col] = None
df_code = df_code[[
    'Company_number', 'Date',
    '1_month', '3_month', '6_month', '12_month',
    '24_month', '36_month', '60_month', 'Sector',
]]
df_code = df_code.sort_values(['Company_number', 'Date']).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------
df_long.to_csv(LONG_OUTPUT_PATH, index=False)
df_code.to_csv(CODE_OUTPUT_PATH, index=False)
print(f"\n✓ Saved {len(df_long):,} rows to {LONG_OUTPUT_PATH}")
print(f"✓ Saved {len(df_code):,} rows to {CODE_OUTPUT_PATH}")
print("\nSector distribution:")
print(df_code['Sector'].value_counts().to_string())
print("\nSample code-ready output:")
print(df_code.head(10).to_string())
