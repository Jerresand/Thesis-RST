"""Data loading and preparation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd


def clean_dataframe(df: pd.DataFrame, date_col_idx: int, value_col_idx: int) -> pd.DataFrame:
    """Clean dataframe by removing BOMs, parsing dates, and coercing numeric values."""
    df = df.copy()
    df.columns = df.columns.str.replace('\ufeff', '', regex=False)

    date_col = df.columns[date_col_idx]
    value_col = df.columns[value_col_idx]

    parsed = pd.to_datetime(df[date_col], format='%Y-%m', errors='coerce')
    fallback = pd.to_datetime(df[date_col], errors='coerce')
    df[date_col] = parsed.where(parsed.notna(), fallback).dt.to_period('M').dt.to_timestamp()

    if df[value_col].dtype == 'object':
        df[value_col] = (
            df[value_col]
            .astype(str)
            .str.replace(',', '.', regex=False)
        )
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    return df


def load_macro_data(
    gdp_path: str,
    interest_path: str,
    brent_path: str,
    fuel_path: str,
    cpi_path: str,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load and clean macroeconomic datasets."""
    df_gdp = pd.read_csv(gdp_path, sep=None, engine='python')
    df_interest = pd.read_csv(interest_path, sep=None, engine='python')
    df_brent = pd.read_csv(brent_path, sep=None, engine='python')
    df_fuel = pd.read_csv(fuel_path, sep=None, engine='python')
    df_cpi = pd.read_csv(cpi_path, sep=None, engine='python')

    df_gdp_cleaned = clean_dataframe(df_gdp, 0, 1)
    df_gdp_cleaned = df_gdp_cleaned.rename(columns={
        df_gdp_cleaned.columns[0]: 'Date',
        df_gdp_cleaned.columns[1]: 'GDP_Growth',
    }).dropna(subset=['Date', 'GDP_Growth'])

    df_interest_cleaned = clean_dataframe(df_interest, 0, 1)
    df_interest_cleaned = df_interest_cleaned.rename(columns={
        df_interest_cleaned.columns[0]: 'Date',
        df_interest_cleaned.columns[1]: 'Interest_Rate',
    })

    df_brent_cleaned = clean_dataframe(df_brent, 0, 1)
    df_brent_cleaned = df_brent_cleaned.rename(columns={
        df_brent_cleaned.columns[0]: 'Date',
        df_brent_cleaned.columns[1]: 'Brent_Oil',
    }).dropna(subset=['Date', 'Brent_Oil'])

    df_fuel_cleaned = clean_dataframe(df_fuel, 0, 1)
    df_fuel_cleaned = df_fuel_cleaned.rename(columns={
        df_fuel_cleaned.columns[0]: 'Date',
        df_fuel_cleaned.columns[1]: 'Fuel_Index',
    }).dropna(subset=['Date', 'Fuel_Index'])

    df_cpi_cleaned = clean_dataframe(df_cpi, 0, 1)
    df_cpi_cleaned = df_cpi_cleaned.rename(columns={
        df_cpi_cleaned.columns[0]: 'Date',
        df_cpi_cleaned.columns[1]: 'CPI',
    }).dropna(subset=['Date', 'CPI'])

    if verbose:
        print("Cleaned df_gdp head:")
        print(df_gdp_cleaned.head())
        print("\nCleaned df_interest head:")
        print(df_interest_cleaned.head())
        print("\nCleaned df_brent head:")
        print(df_brent_cleaned.head())
        print("\nCleaned df_fuel head:")
        print(df_fuel_cleaned.head())
        print("\nCleaned df_cpi head:")
        print(df_cpi_cleaned.head())

    return {
        'gdp': df_gdp_cleaned,
        'interest': df_interest_cleaned,
        'brent': df_brent_cleaned,
        'fuel': df_fuel_cleaned,
        'cpi': df_cpi_cleaned,
    }


def load_gpr_data(gpr_path: str, verbose: bool = True) -> pd.DataFrame:
    """Load and clean the geopolitical risk dataset."""
    df_gpr = pd.read_csv(gpr_path, sep=None, engine='python')
    df_gpr.columns = df_gpr.columns.str.replace('\ufeff', '', regex=False)
    df_gpr['month'] = pd.to_datetime(df_gpr['month'], format='%Y-%m-%d', errors='coerce')

    df_gpr_cleaned = df_gpr[['month', 'GPR']].copy()
    df_gpr_cleaned = df_gpr_cleaned.rename(columns={
        'month': 'Date',
        'GPR': 'GPR_Global',
    })

    df_gpr_cleaned['GPR_Global'] = pd.to_numeric(df_gpr_cleaned['GPR_Global'], errors='coerce')
    df_gpr_cleaned = df_gpr_cleaned.dropna(subset=['Date', 'GPR_Global'])

    if verbose:
        print("\nCleaned df_gpr head:")
        print(df_gpr_cleaned.head())

    return df_gpr_cleaned


def merge_macro_data(frames: Dict[str, pd.DataFrame], df_gpr: pd.DataFrame) -> pd.DataFrame:
    """Merge macro and GPR dataframes on Date using outer joins."""
    df_merged = (
        frames['gdp']
        .merge(frames['interest'], on='Date', how='outer')
        .merge(frames['brent'], on='Date', how='outer')
        .merge(frames['fuel'], on='Date', how='outer')
        .merge(df_gpr, on='Date', how='outer')
        .merge(frames['cpi'], on='Date', how='outer')
    )
    return df_merged


def summarize_macro_data(df_merged: pd.DataFrame, cols: Iterable[str], verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Compute covariance/correlation matrices and mean vector."""
    df_complete = df_merged[list(cols)].dropna()
    covariance_matrix = df_complete.cov()
    correlation_matrix = df_complete.corr()
    mean_vector = df_complete.mean()

    if verbose:
        print("\nCovariance Matrix (All variables):")
        print(covariance_matrix)
        print("\nCorrelation Matrix (All variables):")
        print(correlation_matrix)
        print("\nMean Vector (Macro + GPR variables):")
        print(mean_vector)

    return covariance_matrix, correlation_matrix, mean_vector


def load_pds_data(
    pds_path: str,
    verbose: bool = True,
    use_sic_sectors: bool = False,
    company_lookup_file: Optional[str | Path] = None,
    isin_to_sic_file: Optional[str | Path] = None,
    sic_codes_file: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Load PD data, normalize date column, and add PDzero column.
    
    Args:
        pds_path: Path to PD data CSV
        verbose: Print diagnostic information
        use_sic_sectors: If True, map sectors using SIC major groups
        company_lookup_file: Path to sectors/issuers CSV with issuer_identifier and issuer_name
                            (required if use_sic_sectors=True)
        isin_to_sic_file: Path to ISIN-to-SIC mapping CSV (required if use_sic_sectors=True)
        sic_codes_file: Path to SIC codes CSV (required if use_sic_sectors=True)
        
    Returns:
        DataFrame with Company_number, Date, PD columns, Sector, and PDzero
    """
    df_pds = pd.read_csv(pds_path)
    df_pds.columns = df_pds.columns.str.replace('\ufeff', '', regex=False)

    if 'Date' in df_pds.columns:
        df_pds['Date'] = pd.to_datetime(df_pds['Date'], format='%Y-%m')
    else:
        date_col_name = df_pds.columns[0]
        df_pds[date_col_name] = pd.to_datetime(df_pds[date_col_name], format='%Y-%m')
        df_pds = df_pds.rename(columns={date_col_name: 'Date'})

    if 'Sector' in df_pds.columns and not use_sic_sectors:
        df_pds = df_pds[['Company_number', 'Date', '12_month', 'Sector']]
    else:
        df_pds = df_pds[['Company_number', 'Date', '12_month']]
    
    if use_sic_sectors:
        if not all([company_lookup_file, isin_to_sic_file, sic_codes_file]):
            raise ValueError(
                "When use_sic_sectors=True, must provide company_lookup_file, "
                "isin_to_sic_file, and sic_codes_file"
            )
        
        from pd_pipeline.sensitivity import map_company_to_sector
        
        df_pds = map_company_to_sector(
            df_pds,
            company_lookup_file=company_lookup_file,
            isin_to_sic_file=isin_to_sic_file,
            sic_codes_file=sic_codes_file,
            company_col='Company_number',
            sector_col='Sector',
            verbose=verbose,
        )
    
    df_pds = df_pds.sort_values(['Company_number', 'Date'])
    first_pd = (
        df_pds.dropna(subset=['12_month'])
        .groupby('Company_number')['12_month']
        .first()
    )
    df_pds['PDzero'] = df_pds['Company_number'].map(first_pd)

    if verbose:
        print("\nColumns in df_pds:", df_pds.columns.tolist())
        print("\nFirst few rows of df_pds:")
        print(df_pds.head())

    return df_pds


def merge_pds_macro(df_pds: pd.DataFrame, df_macro: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Merge PDs with macro data on Date."""
    if verbose:
        print("\nBefore merge - PD data shape:", df_pds.shape)
        print("Before merge - Macro data shape:", df_macro.shape)
        if 'Sector' in df_pds.columns:
            non_null = df_pds['Sector'].notna().sum()
            print(f"Non-null Sector values: {non_null} ({100*non_null/len(df_pds):.1f}%)")

    df_final = pd.merge(df_pds, df_macro, on='Date', how='inner')

    if verbose:
        print("\nAfter merge - Final data shape:", df_final.shape)
        print(
            f"Retained {df_final.shape[0]} out of {df_pds.shape[0]} PD datapoints "
            f"({100*df_final.shape[0]/df_pds.shape[0]:.1f}%)"
        )
        print("\nFirst few rows of the merged DataFrame (df_final):")
        print(df_final.head())
        print("\nShape of df_final:")
        print(df_final.shape)
        print("\nInfo of df_final:")
        df_final.info()

    return df_final


def prepare_model_data(
    df_final: pd.DataFrame,
    predictor_cols: Iterable[str],
    sector_col: str = 'Sector',
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop rows with missing predictors/sector and return cleaned dataframe."""
    df_cleaned = df_final.dropna(subset=list(predictor_cols) + [sector_col]).copy()

    if verbose:
        print("Shape of df_final before dropping NaNs:", df_final.shape)
        print("Shape of df_final after dropping NaNs (predictors + Sector):", df_cleaned.shape)
        print(f"Retained {100*df_cleaned.shape[0]/df_final.shape[0]:.1f}% of observations")
        print("\nInfo of df_final_cleaned after handling NaNs:")
        df_cleaned.info()

    return df_cleaned


SIC_DIV2_RANGES = [
    (1000, 1999, 'Mining & Construction'),
    (2000, 2999, 'Light Manufacturing'),
    (3000, 3999, 'Heavy Manufacturing'),
    (4000, 4799, 'Transportation'),
    (4800, 4899, 'Communications'),
    (4900, 4999, 'Utilities'),
    (5000, 5999, 'Wholesale & Retail Trade'),
    (6000, 6999, 'Finance, Insurance & Real Estate'),
    (7000, 7999, 'Services'),
    (8000, 8999, 'Health, Legal & Educational Services'),
    (9000, 9999, 'Public Administration'),
]


def sic_to_div2_sector(sic) -> str:
    """Map a 4-digit SIC code to a div2 sector name."""
    try:
        sic_int = int(str(sic).strip())
    except (ValueError, TypeError):
        return 'Unassigned'
    for lo, hi, name in SIC_DIV2_RANGES:
        if lo <= sic_int <= hi:
            return name
    return 'Unassigned'


def build_sic_div2_pds_file(
    pds_path: str,
    isin_path: str,
    output_path: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate a PD data file with SIC div2 sector classification.

    Reads the base PD file and the ISIN→SIC mapping file, maps each company
    to its div2 sector using 4-digit SIC code ranges, and writes the result.

    Div2 sectors:
        Mining & Construction      1000–1999
        Light Manufacturing        2000–2999
        Heavy Manufacturing        3000–3999
        Transportation             4000–4799
        Communications             4800–4899
        Utilities                  4900–4999
        Wholesale & Retail Trade   5000–5999
        Finance, Insurance & RE    6000–6999
        Services                   7000–7999
        Health/Legal/Education     8000–8999
        Public Administration      9000–9999
        Unassigned                 (no SIC / outside ranges)
    """
    df_pds = pd.read_csv(pds_path)
    df_pds.columns = df_pds.columns.str.replace('\ufeff', '', regex=False)

    df_isin = pd.read_csv(isin_path, sep=';')
    df_isin.columns = df_isin.columns.str.replace('\ufeff', '', regex=False)

    df_isin['Company_number'] = pd.to_numeric(df_isin['issuer_identifier'], errors='coerce')
    company_sic = (
        df_isin[['Company_number', 'SIC']]
        .dropna(subset=['Company_number'])
        .drop_duplicates(subset=['Company_number'])
        .set_index('Company_number')['SIC']
    )

    df_pds['Sector'] = df_pds['Company_number'].map(company_sic).apply(sic_to_div2_sector)

    df_pds.to_csv(output_path, index=False)

    if verbose:
        print(f"✓ Saved {len(df_pds):,} rows to {output_path}")
        print("\nSector distribution:")
        print(df_pds['Sector'].value_counts().to_string())

    return df_pds


def add_macro_lags(
    df_macro: pd.DataFrame,
    cols_to_lag: Iterable[str],
    n_lags: int = 12,
) -> pd.DataFrame:
    """Add lagged columns for each macro variable up to n_lags months back.

    The input DataFrame must have a 'Date' column at monthly frequency.
    For each column in cols_to_lag, creates columns named '{col}_lag1' through
    '{col}_lag{n_lags}' where lag k is the value k months prior to the current date.
    """
    df = df_macro.sort_values('Date').copy()
    for col in cols_to_lag:
        if col not in df.columns:
            continue
        for k in range(1, n_lags + 1):
            df[f'{col}_lag{k}'] = df[col].shift(k)
    return df


def normalize_macro_columns(
    df: pd.DataFrame,
    cols: Iterable[str],
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Standardise macro (and GPR) columns in-place: subtract mean, divide by std.

    All lagged variants of a column are normalised with the **same** mean and std
    as the base column so that the scale is consistent across lags.

    Returns
    -------
    df : DataFrame with the named columns (and their *_lagN counterparts) replaced
         by their z-scores.
    scaler : Series indexed by *base* column name, values = (mean, std) tuples —
             stored for back-transformation if needed.
    """
    df = df.copy()
    base_cols = list(cols)
    stats: dict[str, tuple[float, float]] = {}

    for col in base_cols:
        if col not in df.columns:
            continue
        mu = float(df[col].mean())
        sigma = float(df[col].std(ddof=1))
        if sigma == 0:
            sigma = 1.0
        stats[col] = (mu, sigma)

        # normalise the base column
        df[col] = (df[col] - mu) / sigma

        # normalise all lag columns with the *same* mu/sigma
        k = 1
        while f'{col}_lag{k}' in df.columns:
            df[f'{col}_lag{k}'] = (df[f'{col}_lag{k}'] - mu) / sigma
            k += 1

    scaler = pd.Series({c: v for c, v in stats.items()})
    if verbose:
        print('Macro columns normalised (z-score, pooled across all rows):')
        for c, (m, s) in stats.items():
            print(f'  {c:35s}: mean={m:9.3f},  std={s:8.3f}')
    return df, scaler


def export_dataframe(df: pd.DataFrame, output_file: str, verbose: bool = True) -> None:
    """Export a dataframe to CSV with a lightweight summary."""
    df.to_csv(output_file, index=False)
    if verbose:
        print(f"✓ Successfully exported dataframe to: {output_file}")
        print(f"  - Rows: {df.shape[0]:,}")
        print(f"  - Columns: {df.shape[1]}")
        size_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  - File size: {size_mb:.2f} MB (approximate)")
