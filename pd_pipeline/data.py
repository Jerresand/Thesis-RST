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


def load_gdprealglobal_monthly(gdp_path: str) -> pd.DataFrame:
    """Load the GDPREALGLOBAL monthly file with explicit parsing and validation.

    The source file is semicolon-delimited with no header and may contain a few
    initial months with missing values. Those rows are dropped explicitly.
    """
    df_gdp = pd.read_csv(
        gdp_path,
        sep=';',
        header=None,
        names=['Date', 'GDP_Growth'],
        engine='python',
    )
    df_gdp.columns = df_gdp.columns.str.replace('\ufeff', '', regex=False)

    df_gdp['Date'] = pd.to_datetime(df_gdp['Date'], format='%Y-%m', errors='coerce')
    df_gdp['GDP_Growth'] = (
        df_gdp['GDP_Growth']
        .astype(str)
        .str.replace(',', '.', regex=False)
    )
    df_gdp['GDP_Growth'] = pd.to_numeric(df_gdp['GDP_Growth'], errors='coerce')

    missing_before = int(df_gdp['GDP_Growth'].isna().sum())
    df_gdp = df_gdp.dropna(subset=['Date', 'GDP_Growth']).copy()
    df_gdp = df_gdp.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')

    if df_gdp.empty:
        raise ValueError(f'No valid GDP rows were parsed from {gdp_path}')
    if df_gdp['Date'].duplicated().any():
        raise ValueError(f'Duplicate GDP dates remain after cleaning in {gdp_path}')
    if df_gdp['Date'].isna().any() or df_gdp['GDP_Growth'].isna().any():
        raise ValueError(f'Missing GDP date/value remains after cleaning in {gdp_path}')

    expected = pd.date_range(df_gdp['Date'].min(), df_gdp['Date'].max(), freq='MS')
    missing_months = expected.difference(df_gdp['Date'])
    if len(missing_months) > 0:
        raise ValueError(
            f'GDP series has {len(missing_months)} missing month(s) after cleaning; '
            f'first missing month: {missing_months[0].date()}'
        )

    df_gdp.attrs['dropped_missing_rows'] = missing_before
    return df_gdp


def load_gdprealglobal_quarterly(gdp_path: str) -> pd.DataFrame:
    """Load the non-interpolated quarterly GDPREALGLOBAL file.

    Quarter labels are mapped to the first day of the quarter-ending month so
    they align with the monthly macro series used elsewhere in the pipeline.
    """
    df_gdp = pd.read_csv(
        gdp_path,
        sep=';',
        header=None,
        names=['Date', 'GDP_Growth'],
        engine='python',
    )
    df_gdp.columns = df_gdp.columns.str.replace('\ufeff', '', regex=False)
    df_gdp['GDP_Growth'] = (
        df_gdp['GDP_Growth']
        .astype(str)
        .str.replace(',', '.', regex=False)
    )
    df_gdp['GDP_Growth'] = pd.to_numeric(df_gdp['GDP_Growth'], errors='coerce')
    df_gdp = df_gdp.dropna(subset=['GDP_Growth']).copy()

    quarter_period = pd.PeriodIndex(df_gdp['Date'].astype(str), freq='Q')
    df_gdp['Date'] = quarter_period.asfreq('M', how='end').to_timestamp()
    df_gdp = df_gdp.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')

    if df_gdp.empty:
        raise ValueError(f'No valid quarterly GDP rows were parsed from {gdp_path}')
    if df_gdp['Date'].duplicated().any():
        raise ValueError(f'Duplicate quarterly GDP dates remain after cleaning in {gdp_path}')

    return df_gdp[['Date', 'GDP_Growth']]


def load_macro_data(
    gdp_path: str,
    interest_path: str,
    brent_path: str,
    fuel_path: str,
    cpi_path: str,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load and clean macroeconomic datasets."""
    df_interest = pd.read_csv(interest_path, sep=None, engine='python')
    df_brent = pd.read_csv(brent_path, sep=None, engine='python')
    df_fuel = pd.read_csv(fuel_path, sep=None, engine='python')
    df_cpi = pd.read_csv(cpi_path, sep=None, engine='python')

    df_gdp_cleaned = load_gdprealglobal_monthly(gdp_path)

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
        print(
            "\nGDP validation:"
            f"\n  rows={len(df_gdp_cleaned)}"
            f"\n  date range={df_gdp_cleaned['Date'].min().date()} -> {df_gdp_cleaned['Date'].max().date()}"
            f"\n  dropped missing rows={df_gdp_cleaned.attrs.get('dropped_missing_rows', 0)}"
            f"\n  duplicates remaining={int(df_gdp_cleaned['Date'].duplicated().sum())}"
        )
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

    has_sector = 'Sector' in df_pds.columns and not use_sic_sectors
    has_rating_cat = 'rating_category' in df_pds.columns

    if has_sector and has_rating_cat:
        df_pds = df_pds[['Company_number', 'Date', '12_month', 'Sector', 'rating_category']]
    elif has_sector:
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

    if verbose:
        print("\nColumns in df_pds:", df_pds.columns.tolist())
        print("\nFirst few rows of df_pds:")
        print(df_pds.head())

    return df_pds


def expand_pds_to_monthly_panel(
    df_dedup: pd.DataFrame,
    end_date: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Expand the dedup PD log into a full monthly panel.

    The dedup file is treated as a changelog: a rating event is valid until
    the next event.  The rules are:

    * A company enters the panel on the month of its first Standard rating.
    * The most recent Standard PD is carried forward (forward-fill) until
      either a new Standard event or a WD event.
    * On a WD event the company is removed from the panel.
    * If the company is re-rated (Standard) after a WD it re-enters the panel
      from that month onward.  Multiple WD / re-rating cycles are supported.
    * Active companies with no final WD are kept through *end_date* (defaults
      to the latest date present in *df_dedup*).

    Parameters
    ----------
    df_dedup : DataFrame
        Output of ``load_pds_data`` that includes the ``rating_category``
        column (values ``'Standard'`` or ``'WD'``).
    end_date : str or None
        Last month to include, e.g. ``'2026-03'``.  Defaults to the maximum
        date in *df_dedup*.
    verbose : bool
        Print summary statistics.

    Returns
    -------
    DataFrame with columns: Company_number, Date, 12_month, Sector
    """
    df = df_dedup.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    if end_date is None:
        panel_end = df['Date'].max()
    else:
        panel_end = pd.to_datetime(end_date, format='%Y-%m')

    chunks: list[pd.DataFrame] = []

    for company_id, group in df.groupby('Company_number', sort=False):
        group = group.sort_values('Date').reset_index(drop=True)
        sector = group['Sector'].dropna().iloc[0] if group['Sector'].notna().any() else None

        std = (
            group.loc[group['rating_category'] == 'Standard', ['Date', '12_month']]
            .drop_duplicates(subset=['Date'])
            .sort_values('Date')
            .reset_index(drop=True)
        )
        wd = (
            group.loc[group['rating_category'] == 'WD', ['Date']]
            .drop_duplicates()
            .sort_values('Date')
            .reset_index(drop=True)
        )

        if std.empty:
            continue

        months = pd.DataFrame(
            {'Date': pd.date_range(std['Date'].min(), panel_end, freq='MS')}
        )

        # Carry forward last Standard PD and record its date
        std_aug = std.copy()
        std_aug['last_std_date'] = std_aug['Date']
        months = pd.merge_asof(months, std_aug, on='Date', direction='backward')

        # Carry forward last WD date (NaT when none has occurred yet)
        if not wd.empty:
            wd_aug = wd.copy()
            wd_aug['last_wd_date'] = wd_aug['Date']
            months = pd.merge_asof(
                months,
                wd_aug.rename(columns={'Date': 'wd_key'})[['wd_key', 'last_wd_date']],
                left_on='Date', right_on='wd_key',
                direction='backward',
            ).drop(columns='wd_key')
        else:
            months['last_wd_date'] = pd.NaT

        # Active = last Standard event is at least as recent as last WD event
        active = months['last_wd_date'].isna() | (months['last_std_date'] >= months['last_wd_date'])
        months = months.loc[active & months['12_month'].notna()].copy()

        if months.empty:
            continue

        months['Company_number'] = company_id
        months['Sector'] = sector
        chunks.append(months[['Company_number', 'Date', '12_month', 'Sector']])

    if not chunks:
        return pd.DataFrame(columns=['Company_number', 'Date', '12_month', 'Sector'])

    panel = pd.concat(chunks, ignore_index=True).sort_values(['Company_number', 'Date'])

    if verbose:
        n_companies = panel['Company_number'].nunique()
        date_min = panel['Date'].min().strftime('%Y-%m')
        date_max = panel['Date'].max().strftime('%Y-%m')
        print(f"Monthly panel: {len(panel):,} rows | {n_companies:,} companies | {date_min} – {date_max}")

    return panel


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


FITCH_LONG_TERM_TYPES: frozenset[str] = frozenset({
    'Long Term Rating',
    'Long Term Issuer Default Rating',
    'Local Currency Long Term Issuer Default Rating',
    'Unenhanced Long Term Rating',
})

FITCH_SPECIAL_RATINGS: frozenset[str] = frozenset({'WD', 'NR'})

FITCH_PD_MAP: dict[str, float] = {
    'AAA':  0.0011,   'AA+':  0.000967, 'AA':   0.000833, 'AA-':  0.0007,
    'A+':   0.00065,  'A':    0.0006,   'A-':   0.0006,
    'BBB+': 0.0008,   'BBB':  0.0007,   'BBB-': 0.0020,
    'BB+':  0.0024,   'BB':   0.0050,   'BB-':  0.0103,
    'B+':   0.0137,   'B':    0.0193,   'B-':   0.0310,
    'CCC+': 0.2387,   'CCC':  0.2387,   'CCC-': 0.2387,
    'CC':   0.2387,   'C':    0.2387,   'D':    1.0000,
}

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


def _sic_to_div2_range(sic) -> str:
    """Return the SIC div-2 numeric range string for a given SIC code."""
    try:
        sic_int = int(sic)
    except (ValueError, TypeError):
        return '—'
    if 1000 <= sic_int <= 1999: return '1000-1999'
    if 2000 <= sic_int <= 2999: return '2000-2999'
    if 3000 <= sic_int <= 3999: return '3000-3999'
    if 4000 <= sic_int <= 4799: return '4000-4799'
    if 4800 <= sic_int <= 4899: return '4800-4899'
    if 4900 <= sic_int <= 4999: return '4900-4999'
    if 5000 <= sic_int <= 5999: return '5000-5999'
    if 6000 <= sic_int <= 6999: return '6000-6999'
    if 7000 <= sic_int <= 7999: return '7000-7999'
    if 8000 <= sic_int <= 8999: return '8000-8999'
    if 9000 <= sic_int <= 9999: return '9000-9999'
    return '—'


def build_fitch_pd_data(
    fitch_path: str,
    isin_sic_path: str,
    long_output_path: str,
    code_output_path: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the two Fitch PD output files from the raw Corporate extract.

    Outputs written to disk:

    1. *long_output_path* (``fitch_long_term_pds_with_sic.csv``)
       Instrument-level long-term ratings with mapped 12-month PDs, SIC
       metadata, and a ``rating_category`` column.  Includes WD/NR events for
       companies that have at least one mappable standard rating.

    2. *code_output_path* (``fitch_pds_20260301_sic_div2_dedup.csv``)
       Deduplicated company-month file ready for ``load_pds_data``.  Standard
       rows carry the mean 12-month PD; WD/NR rows carry ``NaN`` for all PD
       horizons.  A ``rating_category`` column marks each row as
       ``'Standard'``, ``'WD'``, or ``'NR'``.  No company appears that is not
       already present in the standard output.

    Returns
    -------
    df_long : instrument-level DataFrame (written to long_output_path)
    df_code : deduplicated company-month DataFrame (written to code_output_path)
    """
    if verbose:
        print("Loading IsinCusiptoSic.csv …")
    df_sic_raw = pd.read_csv(isin_sic_path, sep=';', dtype=str)
    df_sic_raw.columns = df_sic_raw.columns.str.replace('\ufeff', '', regex=False)

    isin_mask  = df_sic_raw['isin'].str.startswith('I_', na=False)
    cusip_mask = df_sic_raw['cusip'].str.startswith('CSP_', na=False)

    isin_to_sic = (
        df_sic_raw.loc[isin_mask, ['isin', 'SIC']]
        .assign(isin_clean=lambda d: d['isin'].str[2:])
        .dropna(subset=['SIC'])
        .query("SIC != '' and SIC != 'nan'")
        .drop_duplicates(subset=['isin_clean'])
        .set_index('isin_clean')['SIC']
    )
    cusip_to_sic = (
        df_sic_raw.loc[cusip_mask, ['cusip', 'SIC']]
        .assign(cusip_clean=lambda d: d['cusip'].str[4:])
        .dropna(subset=['SIC'])
        .query("SIC != '' and SIC != 'nan'")
        .drop_duplicates(subset=['cusip_clean'])
        .set_index('cusip_clean')['SIC']
    )
    if verbose:
        print(f"  ISIN lookup : {len(isin_to_sic):,} entries")
        print(f"  CUSIP lookup: {len(cusip_to_sic):,} entries")

    if verbose:
        print("\nLoading Fitch Ratings Corporate file …")
    df_all = pd.read_csv(fitch_path, dtype=str, low_memory=False)
    df_all.columns = df_all.columns.str.replace('\ufeff', '', regex=False)
    if verbose:
        print(f"  Raw rows: {len(df_all):,}")

    df_all = df_all[df_all['rating_type'].isin(FITCH_LONG_TERM_TYPES)].copy()
    if verbose:
        print(f"  After long-term filter: {len(df_all):,}")

    # SIC lookup: ISIN first, then CUSIP fallback (all rows including WD/NR)
    df_all['SIC'] = pd.NA
    isin_rows = df_all['instrument_identifier_schema'] == 'ISIN'
    df_all.loc[isin_rows, 'SIC'] = df_all.loc[isin_rows, 'instrument_identifier'].map(isin_to_sic)
    cusip_rows = df_all['SIC'].isna() & df_all['CUSIP_number'].notna() & (df_all['CUSIP_number'] != '')
    df_all.loc[cusip_rows, 'SIC'] = df_all.loc[cusip_rows, 'CUSIP_number'].map(cusip_to_sic)
    if verbose:
        print(f"  Rows with SIC found: {df_all['SIC'].notna().sum():,}")

    # Company-level SIC derived from standard (non-WD/NR) rows only
    df_all['Company_number'] = pd.to_numeric(df_all['issuer_identifier'], errors='coerce')
    df_std_for_sic = df_all[~df_all['rating'].isin(FITCH_SPECIAL_RATINGS)]
    company_sic = (
        df_std_for_sic[['Company_number', 'SIC']]
        .dropna(subset=['Company_number', 'SIC'])
        .query("SIC != '' and SIC != 'nan'")
        .drop_duplicates(subset=['Company_number'])
        .set_index('Company_number')['SIC']
    )
    if verbose:
        print(f"  Unique companies with SIC (from standard ratings): {len(company_sic):,}")
    df_all['CompanySIC'] = df_all['Company_number'].map(company_sic)

    # Split standard vs WD/NR
    df_std   = df_all[~df_all['rating'].isin(FITCH_SPECIAL_RATINGS)].copy()
    df_wd_nr = df_all[df_all['rating'].isin(FITCH_SPECIAL_RATINGS)].copy()

    before = len(df_std)
    df_std = df_std.dropna(subset=['CompanySIC'])
    if verbose:
        print(f"  Dropped {before - len(df_std):,} standard rows (obligors with no SIC)")

    df_std['12_month'] = df_std['rating'].map(FITCH_PD_MAP)
    before = len(df_std)
    df_std = df_std.dropna(subset=['12_month']).copy()
    if verbose:
        print(f"  Dropped {before - len(df_std):,} standard rows (unmapped long-term ratings)")

    valid_companies = set(df_std['Company_number'].dropna().unique())
    if verbose:
        print(f"  Valid companies (standard ratings, has SIC + mappable PD): {len(valid_companies):,}")

    df_wd_nr = df_wd_nr[df_wd_nr['Company_number'].isin(valid_companies)].copy()
    df_wd_nr = df_wd_nr.dropna(subset=['CompanySIC'])
    df_wd_nr['12_month'] = float('nan')
    if verbose:
        print(f"  WD/NR rows kept (valid companies only): {len(df_wd_nr):,}")
        print(f"    WD: {(df_wd_nr['rating'] == 'WD').sum():,}  |  NR: {(df_wd_nr['rating'] == 'NR').sum():,}")

    for df_part in [df_std, df_wd_nr]:
        df_part['Sector'] = df_part['CompanySIC'].apply(sic_to_div2_sector)
        df_part['Date'] = pd.to_datetime(df_part['rating_action_date'], errors='coerce').dt.strftime('%Y-%m')
        df_part['Div2_range'] = df_part['CompanySIC'].apply(_sic_to_div2_range)

    df_std['rating_category']   = 'Standard'
    df_wd_nr['rating_category'] = df_wd_nr['rating']

    long_cols = [
        'Company_number', 'issuer_name', 'Date', 'rating', 'rating_category',
        'rating_type', 'rating_action_class', 'object_type_rated', 'instrument_name',
        'CUSIP_number', 'instrument_identifier', 'instrument_identifier_schema',
        '12_month', 'CompanySIC', 'Div2_range', 'Sector',
    ]
    df_long = pd.concat([df_std[long_cols], df_wd_nr[long_cols]], ignore_index=True)
    df_long = df_long.rename(columns={'issuer_name': 'Company_name', 'CompanySIC': 'SIC'})
    df_long = df_long.sort_values(['Company_number', 'Date', 'instrument_name', 'rating']).reset_index(drop=True)

    df_code_std = (
        df_std.groupby(['Company_number', 'Date', 'Sector'], as_index=False)['12_month'].mean()
    )
    df_code_std['rating_category'] = 'Standard'

    df_code_wdnr = (
        df_wd_nr.loc[df_wd_nr['rating'] == 'WD', ['Company_number', 'Date', 'Sector', 'rating', '12_month']]
        .drop_duplicates(subset=['Company_number', 'Date', 'rating'])
        .rename(columns={'rating': 'rating_category'})
        .copy()
    )

    df_code = pd.concat([df_code_std, df_code_wdnr], ignore_index=True)
    for col in ['1_month', '3_month', '6_month', '24_month', '36_month', '60_month']:
        df_code[col] = None
    df_code = df_code[[
        'Company_number', 'Date',
        '1_month', '3_month', '6_month', '12_month',
        '24_month', '36_month', '60_month', 'Sector', 'rating_category',
    ]]
    df_code = df_code.sort_values(['Company_number', 'Date', 'rating_category']).reset_index(drop=True)

    df_long.to_csv(long_output_path, index=False)
    df_code.to_csv(code_output_path, index=False)

    if verbose:
        print(f"\n✓ Saved {len(df_long):,} rows to {long_output_path}")
        print(f"✓ Saved {len(df_code):,} rows to {code_output_path}")
        print("\nrating_category distribution in dedup file:")
        print(df_code['rating_category'].value_counts().to_string())
        print("\nSector distribution (Standard rows only):")
        print(df_code[df_code['rating_category'] == 'Standard']['Sector'].value_counts().to_string())

    return df_long, df_code


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
            df[f'{col}_lag{k}'] = df[col].shift(k * 3)
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
