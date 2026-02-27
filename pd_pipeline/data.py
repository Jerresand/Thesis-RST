"""Data loading and preparation utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd


def clean_dataframe(df: pd.DataFrame, date_col_idx: int, value_col_idx: int) -> pd.DataFrame:
    """Clean dataframe by removing BOMs, parsing dates, and coercing numeric values."""
    df = df.copy()
    df.columns = df.columns.str.replace('\ufeff', '', regex=False)

    date_col = df.columns[date_col_idx]
    value_col = df.columns[value_col_idx]

    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m', errors='coerce')

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
    unemployment_path: str,
    housing_path: str,
    cpi_path: str,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Load and clean macroeconomic datasets."""
    df_gdp = pd.read_csv(gdp_path, sep=None, engine='python')
    df_interest = pd.read_csv(interest_path, sep=None, engine='python')
    df_unemployment = pd.read_csv(unemployment_path, sep=None, engine='python')
    df_housing = pd.read_csv(housing_path, sep=None, engine='python')
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

    df_unemployment_cleaned = clean_dataframe(df_unemployment, 0, 1)
    df_unemployment_cleaned = df_unemployment_cleaned.rename(columns={
        df_unemployment_cleaned.columns[0]: 'Date',
        df_unemployment_cleaned.columns[1]: 'Unemployment_Rate',
    })

    df_housing_cleaned = clean_dataframe(df_housing, 0, 1)
    df_housing_cleaned = df_housing_cleaned.rename(columns={
        df_housing_cleaned.columns[0]: 'Date',
        df_housing_cleaned.columns[1]: 'Housing_Prices',
    }).dropna(subset=['Date', 'Housing_Prices'])

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
        print("\nCleaned df_unemployment head:")
        print(df_unemployment_cleaned.head())
        print("\nCleaned df_housing head:")
        print(df_housing_cleaned.head())
        print("\nCleaned df_cpi head:")
        print(df_cpi_cleaned.head())

    return {
        'gdp': df_gdp_cleaned,
        'interest': df_interest_cleaned,
        'unemployment': df_unemployment_cleaned,
        'housing': df_housing_cleaned,
        'cpi': df_cpi_cleaned,
    }


def load_gpr_data(gpr_path: str, verbose: bool = True) -> pd.DataFrame:
    """Load and clean the geopolitical risk dataset."""
    df_gpr = pd.read_csv(gpr_path, sep=None, engine='python')
    df_gpr.columns = df_gpr.columns.str.replace('\ufeff', '', regex=False)
    df_gpr['month'] = pd.to_datetime(df_gpr['month'], format='%Y-%m-%d', errors='coerce')

    df_gpr_cleaned = df_gpr[['month', 'GPR', 'GPRC_SWE']].copy()
    df_gpr_cleaned = df_gpr_cleaned.rename(columns={
        'month': 'Date',
        'GPR': 'GPR_Global',
        'GPRC_SWE': 'GPR_Sweden',
    })

    df_gpr_cleaned['GPR_Global'] = pd.to_numeric(df_gpr_cleaned['GPR_Global'], errors='coerce')
    df_gpr_cleaned['GPR_Sweden'] = pd.to_numeric(df_gpr_cleaned['GPR_Sweden'], errors='coerce')
    df_gpr_cleaned = df_gpr_cleaned.dropna(subset=['Date', 'GPR_Global', 'GPR_Sweden'])

    if verbose:
        print("\nCleaned df_gpr head:")
        print(df_gpr_cleaned.head())

    return df_gpr_cleaned


def merge_macro_data(frames: Dict[str, pd.DataFrame], df_gpr: pd.DataFrame) -> pd.DataFrame:
    """Merge macro and GPR dataframes on Date using outer joins."""
    df_merged = (
        frames['gdp']
        .merge(frames['interest'], on='Date', how='outer')
        .merge(frames['unemployment'], on='Date', how='outer')
        .merge(frames['housing'], on='Date', how='outer')
        .merge(df_gpr, on='Date', how='outer')
        .merge(frames['cpi'], on='Date', how='outer')
    )
    return df_merged


def summarize_macro_data(df_merged: pd.DataFrame, cols: Iterable[str], verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Compute covariance/correlation matrices and mean vector."""
    covariance_matrix = df_merged[list(cols)].cov()
    correlation_matrix = df_merged[list(cols)].corr()
    mean_vector = df_merged[list(cols)].mean()

    if verbose:
        print("\nCovariance Matrix (All variables):")
        print(covariance_matrix)
        print("\nCorrelation Matrix (All variables):")
        print(correlation_matrix)
        print("\nMean Vector (Macro + GPR variables):")
        print(mean_vector)

    return covariance_matrix, correlation_matrix, mean_vector


def load_pds_data(pds_path: str, verbose: bool = True) -> pd.DataFrame:
    """Load PD data, normalize date column, and add PDzero column."""
    df_pds = pd.read_csv(pds_path)
    df_pds.columns = df_pds.columns.str.replace('\ufeff', '', regex=False)

    if 'Date' in df_pds.columns:
        df_pds['Date'] = pd.to_datetime(df_pds['Date'], format='%Y-%m')
    else:
        date_col_name = df_pds.columns[0]
        df_pds[date_col_name] = pd.to_datetime(df_pds[date_col_name], format='%Y-%m')
        df_pds = df_pds.rename(columns={date_col_name: 'Date'})

    df_pds = df_pds[['Company_number', 'Date', '12_month', 'Sector']]
    df_pds = df_pds.sort_values(['Company_number', 'Date'])
    first_pd = df_pds.groupby('Company_number')['12_month'].first()
    df_pds['PDzero'] = df_pds['Company_number'].map(first_pd)

    if verbose:
        print("Columns in df_pds:", df_pds.columns.tolist())
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


def export_dataframe(df: pd.DataFrame, output_file: str, verbose: bool = True) -> None:
    """Export a dataframe to CSV with a lightweight summary."""
    df.to_csv(output_file, index=False)
    if verbose:
        print(f"✓ Successfully exported dataframe to: {output_file}")
        print(f"  - Rows: {df.shape[0]:,}")
        print(f"  - Columns: {df.shape[1]}")
        size_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  - File size: {size_mb:.2f} MB (approximate)")
