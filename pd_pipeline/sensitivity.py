"""OLS sensitivity analysis for PD changes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import statsmodels.api as sm


MAJOR_GROUP_SECTOR_NAMES = {
    '01': 'Agricultural Production - Crops',
    '02': 'Agricultural Production - Livestock',
    '07': 'Agricultural Services',
    '08': 'Forestry',
    '09': 'Fishing & Aquaculture',
    '10': 'Metal Mining',
    '12': 'Coal Mining',
    '13': 'Oil & Gas Extraction',
    '14': 'Nonmetallic Minerals Mining',
    '15': 'General Building Contractors',
    '16': 'Heavy Construction',
    '17': 'Special Trade Contractors',
    '20': 'Food & Beverage Manufacturing',
    '21': 'Tobacco Manufacturing',
    '22': 'Textile Manufacturing',
    '23': 'Apparel Manufacturing',
    '24': 'Lumber & Wood Products',
    '25': 'Furniture Manufacturing',
    '26': 'Paper Manufacturing',
    '27': 'Printing & Publishing',
    '28': 'Chemicals Manufacturing',
    '29': 'Petroleum Refining',
    '30': 'Rubber & Plastics Products',
    '31': 'Leather Manufacturing',
    '32': 'Stone, Clay & Glass Products',
    '33': 'Primary Metal Industries',
    '34': 'Fabricated Metal Products',
    '35': 'Industrial Machinery & Equipment',
    '36': 'Electronic & Electrical Equipment',
    '37': 'Transportation Equipment Manufacturing',
    '38': 'Instruments & Related Products',
    '39': 'Miscellaneous Manufacturing',
    '40': 'Railroad Transportation',
    '41': 'Local Passenger Transportation',
    '42': 'Trucking & Warehousing',
    '43': 'Postal Service',
    '44': 'Water Transportation',
    '45': 'Air Transportation',
    '46': 'Pipelines',
    '47': 'Transportation Services',
    '48': 'Communications',
    '49': 'Electric, Gas & Sanitary Services',
    '50': 'Wholesale Trade - Durable Goods',
    '51': 'Wholesale Trade - Nondurable Goods',
    '52': 'Building Materials & Garden Supplies',
    '53': 'General Merchandise Stores',
    '54': 'Food Stores',
    '55': 'Automotive Dealers & Service Stations',
    '56': 'Apparel & Accessory Stores',
    '57': 'Home Furniture & Equipment Stores',
    '58': 'Eating & Drinking Places',
    '59': 'Miscellaneous Retail',
    '60': 'Depository Institutions',
    '61': 'Nondepository Credit Institutions',
    '62': 'Security & Commodity Brokers',
    '63': 'Insurance Carriers',
    '64': 'Insurance Agents & Brokers',
    '65': 'Real Estate',
    '67': 'Holding & Investment Offices',
    '70': 'Hotels & Lodging',
    '72': 'Personal Services',
    '73': 'Business Services',
    '75': 'Automotive Repair & Services',
    '76': 'Miscellaneous Repair Services',
    '78': 'Motion Pictures',
    '79': 'Amusement & Recreation Services',
    '80': 'Health Services',
    '81': 'Legal Services',
    '82': 'Educational Services',
    '83': 'Social Services',
    '84': 'Museums & Cultural Services',
    '86': 'Membership Organizations',
    '87': 'Engineering & Management Services',
    '88': 'Private Households',
    '89': 'Miscellaneous Services',
    '91': 'Executive & Legislative Offices',
    '92': 'Justice & Public Safety',
    '93': 'Public Finance',
    '94': 'Administration of Programs',
    '95': 'Environmental Programs',
    '96': 'Regulation & Administration',
    '97': 'National Security & International Affairs',
    '99': 'Nonclassifiable Establishments',
}


def load_sic_to_major_group_mapping(sic_codes_file: str | Path) -> dict:
    """Load SIC codes file and create mapping from SIC code to Major Group sector.
    
    Args:
        sic_codes_file: Path to the SIC codes CSV file
        
    Returns:
        Dictionary mapping 4-digit SIC codes to sector names
    """
    df_sic = pd.read_csv(sic_codes_file)
    
    df_sic['Major_Group'] = df_sic['Major Group'].astype(str).str.zfill(2)
    df_sic['SIC_str'] = df_sic['SIC'].astype(str).str.zfill(4)
    df_sic['Sector_Name'] = df_sic['Major_Group'].map(MAJOR_GROUP_SECTOR_NAMES)
    df_sic['Sector_Name'] = df_sic['Sector_Name'].fillna('Unclassified')
    
    sic_to_sector = dict(zip(df_sic['SIC_str'], df_sic['Sector_Name']))
    return sic_to_sector


def load_company_to_sic_mapping(isin_to_sic_file: str | Path) -> pd.DataFrame:
    """Load ISIN to SIC mapping and extract company name to SIC mapping.
    
    Args:
        isin_to_sic_file: Path to the ISIN to SIC CSV file
        
    Returns:
        DataFrame with issuer_name and SIC columns
    """
    df_isin = pd.read_csv(isin_to_sic_file, sep=';')
    
    df_isin = df_isin[['issuer_name', 'SIC']].copy()
    df_isin['SIC'] = df_isin['SIC'].astype(str).str.strip()
    
    df_isin = df_isin[df_isin['SIC'] != 'na'].drop_duplicates()
    
    return df_isin


def normalize_company_name(name: str) -> str:
    """Normalize company name for matching by removing common variations."""
    if pd.isna(name):
        return ''
    
    name = str(name).strip().lower()
    
    suffixes_to_remove = [
        r'\s+inc\.?$', r'\s+corp\.?$', r'\s+corporation$', r'\s+company$',
        r'\s+ltd\.?$', r'\s+limited$', r'\s+plc$', r'\s+ag$', r'\s+nv$',
        r'\s+sa$', r'\s+s\.a\.?$', r'\s+ab$', r'\s+asa$', r'\s+gmbh$',
        r'\s+llc$', r'\s+lp$', r'\s+b\.v\.?$', r'\s+spa$', r'\s+s\.p\.a\.?$',
        r'\s+co\.?$', r'\s+the$', r'\(the\)$',
    ]
    
    import re
    for suffix in suffixes_to_remove:
        name = re.sub(suffix, '', name)
    
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name


def map_company_to_sector(
    df: pd.DataFrame,
    company_lookup_file: str | Path,
    isin_to_sic_file: str | Path,
    sic_codes_file: str | Path,
    company_col: str = 'Company_number',
    sector_col: str = 'Sector',
    verbose: bool = True,
) -> pd.DataFrame:
    """Map Company_number to sectors based on SIC major groups.
    
    This function performs the following mapping chain:
    Company_number -> issuer_identifier -> issuer_name -> SIC -> Major_Group -> Sector
    
    Uses fuzzy name matching to handle variations in company names.
    
    Args:
        df: DataFrame containing Company_number
        company_lookup_file: Path to company/issuer names lookup CSV with issuer_identifier
        isin_to_sic_file: Path to ISIN-to-SIC mapping CSV
        sic_codes_file: Path to SIC codes CSV
        company_col: Name of company number column in df
        sector_col: Name of sector column to create
        verbose: Print mapping statistics
        
    Returns:
        DataFrame with sector column added based on SIC major groups
    """
    df_issuers = pd.read_csv(company_lookup_file, sep=';')
    if 'issuer_identifier' in df_issuers.columns:
        df_issuers = df_issuers.rename(columns={'issuer_identifier': 'Company_number'})
    
    df_company_to_sic = load_company_to_sic_mapping(isin_to_sic_file)
    
    sic_to_sector_dict = load_sic_to_major_group_mapping(sic_codes_file)
    
    df = df.merge(df_issuers[['Company_number', 'issuer_name']], on='Company_number', how='left')
    
    df['issuer_name_normalized'] = df['issuer_name'].apply(normalize_company_name)
    df_company_to_sic['issuer_name_normalized'] = df_company_to_sic['issuer_name'].apply(normalize_company_name)
    
    df = df.merge(
        df_company_to_sic[['issuer_name_normalized', 'SIC']],
        on='issuer_name_normalized',
        how='left'
    )
    
    df['SIC_4digit'] = df['SIC'].astype(str).str.strip().str.zfill(4)
    df[sector_col] = df['SIC_4digit'].map(sic_to_sector_dict)
    df[sector_col] = df[sector_col].fillna('Unclassified')
    
    if verbose:
        total_companies = df[company_col].nunique()
        mapped_companies = df[df[sector_col] != 'Unclassified'][company_col].nunique()
        print(f"\n✓ Sector mapping complete:")
        print(f"  - Total unique companies: {total_companies}")
        print(f"  - Companies mapped to sectors: {mapped_companies} ({100*mapped_companies/total_companies:.1f}%)")
        print(f"  - Companies unclassified: {total_companies - mapped_companies}")
        print(f"\n  Sector distribution (top 15):")
        sector_counts = df.drop_duplicates(subset=company_col)[sector_col].value_counts()
        for sector, count in sector_counts.head(15).items():
            print(f"    {sector}: {count}")
    
    df = df.drop(columns=['issuer_name', 'issuer_name_normalized', 'SIC', 'SIC_4digit'], errors='ignore')
    
    return df


def calculate_logit(p: np.ndarray | pd.Series) -> np.ndarray:
    """Compute log-odds with clipping to avoid infinities."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def run_sector_ols(
    df: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    pd_col: str,
    pdzero_col: str,
    min_obs: int = 10,
):
    """Fit OLS for a single sector and PD horizon."""
    sector_df = df.copy()
    sector_df['logit_pd'] = calculate_logit(sector_df[pd_col])
    sector_df['logit_pd_zero'] = calculate_logit(sector_df[pdzero_col])
    sector_df['delta_logit'] = sector_df['logit_pd'] - sector_df['logit_pd_zero']

    y = sector_df['delta_logit']
    X = pd.concat([sector_df[macro_cols], sector_df[gpr_cols]], axis=1)
    X = sm.add_constant(X)

    valid_idx = ~(y.isna() | X.isna().any(axis=1))
    y = y[valid_idx]
    X = X[valid_idx]

    if len(y) < min_obs:
        return None, None, None

    model = sm.OLS(y, X).fit()
    conf_int = model.conf_int(alpha=0.05)
    return model, conf_int, len(y)


def run_sensitivity_analysis(
    df_final_cleaned: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
    sector_col: str,
    pd_maturity_cols: Iterable[str],
    pdzero_col: str = 'PDzero',
    min_obs: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run OLS sensitivity analysis across sectors and PD horizons."""
    sensitivities_data = []

    for sector in df_final_cleaned[sector_col].unique():
        sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()
        if verbose:
            print(f"\nProcessing sector: {sector} (n={len(sector_df)})")

        for pd_col in pd_maturity_cols:
            try:
                model, conf_int, n_obs = run_sector_ols(
                    sector_df,
                    macro_cols,
                    gpr_cols,
                    pd_col,
                    pdzero_col,
                    min_obs=min_obs,
                )

                if model is None:
                    if verbose:
                        print(f"  Skipping {pd_col}: insufficient data (n={len(sector_df)})")
                    continue

                result = {
                    'Sector': sector,
                    'PD_Horizon': pd_col,
                    'Intercept': model.params['const'],
                    'N_observations': n_obs,
                    'R_squared': model.rsquared,
                }

                for col in macro_cols:
                    result[f'β_{col}'] = model.params[col]
                    result[f'β_{col}_CI_lower'] = conf_int.loc[col, 0]
                    result[f'β_{col}_CI_upper'] = conf_int.loc[col, 1]

                for col in gpr_cols:
                    result[f'δ_{col}'] = model.params[col]
                    result[f'δ_{col}_CI_lower'] = conf_int.loc[col, 0]
                    result[f'δ_{col}_CI_upper'] = conf_int.loc[col, 1]

                sensitivities_data.append(result)
                if verbose:
                    print(f"  ✓ {pd_col}: R²={model.rsquared:.3f}, N={n_obs}")

            except Exception as exc:  # noqa: BLE001 - keep parity with notebook output
                if verbose:
                    print(f"  ✗ Could not fit model for {pd_col}: {exc}")

    return pd.DataFrame(sensitivities_data)


def export_sensitivities(df_sensitivities: pd.DataFrame, output_file: str) -> None:
    """Export sensitivity results to CSV."""
    df_sensitivities.to_csv(output_file, index=False)
    print(f"✓ Sensitivity results with 95% confidence intervals exported to: {output_file}")
    print(f"  Total sectors analyzed: {len(df_sensitivities)}")
    print("\nColumns include:")
    print("  - Point estimates: β_[variable] and δ_[variable]")
    print("  - 95% CI lower bounds: β_[variable]_CI_lower and δ_[variable]_CI_lower")
    print("  - 95% CI upper bounds: β_[variable]_CI_upper and δ_[variable]_CI_upper")


def print_sensitivity_tables(
    df_sensitivities: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
) -> None:
    """Print macro and GPR sensitivity tables for readability."""
    print("=" * 80)
    print("MACRO SENSITIVITIES (β) - Impact of macroeconomic variables on PD")
    print("=" * 80)
    beta_cols = ['Sector', 'PD_Horizon', 'N_observations', 'R_squared']
    for col in macro_cols:
        beta_cols.extend([f'β_{col}', f'β_{col}_CI_lower', f'β_{col}_CI_upper'])
    print(df_sensitivities[beta_cols])

    print("\n" + "=" * 80)
    print("GPR SENSITIVITIES (δ) - Impact of geopolitical risk on PD")
    print("=" * 80)
    delta_cols = ['Sector', 'PD_Horizon', 'N_observations', 'R_squared']
    for col in gpr_cols:
        delta_cols.extend([f'δ_{col}', f'δ_{col}_CI_lower', f'δ_{col}_CI_upper'])
    print(df_sensitivities[delta_cols])


def print_confidence_interval_summary(
    df_sensitivities: pd.DataFrame,
    gpr_cols: List[str],
) -> None:
    """Print confidence interval summary for the first three sectors."""
    print("=" * 80)
    print("SENSITIVITY ESTIMATES WITH 95% CONFIDENCE INTERVALS (First 3 Sectors)")
    print("=" * 80)

    for _, row in df_sensitivities.head(3).iterrows():
        print(f"\n{row['Sector']} - {row['PD_Horizon']} (N={int(row['N_observations'])}, R²={row['R_squared']:.3f})")
        print("\nβ (Macro):")
        for col in ['GDP_Growth', 'Interest_Rate']:
            print(f"  {col}: {row[f'β_{col}']:.4f} [{row[f'β_{col}_CI_lower']:.4f}, {row[f'β_{col}_CI_upper']:.4f}]")
        print("\nδ (GPR):")
        for col in gpr_cols:
            print(f"  {col}: {row[f'δ_{col}']:.4f} [{row[f'δ_{col}_CI_lower']:.4f}, {row[f'δ_{col}_CI_upper']:.4f}]")


def print_sensitivity_details(
    df_sensitivities: pd.DataFrame,
    macro_cols: List[str],
    gpr_cols: List[str],
) -> None:
    """Print detailed sensitivity estimates with confidence intervals."""
    print("\n" + "=" * 80)
    print("SENSITIVITY ESTIMATES WITH 95% CONFIDENCE INTERVALS")
    print("=" * 80)

    for _, row in df_sensitivities.iterrows():
        print(f"\n{'='*80}")
        print(
            f"Sector: {row['Sector']} | PD Horizon: {row['PD_Horizon']} | "
            f"R²={row['R_squared']:.3f} | N={int(row['N_observations'])}"
        )
        print(f"{'='*80}")

        print("\nMACRO SENSITIVITIES (β):")
        print("-" * 80)
        for col in macro_cols:
            beta = row[f'β_{col}']
            ci_lower = row[f'β_{col}_CI_lower']
            ci_upper = row[f'β_{col}_CI_upper']
            ci_width = ci_upper - ci_lower
            print(
                f"  {col:25s}: β = {beta:8.4f}  "
                f"[95% CI: {ci_lower:8.4f}, {ci_upper:8.4f}]  (width: {ci_width:.4f})"
            )

        print("\nGPR SENSITIVITIES (δ):")
        print("-" * 80)
        for col in gpr_cols:
            delta = row[f'δ_{col}']
            ci_lower = row[f'δ_{col}_CI_lower']
            ci_upper = row[f'δ_{col}_CI_upper']
            ci_width = ci_upper - ci_lower
            print(
                f"  {col:25s}: δ = {delta:8.4f}  "
                f"[95% CI: {ci_lower:8.4f}, {ci_upper:8.4f}]  (width: {ci_width:.4f})"
            )

    print("\n" + "=" * 80)
