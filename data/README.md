# Data Folder Structure

This folder contains all CSV and Excel data files organized by category.

## Folder Organization

### 📁 `macro/` - Macroeconomic Data
Contains macroeconomic indicators and market data for Sweden:
- `sweden_gdp_monthly.csv` - Monthly GDP growth rates
- `sweden_interest_rate_monthly.csv` - Monthly interest rates
- `sweden_unemployment_monthly.csv` - Monthly unemployment rates
- `Bostadspriser ScB 1997-2025_transposed.csv` - Residential property prices (monthly)
- `Commersial real estate prices 1994-2025 ScB Sverige_transposed.csv` - Commercial real estate prices (quarterly)
- `data_gpr_Data_GPR.csv` - Global Policy Risk index data
- `data_gpr_Förklaring_av_variabler.csv` - GPR variable explanations

### 📁 `company/` - Company PD and Sector Data
Contains company-level probability of default data and sector classifications:
- `pdsFitchData.csv` - Complete Fitch PD data for all companies
- `company_pds_with_sectors.csv` - Company PDs with sector classifications
- `company_pds_with_sectors_6m_forward.csv` - Company PDs with 6-month forward data
- `company_names_lookup.csv` - Company identifier lookup table
- `sectors_unique_companies_classified.csv` - Unique companies with sector classifications

### 📁 `analysis/` - Processed Analysis Results
Contains processed data and analysis outputs:
- `pdsFitchData_latest_exposures.csv` - Latest exposure data for Basel calculations
- `pdsFitchData_latest_with_basel_correlation.csv` - Latest data with Basel asset correlations
- `sensitivity_results.csv` - Macro sensitivity analysis results
- `sensitivity_results_6m_delay.csv` - Sensitivity results with 6-month delay
- `df_final_cleaned.csv` - Cleaned merged dataset for modeling

### 📁 `raw_excel/` - Original Excel Files
Contains original Excel files before conversion to CSV format.

## Data Format

All CSV files use one of two formats:

**Format 1: Long format with semicolon separator (macro data)**
```
;Column Name
YYYY-MM;Value
2000-01;1,8
2000-02;0,5
```

**Format 2: Standard CSV with comma separator (company/analysis data)**
```
Column1,Column2,Column3
value1,value2,value3
```

## Notes

- Dates are in `YYYY-MM` format
- Quarterly data uses the end month of each quarter (03, 06, 09, 12)
- Decimal separator varies by file (comma for Swedish data, period for others)
- Files marked as "_transposed" have been reformatted from wide to long format

