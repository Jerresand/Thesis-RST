"""
Macroeconomic Sensitivity Analysis for Company Probability of Defaults (PDs)

This script:
1. Loads and cleans Swedish macroeconomic data (GDP, Interest Rate, Unemployment)
2. Merges with company PD data by date and sector
3. Performs sensitivity analysis using linear regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import statsmodels.api as sm


# =============================================================================
# SECTION 1: Load and Clean Macroeconomic Data
# =============================================================================

df_gdp = pd.read_csv('sweden_gdp_monthly.csv', sep=None, engine='python')
df_interest = pd.read_csv('sweden_interest_rate_monthly.csv', sep=None, engine='python')
df_unemployment = pd.read_csv('sweden_unemployment_monthly.csv', sep=None, engine='python')

def clean_dataframe(df, date_col_idx, value_col_idx):
    """
    Clean dataframe by removing BOM characters, converting dates, and handling numeric values.
    
    Args:
        df: DataFrame to clean
        date_col_idx: Index of the date column
        value_col_idx: Index of the value column
    
    Returns:
        Cleaned DataFrame
    """
    # Remove BOM character from column names
    df.columns = df.columns.str.replace('\ufeff', '', regex=False)

    # Get the actual (cleaned) column names using their indices
    current_date_col_name = df.columns[date_col_idx]
    current_value_col_name = df.columns[value_col_idx]

    # Convert date column to datetime objects, coercing errors
    df[current_date_col_name] = pd.to_datetime(df[current_date_col_name], format='%Y-%m', errors='coerce')

    # Convert value column to numeric, handling comma as decimal separator
    if df[current_value_col_name].dtype == 'object':
        df[current_value_col_name] = df[current_value_col_name].astype(str).str.replace(',', '.', regex=False)
        df[current_value_col_name] = pd.to_numeric(df[current_value_col_name], errors='coerce')

    return df

# Clean df_gdp and rename columns
df_gdp_cleaned = clean_dataframe(df_gdp.copy(), 0, 1)
df_gdp_cleaned = df_gdp_cleaned.rename(columns={df_gdp_cleaned.columns[0]: 'Date', df_gdp_cleaned.columns[1]: 'GDP_Growth'})
df_gdp_cleaned = df_gdp_cleaned.dropna(subset=['Date', 'GDP_Growth'])

# Clean df_interest and rename columns
df_interest_cleaned = clean_dataframe(df_interest.copy(), 0, 1)
df_interest_cleaned = df_interest_cleaned.rename(columns={df_interest_cleaned.columns[0]: 'Date', df_interest_cleaned.columns[1]: 'Interest_Rate'})

# Clean df_unemployment and rename columns
df_unemployment_cleaned = clean_dataframe(df_unemployment.copy(), 0, 1)
df_unemployment_cleaned = df_unemployment_cleaned.rename(columns={df_unemployment_cleaned.columns[0]: 'Date', df_unemployment_cleaned.columns[1]: 'Unemployment_Rate'})

print("Cleaned df_gdp head:")
print(df_gdp_cleaned.head())
print("\nCleaned df_interest head:")
print(df_interest_cleaned.head())
print("\nCleaned df_unemployment head:")
print(df_unemployment_cleaned.head())


# =============================================================================
# SECTION 2: Merge Macroeconomic Data
# =============================================================================

df_merged = df_gdp_cleaned.merge(df_interest_cleaned, on='Date', how='outer').merge(df_unemployment_cleaned, on='Date', how='outer')
print("\nMerged macroeconomic data:")
print(df_merged.head())


# =============================================================================
# SECTION 3: Calculate Covariance Matrix and Mean Vector
# =============================================================================

numerical_cols = ['GDP_Growth', 'Interest_Rate', 'Unemployment_Rate']
covariance_matrix = df_merged[numerical_cols].cov()
print("\nCovariance Matrix:")
print(covariance_matrix)

mean_vector = df_merged[numerical_cols].mean()
print("\nMean Vector:")
print(mean_vector)


# =============================================================================
# SECTION 4: Multivariate Normal Distribution Visualization
# =============================================================================

mvn = multivariate_normal(mean=mean_vector, cov=covariance_matrix)

# Define appropriate ranges for each variable
gdp_range = np.linspace(df_merged['GDP_Growth'].min() - 1, df_merged['GDP_Growth'].max() + 1, 100)
interest_range = np.linspace(df_merged['Interest_Rate'].min() - 2, df_merged['Interest_Rate'].max() + 2, 100)
unemployment_range = np.linspace(df_merged['Unemployment_Rate'].min() - 1, df_merged['Unemployment_Rate'].max() + 1, 100)

plt.figure(figsize=(15, 5))

# 1. GDP_Growth vs. Interest_Rate
plt.subplot(1, 3, 1)
X, Y = np.meshgrid(gdp_range, interest_range)
positions = np.array([X.ravel(), Y.ravel(), np.full(X.size, mean_vector['Unemployment_Rate'])]).T
Z = mvn.pdf(positions).reshape(X.shape)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.xlabel('GDP Growth')
plt.ylabel('Interest Rate')
plt.title('GDP Growth vs. Interest Rate Joint Distribution')

# 2. GDP_Growth vs. Unemployment_Rate
plt.subplot(1, 3, 2)
X, Y = np.meshgrid(gdp_range, unemployment_range)
positions = np.array([X.ravel(), np.full(X.size, mean_vector['Interest_Rate']), Y.ravel()]).T
Z = mvn.pdf(positions).reshape(X.shape)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.xlabel('GDP Growth')
plt.ylabel('Unemployment Rate')
plt.title('GDP Growth vs. Unemployment Rate Joint Distribution')

# 3. Interest_Rate vs. Unemployment_Rate
plt.subplot(1, 3, 3)
X, Y = np.meshgrid(interest_range, unemployment_range)
positions = np.array([np.full(X.size, mean_vector['GDP_Growth']), X.ravel(), Y.ravel()]).T
Z = mvn.pdf(positions).reshape(X.shape)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.xlabel('Interest Rate')
plt.ylabel('Unemployment Rate')
plt.title('Interest Rate vs. Unemployment Rate Joint Distribution')

plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 5: Load and Merge Company PD Data
# =============================================================================

df_pds = pd.read_csv('company_pds_with_sectors.csv', sep=None, engine='python')

# Remove BOM character from column names
df_pds.columns = df_pds.columns.str.replace('\ufeff', '', regex=False)

# Convert 'Date' column to datetime with YYYY-MM format (matches macro data format)
if 'Date' in df_pds.columns:
    df_pds['Date'] = pd.to_datetime(df_pds['Date'], format='%Y-%m')
else:
    # Fallback: if no 'Date' column, use the first column
    date_col_name = df_pds.columns[0]
    df_pds[date_col_name] = pd.to_datetime(df_pds[date_col_name], format='%Y-%m')
    df_pds = df_pds.rename(columns={date_col_name: 'Date'})

print("\nColumns in df_pds:", df_pds.columns.tolist())
print("\nFirst few rows of df_pds:")
print(df_pds.head())

print("\nBefore merge - PD data shape:", df_pds.shape)
print("Before merge - Macro data shape:", df_merged.shape)

# Merge PDs with macroeconomic data
# Both datasets now have YYYY-MM date format and will merge properly
df_final = pd.merge(df_pds, df_merged, on='Date', how='inner')

print("\nAfter merge - Final data shape:", df_final.shape)
print(f"Retained {df_final.shape[0]} out of {df_pds.shape[0]} PD datapoints ({100*df_final.shape[0]/df_pds.shape[0]:.1f}%)")

print("\nFirst few rows of the merged DataFrame (df_final):")
print(df_final.head())

print("\nShape of df_final:")
print(df_final.shape)

print("\nInfo of df_final:")
df_final.info()


# =============================================================================
# SECTION 6: Clean Data - Remove NaN Values
# =============================================================================

macro_cols = ['GDP_Growth', 'Interest_Rate', 'Unemployment_Rate']
df_final_cleaned = df_final.dropna(subset=macro_cols).copy()

print("\nShape of df_final before dropping NaNs:", df_final.shape)
print("Shape of df_final after dropping NaNs:", df_final_cleaned.shape)
print("\nInfo of df_final_cleaned after handling NaNs:")
df_final_cleaned.info()


# =============================================================================
# SECTION 7: Create Lagged PD Columns (t-1) and Define Variables
# =============================================================================

pd_maturity_cols = ['1_month', '3_month', '6_month', '12_month', '24_month', '36_month', '60_month']
macro_cols = ['GDP_Growth', 'Interest_Rate', 'Unemployment_Rate']
sector_col = 'Sector'

print("\nDependent variables (PD Maturity Columns):", pd_maturity_cols)
print("Independent variables (Macroeconomic Indicators):", macro_cols)
print("Categorical variable (Sector Column):", sector_col)

# Create lagged PD columns (t-1) for each company
# Sort by company and date to ensure correct lagging
df_final_cleaned = df_final_cleaned.sort_values(['Company_number', 'Date'])

for pd_col in pd_maturity_cols:
    # Create lagged column (previous time period) per company
    df_final_cleaned[f'{pd_col}_t1'] = df_final_cleaned.groupby('Company_number')[pd_col].shift(1)

# Drop rows where lagged values are NaN (first observation for each company)
df_final_cleaned = df_final_cleaned.dropna(subset=[f'{pd_col}_t1' for pd_col in pd_maturity_cols])

print(f"\nAfter creating lagged PD columns and dropping NaNs: {df_final_cleaned.shape}")
print(f"Lagged columns created: {[f'{col}_t1' for col in pd_maturity_cols]}")


# =============================================================================
# SECTION 8: Define Logit Function and Perform Delta-Logit Regression
# =============================================================================

def calculate_logit(p):
    """
    Computes the log-odds (logit) of a probability.
    Clips values to avoid log(0) or division by zero.
    """
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


sensitivities_data = []

for sector in df_final_cleaned[sector_col].unique():
    sector_df = df_final_cleaned[df_final_cleaned[sector_col] == sector].copy()
    
    print(f"\nProcessing sector: {sector} (n={len(sector_df)})")

    for pd_col in pd_maturity_cols:
        pdzero_col = f'{pd_col}_t1'  # Lagged PD column (t-1)
        
        try:
            # Step 1: Transform the Data - Calculate the 'Change in Logit'
            sector_df['logit_pd'] = calculate_logit(sector_df[pd_col])
            sector_df['logit_pd_zero'] = calculate_logit(sector_df[pdzero_col])
            
            # This delta is what the Macro factors actually explain
            sector_df['delta_logit'] = sector_df['logit_pd'] - sector_df['logit_pd_zero']
            
            # Step 2: Prepare Regression Variables
            y = sector_df['delta_logit']
            X = sector_df[macro_cols]
            
            # Add a constant (intercept) to capture any systematic trend
            X = sm.add_constant(X)
            
            # Drop any remaining NaN values
            valid_idx = ~(y.isna() | X.isna().any(axis=1))
            y = y[valid_idx]
            X = X[valid_idx]
            
            if len(y) < 10:  # Need minimum observations
                print(f"  Skipping {pd_col}: insufficient data (n={len(y)})")
                continue
            
            # Step 3: Run the OLS Regression
            model = sm.OLS(y, X).fit()
            
            # Step 4: Store the Sensitivities
            result = {
                'Sector': sector,
                'PD_Horizon': pd_col,
                'Intercept': model.params['const'],
                'N_observations': len(y),
                'R_squared': model.rsquared
            }
            
            # Add sensitivities for each macroeconomic variable
            for col in macro_cols:
                result[col] = model.params[col]
            
            sensitivities_data.append(result)
            print(f"  ✓ {pd_col}: R²={model.rsquared:.3f}, N={len(y)}")

        except Exception as e:
            print(f"  ✗ Could not fit model for {pd_col}: {e}")

df_sensitivities = pd.DataFrame(sensitivities_data)
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS RESULTS")
print("="*80)
print(df_sensitivities.to_string())

# Save results to CSV
df_sensitivities.to_csv('sensitivity_results.csv', index=False)
print("\n✓ Sensitivity results saved to 'sensitivity_results.csv'")
