"""Default configuration constants for the PD analysis pipeline."""

MACRO_COLS = [
    'GDP_Growth',
    'Interest_Rate',
    'Unemployment_Rate',
    'Housing_Prices',
    'CPI',
]

GPR_COLS = ['GPR_Global']

N_LAGS = 4  # one lag per quarter (Q1…Q4)

LAGGED_MACRO_COLS = [f'{col}_lag{k}' for col in MACRO_COLS for k in range(1, N_LAGS + 1)]
LAGGED_GPR_COLS = [f'{col}_lag{k}' for col in GPR_COLS for k in range(1, N_LAGS + 1)]

# Current + lagged versions of each variable group
ALL_MACRO_COLS = MACRO_COLS + LAGGED_MACRO_COLS
ALL_GPR_COLS = GPR_COLS + LAGGED_GPR_COLS

# Without lags – used for scenario/covariance analysis (unchanged)
ALL_PREDICTOR_COLS = MACRO_COLS + GPR_COLS

# With lags – used for regression and LASSO
ALL_PREDICTOR_COLS_WITH_LAGS = ALL_MACRO_COLS + ALL_GPR_COLS

PD_MATURITY_COLS = ['12_month']

SECTOR_COL = 'Sector'
PDZERO_COL = 'PDzero'

DEFAULT_PD_TENORS = ['1_month', '3_month', '6_month', '12_month', '24_month', '36_month', '60_month']

DEFAULT_RWA_TENORS = ['12_month', '24_month', '36_month', '60_month']
