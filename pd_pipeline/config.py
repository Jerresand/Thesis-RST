"""Default configuration constants for the PD analysis pipeline."""

MACRO_COLS = [
    'GDP_Growth',
    'Interest_Rate',
    'Unemployment_Rate',
    'Housing_Prices',
    'CPI',
]

GPR_COLS = ['GPR_Global', 'GPR_Sweden']

ALL_PREDICTOR_COLS = MACRO_COLS + GPR_COLS

PD_MATURITY_COLS = ['12_month']

SECTOR_COL = 'Sector'
PDZERO_COL = 'PDzero'

DEFAULT_PD_TENORS = ['1_month', '3_month', '6_month', '12_month', '24_month', '36_month', '60_month']

DEFAULT_RWA_TENORS = ['12_month', '24_month', '36_month', '60_month']
