#!/usr/bin/env python
# coding: utf-8

# # CET1-Ratio Constrained Reverse Stress Test
# 
# **Goal**: Find the *most plausible* macro-geopolitical deviation $\boldsymbol{\Delta}$ from the historical
# mean that drives the **stressed CET1 ratio below a breakdown threshold** $R^\omega$.
# 
# Formally (Definition 2, Hurlin et al. 2026):
# 
# $$\min_{\boldsymbol{\Delta} \in \mathbb{R}^5} \; D_M^2 = \boldsymbol{\Delta}^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\Delta}$$
# 
# $$\text{subject to} \quad R(\boldsymbol{\Delta}) \leq R^\omega$$
# 
# where $R(\boldsymbol{\Delta}) = \frac{\text{CET1}^0 - L_q(\boldsymbol{\Delta})}{\text{RWA}(\boldsymbol{\Delta})}$ is the stressed CET1 ratio,
# $R^\omega = R^0 \cdot (1 - \epsilon)$ is the breakdown threshold (ECB 2026: $\epsilon = 3\%$, i.e. 300 bps depletion).
# 
# ---
# 
# ### Methodology
# 
# **Step 1 — Macro $\to$ stressed PD** (long-run sensitivity model):
# 
# $$\text{PD}_i^*(\boldsymbol{\Delta}) = \sigma\!\left(\text{logit}(\text{PD}_i^0) + \mathbf{b}_{s(i)}^{\text{total}} \cdot \boldsymbol{\Delta}\right)$$
# 
# **Step 2 — Portfolio tail loss** (Gordy 2003, 99.9%):
# 
# $$L_q(\boldsymbol{\Delta}) = \sum_{i=1}^{n} \text{EAD}_i \cdot \text{LGD} \cdot \Phi\!\left(\frac{\Phi^{-1}(\text{PD}_i^*) + \sqrt{\rho_i}\,\Phi^{-1}(q)}{\sqrt{1-\rho_i}}\right)$$
# 
# **Step 3 — Stressed RWA** (Basel IRB formula, eq. 19–20 in Hurlin et al.):
# 
# $$\text{RWA}(\boldsymbol{\Delta}) = \sum_{i=1}^{n} \text{EAD}_i \cdot 12.5 \cdot K_i(\text{PD}_i^*)$$
# 
# **Step 4 — Stressed CET1 ratio** (eq. 22):
# 
# $$R(\boldsymbol{\Delta}) = \frac{\text{CET1}^0 - L_q(\boldsymbol{\Delta})}{\text{RWA}(\boldsymbol{\Delta})}$$
# 
# **Solver**: SLSQP (Sequential Least Squares Programming), scipy.optimize.

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
from scipy.optimize import minimize, approx_fprime
import sys
import pathlib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if not (PROJECT_ROOT / 'pd_pipeline').exists():
    PROJECT_ROOT = next(
        (p for p in PROJECT_ROOT.parents if (p / 'pd_pipeline').exists()),
        PROJECT_ROOT,
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / 'data'

from pd_pipeline.basel import asset_correlation_formula, calculate_capital_requirement
from pd_pipeline import data, config

# ── Shared problem parameters ──────────────────────────────────────────────────
LGD             = 0.40
ASRF_QUANTILE   = 0.999    # shared confidence level for Gordy loss and Basel IRB capital
MATURITY        = 2.5       # effective maturity (years)

# ── CET1 and capital-structure assumptions ─────────────────────────────────────
# CET1_RATIO_0  : assumed bank-wide CET1 ratio at baseline
# EAD_CET1_RATIO: assumed leverage ratio  EAD_portfolio / CET1  = 10
# DEPLETION     : absolute CET1 ratio depletion for breakdown (300 bps = 3 pp)
CET1_RATIO_0   = 0.17        # 17% initial CET1 ratio
EAD_CET1_RATIO = 6.0        # EAD of corporate portfolio = 10 × CET1 capital
DEPLETION      = 0.03       # 300 bps absolute depletion (ECB 2026 benchmark)
R_OMEGA        = CET1_RATIO_0 - DEPLETION   # breakdown threshold = 14%

ALL_BASE_VARS  = config.MACRO_COLS + config.GPR_COLS
ALL_PREDICTOR_VARS = config.ALL_PREDICTOR_COLS_WITH_LAGS

_BASE_LABELS = {
    'GDP_Growth'    : 'GDP',
    'Interest_Rate' : 'IR',
    'Brent_Oil'     : 'Oil',
    'Fuel_Index'    : 'CIX',
    'CPI'           : 'CPI',
    'GPR_Global'    : 'GPR',
}

def _var_label(v: str) -> str:
    base = v.split('_lag')[0]
    lag  = int(v.split('_lag')[1]) if '_lag' in v else 0
    lbl  = _BASE_LABELS.get(base, base)
    return lbl if lag == 0 else f'{lbl}[t-{lag}]'

def _infer_sensitivity_vars(df: pd.DataFrame) -> list[str]:
    """Infer usable predictor coefficient columns from a sensitivity CSV."""
    return [col for col in ALL_PREDICTOR_VARS if col in df.columns]


def _base_var(v: str) -> str:
    return v.split('_lag')[0]


SENSITIVITY_CSV = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1 else DATA_DIR / 'final' / 'per_sector_ols_betas_with_lags.csv'
)

print(f'LGD                 = {LGD:.0%}')
print(f'ASRF quantile       = {ASRF_QUANTILE:.1%}')
print(f'Initial CET1        = {CET1_RATIO_0:.1%}')
print(f'EAD / CET1          = {EAD_CET1_RATIO:.0f}x  (portfolio leverage assumption)')
print(f'Depletion ε         = {DEPLETION:.0%}  ({int(DEPLETION*10000)} bps absolute)')
print(f'Threshold R_ω       = {R_OMEGA:.4%}')
print(f'Sensitivity CSV     = {SENSITIVITY_CSV}')


# ## 1. Portfolio Exposures

# In[2]:


df_port = pd.read_csv(DATA_DIR / 'portfolio_simulation.csv')
df_port['rho']        = df_port['pd'].apply(asset_correlation_formula)
df_port['pd_clipped'] = np.clip(df_port['pd'], 1e-9, 1 - 1e-9)
print(f'Exposures   : {len(df_port):,}')
print(f'Total EAD   : {df_port["ead_eur_m"].sum():,.1f} EUR million')
ead_w_pd = (df_port['pd'] * df_port['ead_eur_m']).sum() / df_port['ead_eur_m'].sum()
print(f'Average PD          : {df_port["pd"].mean()*100:.2f}%')
print(f'EAD-weighted PD     : {ead_w_pd*100:.2f}%')
print()
print(df_port['sector'].value_counts().rename('count').to_frame().to_string())


# ## 2. Sensitivity Coefficients
# 
# The optimization dimension is inferred from the passed sensitivity CSV.
# If the file only contains current-period coefficients, the problem stays in the
# current-period space. If lagged coefficients are present, they are included too.

# In[3]:


# Read per-sector sensitivities from the passed CSV.
df_ols = pd.read_csv(SENSITIVITY_CSV)
ACTIVE_VARS = _infer_sensitivity_vars(df_ols)
if not ACTIVE_VARS:
    raise ValueError(
        f'No usable predictor columns found in {SENSITIVITY_CSV}. '
        f'Expected one or more of: {ALL_PREDICTOR_VARS}'
    )

ACTIVE_BASE_VARS: list[str] = []
for v in ACTIVE_VARS:
    base = _base_var(v)
    if base not in ACTIVE_BASE_VARS:
        ACTIVE_BASE_VARS.append(base)

N_VARS      = len(ACTIVE_VARS)
N_BASE_VARS = len(ACTIVE_BASE_VARS)
VAR_LABELS  = {v: _var_label(v) for v in ACTIVE_VARS}

sens_total: dict[str, dict[str, float]] = {}
for _, row in df_ols.iterrows():
    sector = row[config.SECTOR_COL]
    betas: dict[str, float] = {}
    for v in ACTIVE_VARS:
        val = row.get(v, np.nan)
        betas[v] = 0.0 if pd.isna(val) else float(val)
    sens_total[sector] = betas

tbl = pd.DataFrame(sens_total).T[ACTIVE_VARS].rename(columns=VAR_LABELS)
print('Sensitivity coefficients  [Δ logit-PD per unit of Δ macro var]')
print(f'Active factors inferred from CSV ({N_VARS}): {", ".join(ACTIVE_VARS)}')
print(tbl.round(4).to_string())


# In[4]:


# Filter portfolio to sectors with sensitivities
df_valid = df_port[df_port['sector'].isin(sens_total)].copy().reset_index(drop=True)
print(f'Exposures matched to sensitivity sectors: {len(df_valid):,} / {len(df_port):,}')
ead_valid     = df_valid['ead_eur_m'].values
ead_w_pd_v    = float(np.sum(df_valid['pd_clipped'] * df_valid['ead_eur_m']) / df_valid['ead_eur_m'].sum())
print(f'Total EAD (matched) : {df_valid["ead_eur_m"].sum():,.1f} EUR million')
print(f'EAD-weighted PD     : {ead_w_pd_v*100:.2f}%')

n_exp = len(df_valid)
pd0   = df_valid['pd_clipped'].values
rho   = df_valid['rho'].values
ead   = df_valid['ead_eur_m'].values

# Build sensitivity matrix B_total : shape (n_exp, N_VARS)
B_total = np.zeros((n_exp, N_VARS))
for i, (_, row) in enumerate(df_valid.iterrows()):
    s = row['sector']
    for j, v in enumerate(ACTIVE_VARS):
        B_total[i, j] = sens_total[s][v]

# Pre-compute constants for the ASRF/Gordy formulas
sqrt_rho   = np.sqrt(rho)
sqrt_1mrho = np.sqrt(1 - rho)
inv_q      = norm.ppf(ASRF_QUANTILE)
logit_pd0  = np.log(pd0 / (1 - pd0))   # logit(PD_base)

# ## 3. Historical Distribution Parameters

# In[5]:


macro_frames = data.load_macro_data(
    gdp_path      = str(DATA_DIR / 'macro' / 'GDPREALGLOBAL_monthly.csv'),
    interest_path = str(DATA_DIR / 'macro' / 'intrest FRED.csv'),
    brent_path    = str(DATA_DIR / 'macro' / 'brent_oil_monthly.csv'),
    fuel_path     = str(DATA_DIR / 'macro' / 'fuel_index_monthly.csv'),
    cpi_path      = str(DATA_DIR / 'macro' / 'global_cpi_mom_growth.csv'),
    verbose       = False,
)
df_gpr    = data.load_gpr_data(str(DATA_DIR / 'geopolitical' / 'data_gpr_Data_GPR.csv'), verbose=False)
df_merged = data.merge_macro_data(macro_frames, df_gpr)

needs_lags = any('_lag' in v for v in ACTIVE_VARS)
df_macro_for_metric = (
    data.add_macro_lags(df_merged, config.MACRO_COLS + config.GPR_COLS, n_lags=config.N_LAGS)
    if needs_lags else
    df_merged.copy()
)

# Covariance and mean over exactly the factors used by the loaded sensitivity CSV.
cov_df, _, mean_series = data.summarize_macro_data(
    df_macro_for_metric, ACTIVE_VARS, verbose=False
)


mu        = mean_series.values
Sigma     = cov_df.values
Sigma_inv = np.linalg.inv(Sigma)
stds      = np.sqrt(np.diag(Sigma))

# Last observed macro values restricted to the active sensitivity dimensions.
x_last        = df_macro_for_metric[ACTIVE_VARS].dropna().iloc[-1].values
delta_baseline = x_last - mu   # current conditions expressed as deviation from hist. mean

print('Historical mean μ vs last-date values (= regression X-origin):')
print(f'  {"Variable":40s}  {"μ":>10}  {"σ":>8}  {"x_last":>10}  {"δ_base/σ":>10}')
for v, m, s, xl in zip(ACTIVE_VARS, mu, stds, x_last):
    print(f'  {v:40s}: {m:>10.3f}  {s:>8.3f}  {xl:>10.3f}  {(xl-m)/s:>+10.3f}σ')


# ## 4. Baseline Capital Metrics
# 
# We compute baseline corporate RWA at the current macro state
# $\boldsymbol{\Delta} = \boldsymbol{\Delta}_{\text{baseline}}$, where
# stressed PDs equal the portfolio base PDs by construction. We then derive
# $\text{CET1}^0$ and the implied fixed non-corporate RWA block.

# In[6]:


# ── Step 1: Corporate portfolio RWA at current macro baseline (\u0394 = \u03b4_baseline) ──
K_base     = calculate_capital_requirement(pd0, LGD, rho, maturity=MATURITY, quantile=ASRF_QUANTILE)

RWA_corp_0 = float(np.sum(ead * K_base * 12.5))
EAD_total  = float(np.sum(ead))

# ── Step 2: CET1 capital from EAD / CET1 = EAD_CET1_RATIO assumption ──────────
CET1_0 = EAD_total / EAD_CET1_RATIO

# ── Step 3: Total bank RWA from baseline CET1 ratio ───────────────────────────
RWA_total_0 = CET1_0 / CET1_RATIO_0

# ── Step 4: Non-corporate RWA — held constant under any stress scenario ────────
# Only corporate portfolio RWA responds to macro shocks; everything else is fixed.
RWA_other = RWA_total_0 - RWA_corp_0

print('\u2500' * 55)
print('Corporate portfolio')
print('\u2500' * 55)
print(f'  Total EAD              : {EAD_total:>10,.1f} EUR million')
print(f'  Corporate RWA (\u03b4_base) : {RWA_corp_0:>10,.1f} EUR million')
print()
print('\u2500' * 55)
print('Bank-wide capital structure (baseline)')
print('\u2500' * 55)
print(f'  CET1\u2070  = EAD / {EAD_CET1_RATIO:.0f}       : {CET1_0:>10,.1f} EUR million')
print(f'  Total RWA\u2070 = CET1\u2070 / {CET1_RATIO_0:.0%} : {RWA_total_0:>10,.1f} EUR million')
print(f'  Non-corporate RWA      : {RWA_other:>10,.1f} EUR million  (constant)')
print(f'  Corporate RWA          : {RWA_corp_0:>10,.1f} EUR million  (stress-sensitive)')
print(f'  Corporate RWA share    : {RWA_corp_0/RWA_total_0:.1%} of total RWA')
print()
print(f'  Baseline CET1 ratio    : {CET1_0/RWA_total_0:.4%}  (= {CET1_RATIO_0:.0%} by construction)')
print(f'  Breakdown threshold R_\u03c9: {R_OMEGA:.4%}  ({CET1_RATIO_0:.0%} \u2212 {int(DEPLETION*10000)} bps)')


# ## 5. Functions: Loss, RWA, CET1 Ratio
# 
# - `portfolio_loss` / `portfolio_loss_grad` — unchanged from notebook 04 (CET1 numerator)
# - `stressed_rwa` — new: Basel IRB RWA with stressed PDs (CET1 denominator)
# - `cet1_ratio` — new: eq. (22) of Hurlin et al.
# - `cet1_constraint` / `cet1_constraint_grad` — new: constraint $R^\omega - R(\boldsymbol{\Delta}) \geq 0$

# In[7]:


# ── Shared with notebook 04 ────────────────────────────────────────────────────´
#
# The betas in B_total were estimated on X = X_t − X_last (macro relative to last
# observed date), so the logit-PD adjustment for a scenario at deviation δ from the
# historical mean μ is:
#
#   adj = B_total @ (δ − δ_baseline)   where δ_baseline = x_last − μ
#
# At δ = δ_baseline (scenario = current conditions) → adj = 0 → PD unchanged ✓

def portfolio_loss(delta: np.ndarray) -> float:
    """Gordy/ASRF portfolio loss (EUR million) under macro deviation \u0394 from \u03bc."""
    adj        = np.clip(B_total @ (delta - delta_baseline), -50, 50)
    logit_pd_s = logit_pd0 + adj
    pd_s       = 1 / (1 + np.exp(-logit_pd_s))
    pd_s       = np.clip(pd_s, 1e-9, 1 - 1e-9)
    inv_pd_s   = norm.ppf(pd_s)
    cond_pd    = norm.cdf((inv_pd_s + sqrt_rho * inv_q) / sqrt_1mrho)
    return float(np.sum(ead * LGD * cond_pd))


def portfolio_loss_grad(delta: np.ndarray) -> np.ndarray:
    """Analytical gradient of portfolio_loss w.r.t. delta."""
    adj     = np.clip(B_total @ (delta - delta_baseline), -50, 50)
    pd_s    = np.clip(1 / (1 + np.exp(-(logit_pd0 + adj))), 1e-9, 1 - 1e-9)
    inv_pds = norm.ppf(pd_s)
    a       = (inv_pds + sqrt_rho * inv_q) / sqrt_1mrho
    phi_a   = norm.pdf(a)
    phi_inv = norm.pdf(inv_pds)
    var_pd  = pd_s * (1 - pd_s)
    return B_total.T @ (ead * LGD * phi_a * var_pd / (sqrt_1mrho * phi_inv))


def mahal_sq(delta: np.ndarray) -> float:
    """Squared Mahalanobis distance from today's macro conditions.

    D_M² = (δ − δ_baseline)ᵀ Σ⁻¹ (δ − δ_baseline)

    Uses the historical covariance structure Σ but centres on x_last so that
    D_M = 0 at today's conditions.  The optimizer therefore finds the *closest*
    scenario to today (in the historical metric) that breaks the CET1 threshold.
    """
    d = delta - delta_baseline
    return float(d @ Sigma_inv @ d)


def mahal_sq_grad(delta: np.ndarray) -> np.ndarray:
    """Analytical gradient of the displaced squared Mahalanobis distance."""
    return 2 * Sigma_inv @ (delta - delta_baseline)


# ── New: CET1-ratio functions ──────────────────────────────────────────────────

def stressed_rwa(delta: np.ndarray) -> float:
    """Total bank RWA under scenario \u0394.

    RWA_total(\u0394) = RWA_other  +  RWA_corp(\u0394)

    Non-corporate RWA (RWA_other) is held constant; only the corporate
    portfolio RWA responds to stressed PDs.  Eq. (20).
    """
    adj      = np.clip(B_total @ (delta - delta_baseline), -50, 50)
    pd_s     = np.clip(1 / (1 + np.exp(-(logit_pd0 + adj))), 1e-9, 1 - 1e-9)
    K_s      = calculate_capital_requirement(
        pd_s, LGD, rho, maturity=MATURITY, quantile=ASRF_QUANTILE
    )
    rwa_corp = float(np.sum(ead * K_s * 12.5))
    return RWA_other + rwa_corp


def cet1_ratio(delta: np.ndarray) -> float:
    """Stressed CET1 ratio: corporate losses deplete CET1, total bank RWA changes.

    R(\u0394) = (CET1_0 - \u0394L(\u0394)) / RWA_total(\u0394)

    where  \u0394L(\u0394)              = L_q(\u0394) - L_base         (incremental loss vs. current)
    and    RWA_total(\u0394)       = RWA_other + RWA_corp(\u0394)
    and    L_base = L_q(\u03b4_baseline)  = loss at current macro conditions

    R(\u03b4_baseline) = CET1_0 / RWA_total_0 = CET1_RATIO_0 exactly.  Eq. (22).
    """
    incr_loss = portfolio_loss(delta) - loss_base
    return (CET1_0 - incr_loss) / stressed_rwa(delta)


def cet1_constraint(delta: np.ndarray) -> float:
    """R_\u03c9 - R(\u0394) \u2265 0  \u2194  ratio falls below threshold.  Eq. (23)."""
    return R_OMEGA - cet1_ratio(delta)


def cet1_constraint_grad(delta: np.ndarray) -> np.ndarray:
    """Numerical gradient of the CET1 constraint via forward differences."""
    return approx_fprime(delta, cet1_constraint, 1e-6)


# ── Baseline metrics at current macro conditions (δ = δ_baseline) ─────────────
# loss_base must be defined before cet1_ratio is called.
# At δ = δ_baseline: adj = B @ (δ_baseline - δ_baseline) = 0, so PD = pd0 exactly.
loss_base = portfolio_loss(delta_baseline)
rwa_base  = stressed_rwa(delta_baseline)
r0_check  = cet1_ratio(delta_baseline)   # must equal CET1_RATIO_0 exactly

print(f'Baseline portfolio loss (current macro, q={ASRF_QUANTILE:.1%}) : {loss_base:>10,.1f} EUR million')
print(f'Baseline RWA  (current macro)                    : {rwa_base:>10,.1f} EUR million')
print(f'Baseline CET1 ratio  R\u2070                          : {r0_check:.4%}  (should equal {CET1_RATIO_0:.2%})')
print(f'Breakdown threshold  R_\u03c9                          : {R_OMEGA:.4%}  (gap = {(r0_check - R_OMEGA)*10000:.0f} bps)')


# ## 6. CET1 Ratio — Single-Factor Sensitivity
# 
# Each panel shows how the CET1 ratio responds to a ±3σ shift in one macro variable
# while all others remain at their historical mean.

def _permanent_delta(base_var: str, dev: float) -> np.ndarray:
    """Scenario vector with dev applied to the chosen base-variable family.

    'dev' is a raw-unit deviation ADDED to the current-period macro value.
    The axis in the sensitivity chart is therefore (x_scenario - x_last) / σ.
    """
    d = delta_baseline.copy()
    for k, v in enumerate(ACTIVE_VARS):
        if v == base_var or v.startswith(base_var + '_lag'):
            d[k] += dev
    return d

# ## 7. Constrained Optimisation
# 
# We minimise the **squared Mahalanobis distance** subject to the CET1 ratio falling
# below the breakdown threshold $R^\omega$.
# Multiple starting points are used for robustness.

# In[9]:


constraint = {
    'type': 'ineq',
    'fun' : cet1_constraint,
    'jac' : cet1_constraint_grad,
}

rng = np.random.default_rng(42)

DEFAULT_START_MULTS = [
    {'GDP_Growth': -1.5, 'Interest_Rate': +1.0, 'Brent_Oil': +2.0, 'Fuel_Index': +0.5, 'CPI': +0.5, 'GPR_Global': +1.0},
    {'GDP_Growth': -1.0, 'Interest_Rate': +2.0, 'Brent_Oil': +2.5, 'Fuel_Index': +1.5, 'CPI': +0.5, 'GPR_Global': +0.5},
    {'GDP_Growth': -2.0, 'Interest_Rate': +0.5, 'Brent_Oil': +1.5, 'Fuel_Index': +1.0, 'CPI': +1.0, 'GPR_Global': +2.0},
    {'GDP_Growth': -0.5, 'Interest_Rate': +1.5, 'Brent_Oil': +3.0, 'Fuel_Index': +2.0, 'CPI': +0.5, 'GPR_Global': +0.5},
    {'GDP_Growth': -2.5, 'Interest_Rate': +2.0, 'Brent_Oil': +1.0, 'Fuel_Index': +0.5, 'CPI': +1.0, 'GPR_Global': +1.5},
]

def _permanent_start(base_sigmas: dict[str, float]) -> np.ndarray:
    """Build a starting point from per-base-variable σ-multiples."""
    d = np.zeros(N_VARS)
    for bv in ACTIVE_BASE_VARS:
        mult = base_sigmas.get(bv, 0.0)
        for k, v in enumerate(ACTIVE_VARS):
            if v == bv or v.startswith(bv + '_lag'):
                d[k] = mult * stds[k]
    return d

delta0_list = [
    delta_baseline,                                              # current conditions
    np.zeros(N_VARS),                                           # historical mean
    *(_permanent_start(mults) for mults in DEFAULT_START_MULTS),
    *(rng.uniform(-3, 3, (8, N_VARS)) * stds),
]

best = None
for k, d0 in enumerate(delta0_list):
    res = minimize(
        mahal_sq,
        d0,
        method      = 'SLSQP',
        jac         = mahal_sq_grad,
        constraints = constraint,
        options     = dict(maxiter=3000, ftol=1e-12),
    )
    R    = cet1_ratio(res.x)
    D    = np.sqrt(mahal_sq(res.x))
    ok   = R <= R_OMEGA + 1e-4
    print(f'Run {k+1:2d}: success={res.success}, feasible={ok}, '
          f'D_M={D:.4f},  CET1={R:.4%}')
    if ok and (best is None or res.fun < best.fun):
        best = res

assert best is not None, 'No feasible solution found — try wider starting points'
delta_opt = best.x
D_opt     = np.sqrt(mahal_sq(delta_opt))
R_opt     = cet1_ratio(delta_opt)
L_opt     = portfolio_loss(delta_opt)
RWA_opt   = stressed_rwa(delta_opt)
x_opt     = mu + delta_opt

print(f'\n✓ Optimal Δ*:  D_M = {D_opt:.4f},  CET1 = {R_opt:.4%},  Loss = {L_opt:,.1f} EUR m')


# ## 8. Optimal Scenario — Results Table

# In[10]:


# Deviation from today (= what the optimizer actually minimises)
shock_from_today = delta_opt - delta_baseline   # x_scenario - x_last
shock_sigma      = shock_from_today / stds      # in σ units

results_df = pd.DataFrame({
    'Variable'            : ACTIVE_VARS,
    'Label'               : [VAR_LABELS[v] for v in ACTIVE_VARS],
    'x_last (today)'      : x_last,
    'x* = x_last + shock' : x_last + shock_from_today,
    'shock (raw)'         : shock_from_today,
    'shock / \u03c3'          : shock_sigma,
})

print('=' * 105)
print(f'OPTIMAL MACRO SCENARIO  \u2014  closest scenario to TODAY with CET1 \u2264 {R_OMEGA:.2%}')
print('=' * 105)
print(results_df.to_string(index=False, float_format=lambda v: f'{v:>10.3f}'))

# Summary aggregated to base variables
print()
print('Aggregated view (base variables):')
agg_rows = []
for bv in ACTIVE_BASE_VARS:
    idx_bv = [k for k, v in enumerate(ACTIVE_VARS) if v == bv or v.startswith(bv + '_lag')]
    j = ACTIVE_VARS.index(bv)
    agg_rows.append({
        'Base variable'     : _BASE_LABELS[bv],
        'shock[t] (raw)'    : shock_from_today[j],
        'shock[t] / \u03c3' : shock_sigma[j],
        'sum|shock/\u03c3|' : np.sum(np.abs(shock_sigma[idx_bv])),
    })
print(pd.DataFrame(agg_rows).to_string(index=False, float_format=lambda v: f'{v:>8.3f}'))

# ── Plausibility diagnostics ──────────────────────────────────────────────────
#
# D_M is measured from TODAY (displaced Mahalanobis).
# We also compute:
#   (a) D_M from the HISTORICAL MEAN to the optimal scenario — sanity check;
#       should be larger since today is already stressed.
#   (b) D_M from historical mean to TODAY — quantifies how far current
#       macro conditions already are from the historical norm.
#   (c) Effective degrees of freedom using the Satterthwaite approximation
#       eff_df = trace(Σ)² / trace(Σ²), which corrects for correlated lags.

eigenvalues  = np.linalg.eigvalsh(Sigma)
eff_df       = float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))

D_today_from_hist = float(np.sqrt(delta_baseline @ Sigma_inv @ delta_baseline))
D_opt_from_hist   = float(np.sqrt(delta_opt @ Sigma_inv @ delta_opt))

# Chi² using effective df (correct reference for correlated variables)
p_excl_eff = float(chi2.sf(D_opt**2, df=eff_df))
# Chi² from historical mean to optimal scenario
p_excl_hist = float(chi2.sf(D_opt_from_hist**2, df=eff_df))

print()
print('─' * 70)
print('PLAUSIBILITY DIAGNOSTICS')
print('─' * 70)
print(f'  Effective degrees of freedom (Satterthwaite): {eff_df:.1f}  '
      f'(vs nominal {N_VARS})')
print()
print(f'  D_M(today → optimal scenario)    = {D_opt:.4f}   '
      f'[χ²({eff_df:.0f}) tail = {p_excl_eff*100:.1f}%]')
print(f'  D_M(hist. mean → today)          = {D_today_from_hist:.4f}   '
      f'(current macro is already this far from norm)')
print(f'  D_M(hist. mean → optimal)        = {D_opt_from_hist:.4f}   '
      f'[χ²({eff_df:.0f}) tail = {p_excl_hist*100:.1f}%]')
print()
print(f'  Interpretation:')
print(f'    Today is {D_today_from_hist:.2f}σ from the historical mean.')
print(f'    The optimal scenario is {D_opt:.2f}σ FURTHER from today,')
print(f'    or {D_opt_from_hist:.2f}σ from the historical mean in total.')
print(f'    Using eff. df={eff_df:.0f}: the scenario has a χ² tail of '
      f'{p_excl_hist*100:.1f}% (from hist. mean).')
print('─' * 70)
print()
print(f'Baseline CET1 ratio  R\u2070             = {CET1_RATIO_0:.4%}')
print(f'Stressed CET1 ratio  R(\u0394*)          = {R_opt:.4%}')
print(f'CET1 depletion                     = {(CET1_RATIO_0 - R_opt)*10000:.1f} bps')
print(f'Breakdown threshold  R_\u03c9            = {R_OMEGA:.4%}')
print()
print(f'Portfolio loss at \u0394*       (optimum) = {L_opt:>10,.1f} EUR million')
print(f'Portfolio loss at today  (baseline) = {loss_base:>10,.1f} EUR million')
print(f'Stressed RWA  at \u0394*                 = {RWA_opt:>10,.1f} EUR million')
print(f'Baseline RWA  at \u03b4_base             = {RWA_total_0:>10,.1f} EUR million')


# ## 9. Stressed PDs and Capital at the Optimal Scenario

# In[11]:


adj_opt     = np.clip(B_total @ (delta_opt - delta_baseline), -50, 50)
pd_stressed = 1 / (1 + np.exp(-(logit_pd0 + adj_opt)))
pd_stressed = np.clip(pd_stressed, 1e-9, 1 - 1e-9)
K_stressed  = calculate_capital_requirement(
    pd_stressed, LGD, rho, maturity=MATURITY, quantile=ASRF_QUANTILE
)

df_valid = df_valid.copy()
df_valid['pd_base']     = pd0
df_valid['pd_stressed'] = pd_stressed
df_valid['pd_mult']     = pd_stressed / pd0
df_valid['K_base']      = K_base
df_valid['K_stressed']  = K_stressed
df_valid['rwa_base']    = ead * K_base * 12.5
df_valid['rwa_stressed']= ead * K_stressed * 12.5

tbl_pd = (
    df_valid.groupby('sector')
    .agg(
        n              = ('pd_base', 'count'),
        pd_base_pct    = ('pd_base',     lambda x: f"{x.mean()*100:.2f}%"),
        pd_stress_pct  = ('pd_stressed', lambda x: f"{x.mean()*100:.2f}%"),
        stress_mult    = ('pd_mult',     lambda x: f"{x.mean():.2f}\u00d7"),
        rwa_base_m     = ('rwa_base',    lambda x: f"{x.sum():,.0f}"),
        rwa_stress_m   = ('rwa_stressed',lambda x: f"{x.sum():,.0f}"),
    )
)
print('Sector breakdown at optimal \u0394* (EUR million for RWA):')
print(tbl_pd.to_string())
print()
print(f'Portfolio-wide base PD    : {pd0.mean()*100:.2f}%')
print(f'Portfolio-wide stressed PD: {pd_stressed.mean()*100:.2f}%')
print(f'Average stress multiplier : {(pd_stressed / pd0).mean():.2f}\u00d7')
print(f'Total base RWA            : {df_valid["rwa_base"].sum():,.1f} EUR million')
print(f'Total stressed RWA        : {df_valid["rwa_stressed"].sum():,.1f} EUR million')


# ## 10. Visualisation — Optimal Scenario

# In[12]:


# ## 11. Pareto Frontier — Plausibility vs. CET1 Depletion
# 
# Sweep the breakdown threshold from a mild depletion to a severe one,
# and plot the minimum Mahalanobis distance required at each level.
# This is the CET1-ratio analogue of the loss frontier in notebook 04.

# In[13]:


# ## 12. Hurlin et al. Figures 1 & 2 — CET1 Frontier Geometry
# 
# 2-D projections onto the two most impactful scenario dimensions, showing:
# - **Fig. 1 analog**: CET1 breach region, plausibility contours, local neighbourhood $\mathcal{S}_\rho$
# - **Fig. 2 analog**: Near-optimal set $\mathcal{N}_\varepsilon$
# 
# The CET1-ratio breakdown frontier replaces the loss frontier of notebook 04.
# 
# Shocks on the **non-displayed** dimensions are held at the optimal **18-D** design point $\Delta^*$ (`delta_opt`), not at zero.
# 

# In[14]:


# Each panel conditions on the **other 16 shocks at** $\Delta^*$ (`delta_opt`).
# 
# ## 13. Frontier Geometry — All Factor Pairings (base variables)
# 
# The Fig. 1-style plot for every pair of the 6 *base* (current-period) macro variables.
# The star marks $(\Delta^*_i,\Delta^*_j)$ from the global 18-D optimum `delta_opt`;
# tangent plausibility levels use $D_M^2 = \Delta^{*\top}\Sigma^{-1}\Delta^*$.
# Non-displayed dimensions are held at $\Delta^*$.
# 

# In[15]:


# ## Summary
# 
# | | |
# |---|---|
# | **Initial CET1 ratio** $R^0$ | 14.00% |
# | **Depletion threshold** $\varepsilon$ | 300 bps |
# | **Breakdown threshold** $R^\omega = R^0 - \varepsilon$ | 11.00% |
# | **Optimal Mahalanobis distance** $D_M^*$ | see cell 8 |
# | **Scenario probability** ($\chi^2(6)$ tail) | see cell 8 |
# | **ASRF quantile** | 99.9% |
# | **LGD** | 40% |
# | **Maturity** | 2.5 years |
