"""
Macro Scenario Probability Landscape
=====================================
Standalone script – no CET1 constraint.

Four plot types:
  1. Single-factor probability curves   (6 panels)
  2. 2-D probability heatmaps           (15 panels, all base-variable pairs)
  3. Historical time series + today     (6 panels)
  4. Probability threshold table        (how far each variable must move to
                                         cross the 50 / 10 / 5 / 1 % thresholds)

All probabilities use the displaced Mahalanobis centred on today (x_last),
with the Satterthwaite effective degrees of freedom to correct for correlated
lag dimensions.
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from itertools import combinations
from scipy.stats import chi2, norm

# ── Path setup ─────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cet1_macro_optimization as model  # run the model, then read its globals

# ── Shorthand references ────────────────────────────────────────────────────
ALL_BASE_VARS    = model.ALL_BASE_VARS
ALL_VARS         = model.ALL_VARS
_BASE_LABELS     = model._BASE_LABELS
mu               = model.mu
stds             = model.stds
x_last           = model.x_last
delta_baseline   = model.delta_baseline
Sigma            = model.Sigma
Sigma_inv        = model.Sigma_inv
df_merged_lagged = model.df_merged_lagged
N_VARS           = model.N_VARS

# ── Effective degrees of freedom (Satterthwaite) ────────────────────────────
eigenvalues = np.linalg.eigvalsh(Sigma)
eff_df = float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2))
print(f'Effective degrees of freedom: {eff_df:.1f}  (nominal {N_VARS})')

# ── Helper functions ────────────────────────────────────────────────────────
def mahal_sq_today(delta: np.ndarray) -> float:
    """Squared Mahalanobis distance from today's conditions."""
    d = delta - delta_baseline
    return float(d @ Sigma_inv @ d)


def scenario_prob(delta: np.ndarray) -> float:
    """χ²(eff_df) tail probability: how unusual is this scenario from today?"""
    return float(chi2.sf(mahal_sq_today(delta), df=eff_df))


def permanent_delta(base_var: str, dev: float) -> np.ndarray:
    """Permanent shock: delta_baseline + dev on base_var AND all its lags."""
    d = delta_baseline.copy()
    for k, v in enumerate(ALL_VARS):
        if v == base_var or v.startswith(base_var + '_lag'):
            d[k] += dev
    return d


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Single-factor probability curves
# For each base variable: probability of being at least as far from today
# as a permanent shock of ±k·σ.
# ═══════════════════════════════════════════════════════════════════════════
print('\nPlot 1: single-factor probability curves…')

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes_flat = axes.flatten()

sweep_sigma = np.linspace(-3.5, 3.5, 200)

for idx, base_var in enumerate(ALL_BASE_VARS):
    ax  = axes_flat[idx]
    j   = ALL_VARS.index(base_var)
    std = stds[j]
    x_today_val = x_last[j]

    # Raw-unit sweep centred on today
    sweep_raw = sweep_sigma * std          # in original units, relative to today
    x_vals    = x_today_val + sweep_raw    # absolute level on x-axis

    probs = [scenario_prob(permanent_delta(base_var, s * std)) * 100
             for s in sweep_sigma]

    ax.plot(x_vals, probs, color='steelblue', lw=2)

    # Reference probability lines
    for pct, col, ls in [(50, 'grey', ':'), (10, 'darkorange', '--'),
                         (5, 'crimson', '--'), (1, 'darkred', '-.')]:
        ax.axhline(pct, color=col, lw=1.2, ls=ls, label=f'{pct}%')

    # Mark today
    ax.axvline(x_today_val, color='black', lw=1.0, ls=':')
    p_today = scenario_prob(delta_baseline) * 100
    ax.plot(x_today_val, p_today, 'ko', ms=7, zorder=5,
            label=f'Today: {x_today_val:.2f}')

    # Secondary x-axis in σ units
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    sigma_ticks = np.array([-3, -2, -1, 0, 1, 2, 3])
    ax2.set_xticks(x_today_val + sigma_ticks * std)
    ax2.set_xticklabels([f'{s:+d}σ' for s in sigma_ticks], fontsize=7)
    ax2.tick_params(length=3)

    ax.set_xlabel(f'{_BASE_LABELS[base_var]}  (absolute level)', fontsize=9)
    ax.set_ylabel('Tail probability  P(%)', fontsize=9)
    ax.set_title(_BASE_LABELS[base_var], fontweight='bold', pad=18)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=7.5, loc='upper right')
    ax.grid(alpha=0.3)

fig.suptitle(
    'Scenario Plausibility — Single-Factor Permanent Shocks\n'
    f'P = χ²({eff_df:.0f} eff. df) tail (displaced Mahalanobis from today)',
    fontsize=12, fontweight='bold')
fig.tight_layout()
plt.savefig(HERE / 'scenario_prob_single_factor.png', dpi=150, bbox_inches='tight')
plt.show()
print('  Saved → scenario_prob_single_factor.png')


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 — 2-D probability heatmaps for all 15 base-variable pairs
# ═══════════════════════════════════════════════════════════════════════════
print('Plot 2: 2-D probability heatmaps…')

N_GRID  = 80
CONTOUR_LEVELS = [50, 20, 10, 5, 1]   # probability iso-lines (%)
all_pairs = list(combinations(ALL_BASE_VARS, 2))

fig, axes = plt.subplots(3, 5, figsize=(22, 13))
axes_flat = axes.flatten()

for plot_idx, (bv_a, bv_b) in enumerate(all_pairs):
    ax  = axes_flat[plot_idx]
    ja  = ALL_VARS.index(bv_a)
    jb  = ALL_VARS.index(bv_b)
    sa  = stds[ja]
    sb  = stds[jb]

    ga = np.linspace(-3.5 * sa, 3.5 * sa, N_GRID)
    gb = np.linspace(-3.5 * sb, 3.5 * sb, N_GRID)
    GA, GB = np.meshgrid(ga, gb)

    # Build all (N_GRID²) delta vectors at once: fix at delta_baseline,
    # override the two displayed dimensions
    npts    = N_GRID * N_GRID
    d_grid  = np.broadcast_to(delta_baseline, (npts, N_VARS)).copy()
    d_grid[:, ja] = delta_baseline[ja] + GA.ravel()
    d_grid[:, jb] = delta_baseline[jb] + GB.ravel()
    # Also shift all lags for each variable (permanent shock)
    for k, v in enumerate(ALL_VARS):
        base = v.split('_lag')[0]
        if base == bv_a and k != ja:
            d_grid[:, k] = delta_baseline[k] + GA.ravel()
        if base == bv_b and k != jb:
            d_grid[:, k] = delta_baseline[k] + GB.ravel()

    # D_M² for each grid point
    diff  = d_grid - delta_baseline
    d2    = ((diff @ Sigma_inv) * diff).sum(axis=1).reshape(N_GRID, N_GRID)
    P_grid = chi2.sf(d2, df=eff_df) * 100   # in %

    gx = GA / sa
    gy = GB / sb

    # Filled contour (probability colour)
    cf = ax.contourf(gx, gy, P_grid, levels=50, cmap='RdYlGn_r',
                     vmin=0, vmax=100, alpha=0.85)

    # Iso-probability lines
    for pct in CONTOUR_LEVELS:
        cs = ax.contour(gx, gy, P_grid, levels=[pct],
                        colors=['white'], linewidths=1.0, linestyles='--')
        ax.clabel(cs, fmt={pct: f'{pct}%'}, fontsize=5.5, inline=True)

    # Today = origin
    ax.plot(0, 0, 'wo', ms=6, zorder=6, markeredgecolor='black', lw=0.8)
    ax.axhline(0, color='white', lw=0.4, ls=':')
    ax.axvline(0, color='white', lw=0.4, ls=':')

    ax.set_xlabel(f'{_BASE_LABELS[bv_a]}  (σ from today)', fontsize=6.5)
    ax.set_ylabel(f'{_BASE_LABELS[bv_b]}  (σ from today)', fontsize=6.5)
    ax.set_title(f'{_BASE_LABELS[bv_a]} vs {_BASE_LABELS[bv_b]}',
                 fontsize=7, fontweight='bold')
    ax.tick_params(labelsize=5.5)
    ax.set_xlim(gx.min(), gx.max())
    ax.set_ylim(gy.min(), gy.max())

    print(f'  Pair {plot_idx+1:2d}/15  ({bv_a} vs {bv_b})')

# Shared colourbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap='RdYlGn_r',
                            norm=plt.Normalize(vmin=0, vmax=100))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Tail probability P  (%)\n[green = common, red = rare]', fontsize=9)

fig.suptitle(
    f'Scenario Probability Landscape — All 15 Base-Variable Pairs\n'
    f'(permanent shocks from today  |  χ²({eff_df:.0f} eff. df)  |  '
    f'white dot = today)',
    fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 0.89, 1])
plt.savefig(HERE / 'scenario_prob_2d_heatmaps.png', dpi=150, bbox_inches='tight')
plt.show()
print('  Saved → scenario_prob_2d_heatmaps.png')


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Historical time series with today's position highlighted
# ═══════════════════════════════════════════════════════════════════════════
print('Plot 3: historical time series…')

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes_flat = axes.flatten()

for idx, base_var in enumerate(ALL_BASE_VARS):
    ax  = axes_flat[idx]
    j   = ALL_VARS.index(base_var)

    series = df_merged_lagged[['Date', base_var]].dropna()
    dates  = pd.to_datetime(series['Date'])
    vals   = series[base_var].values

    m_hist   = mu[j]
    s_hist   = stds[j]
    x_today  = x_last[j]

    ax.plot(dates, vals, color='steelblue', lw=1.4, label='History')
    ax.axhline(m_hist, color='grey',      lw=1.2, ls='--', label='Hist. mean μ')
    ax.axhline(m_hist + s_hist,   color='darkorange', lw=0.8, ls=':', label='μ ± 1σ')
    ax.axhline(m_hist - s_hist,   color='darkorange', lw=0.8, ls=':')
    ax.axhline(m_hist + 2*s_hist, color='crimson',    lw=0.8, ls=':', label='μ ± 2σ')
    ax.axhline(m_hist - 2*s_hist, color='crimson',    lw=0.8, ls=':')

    # Today
    ax.axhline(x_today, color='black', lw=1.5,
               label=f'Today: {x_today:.2f} ({(x_today-m_hist)/s_hist:+.1f}σ)')
    ax.plot(dates.iloc[-1], x_today, 'k^', ms=8, zorder=6)

    ax.set_title(_BASE_LABELS[base_var], fontweight='bold')
    ax.set_ylabel('Level')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', labelsize=7.5)

fig.suptitle('Historical Macro Variables — Where Are We Today?',
             fontsize=13, fontweight='bold')
fig.tight_layout()
plt.savefig(HERE / 'scenario_prob_history.png', dpi=150, bbox_inches='tight')
plt.show()
print('  Saved → scenario_prob_history.png')


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 — Probability threshold table
# For each base variable: how large a permanent shock (σ) is needed to reach
# the 50 / 20 / 10 / 5 / 1 % tail probability thresholds?
# ═══════════════════════════════════════════════════════════════════════════
print('Plot 4: probability threshold bars…')

TARGET_PROBS = [0.50, 0.20, 0.10, 0.05, 0.01]
PROB_COLORS  = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
PROB_LABELS  = ['50%', '20%', '10%', '5%', '1%']

# For each base var and target prob, find the +/- shock needed
threshold_sigmas = {bv: {} for bv in ALL_BASE_VARS}

for bv in ALL_BASE_VARS:
    j   = ALL_VARS.index(bv)
    std = stds[j]
    for pct, target in zip(PROB_LABELS, TARGET_PROBS):
        # Binary search for the positive shock that hits target probability
        lo, hi = 0.0, 6.0
        for _ in range(50):
            mid  = (lo + hi) / 2
            p    = scenario_prob(permanent_delta(bv, mid * std))
            if p > target:
                lo = mid
            else:
                hi = mid
        threshold_sigmas[bv][pct] = (lo + hi) / 2

fig, ax = plt.subplots(figsize=(12, 6))

n_vars = len(ALL_BASE_VARS)
n_thr  = len(TARGET_PROBS)
bar_w  = 0.14
x_pos  = np.arange(n_vars)

for i, (pct, col) in enumerate(zip(PROB_LABELS, PROB_COLORS)):
    vals = [threshold_sigmas[bv][pct] for bv in ALL_BASE_VARS]
    offset = (i - n_thr / 2 + 0.5) * bar_w
    bars = ax.bar(x_pos + offset, vals, width=bar_w,
                  color=col, label=pct, edgecolor='white', alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                f'{val:.1f}σ', ha='center', va='bottom', fontsize=7)

ax.set_xticks(x_pos)
ax.set_xticklabels([_BASE_LABELS[bv] for bv in ALL_BASE_VARS], fontsize=11)
ax.set_ylabel('Permanent shock size needed  (σ from today)', fontsize=11)
ax.set_title(
    'How Large a Shock is Needed to Reach Each Probability Threshold?\n'
    f'(permanent shock, all lags move together  |  χ²({eff_df:.0f} eff. df))',
    fontsize=11, fontweight='bold')
ax.legend(title='Tail prob.', fontsize=9, title_fontsize=9,
          loc='upper right', ncol=5)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

fig.tight_layout()
plt.savefig(HERE / 'scenario_prob_thresholds.png', dpi=150, bbox_inches='tight')
plt.show()
print('  Saved → scenario_prob_thresholds.png')


# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════
print()
print('=' * 75)
print('PROBABILITY THRESHOLDS — σ of permanent shock needed from today')
print(f'(χ²  eff. df = {eff_df:.1f})')
print('=' * 75)
header = f'{"Variable":20s}' + ''.join(f'{p:>10s}' for p in PROB_LABELS)
print(header)
print('-' * 75)
for bv in ALL_BASE_VARS:
    row = f'{_BASE_LABELS[bv]:20s}'
    for pct in PROB_LABELS:
        row += f'{threshold_sigmas[bv][pct]:>10.2f}σ'
    j = ALL_VARS.index(bv)
    row += f'    [today: {(x_last[j]-mu[j])/stds[j]:+.2f}σ from hist. mean]'
    print(row)
print('=' * 75)
print()
print('Files saved:')
print('  scenario_prob_single_factor.png')
print('  scenario_prob_2d_heatmaps.png')
print('  scenario_prob_history.png')
print('  scenario_prob_thresholds.png')
