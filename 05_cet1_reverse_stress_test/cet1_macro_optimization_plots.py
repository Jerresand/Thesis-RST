import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from itertools import combinations
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import approx_fprime, minimize
from scipy.stats import chi2, norm


def plot_cet1_sensitivity(
    all_base_vars,
    all_vars,
    base_labels,
    stds,
    cet1_ratio,
    permanent_delta,
    r_omega,
):
    n = len(all_base_vars)
    n_cols = min(3, max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.0 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, base_var in enumerate(all_base_vars):
        ax = axes_flat[idx]
        j = all_vars.index(base_var)
        std = stds[j]

        sweep = np.linspace(-3 * std, 3 * std, 80)
        ratios = [cet1_ratio(permanent_delta(base_var, dev)) * 100 for dev in sweep]

        ax.plot(sweep / std, ratios, color='steelblue', lw=2)
        ax.axhline(r_omega * 100, color='crimson', ls='--', lw=1.5,
                   label=f'Rω = {r_omega:.2%}')
        ax.axvline(0, color='grey', ls=':', lw=1, label='Current conditions')
        ax.set_title(base_labels[base_var], fontweight='bold')
        ax.set_xlabel('Δ vs current  (σ units)')
        ax.set_ylabel('CET1 ratio (%)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    has_lags = any('_lag' in v for v in all_vars)
    subtitle = '(current + matching lags shift simultaneously)' if has_lags else '(current-period factors only)'
    fig.suptitle('CET1 Ratio Sensitivity to Single-Factor Permanent Macro Shocks\n'
                 f'{subtitle}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig('cet1_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Chart saved → cet1_sensitivity.png')


def plot_optimal_results(
    all_vars,
    var_labels,
    shock_sigma,
    df_valid,
    cet1_ratio_0,
    cet1_0,
    l_opt,
    loss_base,
    rwa_opt,
    rwa_total_0,
    r_omega,
    r_opt,
    d_opt,
):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    colors_a = ['#c0392b' if d < 0 else '#2980b9' for d in shock_sigma]
    labels_a = [var_labels[v] for v in all_vars]
    bars = ax_a.barh(labels_a, shock_sigma,
                     color=colors_a, edgecolor='white', height=0.55)
    ax_a.axvline(0, color='black', lw=0.8)
    for bar, val in zip(bars, shock_sigma):
        if abs(val) > 0.05:
            ax_a.text(val + (0.04 if val >= 0 else -0.04), bar.get_y() + bar.get_height() / 2,
                      f'{val:+.2f}σ', va='center',
                      ha='left' if val >= 0 else 'right', fontsize=6.5)
    ax_a.set_xlabel('shock / σ  (standard deviations from today)')
    has_lags = any('_lag' in v for v in all_vars)
    title_suffix = '(current + lagged variables)' if has_lags else '(current-period variables)'
    ax_a.set_title(f'Optimal Scenario — shock from today\n{title_suffix}', fontweight='bold')
    ax_a.tick_params(axis='y', labelsize=7)
    ax_a.grid(axis='x', alpha=0.3)

    ax_b = fig.add_subplot(gs[0, 1])
    sector_mult = df_valid.groupby('sector')['pd_mult'].mean().sort_values(ascending=True)
    ax_b.barh(sector_mult.index, sector_mult.values - 1,
              color='#e67e22', edgecolor='white', height=0.55)
    ax_b.axvline(0, color='black', lw=0.8)
    ax_b.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax_b.set_xlabel('PD stress increase (relative)')
    ax_b.set_title('Sector PD Stress at Δ*', fontweight='bold')
    ax_b.grid(axis='x', alpha=0.3)

    ax_c = fig.add_subplot(gs[1, 0])
    incr_loss_opt = l_opt - loss_base
    delta_loss_cet1 = -incr_loss_opt / rwa_total_0
    r_after_loss = cet1_ratio_0 + delta_loss_cet1
    delta_rwa_cet1 = (cet1_0 - incr_loss_opt) / rwa_opt - (cet1_0 - incr_loss_opt) / rwa_total_0
    waterfall_vals = [cet1_ratio_0, delta_loss_cet1, delta_rwa_cet1, r_opt]
    waterfall_lbls = ['Baseline\nR⁰', 'Δ from\nloss', 'Δ from\nRWA', 'Stressed\nR(Δ*)']
    bottoms = [0, cet1_ratio_0, r_after_loss, 0]
    bar_colors = ['#2980b9', '#c0392b', '#e67e22', '#27ae60']
    ax_c.bar(waterfall_lbls, waterfall_vals, bottom=bottoms, color=bar_colors,
             edgecolor='white', width=0.5)
    ax_c.axhline(r_omega, color='crimson', ls='--', lw=1.5,
                 label=f'Threshold R_ω = {r_omega:.2%}')
    ax_c.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax_c.set_title('CET1 Ratio Decomposition', fontweight='bold')
    ax_c.legend(fontsize=9)
    ax_c.grid(axis='y', alpha=0.3)

    ax_d = fig.add_subplot(gs[1, 1])
    rwa_by_sector = df_valid.groupby('sector')[['rwa_base', 'rwa_stressed']].sum() / 1000
    rwa_by_sector = rwa_by_sector.sort_values('rwa_stressed', ascending=True)
    y_pos = np.arange(len(rwa_by_sector))
    ax_d.barh(y_pos - 0.2, rwa_by_sector['rwa_base'], height=0.35,
              color='steelblue', label='Baseline RWA')
    ax_d.barh(y_pos + 0.2, rwa_by_sector['rwa_stressed'], height=0.35,
              color='#e74c3c', label='Stressed RWA')
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(rwa_by_sector.index, fontsize=8)
    ax_d.set_xlabel('RWA (EUR billion)')
    ax_d.set_title('RWA Base vs Stressed by Sector', fontweight='bold')
    ax_d.legend(fontsize=9)
    ax_d.grid(axis='x', alpha=0.3)

    fig.suptitle(f'CET1 Reverse Stress Test — Design Point  '
                 f'(D_M = {d_opt:.2f},  CET1 = {r_opt:.2%})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    plt.savefig('cet1_opt_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Chart saved → cet1_opt_results.png')


def plot_optimal_shocks_grouped_lags(
    all_base_vars,
    all_vars,
    base_labels,
    shock_sigma,
    d_opt,
    r_opt,
):
    """Per base factor: shock from today (σ) for current period and each lag.

    Uses only the already-computed optimal point — no grids or re-optimisation.
    """
    has_lags = any('_lag' in v for v in all_vars)
    if not has_lags:
        print('Skipping lag-grouped shock plot: loaded sensitivity CSV has no lag columns.')
        return

    n = len(all_base_vars)
    n_cols = min(3, max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.6 * n_rows))
    axes_flat = axes.flatten()
    lag_colors = ('#2980b9', '#7fb3d5', '#bdc3c7')
    x_labels = ('t', 't-1', 't-2')

    for ax, base_var in zip(axes_flat, all_base_vars):
        ordered = [base_var]
        for lag in (1, 2):
            name = f'{base_var}_lag{lag}'
            if name in all_vars:
                ordered.append(name)

        idx = [all_vars.index(v) for v in ordered]
        vals = [shock_sigma[i] for i in idx]
        x = np.arange(len(vals))
        cols = lag_colors[: len(vals)]
        ax.bar(x, vals, color=cols, edgecolor='white', width=0.62)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels[: len(vals)])
        ax.set_ylabel('shock / σ')
        ax.set_title(base_labels[base_var], fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for xi, vi in zip(x, vals):
            if abs(vi) >= 0.02:
                ax.text(
                    xi,
                    vi + (0.06 if vi >= 0 else -0.06),
                    f'{vi:+.2f}',
                    ha='center',
                    va='bottom' if vi >= 0 else 'top',
                    fontsize=8,
                )

    for ax in axes_flat[len(all_base_vars):]:
        ax.set_visible(False)

    fig.suptitle(
        f'Optimal scenario Δ* — shocks from today by lag  '
        f'(D_M = {d_opt:.2f},  CET1 = {r_opt:.2%})',
        fontsize=12,
        fontweight='bold',
    )
    fig.tight_layout()
    plt.savefig('cet1_opt_shocks_by_factor_lags.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Chart saved → cet1_opt_shocks_by_factor_lags.png')


def plot_pareto_frontier(
    cet1_ratio_0,
    depletion,
    d_opt,
    mahal_sq,
    mahal_sq_grad,
    cet1_ratio,
    n_vars,
):
    depletions = np.arange(0.01, 0.201, 0.01)
    thresholds = cet1_ratio_0 * (1 - depletions)
    front_d = []
    front_ok = []
    delta_warm = np.zeros(n_vars)

    for thr in thresholds:
        def _con(d, t=thr):
            return t - cet1_ratio(d)

        def _con_jac(d, t=thr):
            return approx_fprime(d, lambda dd: t - cet1_ratio(dd), 1e-6)

        res = minimize(mahal_sq, delta_warm, method='SLSQP',
                       jac=mahal_sq_grad,
                       constraints={'type': 'ineq', 'fun': _con, 'jac': _con_jac},
                       options=dict(maxiter=2000, ftol=1e-10))
        r_val = cet1_ratio(res.x)
        d_val = np.sqrt(mahal_sq(res.x))
        ok = r_val <= thr + 1e-3
        front_d.append(d_val if ok else np.nan)
        front_ok.append(ok)
        if ok:
            delta_warm = res.x.copy()

    dep_ok = depletions[front_ok]
    d_ok = np.array(front_d)[front_ok]
    p_ok = chi2.sf(d_ok**2, df=n_vars) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(dep_ok * 100, d_ok, 'o-', color='steelblue', lw=2, ms=5)
    ax1.axvline(depletion * 100, color='crimson', ls='--', lw=1.5,
                label=f'ECB target {depletion*100:.0f} bps')
    ax1.axhline(d_opt, color='darkorange', ls=':', lw=1.5,
                label=f'D_M* = {d_opt:.2f}')
    ax1.set_xlabel('CET1 Depletion ε (bps)', fontsize=11)
    ax1.set_ylabel('Mahalanobis Distance D_M', fontsize=11)
    ax1.set_title('Plausibility of Worst-Case Scenario\nvs. CET1 Severity', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:.0f}'))

    ax2.plot(dep_ok * 100, p_ok, 's-', color='seagreen', lw=2, ms=5)
    ax2.axvline(depletion * 100, color='crimson', ls='--', lw=1.5,
                label=f'ECB target {depletion*100:.0f} bps')
    ax2.set_xlabel('CET1 Depletion ε (bps)', fontsize=11)
    ax2.set_ylabel('Scenario tail probability  [χ² tail, %]', fontsize=11)
    ax2.set_title('Scenario Probability\nvs. CET1 Severity', fontweight='bold')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*100:.0f}'))

    fig.suptitle('Pareto Frontier: CET1 Depletion vs. Scenario Plausibility',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig('cet1_opt_frontier.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Chart saved → cet1_opt_frontier.png')


def plot_hurlin_analogs(
    all_vars,
    var_labels,
    shock_sigma,          # (delta_opt - delta_baseline) / stds — shock from today
    stds,
    delta_opt,
    delta_baseline,       # x_last - mu_hist
    n_vars,
    logit_pd0,
    b_total,
    sqrt_rho,
    inv_q,
    sqrt_1mrho,
    ead,
    lgd,
    calculate_capital_requirement,
    rho,
    maturity,
    rwa_other,
    cet1_0,
    loss_base,
    sigma_inv,
    mahal_sq,
    cet1_ratio,
    r_omega,
):
    # Pick the two dimensions with largest shock from today
    top2  = np.argsort(np.abs(shock_sigma))[-2:][::-1]
    idx_a = int(top2[0])
    idx_b = int(top2[1])
    lbl_a = var_labels[all_vars[idx_a]]
    lbl_b = var_labels[all_vars[idx_b]]
    print(f'2-D slice:  x = {lbl_a}  |shock/σ| = {abs(shock_sigma[idx_a]):.2f}')
    print(f'            y = {lbl_b}  |shock/σ| = {abs(shock_sigma[idx_b]):.2f}')

    # Grid centred on today's conditions (delta_baseline); ±4.5σ around it
    n_g   = 140
    ga    = np.linspace(delta_baseline[idx_a] - 4.5 * stds[idx_a],
                        delta_baseline[idx_a] + 4.5 * stds[idx_a], n_g)
    gb    = np.linspace(delta_baseline[idx_b] - 4.5 * stds[idx_b],
                        delta_baseline[idx_b] + 4.5 * stds[idx_b], n_g)
    ga_grid, gb_grid = np.meshgrid(ga, gb)

    npts    = n_g * n_g
    d_shock = np.broadcast_to(delta_opt, (npts, n_vars)).copy()
    d_shock[:, idx_a] = ga_grid.ravel()
    d_shock[:, idx_b] = gb_grid.ravel()

    # PD adjustment relative to today (subtract delta_baseline)
    adj_f = np.clip(logit_pd0[None, :] + (d_shock - delta_baseline) @ b_total.T, -50, 50)
    pd_f  = np.clip(1 / (1 + np.exp(-adj_f)), 1e-9, 1 - 1e-9)
    cp_f  = norm.cdf((norm.ppf(pd_f) + sqrt_rho[None, :] * inv_q) / sqrt_1mrho[None, :])
    l_2d  = (ead[None, :] * lgd * cp_f).sum(1).reshape(n_g, n_g)
    k_f   = calculate_capital_requirement(pd_f, lgd, rho[None, :], maturity=maturity)
    rwa_2d = (ead[None, :] * k_f * 12.5).sum(1).reshape(n_g, n_g) + rwa_other
    r_2d   = (cet1_0 - (l_2d - loss_base)) / rwa_2d

    # Mahalanobis D² centred on today (displaced)
    diff_base = d_shock - delta_baseline
    d2_2d = ((diff_base @ sigma_inv) * diff_base).sum(1).reshape(n_g, n_g)

    # Design point position and plausibility level
    shock_opt = delta_opt - delta_baseline          # shock vector from today
    ab_star   = shock_opt[[idx_a, idx_b]]           # 2-D projection
    d2_star   = mahal_sq(delta_opt)
    dm_star   = np.sqrt(d2_star)
    r_star    = cet1_ratio(delta_opt)
    print(f'\n  Projection of optimal shock: {lbl_a} = {ab_star[0]/stds[idx_a]:+.2f}σ, '
          f'{lbl_b} = {ab_star[1]/stds[idx_b]:+.2f}σ')
    print(f'  D_M = {dm_star:.3f},  CET1 at full Δ* = {r_star:.4%}')

    # Local ball centred on design point
    dfl2           = np.zeros((n_g * n_g, n_vars))
    dfl2[:, idx_a] = ga_grid.ravel() - delta_opt[idx_a]
    dfl2[:, idx_b] = gb_grid.ravel() - delta_opt[idx_b]
    d2_dpt = ((dfl2 @ sigma_inv) * dfl2).sum(1).reshape(n_g, n_g)

    rho_ball = d2_star * 0.20
    epsilon  = d2_star * 1.50
    near_opt = (r_2d <= r_omega) & (d2_2d <= d2_star + epsilon)

    # Axes in σ-units relative to TODAY (origin = current conditions)
    gx  = (ga_grid - delta_baseline[idx_a]) / stds[idx_a]
    gy  = (gb_grid - delta_baseline[idx_b]) / stds[idx_b]
    axs = ab_star[0] / stds[idx_a]   # design point on σ-axis
    ays = ab_star[1] / stds[idx_b]

    inner_levs_raw = np.array([0.20, 0.45, 0.70, 0.90]) * d2_star
    inner_levs = inner_levs_raw[inner_levs_raw > 0]

    fig1, ax1 = plt.subplots(figsize=(7.5, 6.5))
    ax1.contourf(gx, gy, r_2d, levels=[r_2d.min(), r_omega], colors=['#fdb8b8'], alpha=0.55)
    ax1.contour(gx, gy, r_2d, levels=[r_omega], colors=['crimson'], linewidths=2.5)
    if len(inner_levs) > 0:
        ax1.contour(gx, gy, d2_2d, levels=inner_levs,
                    colors='steelblue', linewidths=0.9, linestyles='--', alpha=0.55)
    ax1.contour(gx, gy, d2_2d, levels=[d2_star], colors='steelblue', linewidths=2.2, linestyles='--')
    ax1.contour(gx, gy, d2_dpt, levels=[rho_ball], colors='darkorange', linewidths=1.8, linestyles=':')
    ax1.plot(axs, ays, 'k*', ms=15, zorder=6)
    ax1.plot(0, 0, 'ko', ms=7, zorder=5)      # origin = today
    ax1.axhline(0, color='k', lw=0.5, ls=':')
    ax1.axvline(0, color='k', lw=0.5, ls=':')
    ax1.legend(handles=[
        Patch(fc='#fdb8b8', ec='crimson', alpha=0.75,
              label='Breach region  $\\mathcal{S}_{\\mathrm{red}}$'),
        Line2D([0], [0], color='crimson', lw=2.5,
               label=f'CET1 frontier  R = {r_omega*100:.2f}%'),
        Line2D([0], [0], color='steelblue', lw=1.2, ls='--', alpha=0.7,
               label='Plausibility iso-curves  $d^2_\\Sigma$  (from today)'),
        Line2D([0], [0], color='steelblue', lw=2.2, ls='--',
               label=f'Tangent curve  $d^2 = {d2_star:.1f}$'),
        Line2D([0], [0], color='darkorange', lw=1.8, ls=':',
               label=f'Local ball  $B_\\rho$  ($\\rho^2 = {rho_ball:.1f}$)'),
        Line2D([0], [0], marker='*', color='k', ms=11, ls='',
               label=f'Design point  $s^\\omega$  ($D_M = {dm_star:.2f}$)'),
        Line2D([0], [0], marker='o', color='k', ms=7, ls='',
               label='Today\'s conditions (origin)'),
    ], fontsize=8.5, loc='best')
    ax1.set_xlabel(f'shock {lbl_a}  (σ from today)', fontsize=11)
    ax1.set_ylabel(f'shock {lbl_b}  (σ from today)', fontsize=11)
    ax1.set_title(
        'CET1 frontier in the scenario plane\n'
        f'(analog of Hurlin et al. Fig. 1  —  threshold = {r_omega:.2%})',
        fontsize=11, fontweight='bold')
    ax1.set_xlim(gx.min(), gx.max())
    ax1.set_ylim(gy.min(), gy.max())
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    plt.savefig('cet1_hurlin_fig1_analog.png', dpi=150, bbox_inches='tight')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7.5, 6.5))
    ax2.contourf(gx, gy, r_2d, levels=[r_2d.min(), r_omega], colors=['#fdb8b8'], alpha=0.40)
    ax2.contourf(gx, gy, near_opt.astype(float), levels=[0.5, 1.5],
                 colors=['#c0392b'], alpha=0.35, hatches=['////'])
    ax2.contour(gx, gy, r_2d, levels=[r_omega], colors=['crimson'], linewidths=2.5)
    if len(inner_levs) > 0:
        ax2.contour(gx, gy, d2_2d, levels=inner_levs,
                    colors='steelblue', linewidths=0.9, linestyles='--', alpha=0.55)
    ax2.contour(gx, gy, d2_2d, levels=[d2_star], colors='steelblue', linewidths=2.2, linestyles='--')
    ax2.contour(gx, gy, d2_2d, levels=[d2_star + epsilon], colors='steelblue', linewidths=2.0, linestyles='-.')
    ax2.plot(axs, ays, 'k*', ms=15, zorder=6)
    ax2.plot(0, 0, 'ko', ms=7, zorder=5)
    ax2.axhline(0, color='k', lw=0.5, ls=':')
    ax2.axvline(0, color='k', lw=0.5, ls=':')
    ax2.legend(handles=[
        Patch(fc='#fdb8b8', ec='crimson', alpha=0.60,
              label='Breach region  $\\mathcal{S}_{\\mathrm{red}}$'),
        Patch(fc='#c0392b', ec='#c0392b', alpha=0.45, hatch='////',
              label=f'Near-optimal set  $\\mathcal{{N}}_\\varepsilon$  ($\\varepsilon = {epsilon:.1f}$)'),
        Line2D([0], [0], color='crimson', lw=2.5,
               label=f'CET1 frontier  R = {r_omega*100:.2f}%'),
        Line2D([0], [0], color='steelblue', lw=1.2, ls='--', alpha=0.7,
               label='Plausibility iso-curves  $d^2_\\Sigma$  (from today)'),
        Line2D([0], [0], color='steelblue', lw=2.2, ls='--',
               label=f'Tangent curve  $d^2 = {d2_star:.1f}$'),
        Line2D([0], [0], color='steelblue', lw=2.0, ls='-.',
               label=f'Outer contour  $d^2 = {d2_star+epsilon:.1f}$'),
        Line2D([0], [0], marker='*', color='k', ms=11, ls='',
               label=f'Design point  $s^\\omega$  ($D_M = {dm_star:.2f}$)'),
    ], fontsize=8.5, loc='best')
    ax2.set_xlabel(f'shock {lbl_a}  (σ from today)', fontsize=11)
    ax2.set_ylabel(f'shock {lbl_b}  (σ from today)', fontsize=11)
    ax2.set_title(
        'Near-optimal scenario set  $\\mathcal{N}_\\varepsilon$\n'
        f'(analog of Hurlin et al. Fig. 2  —  $\\varepsilon = {epsilon:.1f}$)',
        fontsize=11, fontweight='bold')
    ax2.set_xlim(gx.min(), gx.max())
    ax2.set_ylim(gy.min(), gy.max())
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    plt.savefig('cet1_hurlin_fig2_analog.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('Figures saved  →  cet1_hurlin_fig1_analog.png,  cet1_hurlin_fig2_analog.png')


def plot_all_factor_pairings(
    all_vars,
    all_base_vars,
    var_labels,
    stds,
    delta_opt,
    delta_baseline,       # x_last - mu_hist
    n_vars,
    logit_pd0,
    b_total,
    sqrt_rho,
    inv_q,
    sqrt_1mrho,
    ead,
    lgd,
    calculate_capital_requirement,
    rho,
    maturity,
    rwa_other,
    cet1_0,
    loss_base,
    sigma_inv,
    mahal_sq,
    r_omega,
):
    n_g_all = 100
    if len(all_base_vars) < 2:
        print('Skipping all-factor-pairings plot: need at least two base factors.')
        return
    base_indices = [all_vars.index(v) for v in all_base_vars]
    all_pairs = list(combinations(base_indices, 2))
    n_pairs = len(all_pairs)
    n_cols = min(5, max(1, n_pairs))
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.2 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    d2_global = mahal_sq(delta_opt)
    dm_global = np.sqrt(d2_global)
    shock_opt = delta_opt - delta_baseline

    for plot_idx, (ia, ib) in enumerate(all_pairs):
        ax    = axes_flat[plot_idx]
        lbl_a = var_labels[all_vars[ia]]
        lbl_b = var_labels[all_vars[ib]]

        # Grid centred on today's conditions for each pair dimension
        ga = np.linspace(delta_baseline[ia] - 4.5 * stds[ia],
                         delta_baseline[ia] + 4.5 * stds[ia], n_g_all)
        gb = np.linspace(delta_baseline[ib] - 4.5 * stds[ib],
                         delta_baseline[ib] + 4.5 * stds[ib], n_g_all)
        ga_grid, gb_grid = np.meshgrid(ga, gb)

        npts    = n_g_all * n_g_all
        d_shock = np.broadcast_to(delta_opt, (npts, n_vars)).copy()
        d_shock[:, ia] = ga_grid.ravel()
        d_shock[:, ib] = gb_grid.ravel()

        adj_f  = np.clip(logit_pd0[None, :] + (d_shock - delta_baseline) @ b_total.T, -50, 50)
        pd_f   = np.clip(1 / (1 + np.exp(-adj_f)), 1e-9, 1 - 1e-9)
        cp_f   = norm.cdf((norm.ppf(pd_f) + sqrt_rho[None, :] * inv_q) / sqrt_1mrho[None, :])
        l_2d   = (ead[None, :] * lgd * cp_f).sum(1).reshape(n_g_all, n_g_all)
        k_f    = calculate_capital_requirement(pd_f, lgd, rho[None, :], maturity=maturity)
        rwa_2d = (ead[None, :] * k_f * 12.5).sum(1).reshape(n_g_all, n_g_all) + rwa_other
        r_2d   = (cet1_0 - (l_2d - loss_base)) / rwa_2d

        diff_base = d_shock - delta_baseline
        d2_2d = ((diff_base @ sigma_inv) * diff_base).sum(1).reshape(n_g_all, n_g_all)

        # σ-axes relative to today (origin = current conditions)
        gx = (ga_grid - delta_baseline[ia]) / stds[ia]
        gy = (gb_grid - delta_baseline[ib]) / stds[ib]

        if np.any(r_2d <= r_omega):
            ax.contourf(gx, gy, r_2d, levels=[r_2d.min(), r_omega], colors=['#fdb8b8'], alpha=0.55)
            ax.contour(gx, gy, r_2d, levels=[r_omega], colors=['crimson'], linewidths=1.5)

        inner_levs = np.array([0.20, 0.45, 0.70, 0.90]) * d2_global
        inner_levs = inner_levs[inner_levs > 0]
        if len(inner_levs) > 0:
            ax.contour(gx, gy, d2_2d, levels=inner_levs,
                       colors='steelblue', linewidths=0.6, linestyles='--', alpha=0.45)
        ax.contour(gx, gy, d2_2d, levels=[d2_global], colors='steelblue', linewidths=1.6, linestyles='--')

        # Design point = shock from today projected onto this pair
        axs = shock_opt[ia] / stds[ia]
        ays = shock_opt[ib] / stds[ib]
        ax.plot(axs, ays, 'k*', ms=9, zorder=6, label=f'$s^\\omega$ $D_M$={dm_global:.2f}')
        ax.legend(fontsize=5.5, loc='upper right', handlelength=0.8)

        ax.plot(0, 0, 'ko', ms=4, zorder=5)    # today = origin
        ax.axhline(0, color='k', lw=0.4, ls=':')
        ax.axvline(0, color='k', lw=0.4, ls=':')

        short_a = all_vars[ia].replace('_', ' ')
        short_b = all_vars[ib].replace('_', ' ')
        ax.set_xlabel(f'shock {short_a} (σ)', fontsize=6.5)
        ax.set_ylabel(f'shock {short_b} (σ)', fontsize=6.5)
        ax.set_title(f'{lbl_a}\nvs  {lbl_b}', fontsize=6.5, fontweight='bold')
        ax.tick_params(labelsize=5.5)
        ax.set_xlim(gx.min(), gx.max())
        ax.set_ylim(gy.min(), gy.max())
        ax.grid(alpha=0.2)
        print(f'  Pair {plot_idx+1:2d}/{n_pairs}  ({all_vars[ia]} vs {all_vars[ib]})  done')

    for ax in axes_flat[n_pairs:]:
        ax.set_visible(False)

    legend_elements = [
        Patch(fc='#fdb8b8', ec='crimson', alpha=0.75, label='Breach region'),
        Line2D([0], [0], color='crimson', lw=1.5, label=f'CET1 frontier  R = {r_omega*100:.2f}%'),
        Line2D([0], [0], color='steelblue', lw=0.9, ls='--', alpha=0.6,
               label='Plausibility iso-curves'),
        Line2D([0], [0], color='steelblue', lw=1.6, ls='--', label='Tangent contour'),
        Line2D([0], [0], marker='*', color='k', ms=8, ls='', label='18-D $s^\\omega$ (projected)'),
        Line2D([0], [0], marker='o', color='k', ms=5, ls='', label='Axes at 0, others at Δ*'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.suptitle(
        f'CET1 Frontier Geometry — All {n_pairs} Factor Pairings\n'
        f'(breach region, plausibility iso-curves, global $s^\\omega$ projected, off-axis at $\\Delta^*$  |  threshold = {r_omega*100:.2f}%)',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('cet1_frontier_all_pairs.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Chart saved → cet1_frontier_all_pairs.png')


def run_all_plots(model):
    plot_cet1_sensitivity(
        all_base_vars=model.ACTIVE_BASE_VARS,
        all_vars=model.ACTIVE_VARS,
        base_labels=model._BASE_LABELS,
        stds=model.stds,
        cet1_ratio=model.cet1_ratio,
        permanent_delta=model._permanent_delta,
        r_omega=model.R_OMEGA,
    )

    plot_optimal_results(
        all_vars=model.ACTIVE_VARS,
        var_labels=model.VAR_LABELS,
        shock_sigma=model.shock_sigma,
        df_valid=model.df_valid,
        cet1_ratio_0=model.CET1_RATIO_0,
        cet1_0=model.CET1_0,
        l_opt=model.L_opt,
        loss_base=model.loss_base,
        rwa_opt=model.RWA_opt,
        rwa_total_0=model.RWA_total_0,
        r_omega=model.R_OMEGA,
        r_opt=model.R_opt,
        d_opt=model.D_opt,
    )

    plot_optimal_shocks_grouped_lags(
        all_base_vars=model.ACTIVE_BASE_VARS,
        all_vars=model.ACTIVE_VARS,
        base_labels=model._BASE_LABELS,
        shock_sigma=model.shock_sigma,
        d_opt=model.D_opt,
        r_opt=model.R_opt,
    )

    plot_pareto_frontier(
        cet1_ratio_0=model.CET1_RATIO_0,
        depletion=model.DEPLETION,
        d_opt=model.D_opt,
        mahal_sq=model.mahal_sq,
        mahal_sq_grad=model.mahal_sq_grad,
        cet1_ratio=model.cet1_ratio,
        n_vars=model.N_VARS,
    )

    plot_hurlin_analogs(
        all_vars=model.ACTIVE_VARS,
        var_labels=model.VAR_LABELS,
        shock_sigma=model.shock_sigma,
        stds=model.stds,
        delta_opt=model.delta_opt,
        delta_baseline=model.delta_baseline,
        n_vars=model.N_VARS,
        logit_pd0=model.logit_pd0,
        b_total=model.B_total,
        sqrt_rho=model.sqrt_rho,
        inv_q=model.inv_q,
        sqrt_1mrho=model.sqrt_1mrho,
        ead=model.ead,
        lgd=model.LGD,
        calculate_capital_requirement=model.calculate_capital_requirement,
        rho=model.rho,
        maturity=model.MATURITY,
        rwa_other=model.RWA_other,
        cet1_0=model.CET1_0,
        loss_base=model.loss_base,
        sigma_inv=model.Sigma_inv,
        mahal_sq=model.mahal_sq,
        cet1_ratio=model.cet1_ratio,
        r_omega=model.R_OMEGA,
    )

    plot_all_factor_pairings(
        all_vars=model.ACTIVE_VARS,
        all_base_vars=model.ACTIVE_BASE_VARS,
        var_labels=model.VAR_LABELS,
        stds=model.stds,
        delta_opt=model.delta_opt,
        delta_baseline=model.delta_baseline,
        n_vars=model.N_VARS,
        logit_pd0=model.logit_pd0,
        b_total=model.B_total,
        sqrt_rho=model.sqrt_rho,
        inv_q=model.inv_q,
        sqrt_1mrho=model.sqrt_1mrho,
        ead=model.ead,
        lgd=model.LGD,
        calculate_capital_requirement=model.calculate_capital_requirement,
        rho=model.rho,
        maturity=model.MATURITY,
        rwa_other=model.RWA_other,
        cet1_0=model.CET1_0,
        loss_base=model.loss_base,
        sigma_inv=model.Sigma_inv,
        mahal_sq=model.mahal_sq,
        r_omega=model.R_OMEGA,
    )


if __name__ == '__main__':
    import cet1_macro_optimization as model

    run_all_plots(model)
