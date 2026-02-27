"""CET1 ratio analysis utilities."""

from __future__ import annotations

from typing import Dict


def cet1_analysis(
    rwa_0: float,
    cet1_0: float,
    portfolio_loss: float,
    results_by_tenor: Dict[str, Dict[str, object]],
    tenor: str = '12_month',
    verbose: bool = True,
) -> Dict[str, object]:
    """Compute CET1 impact given portfolio loss and RWA results."""
    if tenor not in results_by_tenor:
        raise ValueError(f"Tenor '{tenor}' not found in results_by_tenor")

    total_rwa = results_by_tenor[tenor]['total_rwa']
    total_ead = results_by_tenor[tenor]['total_ead']

    new_cet1 = cet1_0 - portfolio_loss
    new_rwa = rwa_0 + total_rwa

    initial_cet1_ratio = (cet1_0 / rwa_0) * 100 if rwa_0 else None
    new_cet1_ratio = (new_cet1 / new_rwa) * 100 if new_rwa else None

    minimum_cet1_ratio = 4.5
    capital_conservation_buffer = 2.5
    total_minimum_ratio = minimum_cet1_ratio + capital_conservation_buffer

    minimum_cet1_required = new_rwa * (minimum_cet1_ratio / 100)
    total_cet1_required = new_rwa * (total_minimum_ratio / 100)

    surplus_deficit_minimum = new_cet1 - minimum_cet1_required
    surplus_deficit_buffers = new_cet1 - total_cet1_required

    if verbose:
        print("=" * 70)
        print("CET1 RATIO ANALYSIS")
        print("=" * 70)
        print(f"\nInput Parameters:")
        print(f"  Initial CET1 Capital (CET1_0): {cet1_0:,.0f} SEK")
        print(f"  Initial RWA (RWA_0): {rwa_0:,.0f} SEK")
        print(f"  Total Portfolio Loss (stressed): {portfolio_loss:,.2f} SEK")
        print(f"\nPortfolio Metrics ({tenor} tenor):")
        print(f"  Total EAD: {total_ead:,.0f} SEK")
        print(f"  Total RWA: {total_rwa:,.0f} SEK")

        print("\n" + "=" * 70)
        print("INITIAL STATE")
        print("=" * 70)
        print(f"  Initial CET1 Capital: {cet1_0:,.0f} SEK")
        print(f"  Initial RWA: {rwa_0:,.0f} SEK")
        print(f"  Initial CET1 Ratio: {initial_cet1_ratio:.2f}%" if initial_cet1_ratio is not None else "  Initial CET1 Ratio: N/A")

        print("\n" + "=" * 70)
        print("IMPACT OF PORTFOLIO")
        print("=" * 70)
        print(f"  Portfolio Loss (impact on CET1): -{portfolio_loss:,.2f} SEK")
        print(f"  Additional RWA from portfolio: +{total_rwa:,.0f} SEK")

        print("\n" + "=" * 70)
        print("NEW STATE (AFTER PORTFOLIO IMPACT)")
        print("=" * 70)
        print(f"  New CET1 Capital: {new_cet1:,.2f} SEK")
        print(f"  New RWA: {new_rwa:,.0f} SEK")
        if new_cet1_ratio is not None:
            print(f"  New CET1 Ratio: {new_cet1_ratio:.2f}%")
        else:
            print("  New CET1 Ratio: N/A (RWA is zero)")

        if new_cet1_ratio is not None:
            print("\n" + "=" * 70)
            print("REGULATORY CAPITAL REQUIREMENTS")
            print("=" * 70)
            print(f"  Minimum CET1 Ratio (Basel III): {minimum_cet1_ratio}%")
            print(f"  Minimum CET1 Capital Required: {minimum_cet1_required:,.0f} SEK")
            print(f"\n  CET1 Ratio with Buffers: {total_minimum_ratio}%")
            print(f"  CET1 Capital Required (with buffers): {total_cet1_required:,.0f} SEK")

            print("\n" + "=" * 70)
            print("CAPITAL ADEQUACY ASSESSMENT")
            print("=" * 70)
            print(f"  Surplus/(Deficit) vs. Minimum: {surplus_deficit_minimum:,.2f} SEK")
            print(f"  Surplus/(Deficit) vs. Buffers: {surplus_deficit_buffers:,.2f} SEK")

            if new_cet1_ratio >= total_minimum_ratio:
                print(f"\n  ✓ ADEQUATE: CET1 ratio ({new_cet1_ratio:.2f}%) meets requirements")
            elif new_cet1_ratio >= minimum_cet1_ratio:
                print(f"\n  ⚠ WARNING: CET1 ratio ({new_cet1_ratio:.2f}%) meets minimum but not buffers")
            else:
                print(f"\n  ✗ INADEQUATE: CET1 ratio ({new_cet1_ratio:.2f}%) below minimum requirement")

            if initial_cet1_ratio is not None:
                change_in_ratio = new_cet1_ratio - initial_cet1_ratio
                print(f"\n  Change in CET1 Ratio: {change_in_ratio:+.2f} percentage points")

    return {
        'initial_cet1': cet1_0,
        'initial_rwa': rwa_0,
        'portfolio_loss': portfolio_loss,
        'portfolio_rwa': total_rwa,
        'new_cet1': new_cet1,
        'new_rwa': new_rwa,
        'new_cet1_ratio': new_cet1_ratio,
        'minimum_cet1_required': minimum_cet1_required,
        'total_cet1_required': total_cet1_required,
        'surplus_deficit_minimum': surplus_deficit_minimum,
        'surplus_deficit_buffers': surplus_deficit_buffers,
    }
