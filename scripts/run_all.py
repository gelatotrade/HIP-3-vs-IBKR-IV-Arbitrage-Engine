#!/usr/bin/env python3
"""
Run All — HIP-3 vs IBKR Alpha Search Pipeline.

Executes:
  1. HIP-3 market data generation (synthetic, or live via API)
  2. Walk-forward backtests across all HIP-3 assets
  3. Greeks strategies (gamma scalp, vanna, charm, vomma, speed, color, zomma)
  4. IV arbitrage analysis
  5. Visualizations (3D surfaces, equity curves, heatmaps, dashboards)

Usage:  python3 scripts/run_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    print('='*80)
    print('  SEARCHING FOR ALPHA — HIP-3 vs IBKR Pipeline')
    print('  Hyperliquid HIP-3 Spot | IBKR Options | IV Arbitrage')
    print('='*80)

    # Step 1: Run backtests
    print('\n[1/2] Running HIP-3 backtests ...\n')
    from hip3_backtest import main as run_backtest
    results_df, curves, greeks_results = run_backtest()

    # Step 2: Generate visualizations
    print('\n[2/2] Generating visualizations ...\n')
    from generate_hip3_visualizations import generate_all_visualizations
    generate_all_visualizations(results_df, curves, greeks_results)

    print('\n' + '='*80)
    print('  PIPELINE COMPLETE')
    print('='*80)
    print(f'  Results:        results/hip3_backtest_results.csv')
    print(f'  Visualizations: docs/img/')
    print(f'  Assets tested:  {len(results_df)}')
    print(f'  Mean alpha:     {results_df["alpha"].mean()*100:+.1f}%')
    print(f'  Mean Sharpe:    {results_df["sharpe"].mean():.2f}')


if __name__ == '__main__':
    main()
