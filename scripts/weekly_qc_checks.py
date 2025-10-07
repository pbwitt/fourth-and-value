#!/usr/bin/env python3
"""
Weekly QC Checks for Fourth & Value
Run this after each weekly refresh to ensure data quality and consistency.

Usage:
    python scripts/weekly_qc_checks.py --season 2025 --week 6 \\
        --props data/props/latest_all_props.csv \\
        --params data/props/params_week6.csv \\
        --edges data/props/props_with_model_week6.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import sys

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.END}\n")

def print_pass(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warn(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_fail(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def check_historical_data(season: int, week: int) -> dict:
    """Check that all historical weeks are available."""
    print_header("1. HISTORICAL DATA AVAILABILITY")

    results = {"pass": True, "warnings": [], "errors": []}

    # Check weekly player stats
    parquet_path = Path(f"data/weekly_player_stats_{season}.parquet")
    if not parquet_path.exists():
        print_fail(f"Missing weekly stats: {parquet_path}")
        results["pass"] = False
        results["errors"].append(f"Missing {parquet_path}")
        return results

    df = pd.read_parquet(parquet_path)
    df.columns = [c.lower() for c in df.columns]

    available_weeks = sorted(df['week'].unique())
    expected_weeks = list(range(1, week))

    print(f"Season {season} weekly data:")
    print(f"  Expected weeks: {expected_weeks}")
    print(f"  Available weeks: {available_weeks}")

    if set(expected_weeks).issubset(set(available_weeks)):
        print_pass(f"All weeks 1-{week-1} available")
    else:
        missing = set(expected_weeks) - set(available_weeks)
        print_fail(f"Missing weeks: {missing}")
        results["pass"] = False
        results["errors"].append(f"Missing weeks: {missing}")

    # Check data freshness
    max_week = df['week'].max()
    if max_week < week - 1:
        print_warn(f"Data only through week {max_week}, expected week {week-1}")
        results["warnings"].append(f"Data may be stale (max week: {max_week})")
    else:
        print_pass(f"Data is current (max week: {max_week})")

    return results

def check_props_markets(props_path: str, season: int, week: int) -> dict:
    """Check props data quality and market coverage."""
    print_header("2. PROPS DATA QUALITY")

    results = {"pass": True, "warnings": [], "errors": []}

    if not Path(props_path).exists():
        print_fail(f"Props file not found: {props_path}")
        results["pass"] = False
        results["errors"].append(f"Missing props file")
        return results

    props = pd.read_csv(props_path)

    # Basic counts
    print(f"Total props: {len(props):,}")
    print(f"Unique players: {props['player'].nunique()}")
    print(f"Unique books: {props['bookmaker_title'].nunique()}")

    # Check required columns
    required_cols = ['player', 'market', 'bookmaker_title', 'price', 'point', 'name']
    missing_cols = [c for c in required_cols if c not in props.columns]
    if missing_cols:
        print_fail(f"Missing columns: {missing_cols}")
        results["pass"] = False
        results["errors"].append(f"Missing columns: {missing_cols}")
    else:
        print_pass("All required columns present")

    # Check for NaN critical fields
    critical_fields = ['player', 'market', 'price']
    for field in critical_fields:
        if field in props.columns:
            nan_count = props[field].isna().sum()
            if nan_count > 0:
                print_warn(f"{field} has {nan_count} NaN values ({nan_count/len(props)*100:.1f}%)")
                results["warnings"].append(f"{field} has NaNs")

    # Market distribution
    print("\nMarket distribution:")
    market_counts = props.groupby('market').size().sort_values(ascending=False)
    for market, count in market_counts.head(15).items():
        print(f"  {market:30s}: {count:4d} props")

    # Expected core markets
    from common_markets import MODELED_MARKETS
    if 'market_std' in props.columns:
        modeled = props[props['market_std'].isin(MODELED_MARKETS)]
        coverage = len(modeled) / len(props) * 100
        print(f"\nModeled market coverage: {coverage:.1f}% ({len(modeled)}/{len(props)})")
        if coverage < 50:
            print_warn(f"Low modeled coverage: {coverage:.1f}%")
            results["warnings"].append(f"Low modeled coverage: {coverage:.1f}%")
        else:
            print_pass(f"Modeled coverage: {coverage:.1f}%")

    return results

def check_params_quality(params_path: str, props_path: str, season: int, week: int) -> dict:
    """Check params data quality and join completeness."""
    print_header("3. PARAMS QUALITY & JOIN COVERAGE")

    results = {"pass": True, "warnings": [], "errors": []}

    if not Path(params_path).exists():
        print_fail(f"Params file not found: {params_path}")
        results["pass"] = False
        results["errors"].append("Missing params file")
        return results

    params = pd.read_csv(params_path)
    props = pd.read_csv(props_path)

    # Basic params stats
    print(f"Total param rows: {len(params):,}")
    print(f"Unique players: {params['player'].nunique()}")
    print(f"Unique markets: {params['market_std'].nunique()}")

    # Check required columns
    required = ['player', 'market_std', 'mu', 'sigma', 'dist']
    missing = [c for c in required if c not in params.columns]
    if missing:
        print_fail(f"Missing columns: {missing}")
        results["pass"] = False
        results["errors"].append(f"Missing columns: {missing}")
    else:
        print_pass("All required columns present")

    # Check for NaN in critical fields
    if 'mu' in params.columns:
        normal_params = params[params['dist'] == 'normal']
        nan_mu = normal_params['mu'].isna().sum()
        if nan_mu > 0:
            print_fail(f"Normal params have {nan_mu} NaN mu values")
            results["pass"] = False
            results["errors"].append(f"{nan_mu} NaN mu values")
        else:
            print_pass("No NaN mu values in normal params")

    # Check join coverage
    if 'name_std' in props.columns and 'name_std' in params.columns:
        props_keys = props[['name_std', 'market_std']].drop_duplicates()
        params_keys = params[['name_std', 'market_std']].drop_duplicates()

        merged = props_keys.merge(
            params_keys,
            on=['name_std', 'market_std'],
            how='left',
            indicator=True
        )

        matched = (merged['_merge'] == 'both').sum()
        total = len(merged)
        coverage = matched / total * 100

        print(f"\nJoin coverage (props → params):")
        print(f"  Matched: {matched}/{total} ({coverage:.1f}%)")

        if coverage < 80:
            print_warn(f"Low join coverage: {coverage:.1f}%")
            results["warnings"].append(f"Low join coverage: {coverage:.1f}%")

            # Show unmatched examples
            unmatched = merged[merged['_merge'] == 'left_only']
            if len(unmatched) > 0:
                print("\nTop unmatched (player, market) pairs:")
                for idx, row in unmatched.head(10).iterrows():
                    print(f"  {row['name_std']:30s} {row['market_std']}")
        else:
            print_pass(f"Join coverage: {coverage:.1f}%")

    # Check defensive adjustments were applied
    if 'implied_ypc' in params.columns or 'implied_cr' in params.columns:
        print_pass("Family-based diagnostics columns present")
    else:
        print_warn("Missing family diagnostic columns")
        results["warnings"].append("Missing family diagnostics")

    return results

def check_edge_calculations(edges_path: str, season: int, week: int) -> dict:
    """Check edge calculations and distributions."""
    print_header("4. EDGE CALCULATIONS & DISTRIBUTIONS")

    results = {"pass": True, "warnings": [], "errors": []}

    if not Path(edges_path).exists():
        print_fail(f"Edges file not found: {edges_path}")
        results["pass"] = False
        results["errors"].append("Missing edges file")
        return results

    edges = pd.read_csv(edges_path)
    modeled = edges[edges['model_prob'].notna()].copy()

    print(f"Total rows: {len(edges):,}")
    print(f"Modeled rows: {len(modeled):,} ({len(modeled)/len(edges)*100:.1f}%)")

    # Check for invalid probabilities
    invalid_prob = (
        (modeled['model_prob'] < 0) |
        (modeled['model_prob'] > 1)
    ).sum()

    if invalid_prob > 0:
        print_fail(f"{invalid_prob} rows with invalid model_prob (not in [0,1])")
        results["pass"] = False
        results["errors"].append(f"{invalid_prob} invalid probabilities")
    else:
        print_pass("All model probabilities in valid range [0, 1]")

    # Edge distribution
    if 'edge_bps' in modeled.columns:
        edges_only = modeled['edge_bps'].dropna()

        print(f"\nEdge distribution:")
        print(f"  Mean: {edges_only.mean():+.0f} bps")
        print(f"  Median: {edges_only.median():+.0f} bps")
        print(f"  Std dev: {edges_only.std():.0f} bps")
        print(f"  Min: {edges_only.min():+.0f} bps")
        print(f"  Max: {edges_only.max():+.0f} bps")

        # Edge buckets
        print("\nEdge buckets:")
        buckets = [
            ('>+5000', (edges_only > 5000).sum()),
            ('+3000 to +5000', ((edges_only > 3000) & (edges_only <= 5000)).sum()),
            ('+1000 to +3000', ((edges_only > 1000) & (edges_only <= 3000)).sum()),
            ('+500 to +1000', ((edges_only > 500) & (edges_only <= 1000)).sum()),
            ('-500 to +500', ((edges_only >= -500) & (edges_only <= 500)).sum()),
            ('-1000 to -500', ((edges_only >= -1000) & (edges_only < -500)).sum()),
            ('<-3000', (edges_only < -3000).sum()),
        ]

        for label, count in buckets:
            pct = count / len(edges_only) * 100
            print(f"  {label:20s}: {count:4d} ({pct:5.1f}%)")

        # Warn if too many extreme edges
        extreme_positive = (edges_only > 5000).sum()
        if extreme_positive > 10:
            print_warn(f"{extreme_positive} props with edge > +5000 bps (may indicate model issues)")
            results["warnings"].append(f"{extreme_positive} extreme positive edges")

    # Check consensus coverage
    if 'consensus_line' in modeled.columns:
        has_consensus = modeled['consensus_line'].notna().sum()
        consensus_pct = has_consensus / len(modeled) * 100
        print(f"\nConsensus coverage: {consensus_pct:.1f}% ({has_consensus}/{len(modeled)})")

        if consensus_pct < 50:
            print_warn(f"Low consensus coverage: {consensus_pct:.1f}%")
            results["warnings"].append(f"Low consensus coverage")
        else:
            print_pass(f"Consensus coverage: {consensus_pct:.1f}%")

    return results

def check_model_calibration(edges_path: str) -> dict:
    """Check model confidence vs market disagreement."""
    print_header("5. MODEL CALIBRATION CHECKS")

    results = {"pass": True, "warnings": [], "errors": []}

    edges = pd.read_csv(edges_path)
    modeled = edges[edges['model_prob'].notna()].copy()

    # High confidence disagreements
    extreme_disagree = modeled[
        ((modeled['model_prob'] > 0.99) & (modeled['mkt_prob'] < 0.60)) |
        ((modeled['model_prob'] < 0.01) & (modeled['mkt_prob'] > 0.40))
    ]

    print(f"High confidence disagreements:")
    print(f"  Props where model is 99%+ confident but market disagrees: {len(extreme_disagree)}")

    if len(extreme_disagree) > 50:
        print_warn(f"High number of extreme disagreements: {len(extreme_disagree)}")
        results["warnings"].append(f"{len(extreme_disagree)} extreme disagreements")

        print("\nTop 10 extreme disagreements:")
        top_disagree = extreme_disagree.nlargest(10, 'edge_bps')[
            ['player', 'market_std', 'name', 'point', 'model_prob', 'mkt_prob', 'edge_bps']
        ]
        print(top_disagree.to_string(index=False))
    elif len(extreme_disagree) > 0:
        print_pass(f"Moderate number of disagreements: {len(extreme_disagree)}")
    else:
        print_pass("No extreme disagreements")

    # Check Over/Under symmetry
    paired = modeled[modeled['name'].isin(['Over', 'Under'])].copy()
    if len(paired) > 0:
        paired['key'] = (paired['player'] + '_' +
                        paired['market_std'] + '_' +
                        paired['point'].astype(str) + '_' +
                        paired['bookmaker'])

        asymmetries = []
        for key in paired['key'].unique():
            subset = paired[paired['key'] == key]
            if len(subset) == 2:
                over_prob = subset[subset['name'] == 'Over']['model_prob'].values
                under_prob = subset[subset['name'] == 'Under']['model_prob'].values
                if len(over_prob) > 0 and len(under_prob) > 0:
                    total = over_prob[0] + under_prob[0]
                    if abs(total - 1.0) > 0.01:
                        asymmetries.append((key.split('_')[0], total))

        if len(asymmetries) > 0:
            print_fail(f"{len(asymmetries)} Over/Under pairs don't sum to 1.0")
            results["pass"] = False
            results["errors"].append(f"{len(asymmetries)} asymmetric pairs")
            for player, total in asymmetries[:5]:
                print(f"  {player}: sum = {total:.4f}")
        else:
            print_pass("All Over/Under pairs sum to 1.0")

    return results

def check_methodology_consistency(params_path: str, season: int, week: int) -> dict:
    """Verify methodology is applied consistently."""
    print_header("6. METHODOLOGY CONSISTENCY")

    results = {"pass": True, "warnings": [], "errors": []}

    params = pd.read_csv(params_path)

    # Check L4 data usage (verify season/week consistency)
    if 'season' in params.columns and 'week' in params.columns:
        param_season = params['season'].mode()[0] if len(params) > 0 else None
        param_week = params['week'].mode()[0] if len(params) > 0 else None

        if param_season == season and param_week == week:
            print_pass(f"Params tagged with correct season={season}, week={week}")
        else:
            print_fail(f"Params have season={param_season}, week={param_week} (expected {season}, {week})")
            results["pass"] = False
            results["errors"].append(f"Season/week mismatch")

    # Check family coherence diagnostics
    family_checks = []

    # Rush family: rush_yds ≈ rush_attempts × YPC
    rush = params[params['market_std'].isin(['rush_yds', 'rush_attempts'])].copy()
    if len(rush) > 0 and 'implied_ypc' in rush.columns:
        rush_pivot = rush.pivot_table(
            index='player',
            columns='market_std',
            values='mu',
            aggfunc='first'
        )
        if 'rush_yds' in rush_pivot.columns and 'rush_attempts' in rush_pivot.columns:
            rush_pivot['implied_ypc_calc'] = rush_pivot['rush_yds'] / rush_pivot['rush_attempts']

            # Get actual implied_ypc from params
            ypc_values = rush[rush['market_std'] == 'rush_yds']['implied_ypc'].dropna()

            if len(ypc_values) > 0:
                mean_ypc = ypc_values.mean()
                if 2.5 <= mean_ypc <= 6.5:
                    print_pass(f"Rush YPC in realistic range: {mean_ypc:.2f}")
                    family_checks.append(True)
                else:
                    print_warn(f"Rush YPC outside bounds [2.5, 6.5]: {mean_ypc:.2f}")
                    results["warnings"].append(f"Rush YPC out of bounds: {mean_ypc:.2f}")
                    family_checks.append(False)

    # Receive family: catch rate bounds
    receive = params[params['market_std'] == 'receptions'].copy()
    if len(receive) > 0 and 'implied_cr' in receive.columns:
        cr_values = receive['implied_cr'].dropna()
        if len(cr_values) > 0:
            min_cr = cr_values.min()
            max_cr = cr_values.max()
            mean_cr = cr_values.mean()

            print(f"Catch rate range: {min_cr:.3f} to {max_cr:.3f} (mean: {mean_cr:.3f})")

            if min_cr < 0.5 or max_cr > 0.95:
                print_warn(f"Catch rate outside bounds [0.5, 0.95]")
                results["warnings"].append("Catch rate out of bounds")
                family_checks.append(False)
            else:
                print_pass("Catch rates within bounds [0.5, 0.95]")
                family_checks.append(True)

    if len(family_checks) > 0 and all(family_checks):
        print_pass("Family coherence checks passed")
    elif len(family_checks) > 0:
        print_warn("Some family coherence checks failed")

    # Check defensive adjustments were applied
    # We can infer this by checking if projections vary by opponent
    # For now, just check that we have the diagnostic columns
    if 'implied_ypc' in params.columns:
        print_pass("Defensive adjustment diagnostics present")
    else:
        print_warn("Missing defensive adjustment indicators")
        results["warnings"].append("Missing defensive diagnostics")

    return results

def main():
    parser = argparse.ArgumentParser(description="Weekly QC checks for Fourth & Value")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--props", type=str, required=True, help="Path to props CSV")
    parser.add_argument("--params", type=str, required=True, help="Path to params CSV")
    parser.add_argument("--edges", type=str, required=True, help="Path to edges CSV")

    args = parser.parse_args()

    print(f"\n{Colors.BOLD}Fourth & Value - Weekly QC Report{Colors.END}")
    print(f"{Colors.BOLD}Season {args.season}, Week {args.week}{Colors.END}")
    print(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    # Run all checks
    all_results.append(check_historical_data(args.season, args.week))
    all_results.append(check_props_markets(args.props, args.season, args.week))
    all_results.append(check_params_quality(args.params, args.props, args.season, args.week))
    all_results.append(check_edge_calculations(args.edges, args.season, args.week))
    all_results.append(check_model_calibration(args.edges))
    all_results.append(check_methodology_consistency(args.params, args.season, args.week))

    # Summary
    print_header("SUMMARY")

    total_errors = sum(len(r["errors"]) for r in all_results)
    total_warnings = sum(len(r["warnings"]) for r in all_results)
    all_pass = all(r["pass"] for r in all_results)

    if all_pass and total_warnings == 0:
        print_pass("ALL CHECKS PASSED ✓")
        sys.exit(0)
    elif all_pass:
        print_warn(f"PASSED WITH {total_warnings} WARNINGS ⚠")
        print("\nWarnings:")
        for i, r in enumerate(all_results, 1):
            for w in r["warnings"]:
                print(f"  {i}. {w}")
        sys.exit(0)
    else:
        print_fail(f"FAILED WITH {total_errors} ERRORS AND {total_warnings} WARNINGS ✗")
        print("\nErrors:")
        for i, r in enumerate(all_results, 1):
            for e in r["errors"]:
                print(f"  {i}. {e}")
        print("\nWarnings:")
        for i, r in enumerate(all_results, 1):
            for w in r["warnings"]:
                print(f"  {i}. {w}")
        sys.exit(1)

if __name__ == "__main__":
    main()
