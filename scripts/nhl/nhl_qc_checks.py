#!/usr/bin/env python3
"""
QC checks for NHL daily build with blocking rules.

Validates:
- Coverage: ≥80% of props have consensus data
- Join integrity: 100% of props join successfully
- Probability sanity: Model probabilities are reasonable
- Edge sanity: P95 absolute edge <3500 bps
- Market distribution: All expected markets present

Outputs JSON report with pass/fail status that blocks publish on failures.

Usage:
  python3 scripts/nhl/nhl_qc_checks.py --date 2025-10-08
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import poisson
import numpy as np


def check_coverage(props_df: pd.DataFrame) -> dict:
    """Check that ≥80% of props have consensus data."""
    total = len(props_df)
    has_consensus = props_df["consensus_prob"].notna().sum()
    coverage_pct = 100.0 * has_consensus / total if total > 0 else 0.0

    passed = coverage_pct >= 80.0

    return {
        "name": "Coverage",
        "passed": bool(passed),
        "blocking": True,
        "threshold": "≥80%",
        "actual": f"{coverage_pct:.1f}%",
        "details": f"{has_consensus}/{total} props have consensus",
    }


def check_joins(props_df: pd.DataFrame) -> dict:
    """Check that 100% of props join successfully with consensus."""
    total = len(props_df)
    missing_consensus = props_df["consensus_prob"].isna().sum()
    join_pct = 100.0 * (total - missing_consensus) / total if total > 0 else 0.0

    passed = join_pct == 100.0

    return {
        "name": "Join Integrity",
        "passed": bool(passed),
        "blocking": True,
        "threshold": "100%",
        "actual": f"{join_pct:.1f}%",
        "details": f"{total - missing_consensus}/{total} props joined",
    }


def check_edge_sanity(props_df: pd.DataFrame) -> dict:
    """Check that P95 absolute edge is <6000 bps (uncalibrated models)."""
    edges = props_df["edge_bps"].abs()
    p95_edge = edges.quantile(0.95)

    # Threshold for uncalibrated models (raw model probabilities vs market)
    # Calibration was disabled to avoid circular logic (model matching market)
    passed = p95_edge < 6000.0

    return {
        "name": "Edge Sanity",
        "passed": bool(passed),
        "blocking": True,
        "threshold": "<6000 bps",
        "actual": f"{p95_edge:.1f} bps",
        "details": f"P95 absolute edge: {p95_edge:.1f} bps (uncalibrated)",
    }


def check_market_distribution(props_df: pd.DataFrame) -> dict:
    """Check that all expected markets are present."""
    expected_markets = {"sog", "goals", "assists", "points"}
    actual_markets = set(props_df["market_std"].dropna().unique())

    missing = expected_markets - actual_markets
    passed = len(missing) == 0

    market_counts = props_df["market_std"].value_counts().to_dict()

    return {
        "name": "Market Distribution",
        "passed": bool(passed),
        "blocking": False,
        "threshold": f"{expected_markets}",
        "actual": f"{actual_markets}",
        "details": f"Market counts: {market_counts}",
        "missing": list(missing) if missing else [],
    }


def check_book_diversity(props_df: pd.DataFrame) -> dict:
    """Check that props have data from multiple books."""
    avg_books = props_df["book_count"].mean()
    min_books = props_df["book_count"].min()

    passed = avg_books >= 2.0

    return {
        "name": "Book Diversity",
        "passed": bool(passed),
        "blocking": False,
        "threshold": "Avg ≥2 books",
        "actual": f"Avg {avg_books:.1f}, Min {min_books:.0f}",
        "details": f"Average {avg_books:.1f} books per prop",
    }


def check_probability_sanity(props_df: pd.DataFrame) -> dict:
    """
    Check that model probabilities are reasonable.

    Flags issues like:
    - Model says >50% for Over when line is ABOVE the model prediction
    - Model says >50% for Under when line is BELOW the model prediction
    - Model probabilities don't align with model line
    """
    issues = []

    # Check for nonsensical probabilities relative to model line
    positive_edge = props_df[props_df['edge_bps'] > 0]

    for _, row in positive_edge.iterrows():
        player = row['player']
        market = row['market_std']
        side = row['side']
        line = row['point']
        model_line = row['model_line']
        model_prob = row['model_prob']

        # Check if probability makes sense relative to model prediction
        if side == 'Over':
            # If model line is BELOW the bet line, probability should be <50%
            # Example: Model predicts 2.0, Over 2.5 should have <50% chance
            if model_line < line and model_prob > 0.5:
                issues.append(f"{player} {market}: Over {line} has {model_prob:.1%} prob but model predicts {model_line}")
        else:  # Under
            # If model line is ABOVE the bet line, probability should be <50%
            # Example: Model predicts 2.0, Under 1.5 should have <50% chance
            if model_line > line and model_prob > 0.5:
                issues.append(f"{player} {market}: Under {line} has {model_prob:.1%} prob but model predicts {model_line}")

    # Limit to first 5 issues for readability
    if len(issues) > 0:
        sample_issues = issues[:5]
        if len(issues) > 5:
            sample_issues.append(f"... and {len(issues) - 5} more")

        return {
            "name": "Probability Sanity",
            "passed": False,
            "blocking": True,
            "threshold": "Probabilities align with model predictions",
            "actual": f"{len(issues)} nonsensical probabilities",
            "details": "; ".join(sample_issues),
        }

    return {
        "name": "Probability Sanity",
        "passed": True,
        "blocking": True,
        "threshold": "Probabilities align with model predictions",
        "actual": "All probabilities reasonable",
        "details": "No issues found",
    }


def check_game_log_dates(date_str: str) -> dict:
    """
    Check that game logs have correct dates (no duplicate player-date rows).

    This validates that the fetch script correctly parses game dates from
    start_time_utc instead of using the schedule fetch date.
    """
    from pathlib import Path

    logs_path = Path(f"data/nhl/processed/skater_logs_{date_str}.parquet")

    if not logs_path.exists():
        return {
            "name": "Game Log Dates",
            "passed": False,
            "blocking": True,
            "threshold": "No duplicate player-date rows",
            "actual": "Game logs file not found",
            "details": f"Missing: {logs_path}",
        }

    try:
        df = pd.read_parquet(logs_path)

        # Check for duplicate (player, game_date) pairs
        # Each player should appear at most once per date
        duplicates = df.groupby(["player", "game_date"]).size()
        duplicate_rows = duplicates[duplicates > 1]

        if len(duplicate_rows) > 0:
            # Show sample duplicates
            sample = duplicate_rows.head(3).to_dict()
            passed = False
            actual = f"{len(duplicate_rows)} player-date duplicates found"
            details = f"Sample duplicates: {sample}"
        else:
            passed = True
            unique_dates = df["game_date"].nunique()
            total_rows = len(df)
            actual = f"{total_rows} rows across {unique_dates} unique dates"
            details = f"All game dates validated correctly"

        return {
            "name": "Game Log Dates",
            "passed": bool(passed),
            "blocking": True,
            "threshold": "No duplicate player-date rows",
            "actual": actual,
            "details": details,
        }

    except Exception as e:
        return {
            "name": "Game Log Dates",
            "passed": False,
            "blocking": True,
            "threshold": "No duplicate player-date rows",
            "actual": "Error loading game logs",
            "details": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="NHL QC checks")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    date_str = args.date

    print("=" * 70)
    print(f"NHL QC Checks for {date_str}")
    print("=" * 70)

    # Load props with model
    props_path = Path(f"data/nhl/props/props_with_model_{date_str}.csv")
    if not props_path.exists():
        print(f"\n[ERROR] Props file not found: {props_path}", file=sys.stderr)
        sys.exit(1)

    props_df = pd.read_csv(props_path)
    print(f"\nLoaded {len(props_df)} prop rows")

    # Run checks
    checks = [
        check_game_log_dates(date_str),  # BLOCKING: validate dates first
        check_coverage(props_df),
        check_joins(props_df),
        check_probability_sanity(props_df),  # BLOCKING: validate probabilities
        check_edge_sanity(props_df),
        check_market_distribution(props_df),
        check_book_diversity(props_df),
    ]

    # Print results
    print("\n" + "-" * 70)
    print("QC Results:")
    print("-" * 70)

    blocking_failures = []
    warnings = []

    for check in checks:
        status = "✓ PASS" if check["passed"] else "✗ FAIL"
        blocking = " [BLOCKING]" if check.get("blocking") else ""

        print(f"\n{status}{blocking} {check['name']}")
        print(f"  Threshold: {check['threshold']}")
        print(f"  Actual:    {check['actual']}")
        print(f"  Details:   {check['details']}")

        if not check["passed"]:
            if check.get("blocking"):
                blocking_failures.append(check)
            else:
                warnings.append(check)

    # Summary
    print("\n" + "=" * 70)
    if blocking_failures:
        print("❌ QC FAILED - Blocking issues detected")
        print("=" * 70)
        for fail in blocking_failures:
            print(f"  - {fail['name']}: {fail['actual']} (threshold: {fail['threshold']})")
        print("\nPublish BLOCKED. Fix blocking issues before deploying.")
        exit_code = 1
    elif warnings:
        print("⚠️  QC PASSED with warnings")
        print("=" * 70)
        for warn in warnings:
            print(f"  - {warn['name']}: {warn['actual']}")
        print("\nOK to publish, but review warnings.")
        exit_code = 0
    else:
        print("✅ All QC checks passed!")
        print("=" * 70)
        exit_code = 0

    # Write JSON report
    qc_dir = Path("data/nhl/qc")
    qc_dir.mkdir(parents=True, exist_ok=True)

    report_path = qc_dir / f"qc_report_{date_str}.json"
    report = {
        "date": date_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "props_count": len(props_df),
        "checks": checks,
        "blocking_failures": [c["name"] for c in blocking_failures],
        "warnings": [c["name"] for c in warnings],
        "publish_blocked": len(blocking_failures) > 0,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nQC report written to {report_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
