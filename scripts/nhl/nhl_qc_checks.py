#!/usr/bin/env python3
"""
QC checks for NHL daily build with blocking rules.

Validates:
- Coverage: ≥80% of props have consensus data
- Join integrity: 100% of props join successfully
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
    """Check that P95 absolute edge is <3500 bps."""
    edges = props_df["edge_bps"].abs()
    p95_edge = edges.quantile(0.95)

    passed = p95_edge < 3500.0

    return {
        "name": "Edge Sanity",
        "passed": bool(passed),
        "blocking": True,
        "threshold": "<3500 bps",
        "actual": f"{p95_edge:.1f} bps",
        "details": f"P95 absolute edge: {p95_edge:.1f} bps",
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
        check_coverage(props_df),
        check_joins(props_df),
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
