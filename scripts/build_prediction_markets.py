#!/usr/bin/env python3
"""Build the prediction-market page with compliant, crawlable metadata."""
import argparse
import json
from pathlib import Path
from site_metadata import metadata
from site_notices import apply

ROOT = Path(__file__).resolve().parents[1]


def render():
    output = ROOT/'docs/prediction-markets/index.html'
    tags = metadata(output, 'Kalshi Sports Prices & Fees | Fourth & Value',
                    'Compare Kalshi sports contract prices, available quantity and estimated fees. Read exact settlement rules and see when each snapshot was observed.')
    source = (ROOT/'scripts/prediction_markets/page.html').read_text().replace('{{METADATA}}', tags)
    config = json.loads((ROOT/'config/prediction_markets.json').read_text())
    source = source.replace('{{QUOTE_MINUTES}}', f"{config['public_snapshot_max_seconds']/60:g}")
    return apply(source, 'prediction-markets/index.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    path = ROOT/'docs/prediction-markets/index.html'
    expected = render()
    if args.check:
        if not path.exists() or path.read_text() != expected:
            raise SystemExit('Run python scripts/build_prediction_markets.py')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected)


if __name__ == '__main__':
    main()
