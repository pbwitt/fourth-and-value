#!/usr/bin/env python3
"""Refresh the public Kalshi board with public GETs only; no account required."""
import argparse
import json
from pathlib import Path

from prediction_markets.client import PublicClient
from prediction_markets.feed import ROOT, atomic_json, collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/prediction_markets.json')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/prediction-markets/snapshot.json')
    parser.add_argument('--archive', type=Path, default=ROOT/'data/prediction-markets/snapshots')
    parser.add_argument('--max-markets', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.max_markets is not None:
        config['max_markets'] = args.max_markets
    books = []
    for path in (ROOT/'docs/nfl/data/quotes.json',):
        if path.exists():
            books.extend(json.loads(path.read_text()).get('rows', []))
    client = PublicClient()
    snapshot = collect(client, config, books)
    atomic_json(args.archive/(snapshot['snapshot_id']+'.json'), snapshot, private=True)
    atomic_json(args.output, snapshot)
    print(json.dumps(dict(status=snapshot['status'], contracts=len(snapshot['rows']),
                          requests=client.requests, snapshot_id=snapshot['snapshot_id'],
                          errors=snapshot['errors'])))
    return 1 if snapshot['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
