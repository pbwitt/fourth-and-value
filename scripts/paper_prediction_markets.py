#!/usr/bin/env python3
"""Run or settle private paper intents. Never authenticates or sends real orders."""
import argparse
import json
from pathlib import Path

from prediction_markets.client import PublicClient, ticker
from prediction_markets.feed import ROOT, atomic_json, collect
from prediction_markets.paper import ledger_file, run_intent, settle
from prediction_markets.pricing import utcnow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'settle'))
    parser.add_argument('--intents', type=Path)
    parser.add_argument('--snapshot', type=Path, help='Use a saved snapshot; freshness checks still apply')
    parser.add_argument('--config', type=Path, default=ROOT/'config/prediction_markets.json')
    parser.add_argument('--ledger', type=Path, default=ROOT/'data/prediction-markets/paper-ledger.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    client = PublicClient()
    if args.action == 'run':
        if args.intents is None:
            parser.error('run requires --intents')
        intents = json.loads(args.intents.read_text())
        if not isinstance(intents, list) or len(intents) > 100:
            parser.error('intents must be an array with at most 100 entries')
        if args.snapshot:
            snapshot = json.loads(args.snapshot.read_text())
        else:
            snapshot = collect(client, config, market_tickers=[i['ticker'] for i in intents])
            atomic_json(ROOT/'data/prediction-markets/snapshots'/(snapshot['snapshot_id']+'.json'), snapshot, private=True)
        results = []
        # Commit each attempt before continuing, so a later invalid intent cannot
        # erase earlier reservations/fills. Lock covers read, checks and write.
        for intent in intents:
            with ledger_file(args.ledger) as ledger:
                results.append(run_intent(snapshot, intent, ledger, config, utcnow()))
        print(json.dumps(dict(mode='paper', results=[{k: r.get(k) for k in ('id', 'status', 'reason', 'cost_dollars')} for r in results]), indent=2))
    else:
        with ledger_file(args.ledger) as ledger:
            tickers = {o['ticker'] for o in ledger['orders'] if o.get('simulation') and o.get('settlement') is None}
            markets = {t: client.get('/markets/'+ticker(t))['market'] for t in tickers}
            print(json.dumps(dict(mode='paper', settled=settle(ledger, markets, utcnow()))))


if __name__ == '__main__':
    main()
