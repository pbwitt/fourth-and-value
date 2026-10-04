#!/usr/bin/env python3
"""Owner-operated mock-funds order rehearsal; never connects to production trading."""
import argparse
import json
from pathlib import Path
import sys

from prediction_markets.feed import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/kalshi_demo.json')
    parser.add_argument('--journal', type=Path, default=ROOT/'data/prediction-markets/demo-journal.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect').add_argument('--ticker', required=True)
    sub.add_parser('submit').add_argument('--intent', type=Path, required=True)
    sub.add_parser('cancel').add_argument('--id', required=True)
    sub.add_parser('reconcile')
    sub.add_parser('status')
    args = parser.parse_args()
    try:
        from prediction_markets.demo_client import DemoClient
        from prediction_markets.demo_execution import DemoExecutor, summary
    except ImportError:
        parser.exit(2, 'Install requirements-kalshi-demo.txt in your virtual environment first.\n')
    try:
        engine = DemoExecutor(DemoClient.from_environment(), args.journal, json.loads(args.config.read_text()))
        if args.command == 'inspect':
            result = engine.inspect(args.ticker)
        elif args.command == 'submit':
            result = summary(engine.submit(json.loads(args.intent.read_text())))
        elif args.command == 'cancel':
            result = summary(engine.cancel(args.id))
        else:
            result = [summary(r) for r in getattr(engine, args.command)()]
        print(json.dumps(dict(mode='demo', result=result), indent=2, allow_nan=False))
        records = result if isinstance(result, list) else [result]
        return 1 if any(r.get('state') in ('submitting', 'uncertain', 'cancel_pending', 'open') for r in records) else 0
    except Exception as exc:
        # Never echo remote bodies, authentication headers, PEMs or arbitrary paths.
        from prediction_markets.demo_client import DemoAPIError
        message = str(exc) if isinstance(exc, (ValueError, DemoAPIError)) and not isinstance(exc, json.JSONDecodeError) else type(exc).__name__
        print('Demo operation stopped: '+message, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
