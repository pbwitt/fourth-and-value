"""Explicit opt-in paid API integration test; synthetic evidence, never a published pick."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from nhl.v2 import astra
import research_budget as daily_budget
from nhl.v2.data import ROOT, digest, iso, write_json
from test_nhl_analyst import CONFIG, NOW, board, source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--live', action='store_true')
    p.add_argument('--env-file', type=Path)
    args = p.parse_args()
    b = board(); sources = [source()]
    request = astra.payload(b, sources, NOW, CONFIG)
    request['instructions'] = ('INTEGRATION TEST: all supplied candidates and reporting are fictional synthetic fixtures. '
                               'Test the response schema; do not claim this is actual NHL news.\n'+request['instructions'])
    amount = astra.bounds(request, CONFIG)
    print(json.dumps(dict(synthetic_test=True, model=request['model'], maximum_reserved_usd=amount)))
    if not args.live:
        return
    if os.getenv('GITHUB_ACTIONS') == 'true':
        raise SystemExit('Paid integration tests are manual, never PR checks')
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('API key unavailable')
    now = datetime.now(timezone.utc)
    budget = daily_budget.PATH
    key = 'integration-test:'+now.date().isoformat()
    if daily_budget.reserve(key, now, amount, path=budget, cap=CONFIG['daily_budget_usd']) != 'reserved':
        raise SystemExit('No request: duplicate attempt or insufficient budget')
    path = ROOT/'reports/nhl-analyst-workflow/astra-smoke.json'
    response = None
    try:
        response = astra.call_api(request)
        reviews = astra.parse_response(response, b, sources, NOW)
        result = dict(synthetic_test=True, executed_at=iso(now), model=response.get('model', astra.MODEL),
                      request_sha256=digest(request), response_id=response.get('id'), status='passed',
                      usage=response.get('usage'), conservative_cost_usd=round(response['usage']['input_tokens']*astra.INPUT_RATE+
                            response['usage']['output_tokens']*astra.OUTPUT_RATE, 6), reviews=reviews)
        write_json(path, result)
        print(json.dumps({k: result[k] for k in ('synthetic_test', 'status', 'conservative_cost_usd')}))
    finally:
        daily_budget.settle(key, response.get('usage') if isinstance(response, dict) else None, path=budget)


if __name__ == '__main__':
    main()
