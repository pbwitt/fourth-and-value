"""Publish the quantitative NHL shortlist and optionally run one bounded morning critique."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import requests

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nhl.v2 import astra, evidence
from nhl.v2.candidates import shortlist, exclusion
from nhl.v2.data import ROOT, digest, iso, stamp, write_json

ARCHIVE = ROOT/'artifacts/nhl/analyst'
PUBLIC = ROOT/'docs/nhl/data/candidates.json'
CONFIG = ROOT/'config/nhl_analyst.json'


def immutable(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Immutable analyst archive conflict')
    else:
        write_json(path, value)


def review(board, config, archive=ARCHIVE, clock=lambda: datetime.now(timezone.utc)):
    if not board['candidates']:
        return board
    if not config['astra_enabled']:
        board['review_status'] = 'disabled'
        return board
    if board['session'] != 'morning':
        board['review_status'] = 'afternoon_quantitative_update'
        return board
    if not os.getenv('OPENAI_API_KEY'):
        board['review_status'] = 'api_key_unavailable'
        return board
    key = board['decision_date']+':morning'
    budget = archive/'budget.json'
    if budget.exists() and any(e['key'] == key for e in json.loads(budget.read_text())['entries']):
        board['review_status'] = 'already_attempted_today'
        return board
    sources, diagnostics = evidence.collect(board['candidates'], clock)
    asof = clock()
    board['evidence_status'] = diagnostics
    board['sources'] = [{k: v for k, v in s.items() if k != 'excerpt'} for s in sources]
    if not sources:
        board['review_status'] = 'no_usable_reporting'
        return board
    # Collection can take time; never submit a critique of an already invalid candidate.
    if any(exclusion(r, asof, config) for r in board['candidates']):
        board['review_status'] = 'expired_during_research'
        return board
    request = astra.payload(board, sources, asof, config)
    amount = astra.bounds(request, config)
    request_id = digest(request)[:24]
    packet_path = archive/'requests'/f'{request_id}.json'
    immutable(packet_path, dict(board_id=board['board_id'], request_id=request_id,
                               prepared_at=iso(asof), request=request, source_diagnostics=diagnostics))
    result = astra.reserve(budget, key, asof, amount, config['weekly_budget_usd'])
    if result != 'reserved':
        board['review_status'] = result
        return board
    board['review_request_id'] = request_id
    # In CI this reservation and the exact packet reach origin before the paid request.
    astra.checkpoint([budget, packet_path, archive/'boards'/f"{board['board_id']}.json"])
    response = None
    try:
        response = astra.call_api(request)
        finished = clock()
        immutable(archive/'responses'/f'{request_id}.json', dict(request_id=request_id, received_at=iso(finished), response=response))
        reviews = astra.parse_response(response, board, sources, asof)
        for item in reviews:
            item['evidence_asof'] = item['reviewed_at']
            item['reviewed_at'] = iso(finished)
            item['completed_at'] = iso(finished)
        by_id = {r['candidate_id']: r for r in reviews}
        for row in board['candidates']:
            row['qualitative_review'] = by_id[row['candidate_id']]
        board.update(review_status='completed', review_completed_at=iso(finished),
                     review_requires_price_recheck=any(exclusion(r, finished, config) for r in board['candidates']))
    except (requests.RequestException, ValueError, RuntimeError, TypeError, KeyError):
        board['review_status'] = 'review_unavailable'
    finally:
        astra.settle_budget(budget, key, response.get('usage') if isinstance(response, dict) else None)
    return board


def prepare(state, now, config, run_review=False, archive=ARCHIVE, public=PUBLIC):
    board = shortlist(state, now, config)
    immutable(archive/'boards'/f"{board['board_id']}.json", deepcopy(board))
    if run_review:
        try:
            board = review(board, config, archive)
        except Exception as error:
            # Review/source/budget errors never turn a stale previous shortlist into today's picks.
            board.update(review_status='review_unavailable', review_error=type(error).__name__)
    immutable(archive/'published'/f'{digest(board)[:24]}.json', board)
    write_json(public, board)
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--astra', action='store_true', help='One bounded, sourced morning review; no call on an empty slate')
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    # Serialize local reservations as well as the GitHub concurrency group.
    lock = ROOT/'data/nhl/analyst.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.loads((ROOT/'docs/nhl/data/latest.json').read_text())
        config = json.loads(CONFIG.read_text())
        board = prepare(state, datetime.now(timezone.utc), config, args.astra)
    print(json.dumps({k: board[k] for k in ('board_id', 'status', 'eligible_count', 'review_status')}))


if __name__ == '__main__':
    main()
