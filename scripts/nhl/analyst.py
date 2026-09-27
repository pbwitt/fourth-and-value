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
    """Compatibility entry point; the shared sports queue owns all paid research."""
    board['review_status'] = 'shared_research_queue'
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
    parser.add_argument('--astra', action='store_true', help='Compatibility flag; paid analysis now runs through scripts/analyst_review.py')
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    # Serialize local reservations as well as the GitHub concurrency group.
    lock = ROOT/'data/analyst/review.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json.loads((ROOT/'docs/nhl/data/latest.json').read_text())
        config = json.loads(CONFIG.read_text())
        board = prepare(state, datetime.now(timezone.utc), config, args.astra)
    print(json.dumps({k: board[k] for k in ('board_id', 'status', 'eligible_count', 'review_status')}))


if __name__ == '__main__':
    main()
