"""Import human decisions and compare prospective screened/selected shadow cohorts."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); __package__ = 'nhl.v2'

from .candidates import exclusion
from .data import ROOT, digest, iso, stamp, write_json

ARCHIVE = ROOT/'artifacts/nhl/analyst'


def validate(record, board, now, config):
    allowed = {'board_id', 'candidate_id', 'offer_id', 'forecast_id', 'recorded_at', 'analyst', 'decision',
               'reason', 'double_counting_check', 'source_url', 'source_published_at', 'price_confirmed', 'context_checked'}
    if set(record) != allowed:
        raise ValueError('Unexpected or missing decision fields; probability overrides use the separate review interface')
    if record.get('board_id') != board['board_id']:
        raise ValueError('Decision belongs to another board')
    row = next((r for r in board['candidates'] if r['candidate_id'] == record.get('candidate_id')), None)
    if not row or any(record.get(k) != row[k] for k in ('offer_id', 'forecast_id')):
        raise ValueError('Decision offer/forecast identity mismatch')
    if record.get('decision') not in ('select', 'watch', 'pass'):
        raise ValueError('Invalid human decision')
    if any(not isinstance(record.get(k), str) or not 3 <= len(record[k].strip()) <= 1500 for k in ('analyst', 'reason', 'double_counting_check')):
        raise ValueError('Analyst, reason and double-counting check required')
    at = stamp(record['recorded_at'])
    if not stamp(board['generated_at']) <= at <= now or at >= stamp(row['commence_time']):
        raise ValueError('Decision must be recorded before the game')
    # Imported later is allowed; no historical claim of a verified server timestamp.
    if record['decision'] == 'select':
        if exclusion(row, at, config):
            raise ValueError('Candidate expired or price invalid at recorded decision')
        if record.get('price_confirmed') is not True or record.get('context_checked') is not True:
            raise ValueError('Human price and context checks required')
        url = urlsplit(record.get('source_url') or '')
        if url.scheme != 'https' or not url.hostname or url.username or url.password:
            raise ValueError('Selection needs a sourced contextual check')
        if not stamp(record['source_published_at']) <= at:
            raise ValueError('Future decision evidence')
    return dict(record, decision_id=digest(record)[:24], ingested_at=iso(now),
                timestamp_basis='analyst_reported; ingestion separately recorded',
                evaluation_status='prospective_shadow_only', probability_adjustment=None,
                stake_units=1 if record['decision'] == 'select' else 0)


def record_decision(record, now, config, archive=ARCHIVE, public=ROOT/'docs/nhl/data/candidates.json'):
    bid = record.get('board_id', '')
    if len(bid) != 24 or any(c not in '0123456789abcdef' for c in bid):
        raise ValueError('Invalid board identity')
    board = json.loads((archive/'boards'/f'{bid}.json').read_text())
    result = validate(record, board, now, config)
    # A file per candidate/board is append-only: no replacing a loss with a later pass.
    path = archive/'decisions'/f"{bid}-{result['candidate_id']}.json"
    if path.exists() and json.loads(path.read_text())['decision_id'] != result['decision_id']:
        raise ValueError('Decision already recorded; preserve the original')
    if not path.exists():
        write_json(path, result)
    if public.exists():
        current = json.loads(public.read_text())
        if current.get('board_id') == bid:
            for row in current['candidates']:
                if row['candidate_id'] == result['candidate_id']:
                    row.update(human_decision=result['decision'], human_review=result)
            write_json(public, current)
            write_json(archive/'published'/f'{digest(current)[:24]}.json', current)
    return result


def evaluate(boards, decisions, games, players):
    from .grading import settle, betting_metrics
    games = {g['game_id']: g for g in games}
    players = {(p['game_id'], p['player_id']): p for p in players}
    # First morning observation/game/day only; do not cherry-pick favorable reruns.
    seen, graded = set(), []
    selected = {(d['board_id'], d['candidate_id']): d for d in decisions}
    for board in sorted(boards, key=lambda b: b['generated_at']):
        if board['session'] != 'morning':
            continue
        for r in board['candidates']:
            key = (board['decision_date'], r['nhl_game_id'])
            if key in seen:
                continue
            seen.add(key)
            game = games.get(r['nhl_game_id'])
            player = players.get((r['nhl_game_id'], r.get('player_id')))
            result = settle(r, game, player)
            decision = selected.get((board['board_id'], r['candidate_id']))
            # Ingested after kickoff is retained for auditing but excluded from prospective evaluation.
            timely = bool(decision and stamp(decision['ingested_at']) < stamp(r['commence_time']))
            graded.append(dict(r, board_id=board['board_id'], result=result,
                               human_decision=decision['decision'] if timely else 'unreviewed',
                               decision_timely=timely))
    cohorts = {'all_screened': graded, 'human_selected': [r for r in graded if r['human_decision'] == 'select'],
               'human_passed': [r for r in graded if r['human_decision'] == 'pass']}
    metrics = {}
    for name, rows in cohorts.items():
        resolved = [r for r in rows if r['result'] in ('won', 'lost')]
        # Probabilities conditional on no push, to match binary settled outcomes.
        brier = sum(((r['final_probability']/(1-r['push_probability']))-(r['result'] == 'won'))**2 for r in resolved)/len(resolved) if resolved else None
        metrics[name] = dict(betting_metrics(rows), brier_conditional_non_push=brier,
                             binary_count=len(resolved), unresolved=sum(r['result'] not in ('won', 'lost', 'push', 'void') for r in rows),
                             worse_execution=betting_metrics(rows, worse_execution=.05))
    evaluated_keys = {(r['board_id'], r['candidate_id']) for r in graded}
    excluded = [dict(decision_id=d['decision_id'], reason='outside_first_morning_cohort') for d in decisions
                if (d['board_id'], d['candidate_id']) not in evaluated_keys]
    return dict(status='prospective_shadow_only', metrics=metrics, rows=graded, excluded_decisions=excluded,
                limitations=['No placed wagers or verified fills; original quote prices are hypothetical execution.',
                             'No causal estimate of analyst skill; selection changes the evaluated population.',
                             'No closing-price feed yet; closing-line value unavailable.',
                             'Unconfirmed participation stays unresolved; never treat missing player data as a loss.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    rec = subs.add_parser('record'); rec.add_argument('file', type=Path)
    grade = subs.add_parser('grade'); source = grade.add_mutually_exclusive_group(required=True)
    source.add_argument('--history', type=Path,
        help='Normalized JSON containing games and players, as used by nhl.v2')
    source.add_argument('--cached-history', action='store_true', help='Use the frozen history plus current-season cached game records')
    grade.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'record':
        result = record_decision(json.loads(args.file.read_text()), datetime.now(timezone.utc),
                                 json.loads((ROOT/'config/nhl_analyst.json').read_text()))
        print(result['decision_id'])
    else:
        import gzip
        history_path = args.history or ROOT/'models/nhl/v2/history.json.gz'
        opener = gzip.open if history_path.suffix == '.gz' else open
        with opener(history_path, 'rt') as handle:
            history = json.load(handle)
        if args.cached_history:
            from .data import load
            season = json.loads((ROOT/'docs/nhl/data/latest.json').read_text())['season']
            games, players, _ = load(ROOT/'data/nhl/v2/history', [season])
            for key, values in [('games', games), ('players', players)]:
                history[key] = [r for r in history[key] if r['season'] != season]+values
        boards = [json.loads(p.read_text()) for p in (ARCHIVE/'boards').glob('*.json')]
        decisions = [json.loads(p.read_text()) for p in (ARCHIVE/'decisions').glob('*.json')]
        write_json(args.output, evaluate(boards, decisions, history['games'], history['players']))


if __name__ == '__main__':
    main()
