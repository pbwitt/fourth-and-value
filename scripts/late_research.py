"""Bounded late research check for shortlisted candidates; shares the daily allowance.

After the morning edition, free checks look for material new information about each
shortlisted idea whose game has not started: a new or updated lineup, injury, scratch,
starter or goalie report for that event, or (MLB) a batting order confirmed after the
morning. Only a triggered candidate gets a paid re-review, of its CURRENT exact offer
(same book, line and side; the price may have moved), through the same review client,
validator and `artifacts/analyst/daily-budget.json` ledger. Reservation keys include the
new evidence, so a rerun with the same information cannot spend twice.

The morning edition is never modified. Each reassessment is an immutable record with
its own timestamp, per-day version number and basis edition, plus a day index at
`docs/briefing/reassessments.json`. Configuration: `late_check` in
`config/analyst_review.json` (disabled by default; see RESEARCH_SYSTEM.md).

  python scripts/late_research.py            # dry run: triggers only, never pays
  python scripts/late_research.py --execute  # paid only when enabled and triggered
"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
import re
import subprocess

from nhl.analyst import immutable
from nhl.v2 import evidence
from nhl.v2.data import ROOT, digest, iso, stamp, write_json

import analyst_review
import research_budget as daily_budget

SCHEMA = 'late-reassessment-1'
PUBLIC = ROOT/'docs/briefing/reassessments.json'
RECORDS = ROOT/'docs/briefing/reassessments'
MATERIAL = re.compile(r'lineup|line-up|batting order|injur|scratch|inactive|starter|starting|goalie|ruled out|'
                      r'questionable|doubtful|activated|placed on|out for|game-time decision|pitch count|limited', re.I)
DEFAULTS = dict(enabled=False, min_minutes_before_start=20, max_minutes_before_start=240, max_candidates=6,
                max_batches=2, window_et=[11, 21])


def settings(config):
    return dict(DEFAULTS, **(config.get('late_check') or {}))


def material_sources(row, sources_now, original_sources, now):
    """New or updated, event-matched sources whose subject is lineup/availability/deployment."""
    seen = {s['url']: stamp(s.get('updated_at') or s['published_at']) if s.get('published_at') else None
            for s in original_sources}
    found = []
    for s in sources_now:
        if not evidence.usable(s, row, now):
            continue
        effective = stamp(s.get('updated_at') or s['published_at']) if s.get('published_at') else stamp(s['retrieved_at'])
        kind = s.get('source_kind', 'reporting')
        changed = s['url'] not in seen or (seen[s['url']] and effective > seen[s['url']]) or \
            (kind == 'live_injury_table' and s['url'] in seen)
        subject = kind in ('official_injury_report', 'live_injury_table') or MATERIAL.search(s.get('title', '')+' '+s.get('excerpt', '')[:400])
        if changed and subject:
            found.append(dict(kind='new_or_updated_source', url=s['url'], source_kind=kind,
                              effective_at=iso(effective), title=s.get('title')))
    return found


def lineup_change(card_row, current_row):
    """MLB: batting order confirmed after the morning snapshot (feed field, no inference)."""
    before, after = card_row.get('lineup_status'), current_row.get('lineup_status')
    if card_row.get('sport') == 'MLB' and after and after != before and 'published' in str(after).lower():
        return [dict(kind='lineup_published', detail=str(after))]
    return []


def eligible_rows(card, now, cfg):
    rows = []
    for r in card.get('rows', []):
        start = stamp(r['commence_time'])
        minutes = (start-now).total_seconds()/60
        if cfg['min_minutes_before_start'] <= minutes <= cfg['max_minutes_before_start']:
            rows.append(r)
    return rows[:cfg['max_candidates']]


def current_offer(row, selection):
    """The same outcome at the same book and line in the current screen, if still offered."""
    def identity(r):
        return (r['sport'], str(r['game_id']), r.get('player') or '', r.get('market_std') or r.get('market'),
                str(r['side']).lower(), r.get('line'), r['book'], stamp(r['commence_time']))
    target = identity(row)
    return next((r for r in selection['selected'] if identity(r) == target), None)


def research_state_of(feeds, sport, board, now, review_bet_key):
    """Research state from the shared selector, against the late review board only."""
    merged = deepcopy(feeds)
    merged['Reviews'] = dict(schema_version=1, sports={sport: board})
    result = subprocess.run(['node', str(ROOT/'scripts/analyst_shortlist.cjs')],
        input=json.dumps(dict(feeds=merged, asof=iso(now)), allow_nan=False), capture_output=True, text=True,
        timeout=30, check=True, cwd=ROOT)
    row = next((r for r in json.loads(result.stdout)['selected'] if r['review_bet_key'] == review_bet_key), None)
    return (row or {}).get('research_state')


def run(now, config, *, feeds=None, execute=False, root=ROOT, archive=analyst_review.ARCHIVE,
        public=PUBLIC, records=RECORDS, clock=lambda: datetime.now(timezone.utc)):
    cfg = settings(config)
    day = now.astimezone(analyst_review.ET).date().isoformat()
    summary = dict(schema_version=1, decision_date=day, checked_at=iso(now), status=None, enabled=bool(cfg['enabled']),
                   candidates=[], paid_batches=0)
    try:
        card = json.loads((root/'docs/briefing/morning-card.json').read_text())
    except (OSError, ValueError):
        card = {}
    if card.get('kind') != 'morning' or card.get('decision_date') != day:
        summary['status'] = 'no_same_day_morning_edition'
        return summary
    lo, hi = cfg['window_et']
    if not lo <= now.astimezone(analyst_review.ET).hour < hi:
        summary['status'] = 'outside_late_window'
        return summary
    feeds = feeds if feeds is not None else analyst_review.load_feeds()
    selection = analyst_review.selected(feeds, now)
    queue = []
    for row in eligible_rows(card, now, cfg):
        entry = dict(sport=row['sport'], player=row.get('player'), market=row.get('market_std') or row.get('market'),
                     side=row['side'], line=row['line'], book=row['book'], original_price=row['price'],
                     original_quoted_at=row['quoted_at'], original_state=(row.get('research_state') or {}).get('gate'))
        summary['candidates'].append(entry)
        current = current_offer(row, selection)
        if not current:
            entry['status'] = 'offer_unavailable'
            continue
        normalized = analyst_review.normalized(current)
        try:
            sources, diagnostics = evidence.collect([normalized], clock, sport=row['sport'])
        except Exception as error:
            entry.update(status='source_check_failed', category=evidence.failure_category(error))
            continue
        original = [s for s in row.get('review_sources', []) if row.get('reviewed_candidate', {}).get('candidate_id') in s.get('candidate_ids', [])] \
            or row.get('review_sources', [])
        triggers = material_sources(normalized, sources, original, now)+lineup_change(row, current)
        entry['triggers'] = triggers
        if not triggers:
            entry['status'] = 'no_material_change'
            continue
        entry['status'] = 'triggered'
        queue.append((row, current, normalized, sources, diagnostics, entry))
    if not queue:
        summary['status'] = 'no_triggers' if summary['candidates'] else 'no_eligible_candidates'
        return summary
    if not (execute and cfg['enabled']):
        summary['status'] = 'triggered_dry_run' if cfg['enabled'] else 'disabled'
        return summary
    batches = 0
    for sport in ('NFL', 'MLB', 'NHL'):
        items = [q for q in queue if q[0]['sport'] == sport]
        while items and batches < cfg['max_batches']:
            chunk, items = items[:3], items[3:]
            board = dict(sport=sport, decision_date=day, generated_at=iso(now), session='late',
                         policy_version=config['policy_version'], candidates=[deepcopy(q[2]) for q in chunk],
                         coverage={}, sources=[], review_status='not_requested')
            board['board_id'] = digest(board)[:24]
            immutable(archive/'boards'/f"{board['board_id']}.json", deepcopy(board))
            sources = list({s['source_id']: s for q in chunk for s in q[3]}.values())
            board['_prepared_evidence'] = (sources, chunk[0][4])
            reviewed = analyst_review.review(board, feeds, dict(config, sessions=dict(config['sessions'], late=[lo, hi])),
                                             archive, clock)
            batches += 1
            for row, current, normalized, _, _, entry in chunk:
                late_row = next(r for r in reviewed['candidates'] if r['candidate_id'] == normalized['candidate_id'])
                state = research_state_of(feeds, sport, reviewed, clock(), current['review_bet_key']) \
                    if late_row.get('qualitative_review') else None
                record = dict(schema=SCHEMA, basis_edition_id=card['edition_id'], decision_date=day,
                    reassessed_at=iso(clock()), batch_status=reviewed['review_status'],
                    batch_category=(reviewed.get('batch') or {}).get('category'),
                    original=dict(price=row['price'], quoted_at=row['quoted_at'], research_state=row.get('research_state'),
                                  verdict=((row.get('qualitative_review') or {}).get('assessment') or {}).get('verdict')),
                    current=dict(price=current['price'], quoted_at=current['quoted_at'], forecast_at=current.get('forecast_at'),
                                 candidate_id=normalized['candidate_id'], review=late_row.get('qualitative_review'),
                                 research_failure=late_row.get('research_failure'), research_state=state),
                    identity=dict(sport=sport, game=row['game'], game_id=str(row['game_id']), commence_time=row['commence_time'],
                                  player=row.get('player'), market=entry['market'], side=row['side'], line=row['line'], book=row['book']),
                    triggers=entry['triggers'], probability_adjustment=None, evaluation_status='prospective_shadow_only')
                record['decision_change'] = change(row.get('research_state'), state)
                entry.update(status='reassessed', decision_change=record['decision_change'])
                publish(record, public, records)
    summary.update(status='completed', paid_batches=batches,
                   budget=daily_budget.usage_summary(now, path=archive/'daily-budget.json', config=config))
    return summary


def change(before, after):
    selectable = ('model_case_only', 'verified_context')
    a, b = (before or {}).get('gate'), (after or {}).get('gate')
    if b is None:
        return 'not_assessed'
    if (a in selectable) == (b in selectable):
        return 'unchanged_eligibility'
    return 'now_eligible' if b in selectable else 'now_'+b


def publish(record, public=PUBLIC, records=RECORDS):
    """Immutable record plus a per-day index; versions count reassessments of the same idea."""
    index = json.loads(public.read_text()) if public.exists() else {}
    if index.get('decision_date') != record['decision_date']:
        index = dict(schema_version=1, decision_date=record['decision_date'], reassessments=[])
    identity = digest(record['identity'])[:16]
    record['version'] = 1+sum(r['identity_id'] == identity for r in index['reassessments'])
    record['reassessment_id'] = digest(record)[:24]
    path = records/f"{record['decision_date']}-{record['reassessment_id']}.json"
    immutable(path, record)
    index['reassessments'].append(dict(identity_id=identity, reassessment_id=record['reassessment_id'],
        version=record['version'], reassessed_at=record['reassessed_at'], basis_edition_id=record['basis_edition_id'],
        identity=record['identity'], current_price=record['current']['price'],
        label=((record['current'].get('research_state') or {}).get('label') or 'Late research failed'),
        decision_change=record['decision_change'], archive_url='/'+str(path.relative_to(ROOT/'docs')) if path.is_relative_to(ROOT/'docs') else None))
    write_json(public, index)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Allow paid reviews when enabled and triggered')
    args = parser.parse_args()
    config = json.loads(analyst_review.CONFIG.read_text())
    lock = ROOT/'data/analyst/review.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with lock.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        summary = run(datetime.now(timezone.utc), config, execute=args.execute and bool(os.getenv('OPENAI_API_KEY')))
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
