"""Publish an immutable, dated research edition; market refreshes cannot erase it."""
from copy import deepcopy
import json
import os
import subprocess
from zoneinfo import ZoneInfo
from nhl.v2.data import ROOT, iso, digest, stamp, write_json
from nhl.analyst import immutable

ET=ZoneInfo('America/New_York')

def existing_today(root,now,*,include_test=False):
    """Only a morning edition completes a normal run; tests may reuse either kind."""
    try:
        card=json.loads((root/'docs/briefing/morning-card.json').read_text())
        kinds=('morning','test') if include_test else ('morning',)
        return (card.get('schema_version') == 1 and card.get('kind') in kinds
            and card.get('status') in ('published', 'no_reviewed_candidates')
            and isinstance(card.get('edition_id'), str) and bool(card['edition_id'])
            and isinstance(card.get('rows'), list) and len(card['rows']) <= 10
            and stamp(card['published_at']) <= now
            and card.get('decision_date') == stamp(card['published_at']).astimezone(ET).date().isoformat()
            == now.astimezone(ET).date().isoformat())
    except (OSError,ValueError,TypeError,KeyError,AttributeError):
        return False


def research_health(feeds, reviews, selected, now, refresh_results):
    """Distinguish bounded coverage and a legitimate no-pick day from failures."""
    sports, issues = {}, []
    failed_batches = {'review_unavailable', 'partially_completed', 'api_key_unavailable', 'expired_during_research',
        'outside_review_window', 'disabled', 'budget_halted'}
    for coverage in selected['coverage']:
        sport = coverage['sport']; board = reviews.get('sports', {}).get(sport, {})
        feed = feeds.get(sport) or {}
        count = board.get('candidate_count', len(board.get('candidates', [])))
        done = board.get('reviewed_count', sum(bool(r.get('qualitative_review')) for r in board.get('candidates', [])))
        state = board.get('review_status', 'not_requested')
        sports[sport] = dict(review_status=state, candidate_count=count, reviewed_count=done,
            pending_count=max(0, count-done), source_status=feed.get('status', 'unavailable'),
            refresh_status=refresh_results.get(sport, 'not_supplied'),
            available_at_publication=coverage['available'],
            batch_statuses=board.get('batch_statuses', []),
            # Actionable diagnostics: sanitized categories and affected candidate IDs.
            batches=[{k: b.get(k) for k in ('index', 'status', 'category', 'candidate_ids', 'request_sent',
                                            'charged_usd', 'reserved_usd', 'ledger_status', 'rejected', 'http_status')
                      if b.get(k) is not None} for b in board.get('batches', [])],
            coverage_summary=board.get('coverage_summary'),
            source_diagnostics=source_diagnostics(board))
        # An explicitly empty, successfully refreshed market feed is not a failure.
        try:
            checked = stamp(feed.get('last_success_at') or feed.get('generated_at'))
            empty_market = (feed.get('status') == 'waiting_for_markets' and not feed.get('rows')
                and not feed.get('model_error') and 0 <= (now-checked).total_seconds() <= 5400)
        except (ValueError, TypeError, AttributeError):
            empty_market = False
        if refresh_results.get(sport) in ('failure', 'cancelled', 'skipped'):
            issues.append(sport+': refresh failed or did not finish')
        elif refresh_results.get(sport) == 'reused' and not coverage['available']:
            issues.append(sport+': reused feed expired before publication')
        elif not coverage['available'] and not empty_market:
            issues.append(sport+': current feed unavailable or expired')
        if count and not done:
            reason={'api_key_unavailable':'analysis unavailable', 'review_unavailable':'analysis unavailable',
                'expired_during_research':'inputs expired during research',
                'budget_exhausted':'daily research allowance reached',
                'budget_halted':'research spending paused'}.get(state,'research did not complete')
            issues.append(sport+': no completed assessments ('+reason+')')
        elif failed_batches.intersection(board.get('batch_statuses', [])+[state]):
            summary = board.get('coverage_summary') or {}
            detail = ', '.join(f'{k} ×{v}' for k, v in sorted((summary.get('failure_categories') or {}).items()))
            issues.append(sport+': some assessments failed'+(f' ({detail})' if detail else ''))
    discovery=reviews.get('discovery_coverage') or {}
    discovery_without_completion=(discovery.get('slate_games',0) or 0)>0 and not discovery.get('submitted_games') and reviews.get('discovery_status')!='completed'
    if reviews.get('discovery_status') in ('api_key_unavailable', 'discovery_unavailable', 'budget_halted') or (discovery_without_completion and not any(s['reviewed_count'] for s in sports.values())):
        issues.append('Independent discovery did not complete')
    return dict(completed=not issues, issues=issues, sports=sports,
        candidate_count=sum(s['candidate_count'] for s in sports.values()),
        reviewed_count=sum(s['reviewed_count'] for s in sports.values()),
        run_url=os.getenv('RUN_URL') or None)


def source_diagnostics(board):
    """Sanitized source-retrieval counts; hosts and stages only, never response bodies."""
    status = board.get('evidence_status') or {}
    if not isinstance(status, dict):
        return None
    counts = {}
    for f in status.get('failures', []):
        label = f"{f.get('stage', 'unknown')}:{f.get('category', 'unavailable')}"
        counts[label] = counts.get(label, 0)+1
    rejected = {}
    for r in status.get('rejected', []):
        rejected[r.get('reason', 'unknown')] = rejected.get(r.get('reason', 'unknown'), 0)+1
    coverage = status.get('coverage') or {}
    return dict(status=status.get('status'), attempts=status.get('attempts'), failures=counts, rejected=rejected,
                targeted_failures=len(status.get('targeted_failures') or []),
                candidates_without_sources=sorted(k for k, v in coverage.items() if not v))


def publish_card(feeds,reviews,now,*,root=ROOT,kind='morning',refresh_results=None):
    if kind not in ('morning','test'):raise ValueError('Unknown edition kind')
    merged=deepcopy(feeds);merged['Reviews']=reviews
    result=subprocess.run(['node',str(ROOT/'scripts/analyst_shortlist.cjs')],
        input=json.dumps({'feeds':merged,'asof':iso(now)},allow_nan=False),
        text=True,capture_output=True,check=True,timeout=30,cwd=ROOT)
    selected=json.loads(result.stdout)
    rows=selected['card']
    health=research_health(feeds,reviews,selected,now,refresh_results or {})
    card=dict(schema_version=1,kind=kind,decision_date=now.astimezone(ET).date().isoformat(),
        published_at=iso(now),policy_version='morning-edition-1',rows=rows,
        coverage=selected['coverage'],budget=reviews.get('budget'),
        discovery_status=reviews.get('discovery_status'),
        discovery_coverage=reviews.get('discovery_coverage'),
        research=health,
        status=('published' if rows else 'no_reviewed_candidates') if health['completed'] else 'research_incomplete',
        basis='Original reviewed forecasts and quotes at publication; no automatic intraday reassessment.')
    card['edition_id']=digest(card)[:24]
    relative='docs/briefing/cards/'+card['decision_date']+'-'+card['edition_id']+'.json'
    card['archive_url']='/'+relative.removeprefix('docs/')
    card['decision_ledger']=freeze_decisions(feeds,selected,card,now,root)
    immutable(root/relative,card)
    write_json(root/'docs/briefing/morning-card.json',card)
    return card


def freeze_decisions(feeds,selected,card,now,root):
    """Freeze the three decision versions and the NHL shadow pilot for this edition.

    Evaluation records must not block or alter the reader-facing card; a failure is
    reported on the card instead of being hidden.
    """
    try:
        import decision_ledger
        from nhl.v2 import deployment_pilot
        nhl=[r for r in selected['selected'] if r['sport']=='NHL']
        facts={r.get('offer_id'):(r.get('qualitative_review') or {}).get('facts',[]) for r in nhl}
        records=deployment_pilot.run(nhl,facts,(feeds.get('NHL') or {}).get('model_distribution'),now)
        pilot_path=None
        if records:
            pilot=dict(schema='nhl-deployment-pilot-run-1',edition_id=card['edition_id'],computed_at=iso(now),
                       pilot_version=deployment_pilot.PILOT_VERSION,evaluation_status='shadow_only',records=records)
            pilot_path=root/'artifacts/research/pilot/nhl-shots'/(card['decision_date']+'-'+card['edition_id']+'.json')
            immutable(pilot_path,pilot)
        ledger=decision_ledger.freeze(selected,card['rows'],now,edition_id=card['edition_id'],kind=card['kind'],
            pilot={r['outcome_key']:r for r in records},ledgers=root/'artifacts/research/decisions',
            prompt_version=__import__('analyst_review').PROMPT_VERSION)
        return dict(status='frozen',ledger_id=ledger['ledger_id'],universe=ledger['universe_count'],counts=ledger['counts'],
            path=str((root/'artifacts/research/decisions'/(ledger['decision_date']+'-'+card['edition_id']+'.json')).relative_to(root)),
            pilot_records=len(records),pilot_path=str(pilot_path.relative_to(root)) if pilot_path else None)
    except Exception as error:
        return dict(status='failed',error=type(error).__name__)
