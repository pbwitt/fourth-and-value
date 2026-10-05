"""Cheap recovery gate and end-to-end delivery check; never calls a paid API."""
import argparse
from datetime import datetime, timezone
import json
import os
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from morning_card import ET, ROOT, existing_today
from nhl.v2.data import stamp


SPORTS = ('NFL', 'MLB', 'NHL')
# A reused feed must stay inside its 90-minute quote/model window through research and
# publication. NHL quotes expire after 30 minutes, so NHL is always refreshed.
REUSE_FEED_MINUTES = {'NFL': 60, 'MLB': 60}
FEEDS = {'NFL': ('docs/props/top-picks.json', 'generated_at'), 'MLB': ('docs/mlb/data/latest.json', 'model_checked_at')}
UNHEALTHY_BATCHES = {'review_unavailable', 'partially_completed', 'api_key_unavailable', 'expired_during_research',
                     'outside_review_window', 'disabled', 'budget_halted'}


def feed_age_minutes(root, sport, now):
    try:
        path, field = FEEDS[sport]
        feed = json.loads((root/path).read_text())
        return (now-stamp(feed[field])).total_seconds()/60
    except (KeyError, OSError, ValueError, TypeError):
        return None


def recovery_scope(root, now):
    """Sports a recovery start must refresh again after an incomplete same-day edition.

    A sport is reused only when the earlier edition shows a successful refresh, a feed
    available at publication, research without failed batches, and a published feed
    young enough to stay fresh. Its unchanged feed then lets the research step reuse
    the same reviews (identical price/forecast/evidence keys) without paying again,
    while still reviewing any candidates the earlier run did not reach.
    """
    try:
        card = json.loads((root/'docs/briefing/morning-card.json').read_text())
        same_day = (card.get('schema_version') == 1 and card.get('kind') == 'morning'
                    and card.get('status') == 'research_incomplete'
                    and card.get('decision_date') == now.astimezone(ET).date().isoformat()
                    and stamp(card['published_at']) <= now)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        same_day = False
    if not same_day:
        return dict(refresh=list(SPORTS), reused=[], basis='no_same_day_incomplete_edition')
    reused, reasons = [], {}
    for sport in SPORTS:
        h = (card.get('research') or {}).get('sports', {}).get(sport) or {}
        age = feed_age_minutes(root, sport, now) if sport in REUSE_FEED_MINUTES else None
        why = ('nhl_quotes_expire_in_30_minutes' if sport not in REUSE_FEED_MINUTES else
               'refresh_not_successful' if h.get('refresh_status') not in ('success', 'reused') else
               'feed_unavailable_at_publication' if not h.get('available_at_publication') else
               'failed_research_batches' if UNHEALTHY_BATCHES.intersection(h.get('batch_statuses', [])+[h.get('review_status')]) else
               'feed_too_old_to_reuse' if age is None or not 0 <= age <= REUSE_FEED_MINUTES[sport] else None)
        reasons[sport] = why or 'reused_unchanged_feed'
        if not why:
            reused.append(sport)
    return dict(refresh=[s for s in SPORTS if s not in reused], reused=reused, reasons=reasons,
                basis='same_day_incomplete_edition', edition_id=card.get('edition_id'))


def gate(root, now, *, test_edition=False, replace_card=False):
    if existing_today(root, now, include_test=test_edition) and not replace_card:
        return dict(refresh=False, reason='edition_already_published', scope=dict(refresh=[], reused=[]))
    if not test_edition and not 7 <= now.astimezone(ET).hour < 12:
        raise RuntimeError('No completed edition and outside the 07:00–12:00 Eastern recovery window')
    scope = recovery_scope(root, now) if not (test_edition or replace_card) else dict(refresh=list(SPORTS), reused=[], basis='explicit_operator_request')
    return dict(refresh=True, reason='missing_or_incomplete_edition', scope=scope)


def record_start(now, result, *, source='operator', summary_path=None):
    """Record gate execution time, without guessing which cron occurrence was delayed."""
    local = now.astimezone(ET)
    source = source if source in ('github-schedule', 'supabase', 'operator') else 'unknown'
    record = dict(trigger_source=source, gate_checked_at=now.isoformat(),
                  eastern_time=local.isoformat(), **result)
    print(json.dumps(record))
    if summary_path:
        with open(summary_path, 'a') as output:
            output.write('\n## Morning start\n\n'
                         f'- Source: {source}\n'
                         f'- Gate checked (Eastern): {local.isoformat()}\n'
                         '- Daily target start: 07:05 America/New_York\n'
                         f'- Gate: {result["reason"]}\n'
                         f'- Refresh: {", ".join((result.get("scope") or {}).get("refresh", [])) or "none"}\n'
                         f'- Reused unchanged feeds: {", ".join((result.get("scope") or {}).get("reused", [])) or "none"}\n')
    return record


def verify_live(edition_id, *, root=ROOT, attempts=24, interval=20,
                fetch=urlopen, sleep=time.sleep):
    card=json.loads((root/'docs/briefing/morning-card.json').read_text())
    if not edition_id or card.get('edition_id') != edition_id:
        raise RuntimeError('Local edition changed or missing; delivery cannot be verified')
    for attempt in range(attempts):
        query=urlencode({'edition_id':edition_id, 'delivery_check':str(time.time_ns())})
        request=Request('https://fourthandvalue.com/briefing/morning-card.json?'+query,
            headers={'User-Agent':'FourthAndValue/1.0 morning-delivery-check', 'Cache-Control':'no-cache'})
        try:
            with fetch(request,timeout=15) as response:
                live=json.loads(response.read(10_000_000))
            if isinstance(live,dict) and all(live.get(k)==card.get(k) for k in ('edition_id','decision_date','kind','status')):
                print('Public morning edition verified: '+edition_id)
                return
        except (URLError, OSError, ValueError, TypeError):
            pass
        if attempt+1 < attempts:
            sleep(interval)
    raise RuntimeError('Public morning edition did not match '+edition_id+' within the delivery window')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate',action='store_true')
    parser.add_argument('--test-edition',action='store_true')
    parser.add_argument('--replace-card',action='store_true')
    parser.add_argument('--verify-live',metavar='EDITION_ID')
    parser.add_argument('--attempts',type=int,default=24)
    args=parser.parse_args()
    if args.gate:
        now=datetime.now(timezone.utc)
        try:
            result=gate(ROOT,now,test_edition=args.test_edition,replace_card=args.replace_card)
        except RuntimeError:
            record_start(now,dict(refresh=False,reason='outside_window_without_completed_edition'),
                source=os.getenv('MORNING_TRIGGER_SOURCE','operator'),summary_path=os.getenv('GITHUB_STEP_SUMMARY'))
            raise
        record_start(now,result,source=os.getenv('MORNING_TRIGGER_SOURCE','operator'),
            summary_path=os.getenv('GITHUB_STEP_SUMMARY'))
        if os.getenv('GITHUB_OUTPUT'):
            scope=result.get('scope') or {}
            with open(os.environ['GITHUB_OUTPUT'],'a') as output:
                output.write('refresh='+str(result['refresh']).lower()+'\n')
                for sport in SPORTS:
                    output.write(f'refresh_{sport.lower()}='+str(result['refresh'] and sport in scope.get('refresh',SPORTS)).lower()+'\n')
                output.write('reused='+','.join(scope.get('reused',[]))+'\n')
    if args.verify_live:
        verify_live(args.verify_live,attempts=args.attempts)


if __name__=='__main__':
    main()
