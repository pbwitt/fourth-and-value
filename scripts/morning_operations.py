"""Cheap recovery gate and end-to-end delivery check; never calls a paid API."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from morning_card import ET, ROOT, existing_today


NFL_FEED = 'docs/nfl/data/latest.json'
# NFL quotes stay valid for 90 minutes at publication. A recovery start reuses an NFL
# refresh at most 45 minutes old, which leaves research at least 45 minutes, instead of
# buying the full NFL slate again; NHL (30 minutes) and MLB still refresh every start.
NFL_REUSE = timedelta(minutes=45)


def nfl_refreshed_at(root):
    """When the published NFL feed's model and prices were last refreshed, or None."""
    try:
        feed = json.loads((Path(root) / NFL_FEED).read_text())
        if feed.get('status') != 'ready':
            return None
        stamp = datetime.fromisoformat(str(feed.get('model_checked_at') or feed.get('last_success_at')).replace('Z', '+00:00'))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return stamp if stamp.tzinfo else None


def gate(root, now, *, test_edition=False, replace_card=False):
    if existing_today(root, now, include_test=test_edition) and not replace_card:
        return dict(refresh=False, refresh_nfl=False, reason='edition_already_published')
    if not test_edition and not 7 <= now.astimezone(ET).hour < 12:
        raise RuntimeError('No completed edition and outside the 07:00–12:00 Eastern recovery window')
    result = dict(refresh=True, refresh_nfl=True, reason='missing_or_incomplete_edition')
    refreshed = nfl_refreshed_at(root)
    if not (test_edition or replace_card) and refreshed and timedelta(0) <= now - refreshed <= NFL_REUSE:
        result.update(refresh_nfl=False, nfl_reused_from=refreshed.isoformat())
    return result


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
                         + (f'- NFL: reusing the refresh from {datetime.fromisoformat(result["nfl_reused_from"]).astimezone(ET):%H:%M} ET\n'
                            if result.get('nfl_reused_from') else ''))
    return record


def verify_live(edition_id, *, root=ROOT, attempts=24, interval=20,
                fetch=urlopen, sleep=time.sleep, kind='morning'):
    name='afternoon-card.json' if kind=='afternoon' else 'morning-card.json'
    card=json.loads((root/'docs/briefing'/name).read_text())
    if not edition_id or card.get('edition_id') != edition_id:
        raise RuntimeError('Local edition changed or missing; delivery cannot be verified')
    for attempt in range(attempts):
        query=urlencode({'edition_id':edition_id, 'delivery_check':str(time.time_ns())})
        request=Request(f'https://fourthandvalue.com/briefing/{name}?'+query,
            headers={'User-Agent':'FourthAndValue/1.0 morning-delivery-check', 'Cache-Control':'no-cache'})
        try:
            with fetch(request,timeout=15) as response:
                live=json.loads(response.read(10_000_000))
            if isinstance(live,dict) and all(live.get(k)==card.get(k) for k in ('edition_id','decision_date','kind','status')):
                print(f'Public {kind} edition verified: '+edition_id)
                return
        except (URLError, OSError, ValueError, TypeError):
            pass
        if attempt+1 < attempts:
            sleep(interval)
    raise RuntimeError(f'Public {kind} edition did not match '+edition_id+' within the delivery window')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate',action='store_true')
    parser.add_argument('--test-edition',action='store_true')
    parser.add_argument('--replace-card',action='store_true')
    parser.add_argument('--verify-live',metavar='EDITION_ID')
    parser.add_argument('--attempts',type=int,default=24)
    parser.add_argument('--kind',choices=['morning','afternoon'],default='morning')
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
            with open(os.environ['GITHUB_OUTPUT'],'a') as output:
                output.write('refresh='+str(result['refresh']).lower()+'\n')
                output.write('refresh_nfl='+str(result['refresh_nfl']).lower()+'\n')
    if args.verify_live:
        verify_live(args.verify_live,attempts=args.attempts,kind=args.kind)


if __name__=='__main__':
    main()
