#!/usr/bin/env python3
"""Staggered weekly delivery with idempotent morning recovery.

MLB Monday, NHL Tuesday, NBA Thursday. NFL's Wednesday refresh owns its immutable
week-numbered archive and also recovers through the morning NFL workflow.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
ET=ZoneInfo('America/New_York')
DAYS={'mlb':0,'nhl':1,'nba':3}


def expected_end(sport, now):
    today=now.astimezone(ET).date()
    due=today-timedelta(days=(today.weekday()-DAYS[sport])%7)
    return due-timedelta(days=1)


def completed(sport, end, root=ROOT):
    stem=Path(root)/f'docs/blog/{sport}-recap-{end}'
    if not stem.with_suffix('.html').exists() or not stem.with_suffix('.json').exists():
        return None
    try:
        data=json.loads(stem.with_suffix('.json').read_text())['summary']
    except (ValueError,KeyError,TypeError):
        return None
    from sport_weekly_review import stamp
    if not isinstance(data,dict) or data.get('start')!=str(end-timedelta(days=6)) or not stamp(data.get('generated_at')):
        return None
    if data.get('status')!='published' or data.get('sport')!=sport or data.get('end')!=str(end):
        return None
    return data


def should_run(sport, end, now, previous, root=ROOT):
    report=completed(sport,end,root)
    if report:
        unresolved=report.get('tickets',{}).get('unresolved',0)+sum(x.get('unresolved',0) for x in report.get('board',{}).values())
        # Successful complete reports are immutable under automatic delivery.
        # Later official corrections can be incorporated by an explicit rerun.
        if not unresolved:
            return False
        from sport_weekly_review import stamp
        last=stamp(report.get('generated_at'))
        attempt=stamp(previous.get('checked_at')) if previous.get('end')==str(end) else None
        last=max([x for x in [last,attempt] if x],default=None)
        return (not last or last.astimezone(ET).date()!=now.astimezone(ET).date()) and (now.astimezone(ET).date()-end).days<=3
    if previous.get('end')==str(end) and previous.get('checked_at'):
        from sport_weekly_review import stamp
        last=stamp(previous['checked_at'])
        if last and now-last<timedelta(hours=2):
            return False
    return True


def run(now=None, root=ROOT):
    from sport_weekly_review import review, rebuild_home, stamp
    now=now or datetime.now(timezone.utc)
    root=Path(root);path=root/'docs/recaps/status.json'
    previous={}
    if path.exists():
        try:
            previous={s['sport']:s for s in json.loads(path.read_text()).get('sports',[]) if isinstance(s,dict) and s.get('sport')}
        except (ValueError,TypeError,AttributeError):
            previous={}  # Report artifacts, not an unreadable receipt, decide delivery.
    statuses=[];failures=[];published=False
    for sport in DAYS:
        end=expected_end(sport,now);old=previous.get(sport,{})
        report=completed(sport,end,root)
        if should_run(sport,end,now,old,root):
            try:
                result=review(sport,end,root)
                published=published or result['status']=='published'
                status={k:result[k] for k in ['sport','status','start','end']}
                status['checked_at']=now.isoformat()
            except Exception as exc:
                status=dict(sport=sport,status='failed',end=str(end),checked_at=now.isoformat(),error=str(exc))
                failures.append(f'{sport}: {exc}')
        elif report:
            attempt=stamp(old.get('checked_at')) if old.get('end')==str(end) else None
            if attempt and attempt>stamp(report['generated_at']):
                # A skipped run must retain the newer attempt, including a failed
                # refresh, so the next invocation observes the same retry limit.
                status=dict(old)
            else:
                status=dict(sport=sport,status='published',start=report['start'],end=str(end),checked_at=report['generated_at'])
        else:
            status=old
        if status:
            status['weekday']=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][DAYS[sport]]
            if report or status['status']=='published':status['url']=f'/blog/{sport}-recap-{end}.html'
            statuses.append(status)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(schema=1,sports=statuses,nfl=dict(weekday='Wednesday',recovery='Morning pipeline plus 10 a.m., noon and 6 p.m. ET scheduled starts'))
    text=json.dumps(payload,indent=2)+'\n'
    if not path.exists() or path.read_text()!=text:path.write_text(text)
    if published:rebuild_home()
    print(json.dumps(statuses,indent=2))
    if failures:raise RuntimeError('; '.join(failures))
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    try:run()
    except Exception as exc:print(str(exc),file=sys.stderr);raise SystemExit(1)
