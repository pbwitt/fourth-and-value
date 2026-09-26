"""Private operational reports. No private ideas, draft text, credentials or paid calls."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
import requests
import editorial_schedule as schedule

ROOT=Path(__file__).resolve().parents[1]
SPORTS=('NFL','MLB','NBA','NHL')
STAGES={
 'Refresh market prices and briefing':'prices',
 'Research and publish original analysis':'writing',
 'Verify expected writer execution':'writer_check',
 'Publish public pages only':'publication',
 'Rebuild site':'deployment',
 'Verify daily article delivery':'live_delivery',
}


def github_run():
    token=os.getenv('GH_TOKEN');repo=os.getenv('GITHUB_REPOSITORY');run_id=os.getenv('GITHUB_RUN_ID')
    if not token or not repo or not run_id:return {'available':False,'reason':'Run details unavailable'}
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'}
    base=f'https://api.github.com/repos/{repo}/actions/runs/{run_id}'
    try:
        r=requests.get(base,headers=headers,timeout=20);r.raise_for_status();run=r.json()
        r=requests.get(base+'/jobs',headers=headers,params={'per_page':100},timeout=20);r.raise_for_status()
        stages={}
        for job in r.json().get('jobs',[]):
            for step in job.get('steps',[]):
                if step.get('name') in STAGES:
                    stages[STAGES[step['name']]]={'status':step.get('conclusion') or step.get('status','unknown'),
                        'started_at':step.get('started_at'),'completed_at':step.get('completed_at')}
        return {'available':True,'id':str(run_id),'url':run.get('html_url'),'event':run.get('event'),
                'created_at':run.get('created_at'),'started_at':run.get('run_started_at'),
                'status':run.get('conclusion') or run.get('status'),'stages':stages}
    except (requests.RequestException,ValueError):
        return {'available':False,'reason':'GitHub run details could not be read'}


def recent(value,now,hours):
    age=schedule.age_hours(value,now)
    return age is not None and 0<=age<=hours and schedule.same_et_day(value,now)


def safe_reason(value):
    # A classification, never echo arbitrary model exceptions or private topics.
    text=str(value or '').lower()
    for terms,label in [
        (('audit','unsupported'),'Factual review did not pass'),
        (('publisher','reporting','source'),'Insufficient current reporting'),
        (('budget','reservation'),'Spending limit or existing reservation'),
        (('fund','credit','quota','api_key'),'Writing credentials or funding need attention'),
        (('model inputs','history'),'Model inputs are not current'),
        (('refresh','stale','briefing'),'Current data refresh required'),
        (('no current market','market data'),'No usable current markets'),
        (('network',),'Upstream connection failed'),
        (('private','idea','requested'),'Private idea needs attention in the editorial desk')]:
        if any(term in text for term in terms):return label
    return 'See the workflow checks for details' if text else None


def next_opportunity(now):
    local=now.astimezone(schedule.ET)
    for minute in range(1,24*60+1):
        candidate=(local+timedelta(minutes=minute)).replace(second=0,microsecond=0)
        h,m=candidate.hour,candidate.minute
        if h>=5 and (m==17 or (5<=h<=9 and m in (27,47)) or (h,m) in ((5,7),(6,37),(11,7),(16,7),(21,7))):
            return candidate.isoformat()


def build_report(root=ROOT,now=None,run=None,phase='finish'):
    now=now or datetime.now(timezone.utc);root=Path(root)
    day=now.astimezone(schedule.ET).date().isoformat()
    state=schedule.today_state(root,now);run=run or {'available':False}
    cfg=schedule.config(root);expected=min(int(cfg.get('writer',{}).get('daily_story_limit',2) or 2),2)
    briefing=schedule.load(root/'docs/briefing/latest.json',{})
    data=[]
    for sport in SPORTS:
        board=schedule.load(root/f'docs/{sport.lower()}/data/latest.json',{})
        games=[g for g in briefing.get('games',[]) if g.get('sport')==sport]
        fresh=recent(briefing.get('generated_at'),now,6)
        usable=sum(1 for g in games if schedule.stamp(g.get('commence_time')) and schedule.stamp(g['commence_time'])>now)
        coverage=str(briefing.get('coverage',{}).get(sport,'Not checked'))
        market_status='ready' if fresh and usable else 'unavailable' if 'unavailable' in coverage.lower() or 'credential' in coverage.lower() else 'no_markets' if fresh else 'stale'
        checked=board.get('model_checked_at')
        model_status='unvalidated' if 'not validated' in str(board.get('model_status','')) or 'pending leakage correction and validation' in str(board.get('model_status','')) else 'unknown'
        if checked:
            model_status='current' if recent(checked,now,1.5) else 'stale'
            if sport=='MLB' and (board.get('model_summary',{}).get('history_through')!=(now.astimezone(schedule.ET).date()-timedelta(days=1)).isoformat() or board.get('history_error')):model_status='stale'
        if checked and board.get('status')=='unavailable':model_status='unavailable'
        data.append({'sport':sport,'prices':market_status,'games':usable,'checked_at':briefing.get('generated_at'),
            'model':model_status,'model_checked_at':checked,'model_detail':str(board.get('model_status',''))[:200],'detail':coverage[:180]})
    articles=[]
    catalog={a.get('url'):a for a in schedule.today_catalog(root,now)}
    stored={url for url in catalog if str(url).startswith('/editorial/articles/') and (root/'docs'/url.lstrip('/')).is_file()}
    for key,slot in state.get('slots',{}).items():
        url=slot.get('url');public=catalog.get(url,{})
        provenance=slot.get('source_check',{})
        articles.append({'slot':key,'sport':key.split('-')[-1].upper(),'status':slot.get('status','unknown'),
            'title':public.get('title'),'url':url if url in stored else None,
            'reason':safe_reason(slot.get('reason')),
            'review':'passed' if slot.get('status') in ('published','review') else 'failed' if 'audit' in str(slot.get('reason','')).lower() else 'not_recorded',
            'selection':slot.get('selection'),'source_fallback':provenance.get('fallback_used'), 'source_hosts':provenance.get('hosts',[])})
    sources=[]
    for sport,check in state.get('source_checks',{}).items():
        sources.append({'sport':sport,**{k:check.get(k) for k in ('at','candidates','hosts','fallback_attempted','fallback_used','ready')}})
    blocked=[{'sport':sport,'reason':safe_reason(reason)} for sport,reason in state.get('data_skips',{}).items()]
    need,reason=schedule.writer_need(root,now)
    stages=run.get('stages',{})
    live=stages.get('live_delivery',{}).get('status','not_checked')
    complete=len(stored)>=expected
    status='delivered' if complete and live=='success' else 'saved_not_verified' if complete else 'overdue' if now.astimezone(schedule.ET).hour>=8 else 'pending'
    return {'version':1,'day':day,'observed_at':now.isoformat(),'phase':phase,'status':status,
        'expected':expected,'saved':len(stored),'live_check':live,'run':run,'data':data,'articles':articles,
        'selection_checks':state.get('selection_checks',[]),'sources':sources,'unselected':blocked,'selection_count':len(state.get('allocation',[])),
        'writer_checked_at':state.get('last_writer_check',{}).get('at'),
        'recovery':{'eligible':need,'reason':reason,'next_opportunity':next_opportunity(now) if need else None,
                    'note':'Scheduled opportunity; GitHub may delay or miss a trigger.'},
        'fallback_policy':'All four sports are considered. Two qualifying sports take priority; otherwise a different matchup from the same qualifying sport may fill the second slot. Current model evidence is required. Rejected or uncertain paid attempts are not repeated.'}


def store_report(report):
    base=os.getenv('SUPABASE_URL');key=os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('UPABASE_SERVICE_ROLE_KEY')
    if not base or not key:raise RuntimeError('Diagnostics storage credentials are not configured')
    run=os.getenv('GITHUB_RUN_ID') or report['observed_at']
    identity=f"{run}-{os.getenv('GITHUB_RUN_ATTEMPT','1')}-{report['phase']}"
    row={'id':identity,'edition_day':report['day'],'observed_at':report['observed_at'],'report':report}
    r=requests.post(base.rstrip('/')+'/rest/v1/editorial_pipeline_reports',
        headers={'apikey':key,'Authorization':'Bearer '+key,'Prefer':'resolution=merge-duplicates,return=minimal'},
        params={'on_conflict':'id'},json=row,timeout=25)
    if r.status_code in (404,400):raise RuntimeError('Run the one-time supabase/editorial_diagnostics.sql setup in Supabase')
    if not r.ok:raise RuntimeError(f'Diagnostics storage failed (HTTP {r.status_code})')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='finish',choices=['start','finish','watchdog']);parser.add_argument('--store',action='store_true')
    args=parser.parse_args();report=build_report(run=github_run(),phase=args.phase)
    summary=f"Morning diagnostics: {report['saved']}/{report['expected']} article files; status={report['status']}."
    print(summary)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as out:out.write('\n'+summary+'\n\n[Private dashboard](https://fourthandvalue.com/editorial/diagnostics.html)\n')
    if args.store:
        try:store_report(report)
        except (RuntimeError,requests.RequestException) as exc:
            message=str(exc) if isinstance(exc,RuntimeError) else 'Diagnostics database connection failed'
            raise SystemExit(message)
        print('Private diagnostic report saved')

if __name__=='__main__':main()
