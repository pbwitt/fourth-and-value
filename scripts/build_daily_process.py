"""Generate the public operating description from schedules/policy; --check in CI."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
WORKFLOWS={
 'morning-picks.yml':('Morning research edition','Gate on a completed edition; otherwise refresh NFL, MLB and NHL, wait for all three, research and verify the published card. Later starts recover missing/incomplete work.'),
 'editorial-daily.yml':('Editorial articles and maintenance','Primary article run starts after the morning sports refreshes; target two articles by 8:30 a.m. Recovery needs fresh morning models. Hourly checks handle approved posts, freshness and eligible morning catch-up; they are not hourly article production.'),
 'nfl-weekly.yml':('NFL data and models','Refresh statistics, player estimates, scoring projections and available prices; publish before research and editorial writing. Explicit manual editorial requests can also refresh NFL.'),
 'mlb-daily.yml':('MLB data and models','Refresh history/training cache, prices, lineups and forecasts. Morning refresh is coordinated by Morning Picks Edition.'),
 'nhl-daily.yml':('NHL data and models','Refresh regular-season statistics and prices, run inference, publish all qualifying candidates. After a validated refresh, regrade NHL Market Analytics from saved pregame snapshots and grade archived cross-market checks (descriptive research only; not an input to picks). Later refresh can capture newly posted props.'),
 'editorial-watchdog.yml':('Article recovery monitor','Checks missing article delivery during the morning writing window, with an extra check after the 8:30 a.m. target; no paid writing without fresh model inputs.'),
 'analyst-daily.yml':('Top Picks discovery and review','Runs once after the morning feeds finish, or as an explicit operator test. No automatic intraday research.')}


def times(cron):
    minute,hour,dom,month,dow=cron.split()
    days={'*':'daily','0':'Sunday','3':'Wednesday','4':'Thursday'}
    def expand(v):
        out=[]
        for part in v.split(','):
            if '-' in part:
                lo,hi=map(int,part.split('-'));out.extend(range(lo,hi+1))
            else:out.append(int(part))
        return out
    if '-' in hour and ',' not in hour:
        lo,hi=hour.split('-');label=f'each hour from {int(lo)%12 or 12} {"a.m." if int(lo)<12 else "p.m."} through {int(hi)%12 or 12} {"a.m." if int(hi)<12 else "p.m."}, at minute '+minute.replace(',', ' and ')
    elif hour=='*':label='hourly at :'+minute.zfill(2)
    else:
        values=[f'{h%12 or 12}:{m:02d} {"a.m." if h<12 else "p.m."}' for h in expand(hour) for m in expand(minute)]
        label=', '.join(values)
    return days.get(dow,dow)+': '+label


def render():
    config=json.loads((ROOT/'config/analyst_review.json').read_text())
    nhl=json.loads((ROOT/'config/nhl_analyst.json').read_text())
    rows=[]
    for name,(label,description) in WORKFLOWS.items():
        source=(ROOT/'.github/workflows'/name).read_text()
        crons=re.findall(r"cron: ['\"]([^'\"]+)['\"]",source)
        if source.count('timezone: America/New_York')!=len(crons):
            raise ValueError('Schedule timezone changed: review public timing')
        schedule='<br>'.join(escape(times(c)) for c in crons)
        if name in ('nhl-daily.yml','mlb-daily.yml','nfl-weekly.yml'):
            schedule='daily: 7:05 a.m. via morning workflow; recovery at 7:35, 8:05, 8:35<br>'+schedule
        if name=='editorial-daily.yml':schedule='Primary: after morning sports refreshes<br>Recovery / maintenance checks:<br>'+schedule
        if name=='editorial-watchdog.yml':schedule+='; also after scheduled Morning Picks completes'
        if name=='analyst-daily.yml':schedule='After morning data jobs finish; incomplete-edition recovery or explicit manual tests only otherwise'
        rows.append(f'<tr><td>{escape(label)}</td><td>{schedule}</td><td>{escape(description)}</td></tr>')
    paths=[ROOT/'.github/workflows'/n for n in WORKFLOWS]
    paths += [ROOT/p for p in ['config/analyst_review.json','config/nhl_analyst.json',
        'scripts/editorial_schedule.py','scripts/editorial_writer.py','scripts/editorial_diagnostics.py','scripts/analyst_review.py','scripts/research_discovery.py',
        'scripts/research_budget.py','scripts/morning_card.py','scripts/morning_operations.py','supabase/morning_scheduler.sql','scripts/mlb/predict.py','scripts/nhl/v2/candidates.py','docs/assets/briefing-picks.js']]
    fingerprint=hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()[:20]
    values=dict(POLICY=config['policy_version'],SCHEDULE=''.join(rows),NHL_EV=f"{nhl['minimum_ev']*100:g}",
        MORNING_BUDGET=f"{config['daily_budget_usd']-config['later_reserve_usd']:.2f}",LATER_RESERVE=f"{config['later_reserve_usd']:.2f}",NHL_QUOTE=str(nhl['quote_max_minutes']),NHL_MODEL=str(nhl['model_max_hours']),BUDGET=f"{config['daily_budget_usd']:.2f}",FINGERPRINT=fingerprint)
    result=(ROOT/'scripts/research/daily_process.html').read_text()
    for key,value in values.items():result=result.replace('{{'+key+'}}',value)
    from site_notices import apply
    return apply(result,'research/daily-process.html')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    path=ROOT/'docs/research/daily-process.html';expected=render()
    if args.check:
        if not path.exists() or path.read_text()!=expected:raise SystemExit('Daily process page is stale. Review the process prose, then run python scripts/build_daily_process.py.')
    else:path.write_text(expected)

if __name__=='__main__':main()
