"""Reproduce selection on saved source snapshots; no simulated profits or API calls."""
import json
from pathlib import Path
import subprocess
import tempfile
from collections import Counter
from datetime import timedelta
from nhl.v2.data import ROOT,stamp,iso,write_json,digest
import analyst_review


def replay():
    feeds=analyst_review.load_feeds();feeds.pop('Reviews',None);feeds.pop('Discovery',None)
    report={'input_hashes':{s:digest(feeds.get(s)) for s in ('NFL','NFLContext','MLB')},'evaluation':'Selection replay, not forecast validation or a betting backtest','sports':{}}
    with tempfile.TemporaryDirectory() as td:
        old=Path(td)/'legacy.js'
        old.write_text(subprocess.run(['git','show','611e9c4:docs/assets/briefing-picks.js'],cwd=ROOT,
            check=True,capture_output=True,text=True).stdout)
        for sport in ('NFL','MLB'):
            source=feeds[sport];now=stamp(source.get('generated_at') or source['model_checked_at'])+timedelta(seconds=1)
            current={sport:source,'NFLContext':feeds.get('NFLContext')}
            before=subprocess.run(['node','-e',"const s=require(process.argv[1]);let x=JSON.parse(require('fs').readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(s.collect(x.feeds,Date.parse(x.asof))));",str(old)],
                input=json.dumps(dict(feeds=current,asof=iso(now))),text=True,capture_output=True,check=True)
            before=json.loads(before.stdout);after=analyst_review.selected(current,now)
            reasons=Counter(reason for r in after.get('excluded',[]) for reason in r['exclusion_reasons'])
            fields=('game','player','market_std','market','side','line','price','book','score','screening_ev','forecast_health')
            report['sports'][sport]=dict(asof=iso(now),source_generated_at=source.get('generated_at') or source['model_checked_at'],
                source_rows=len(source['rows']),old_candidates=len(before['selected']),new_candidates=len(after['selected']),
                withheld_rows=len(after.get('excluded',[])),withheld_reasons=dict(reasons),
                new_games=len({r['game_id'] for r in after['selected']}),
                candidates=[{k:r[k] for k in fields if k in r} for r in after['selected']])
    return report

if __name__=='__main__':
    result=replay();write_json(ROOT/'reports/top-picks/selection-replay.json',result)
    print(json.dumps({s:{k:v for k,v in r.items() if k!='candidates'} for s,r in result['sports'].items()},indent=2))
