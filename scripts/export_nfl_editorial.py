"""Publish dated NFL total estimates for editorial use; never invent priced EV."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import editorial_schedule as schedule


def export(root,season,week,now=None):
    root=Path(root);now=now or datetime.now(timezone.utc)
    manifest=schedule.load(root/'data/nfl/refresh/manifest.json',{})
    checked=schedule.stamp(manifest.get('checked_at'))
    if not checked or not 0<=(now-checked).total_seconds()<=5400:
        raise ValueError('NFL input preparation is missing or stale')
    if manifest.get('season')!=season or manifest.get('week')!=week:
        raise ValueError('NFL input manifest belongs to another slate')
    through=datetime.fromisoformat(manifest['history_through']).date()
    if not 0<=(now.astimezone(schedule.ET).date()-through).days<=7:
        raise ValueError('NFL training input cutoff is stale or in the future')
    source=root/'data/nfl/predictions/week_predictions.csv'
    predictions=list(csv.DictReader(source.open()))
    lines=list(csv.DictReader((root/'data/nfl/lines/totals_spreads.csv').open()))
    version=f'nfl-totals-{season}-w{week}-'+hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    rows=[]
    for prediction in predictions:
        if int(prediction['season'])!=season or int(prediction['week'])!=week:continue
        matches=[row for row in lines if row['game']==prediction['game'] and row.get('event_id')]
        if not matches:continue
        line=matches[0];start=schedule.stamp(line.get('commence_time'))
        if not start or start<=now:continue
        if start.astimezone(schedule.ET).date().isoformat()!=prediction['gameday']:continue
        value=float(prediction['total_pred'])
        if not 0<value<150:continue
        rows.append(dict(event_id=line['event_id'],game=prediction['game'],commence_time=start.isoformat(),
            market='totals',model_mean=value,model_mean_label='Estimated combined points',
            model_input_through=through.isoformat(),model_version=version,is_model_pick=False,
            model_status='Independent scoring estimate; calibrated fair price, win probability and injury adjustment are not established',
            model_inputs={'estimated home points':float(prediction['home_pred']),'estimated away points':float(prediction['away_pred'])}))
    board=dict(status='ready' if rows else 'unavailable',model_checked_at=checked.isoformat(),
        last_success_at=now.isoformat() if rows else None,model_status='Independent NFL scoring estimates available' if rows else 'No matching current-slate NFL forecasts',
        model_summary={'history_through':through.isoformat(),'season':season,'week':week},rows=rows)
    path=root/'docs/nfl/data/latest.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(board,indent=2,allow_nan=False)+'\n')
    print(f'Exported {len(rows)} NFL matchup estimates with input cutoff {through}')
    return board

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--season',type=int,required=True);p.add_argument('--week',type=int,required=True);a=p.parse_args()
    export(Path(__file__).resolve().parents[1],a.season,a.week)
