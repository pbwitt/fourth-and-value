"""Select the upcoming regular-season week and prepare reproducible NFL inputs."""
import argparse
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path

import pandas as pd
import requests


def select_week(schedule, now, requested=None):
    games=schedule[schedule['game_type']=='REG'].copy()
    games['kickoff']=pd.to_datetime(games['gameday'].astype(str)+' '+games['gametime'].fillna('13:00').astype(str)).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    upcoming=games[games['kickoff']>now]
    if requested is not None:
        if not 1<=requested<=18:raise ValueError('Week must be between 1 and 18')
        upcoming=upcoming[upcoming['week']==requested]
        if upcoming.empty:raise ValueError('Requested week has no upcoming regular-season games')
    else:
        upcoming=upcoming[upcoming['kickoff']<=now+timedelta(days=14)]
    if upcoming.empty:return None
    return int(upcoming.sort_values('kickoff').iloc[0]['week'])


def prepare(season,week):
    from fetch_weekly_player_stats import fetch
    from nfl_build_team_features import build_team_features
    from nfl_train_totals_model import train_totals_model
    directory=Path('data/nfl/refresh');directory.mkdir(parents=True,exist_ok=True)
    for year in range(season-6,season+1):
        path=Path(f'data/weekly_player_stats_{year}.parquet')
        if year==season or not path.exists():
            try:fetch(year,path)
            except Exception:
                if year==season and week==1:continue
                raise
    if week>1:
        current=pd.read_parquet(f'data/weekly_player_stats_{season}.parquet')
        if int(current['week'].max())<week-1:
            raise ValueError('Latest completed week of player stats is not published yet; refresh stopped')
    columns=['game_id','season','home_team','away_team','week','game_date','total','total_line',
        'home_score','away_score','posteam','defteam','epa','success','pass','rush','down',
        'third_down_converted','yardline_100','touchdown','sack','interception','fumble_lost']
    frames=[]
    for year in range(season-4,season+1):
        path=directory/f'pbp_{year}.parquet'
        if year==season or not path.exists():
            response=requests.get(f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet',timeout=180)
            if response.status_code==404 and year==season and week==1:continue
            response.raise_for_status()
            frame=pd.read_parquet(io.BytesIO(response.content),columns=columns)
            frame.to_parquet(path,index=False)
        frames.append(pd.read_parquet(path))
    pbp=pd.concat(frames,ignore_index=True)
    # Even manual earlier-week runs must never train on that week or later games.
    pbp=pbp[(pbp['season']<season)|((pbp['season']==season)&(pbp['week']<week))]
    if week>1 and not ((pbp['season']==season)&(pbp['week']==week-1)).any():
        raise ValueError('Latest completed week of play-by-play is not available yet')
    combined=directory/'training_pbp.parquet';pbp.to_parquet(combined,index=False)
    build_team_features(str(combined),'data/nfl/processed/team_features.csv')
    train_totals_model('data/nfl/processed/team_features.csv',output_dir='data/nfl/models')
    report=dict(season=season,week=week,checked_at=datetime.now(timezone.utc).isoformat(),
        history_through=str(pbp['game_date'].max()),games=int(pbp['game_id'].nunique()))
    (directory/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select',action='store_true')
    parser.add_argument('--season',type=int)
    parser.add_argument('--week',type=int)
    args=parser.parse_args()
    if args.select:
        import nfl_data_py as nfl
        now=datetime.now(timezone.utc)
        season=args.season or now.year-(now.month<3)
        schedule=nfl.import_schedules([season])
        week=select_week(schedule,now,args.week)
        Path('data').mkdir(exist_ok=True)
        schedule.to_csv(f'data/schedule_{season}.csv',index=False)
        outputs=dict(active=str(week is not None).lower(),season=season,week=week or '')
        print(json.dumps(outputs))
        if os.getenv('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'],'a') as stream:
                for key,value in outputs.items():stream.write(f'{key}={value}\n')
    else:
        if not args.season or not args.week:parser.error('--season and --week are required')
        prepare(args.season,args.week)


if __name__=='__main__':main()
