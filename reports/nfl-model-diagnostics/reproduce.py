"""Reproduce the Murray audit using explicitly supplied immutable input files."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import career_baseline as cb
import make_player_prop_params as model

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--artifact-dir',type=Path,required=True)
p.add_argument('--history-dir',type=Path,required=True)
a=p.parse_args();archive=a.artifact_dir.resolve();history=a.history_dir.resolve()
career=cb.load_career_logs(list(range(2020,2027)),str(history))
logs=career[(career.season==2026)&(career.week<3)].copy()
logs['recent_team']=logs.team;logs['interceptions']=logs.passing_interceptions;logs['rush_attempts']=logs.carries
original=pd.read_csv(archive/'data/props/params_week3.csv')
original=original[original.player.eq('Kyler Murray')&original.market_std.eq('pass_yds')].iloc[0]
# Give the existing relative-path defense loader a private, read-only source link.
with TemporaryDirectory() as temp:
    os.chdir(temp);Path('data').mkdir()
    Path('data/weekly_player_stats_2026.parquet').symlink_to(history/'weekly_player_stats_2026.parquet')
    cands=pd.DataFrame([dict(player='Kyler Murray',market_std='pass_yds')])
    result=model.build_params(cands,logs,2026,3,defensive_ratings=model.calculate_defensive_ratings(2026,3),
        opponent_map={'Kyler Murray':'TB'},home_away_map={'Kyler Murray':False},career_df=career).iloc[0]
    assert abs(result.mu-original.mu)<1e-8,(result.mu,original.mu)
    assert abs(result.sigma-original.sigma)<1e-8,(result.sigma,original.sigma)
    print(json.dumps(dict(mean=result.mu,sigma=result.sigma,trace=json.loads(result.projection_diagnostics)),indent=2))
