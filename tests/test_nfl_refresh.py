import sys
from pathlib import Path
from datetime import datetime,timezone
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_nfl_refresh import select_week
from nfl_predict_totals import live_features

class RefreshTests(unittest.TestCase):
    def test_tuesday_selects_next_week_after_monday(self):
        schedule=pd.DataFrame([dict(game_type='REG',week=2,gameday='2026-09-21',gametime='20:15'),dict(game_type='REG',week=3,gameday='2026-09-24',gametime='20:15')])
        self.assertEqual(select_week(schedule,datetime(2026,9,22,12,tzinfo=timezone.utc)),3)
        self.assertEqual(select_week(schedule,datetime(2026,9,21,12,tzinfo=timezone.utc)),2)
        self.assertIsNone(select_week(schedule,datetime(2026,7,1,tzinfo=timezone.utc)))
        with self.assertRaises(ValueError):select_week(schedule,datetime(2026,9,22,12,tzinfo=timezone.utc),2)
    def test_live_rolling_form_includes_latest_completed_game(self):
        history=pd.DataFrame([dict(team='A',game_date=f'2026-09-{i:02}',game_id=i,points_scored=v,points_scored_L3=-99) for i,v in enumerate([10,20,30,40],1)])
        self.assertEqual(live_features(history,'A',['points_scored_L3','is_home','points_scored_L5'],1),[30,1,25])

if __name__=='__main__':unittest.main()
