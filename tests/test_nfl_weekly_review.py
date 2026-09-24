"""Regression tests for the recurring NFL weekly review pipeline."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import nfl_weekly_review as review


class WeeklyReviewTests(unittest.TestCase):
    def test_ticket_selection_is_one_offer_per_game_player_market(self):
        rows=pd.DataFrame([
            dict(game_id='g',player='P',market_std='rush_yds',name='over',point=50.5,price=-110,
                 bookmaker='a',model_prob=.60,ev_per_100=8.0),
            dict(game_id='g',player='P',market_std='rush_yds',name='over',point=49.5,price=-105,
                 bookmaker='b',model_prob=.62,ev_per_100=12.0),
            dict(game_id='g',player='P',market_std='rush_yds',name='under',point=50.5,price=110,
                 bookmaker='c',model_prob=.40,ev_per_100=4.0),
        ])
        selected=review.select_tickets(rows)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected.iloc[0].bookmaker,'b')
        self.assertEqual(selected.iloc[0].ev_per_100,12.0)

    def test_prop_grading_push_and_missing_player_are_conservative(self):
        props=pd.DataFrame([
            dict(game_id='book1',game='Atlanta Falcons @ Green Bay Packers',home_team='Green Bay Packers',
                 away_team='Atlanta Falcons',player='Example Player',market_std='receptions',
                 name='over',point=3,price=-110,bookmaker='a',model_prob=.55,consensus_prob=.50,mu=3.2,ev_per_100=5),
            dict(game_id='book1',game='Atlanta Falcons @ Green Bay Packers',home_team='Green Bay Packers',
                 away_team='Atlanta Falcons',player='Missing Player',market_std='receptions',
                 name='under',point=2.5,price=-110,bookmaker='a',model_prob=.55,consensus_prob=.50,mu=2.0,ev_per_100=5),
        ])
        schedule=pd.DataFrame([dict(game_id='2026_03_ATL_GB',season=2026,week=3,game_type='REG')])
        stats=pd.DataFrame([dict(game_id='2026_03_ATL_GB',player_display_name='Example Player',
            attempts=0,carries=0,targets=3,receptions=3)])
        graded=review.grade_props(props,stats,schedule,2026,3)
        self.assertEqual(graded.iloc[0].outcome,'push')
        self.assertEqual(graded.iloc[0].units,0)
        self.assertTrue(pd.isna(graded.iloc[1].outcome))
        self.assertEqual(graded.iloc[1].grading_status,'no matching game/player result')

    def test_snapshot_refuses_started_game(self):
        frame=pd.DataFrame([dict(season=2026,week=3,commence_time='2026-09-24T20:00:00Z')])
        with self.assertRaisesRegex(ValueError,'already started'):
            review.validate_snapshot_frame(frame,2026,3,pd.Timestamp('2026-09-24T21:00:00Z'),'props')

    def test_existing_snapshot_is_idempotent_and_tamper_evident(self):
        with TemporaryDirectory() as td:
            root=Path(td);base=root/'reports/nfl-weekly/2026/week-4/pregame';base.mkdir(parents=True)
            evidence=base/'props.csv';evidence.write_text('x\n1\n')
            manifest={'files':[{'path':str(evidence.relative_to(root)),'sha256':review.sha(evidence)}]}
            (base/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(review.snapshot_week(2026,4,datetime(2026,9,30,tzinfo=timezone.utc),root)['files'][0]['sha256'],review.sha(evidence))
            evidence.write_text('x\n2\n')
            with self.assertRaisesRegex(ValueError,'integrity'):
                review.snapshot_week(2026,4,datetime(2026,9,30,tzinfo=timezone.utc),root)

    def test_review_withholds_until_all_games_complete(self):
        with TemporaryDirectory() as td:
            root=Path(td);(root/'data').mkdir()
            pd.DataFrame([dict(season=2026,week=3,game_type='REG',home_score=np.nan,away_score=np.nan)]).to_csv(root/'data/schedule_2026.csv',index=False)
            pd.DataFrame([dict(season=2026,week=3,season_type='REG')]).to_parquet(root/'data/weekly_player_stats_2026.parquet',index=False)
            with self.assertRaisesRegex(ValueError,'not fully completed'):
                review.review_week(2026,3,4,root)

    def test_week2_published_benchmark_still_reconciles(self):
        checks=review.verify_week2()
        self.assertTrue(all(checks.values()))


if __name__=='__main__':
    unittest.main()
