"""Pricing identity, leakage, blank markets and upstream failure regressions."""
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nba.pipeline import compare, add_baselines, refresh, flatten, iso, FeedError

NOW = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)


def quote(book='a', side='Over', line=20.5, market='player_points', price=-110, event='game'):
    from market_math import implied_probability
    return dict(event_id=event, player='Test Player' if market.startswith('player_') else '',
                market=market, line=line, side=side, book=book, price=price,
                book_probability=implied_probability(price), home_team='Home', away_team='Away')


class NBATests(unittest.TestCase):
    def test_exact_line_event_player_and_unique_books(self):
        rows=[quote(),quote(side='Under'),quote(),quote(book='b'),quote(book='b',side='Under'),
              quote(line=21.5),quote(event='other',side='Under')]
        result=compare(rows)
        self.assertEqual(len(result),6)
        self.assertEqual(result[0]['paired_books'],2)
        self.assertEqual(result[0]['fair_probability'],.5)
        self.assertIsNone(result[-1]['fair_probability'])
        self.assertIsNone(result[-2]['fair_probability'])

    def test_spreads_pair_opposite_signs(self):
        paired=compare([quote(side='Home',line=-3.5,market='spreads'),quote(side='Away',line=3.5,market='spreads')])
        self.assertEqual(paired[0]['fair_probability'],.5)
        bad=compare([quote(side='Home',line=-3.5,market='spreads'),quote(side='Away',line=-3.5,market='spreads')])
        self.assertIsNone(bad[0]['fair_probability'])

    def test_conflicting_duplicate_cannot_supply_a_pair(self):
        result=compare([quote(),quote(price=-120),quote(side='Under')])
        self.assertTrue(all(r['fair_probability'] is None for r in result))

    def test_watch_requires_three_other_paired_books(self):
        rows=[quote(book=b,side=s) for b in ['a','b','c'] for s in ['Over','Under']]
        self.assertTrue(all(r['consensus_ev'] is None for r in compare(copy.deepcopy(rows))))
        rows.extend([quote(book='outlier',price=120),quote(book='outlier',side='Under',price=-150)])
        outlier=next(r for r in compare(rows) if r['book']=='outlier' and r['side']=='Over')
        self.assertEqual(outlier['other_book_probability'],.5)
        self.assertAlmostEqual(outlier['consensus_ev'],10)

    def test_baseline_excludes_future_duplicate_and_dnp_games_and_handles_push(self):
        history=[dict(GAME_ID=str(i),PLAYER_ID=1,PLAYER_NAME='Test Player',MIN=30,PTS=20,
                      GAME_DATE=(NOW-timedelta(days=i+1)).date().isoformat()) for i in range(25)]
        history.extend([dict(history[0],PTS=999),dict(history[0],GAME_ID='future',GAME_DATE='2026-10-22',PTS=999),dict(history[0],GAME_ID='dnp',MIN=0,PTS=999)])
        rows=add_baselines([quote(line=20)],{'players':history},NOW)
        self.assertEqual(rows[0]['baseline_games'],25)
        self.assertEqual(rows[0]['baseline_mean'],20)
        self.assertEqual(rows[0]['baseline_push'],1)
        self.assertIsNone(rows[0]['model_probability'])

    def test_no_events_is_success_and_does_not_spend_on_props(self):
        class Empty:
            requests=1
            quota_remaining='100'
            def get(self,suffix,**kwargs):
                self.suffix=suffix
                return []
        client=Empty();result=refresh(client,NOW,{})
        self.assertEqual(result['status'],'waiting_for_markets')
        self.assertEqual(client.suffix,'events')

    def test_provider_errors_propagate_not_empty_success(self):
        class Broken:
            def get(self,*args,**kwargs):raise FeedError('HTTP 401')
        with self.assertRaises(FeedError):refresh(Broken(),NOW,{})

    def test_started_and_stale_quotes_are_excluded(self):
        event=dict(id='g',home_team='Home',away_team='Away',commence_time=iso(NOW+timedelta(hours=2)),bookmakers=[dict(key='a',title='A',last_update=iso(NOW-timedelta(hours=25)),markets=[dict(key='totals',outcomes=[dict(name='Over',point=220.5,price=-110)])])])
        self.assertEqual(flatten(event,NOW),[])
        event['bookmakers'][0]['last_update']=iso(NOW)
        self.assertEqual(len(flatten(event,NOW)),1)
        event['commence_time']=iso(NOW-timedelta(minutes=1))
        self.assertEqual(flatten(event,NOW),[])


if __name__=='__main__':unittest.main()
