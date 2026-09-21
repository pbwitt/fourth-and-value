import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nhl.refresh import (FeedError, SPORT, PROPS, MARKETS, baselines, compare, flatten,
                         historical_rates, iso, refresh, regular_events, schedule, season_for, stats_rows)
from validate_nhl_freshness import validate

NOW = datetime(2026, 9, 21, 14, tzinfo=timezone.utc)
GAME = dict(nhl_game_id=2026020001, season=20262027, game_type=2,
            commence_time='2026-09-29T21:00:00Z', home_team='Carolina Hurricanes', away_team='Florida Panthers')
EVENT = dict(id='event1', sport_key=SPORT, commence_time=GAME['commence_time'],
             home_team=GAME['home_team'], away_team=GAME['away_team'])


def history():
    return dict(fetched_at=iso(NOW), seasons={'20252026': {'players': [dict(playerId=1,
        skaterFullName='Test Player', gamesPlayed=60, shots=180, goals=30, assists=30, points=60)],
        'teams': []}, '20262027': {'players': [], 'teams': []}})


class NHLRefreshTests(unittest.TestCase):
    def test_september_rollover_and_midwinter_season(self):
        self.assertEqual(season_for(NOW), 20262027)
        self.assertEqual(season_for(NOW.replace(month=1)), 20252026)
        self.assertEqual(season_for(NOW.replace(month=8)), 20252026)

    def test_only_unambiguous_matching_regular_events(self):
        self.assertEqual(len(regular_events([EVENT], [GAME])), 1)
        for changed in [dict(home_team='Wrong'), dict(commence_time='2026-09-22T21:00:00Z')]:
            self.assertEqual(regular_events([{**EVENT, **changed}], [GAME]), [])
        self.assertEqual(regular_events([EVENT], [GAME, GAME]), [])

    def test_schedule_excludes_preseason_playoffs_past_and_postponed(self):
        team = lambda place, name: dict(placeName={'default': place}, commonName={'default': name})
        game = dict(id=1, season=20262027, gameType=2, gameState='FUT', gameScheduleState='OK',
                    startTimeUTC=GAME['commence_time'], homeTeam=team('Carolina', 'Hurricanes'),
                    awayTeam=team('Florida', 'Panthers'))
        invalid = [dict(gameType=1), dict(gameType=3), dict(gameScheduleState='PPD'),
                   dict(season=20252026), dict(startTimeUTC='2026-09-20T21:00:00Z'), dict(gameState='LIVE')]
        response = {'gameWeek': [{'games': [game] + [{**game, **v, 'id': i+2} for i,v in enumerate(invalid)]}]}
        with patch('nhl.refresh.official_json', return_value=response):
            result = schedule(NOW)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['nhl_game_id'], 1)

    def test_nhl_shots_pair_devig_and_exact_line(self):
        market = dict(key='player_shots_on_goal', last_update=iso(NOW), outcomes=[
            dict(name='Over', description='Test Player', point=2.5, price=-110),
            dict(name='Under', description='Test Player', point=2.5, price=-110)])
        event = {**EVENT, 'bookmakers': [dict(key='a', title='A', markets=[market])]}
        rows = compare(flatten(event, NOW, SPORT, MARKETS, list(PROPS)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['fair_probability'], .5)
        market['outcomes'][1]['point'] = 3.5
        rows = compare(flatten(event, NOW, SPORT, MARKETS, list(PROPS)))
        self.assertTrue(all(r['fair_probability'] is None for r in rows))

    def test_poisson_integer_under_excludes_push(self):
        rows = [dict(market='player_goals', player='Test Player', side=s, line=1.0) for s in ['Over', 'Under']]
        rows = baselines(rows, history(), NOW)
        self.assertAlmostEqual(sum(r['baseline_probability'] for r in rows) + rows[0]['baseline_push'], 1)
        self.assertTrue(all(r['model_probability'] is None for r in rows))

    def test_current_season_threshold_and_prior_fallback(self):
        h = history()
        h['seasons']['20262027']['players'] = [dict(playerId=1, skaterFullName='Test Player', gamesPlayed=19, shots=76)]
        self.assertEqual(historical_rates(h, 'players', NOW)['testplayer'][1], '20252026')
        h['seasons']['20262027']['players'][0]['gamesPlayed'] = 20
        self.assertEqual(historical_rates(h, 'players', NOW)['testplayer'][1], '20262027')

    def test_stale_error_ambiguous_and_small_history_withheld(self):
        for h in [dict(history(), fetched_at=iso(NOW-timedelta(hours=37))), dict(history(), error='unavailable')]:
            self.assertEqual(historical_rates(h, 'players', NOW), {})
        h = history()
        h['seasons']['20252026']['players'].append(dict(h['seasons']['20252026']['players'][0], playerId=2))
        self.assertEqual(historical_rates(h, 'players', NOW), {})
        h = history();h['seasons']['20252026']['players'][0]['gamesPlayed'] = 19
        self.assertEqual(historical_rates(h, 'players', NOW), {})

    def test_totals_use_same_reference_season_and_no_invented_probability(self):
        h = history()
        h['seasons']['20252026']['teams'] = [dict(teamId=i, teamFullName=n, gamesPlayed=82,
            goalsForPerGame=3.0, goalsAgainstPerGame=2.5) for i,n in enumerate([GAME['home_team'], GAME['away_team']])]
        row = dict(player='', market='totals', home_team=GAME['home_team'], away_team=GAME['away_team'])
        result = baselines([row], h, NOW)[0]
        self.assertEqual(result['baseline_mean'], 5.5)
        self.assertIsNone(result['model_probability'])
        self.assertIsNone(result['baseline_probability'])

    def test_empty_or_preseason_odds_do_not_request_paid_markets(self):
        class Client:
            requests = 1
            quota_remaining = None
            def get(self, suffix, **params):
                assert suffix == 'events'
                return [{**EVENT, 'commence_time': '2026-09-22T21:00:00Z'}]
        state = refresh(Client(), NOW, [GAME], {})
        self.assertEqual(state['status'], 'waiting_for_markets')
        self.assertEqual(state['rows'], [])
        self.assertEqual(state['excluded_events'], 1)

    def test_stats_paginate_and_exclude_today(self):
        with patch('nhl.refresh.official_json', side_effect=[{'data': [{}]*100, 'total': 101}, {'data': [{}], 'total': 101}]) as get:
            self.assertEqual(len(stats_rows('skater', 20252026, NOW)), 101)
            self.assertIn('gameDate<"2026-09-21"', get.call_args.kwargs['cayenneExp'])
            self.assertEqual(get.call_args.kwargs['start'], 100)

    def test_partial_stats_and_schedule_fail_closed(self):
        with patch('nhl.refresh.official_json', return_value={'data': [], 'total': 100}):
            with self.assertRaises(FeedError): stats_rows('skater', 20252026, NOW)
        with patch('nhl.refresh.official_json', return_value={}):
            with self.assertRaises(FeedError): schedule(NOW)

    def test_fresh_empty_state_passes_but_old_content_fails(self):
        state = dict(status='waiting_for_markets', last_success_at=iso(NOW), season=20262027,
                     events=[GAME], rows=[])
        self.assertEqual(validate(state, NOW), [])
        self.assertTrue(validate({**state, 'last_success_at': '2025-12-29T10:00:00Z'}, NOW))
        state['rows'] = [dict(quoted_at=iso(NOW), commence_time='2025-12-30T10:00:00Z')]
        self.assertTrue(validate(state, NOW))


if __name__ == '__main__':
    unittest.main()
