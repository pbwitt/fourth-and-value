"""Completed results must reach morning forecasts without changing historical evidence."""
import copy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nhl.v2.data import digest, history_day, iso, normalize, observed_history, write_json
from nhl.v2.features import build, forecast_ledger, history_at
from nhl.v2.inference import live_history
from nhl.v2 import goalies
from nhl.refresh import load_history, stats_rows

NOW = datetime(2026, 10, 10, 11, 6, 40, tzinfo=timezone.utc)
SEASON = 20262027


def history():
    teams, players = [], []
    for gid, day, toi, shots in [(1, '2026-10-06', 1200, 4), (2, '2026-10-09', 600, 0)]:
        for home in (True, False):
            tid = 1 if home else 2
            teams.append(dict(gameId=gid, gameDate=day, homeRoad='H' if home else 'R',
                teamId=tid, teamFullName=f'Team {tid}', goalsFor=3 if home else 2,
                goalsAgainst=2 if home else 3, shotsForPerGame=30, wins=int(home),
                winsInRegulation=int(home), winsInShootout=0))
            players.append(dict(gameId=gid, playerId=tid, skaterFullName=f'Player {tid}',
                positionCode='C', timeOnIcePerGame=toi, homeRoad='H' if home else 'R',
                teamAbbrev=f'T{tid}', shots=shots, goals=0, assists=0, points=0))
    return normalize(teams, players, SEASON)


class LiveHistoryTests(unittest.TestCase):
    def test_morning_uses_yesterday_for_shots_ice_time_team_form_and_ledger(self):
        games, players = history()
        original = copy.deepcopy((games, players))
        manifests = [dict(season=SEASON, ingested_at=iso(NOW-timedelta(seconds=1)))]
        live_games, live_players = observed_history(games, players, manifests, NOW)
        old = history_at(games, players, NOW)
        fixed = history_at(live_games, live_players, NOW)
        before = old.player_features(1, 'C', '2026-10-10', NOW)
        after = fixed.player_features(1, 'C', '2026-10-10', NOW)
        self.assertEqual(before['last_game'], '2026-10-06')
        self.assertEqual(after['last_game'], '2026-10-09')
        self.assertEqual(after['history_games'], 2)
        self.assertLess(after['projected_toi'], before['projected_toi'])
        self.assertLess(after['opportunity_means'][0], before['opportunity_means'][0])
        self.assertEqual(fixed.teams[1][-1]['game_date'], '2026-10-09')
        # The second game contributes its outcome and its original pre-game expectation.
        ledger = forecast_ledger(live_games, live_players, NOW)
        expected = build(games, players)[1]
        mine = [r for r in expected if r['player_id'] == 1]
        self.assertAlmostEqual(ledger[1][4], sum(r['opportunity_means'][0] for r in mine))
        self.assertEqual((games, players), original, 'stored/backtest timestamps stay unchanged')
        self.assertEqual(live_games[-1]['availability_basis'], 'observed_final_report')
        self.assertIsNone(live_games[-1]['source_published_at'])
        self.assertEqual(live_games[-1]['reconstructed_available_at'], '2026-10-10T12:00:00Z')

    def test_neither_future_observations_nor_current_date_games_are_admitted(self):
        games, players = history()
        with self.assertRaisesRegex(ValueError, 'after the live decision'):
            observed_history(games, players, [dict(season=SEASON, ingested_at=iso(NOW+timedelta(seconds=1)))], NOW)
        games[-1]['game_date'] = '2026-10-10'
        with self.assertRaisesRegex(ValueError, 'earlier completed dates'):
            observed_history(games, players, [dict(season=SEASON, ingested_at=iso(NOW))], NOW)

    def test_evening_utc_boundary_and_winter_morning(self):
        self.assertEqual(history_day(datetime(2026, 10, 10, 1, tzinfo=timezone.utc)), '2026-10-09')
        # 6:05 a.m. Eastern in winter is also before the old noon UTC gate.
        games, players = history()
        for row in games + players:
            row['game_date'] = '2027-01-09'
            row['available_at'] = '2027-01-10T12:00:00Z'
        now = datetime(2027, 1, 10, 11, 5, tzinfo=timezone.utc)
        g, p = observed_history(games, players, [dict(season=SEASON, ingested_at=iso(now))], now)
        self.assertEqual(len(history_at(g, p, now).players[1]), 2)

    def test_live_fetch_refreshes_across_midnight_and_for_each_market_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = root/'models'
            model.mkdir()
            archive = model/'history.json.gz'
            with gzip.open(archive, 'wt') as f:
                json.dump(dict(games=[], players=[]), f)
            write_json(model/'manifest.json', dict(history_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
            directory = root/'data/nhl/v2/history'
            # Only eight hours old, but checked before the Eastern date changed.
            write_json(directory/'live-check.json', dict(checked_at=iso(NOW-timedelta(hours=8)),
                       season=SEASON, through='2026-10-09'))

            def collect(seasons, target, through):
                games, players = history()
                write_json(target/f'{SEASON}.json', dict(games=games, players=players,
                    manifest=dict(season=SEASON, through=through, ingested_at=iso(NOW),
                                  data_sha256=digest([games, players]))))

            with patch('nhl.v2.inference.ROOT', root), patch('nhl.v2.inference.MODEL_DIR', model), \
                 patch('nhl.v2.data.collect', side_effect=collect) as fetch, \
                 patch('nhl.v2.inference.datetime', wraps=datetime) as clock:
                clock.now.return_value = NOW
                games, players, checked = live_history(NOW)
                fetch.assert_called_once_with([SEASON], directory, '2026-10-10')
                self.assertEqual(history_at(games, players, NOW).players[1][-1]['game_date'], '2026-10-09')
                live_history(NOW+timedelta(minutes=1))
                self.assertEqual(fetch.call_count, 1, 'same-run sidecars reuse the checked history')
                live_history(NOW+timedelta(minutes=2), refresh=True)
                self.assertEqual(fetch.call_count, 2, 'a new board refresh checks results again')
                with patch('nhl.v2.data.collect', side_effect=RuntimeError('unavailable')):
                    with self.assertRaises(RuntimeError):
                        live_history(NOW, refresh=True)

    def test_goalie_context_counts_yesterday_before_old_cutoff(self):
        games, players = history()
        games, _ = observed_history(games, players, [dict(season=SEASON, ingested_at=iso(NOW))], NOW)
        rows = [dict(gameId=g['game_id'], playerId=30, goalieFullName='Goalie', homeRoad='H',
                     gamesStarted=1, shotsAgainst=30, goalsAgainst=2) for g in games]
        projection = goalies.project(goalies.normalize(rows, games), 1, '2026-10-10', NOW)
        self.assertTrue(projection['back_to_back'])
        self.assertEqual(projection['goalies'][0]['last_start'], '2026-10-09')

    def test_summary_cache_also_respects_completed_date_and_season(self):
        saved = dict(fetched_at=iso(NOW-timedelta(hours=8)), current_season=SEASON, through_date='2026-10-08')
        with patch('nhl.refresh.read_json', return_value=saved), patch('nhl.refresh.save_json'), \
             patch('nhl.refresh.stats_rows', return_value=[]) as fetch:
            self.assertEqual(load_history(NOW)['through_date'], '2026-10-09')
            self.assertEqual(fetch.call_count, 4)
        with patch('nhl.refresh.official_json', return_value=dict(data=[], total=0)) as fetch:
            stats_rows('team', SEASON, NOW.replace(hour=1))
            self.assertIn('gameDate<"2026-10-09"', fetch.call_args.kwargs['cayenneExp'])

    def test_goalie_cache_cannot_cross_to_next_date_under_twelve_hours(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for season in (SEASON-10001, SEASON):
                write_json(root/'goalies'/f'{season}.json',
                           dict(appearances=[],manifest=dict(data_sha256=digest([]))))
            write_json(root/'goalies/live-check.json', dict(season=SEASON,
                       checked_at=iso(NOW-timedelta(hours=8)),through='2026-10-09'))
            with patch('nhl.v2.goalies.collect') as fetch:
                goalies.live(root, [], NOW, SEASON)
                fetch.assert_called_once_with(SEASON, root, [], '2026-10-10')


if __name__ == '__main__':
    unittest.main()
