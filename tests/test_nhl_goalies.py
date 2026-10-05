from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nhl.v2 import goalies
from nhl.v2.goalies import attach, describe, load, normalize, project, save_rate

NOW = datetime(2026, 1, 20, 15, tzinfo=timezone.utc)


def game(gid, day, home=1, away=2, season=20252026):
    available = (datetime.fromisoformat(day).replace(tzinfo=timezone.utc) + timedelta(days=1, hours=12)).isoformat()
    return dict(game_id=gid, season=season, game_date=day, available_at=available, home_id=home, away_id=away)


def report(gid, pid, home, started, shots=30, goals=3, name=None):
    return dict(gameId=gid, playerId=pid, goalieFullName=name or f'Goalie {pid}', homeRoad='H' if home else 'R',
                gamesStarted=int(started), shotsAgainst=shots, goalsAgainst=goals)


def season(days, starters, team=1, opponent=2):
    """Team 1 at home every game; `starters[i]` starts game i, the other goalie sits."""
    games, rows = [], []
    for i, (day, starter) in enumerate(zip(days, starters)):
        games.append(game(100 + i, day, team, opponent))
        rows.append(report(100 + i, starter, True, True))
        rows.append(report(100 + i, 30 if starter == 31 else 31, True, False, shots=0, goals=0))
        rows.append(report(100 + i, 40, False, True))
    return games, normalize(rows, games)


class NormalizeTests(unittest.TestCase):
    def test_joins_team_ids_and_rejects_bad_reports(self):
        games = [game(1, '2026-01-10', home=7, away=9)]
        out = normalize([report(1, 30, True, True), report(1, 40, False, True, shots=25, goals=2)], games)
        self.assertEqual([(a['player_id'], a['team_id'], a['started']) for a in out], [(30, 7, True), (40, 9, True)])
        self.assertEqual(out[0]['available_at'], games[0]['available_at'], 'box scores count from the next morning')
        with self.assertRaisesRegex(ValueError, 'schema'):
            normalize([dict(gameId=1, playerId=30)], games)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            normalize([report(1, 30, True, True)] * 2, games)
        with self.assertRaisesRegex(ValueError, 'More than one starting goalie'):
            normalize([report(1, 30, True, True), report(1, 31, True, True)], games)
        self.assertEqual(normalize([report(2, 30, True, True)], games), [], 'games not yet in the team history wait')


class ProjectionTests(unittest.TestCase):
    def test_recent_starts_weigh_most_and_only_pre_decision_games_count(self):
        days = [f'2026-01-{d:02}' for d in range(1, 17, 2)]  # 8 games, two days apart
        _, apps = season(days, [30, 30, 30, 30, 31, 31, 31, 31])
        p = project(apps, 1, '2026-01-19', NOW)
        self.assertEqual([g['player_id'] for g in p['goalies']], [31, 30], 'the recent starter leads')
        self.assertAlmostEqual(sum(g['start_probability'] for g in p['goalies']), 1, places=3)
        self.assertGreater(p['goalies'][0]['start_probability'], .5)
        self.assertFalse(p['confirmed'])
        self.assertIn('Not a confirmed starter', p['basis'])
        early = project(apps, 1, '2026-01-19', datetime(2026, 1, 8, 15, tzinfo=timezone.utc))
        self.assertEqual([g['player_id'] for g in early['goalies']], [30], 'later box scores are not known yet')
        self.assertIsNone(project(apps, 99, '2026-01-19', NOW))

    def test_back_to_back_discounts_last_nights_starter(self):
        days = [f'2026-01-{d:02}' for d in range(1, 19, 2)]
        _, apps = season(days, [30] * 8 + [31])
        rested = project(apps, 1, '2026-01-19', NOW)
        b2b = project(apps, 1, '2026-01-18', datetime(2026, 1, 18, 15, tzinfo=timezone.utc))
        self.assertTrue(b2b['back_to_back'])
        self.assertFalse(rested['back_to_back'])
        share = {g['player_id']: g['start_probability'] for g in b2b['goalies']}
        self.assertGreater(share[30], share[31], "yesterday's starter is discounted on the second night")

    def test_current_roster_drops_departed_goalies(self):
        days = [f'2026-01-{d:02}' for d in range(1, 17, 2)]
        _, apps = season(days, [30] * 6 + [31] * 2)
        p = project(apps, 1, '2026-01-19', NOW, current={31})
        self.assertEqual([(g['player_id'], g['start_probability']) for g in p['goalies']], [(31, 1.0)])
        self.assertIsNone(project(apps, 1, '2026-01-19', NOW, current={55}))

    def test_last_season_is_labeled_without_a_roster(self):
        old = [game(1, '2025-04-10', season=20242025)]
        new = [game(2, '2025-10-10')]
        apps = normalize([report(1, 30, True, True)], old) + normalize([report(2, 31, True, True)], new)
        self.assertIn('offseason moves are not reflected', project(apps, 1, '2025-10-20', NOW)['basis'])
        rostered = project(apps, 1, '2025-10-20', NOW, current={30, 31})['basis']
        self.assertNotIn('offseason', rostered)
        self.assertIn('current roster', rostered)

    def test_rosters_read_official_goalie_ids_and_skip_failures(self):
        events = [dict(nhl_game_id=7, commence_time='2026-01-20T23:00:00Z', home_id=1, away_id=2,
                       home_abbrev='BOS', away_abbrev='TOR'),
                  dict(nhl_game_id=8, commence_time='2026-02-20T23:00:00Z', home_id=3, away_id=4,
                       home_abbrev='MTL', away_abbrev='OTT')]

        class Response:
            def __init__(self, url):
                self.url = url
            def raise_for_status(self):
                if 'TOR' in self.url:
                    import requests
                    raise requests.HTTPError('503')
            def json(self):
                return dict(goalies=[dict(id=30), dict(id='31')], forwards=[dict(id=99)])
        with patch('requests.get', side_effect=lambda url, timeout: Response(url)) as get:
            out = goalies.rosters(events, NOW)
        self.assertEqual(out, {1: {30, 31}}, 'a failed team is left out, never guessed')
        self.assertEqual(get.call_count, 2, 'only teams playing within 48 hours are fetched')
        events.insert(0, dict(events[0], nhl_game_id=6, home_id=5, away_id=6, home_abbrev='TOR', away_abbrev='NYR'))
        with patch('requests.get', side_effect=lambda url, timeout: Response(url)) as get:
            self.assertEqual(goalies.rosters(events, NOW), {})
        self.assertEqual(get.call_count, 1, 'a network failure stops further requests')

    def test_save_rate_is_shrunk_toward_league(self):
        hot = [dict(shots_against=100, goals_against=2)]
        rate, shots = save_rate(hot)
        self.assertEqual(shots, 100)
        self.assertLess(rate, .95, 'a hot week barely moves the estimate')
        self.assertGreater(rate, goalies.LEAGUE_SAVE_PCT)
        self.assertEqual(save_rate([])[0], goalies.LEAGUE_SAVE_PCT)


class AttachTests(unittest.TestCase):
    def test_skaters_get_the_opposing_goalie_and_game_lines_get_both(self):
        days = [f'2026-01-{d:02}' for d in range(1, 17, 2)]
        _, apps = season(days, [30] * 8)
        state = dict(events=[dict(nhl_game_id=7, commence_time='2026-01-20T00:00:00Z', home_id=1, away_id=2,
                                  home_team='Boston Bruins', away_team='Toronto Maple Leafs')],
                     rows=[dict(nhl_game_id=7, player='Skater', player_team_id=2, market='player_shots_on_goal'),
                           dict(nhl_game_id=7, player='', market='totals'),
                           dict(nhl_game_id=8, player='', market='totals', goalie_assumption='unchanged'),
                           dict(nhl_game_id=9, player='', market='totals', goalie_assumption='unchanged')])
        state['events'].append(dict(nhl_game_id=9, commence_time='2026-01-25T00:00:00Z', home_id=1, away_id=2))
        out = attach(state, apps, NOW)
        p = out['goalie_projections']['7']
        self.assertEqual(p['home']['goalies'][0]['player_id'], 30)
        self.assertEqual(p['as_of'], '2026-01-20T15:00:00Z')
        skater, total, other, later = out['rows']
        self.assertNotIn('9', out['goalie_projections'], 'games beyond the 48-hour forecast window wait')
        self.assertEqual(later['goalie_assumption'], 'unchanged')
        self.assertTrue(skater['goalie_assumption'].startswith('Projected, not confirmed. Opposing goalie, Boston Bruins: likely Goalie 30'))
        self.assertIn('Boston Bruins', total['goalie_assumption'])
        self.assertIn('Toronto Maple Leafs: likely Goalie 40', total['goalie_assumption'])
        self.assertEqual(other['goalie_assumption'], 'unchanged', 'rows without an official game keep their text')
        self.assertEqual(describe(None, 'Boston Bruins'), 'Boston Bruins: goalie unknown.')
        self.assertLess(len(total['goalie_assumption']), 200, 'reviewer requests are byte-capped')

    def test_enrich_keeps_forecasts_when_goalie_history_fails(self):
        from nhl.v2.inference import enrich
        state = dict(events=[], rows=[])
        with patch('nhl.v2.inference.bundle', return_value=({}, dict(trained_through='2025-04-16'))), \
             patch('nhl.v2.inference.annotate', side_effect=lambda rows, *a, **k: rows), \
             patch('nhl.v2.goalies.live', side_effect=RuntimeError('down')), \
             patch('nhl.v2.inference.live_history', return_value=([], [], NOW.isoformat())):
            out = enrich(state, NOW)
        self.assertIsNone(out['model_error'])
        self.assertIsNone(out['goalie_projections'])
        self.assertIn('RuntimeError', out['goalie_error'])
        with patch('nhl.v2.inference.bundle', return_value=({}, dict(trained_through='2025-04-16'))), \
             patch('nhl.v2.inference.annotate', side_effect=lambda rows, *a, **k: rows), \
             patch('nhl.v2.goalies.live') as fetch:
            out = enrich(dict(events=[], rows=[]), NOW, offline_inputs=([], [], NOW.isoformat()))
        fetch.assert_not_called()
        self.assertIsNone(out['goalie_error'])


class StorageTests(unittest.TestCase):
    def test_collect_writes_checksummed_history_and_live_refreshes_twice_a_day(self):
        games = [game(1, '2026-01-10', season=20252026), game(2, '2025-03-01', season=20242025)]
        pages = {20252026: [report(1, 30, True, True)], 20242025: [report(2, 31, True, True)]}
        calls = []

        def fetch(kind, season, root, through=None, after=None):
            calls.append((kind, season, through))
            return pages[season], [dict(path=f'raw/{season}.json')]
        with tempfile.TemporaryDirectory() as root, patch('nhl.v2.data.fetch_report', side_effect=fetch):
            apps = goalies.live(root, games, NOW, 20252026)
            self.assertEqual(sorted(a['player_id'] for a in apps), [30, 31])
            self.assertEqual(calls, [('goalie', 20242025, None), ('goalie', 20252026, '2026-01-20')])
            goalies.live(root, games, NOW + timedelta(hours=6), 20252026)
            self.assertEqual(len(calls), 2, 'a fresh check is reused')
            goalies.live(root, games, NOW + timedelta(hours=13), 20252026)
            self.assertEqual(len(calls), 3)
            path = Path(root) / 'goalies' / '20252026.json'
            data = json.loads(path.read_text())
            data['appearances'][0]['shots_against'] = 99
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load(root, [20252026])
            pages[20252026] = [report(9, 30, True, True)]
            with self.assertRaisesRegex(ValueError, 'did not join'):
                goalies.collect(20252026, root, games)


if __name__ == '__main__':
    unittest.main()
