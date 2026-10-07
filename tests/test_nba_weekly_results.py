"""NBA recap identity, finality, participation and durable replay regressions."""
import copy
from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import unittest

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nba_weekly_results import STATS, fetch_results, get_actual

DAY = date(2026, 10, 21)


def event(identifier='401000001', final=True):
    return dict(id=identifier, date='2026-10-22T02:00:00Z', competitions=[dict(
        id=identifier, date='2026-10-22T02:00:00Z',
        status=dict(type=dict(completed=final, state='post' if final else 'pre')),
        competitors=[dict(homeAway='home', score='115', team=dict(id='13', displayName='Los Angeles Lakers')),
                     dict(homeAway='away', score='108', team=dict(id='12', displayName='LA Clippers'))])])


def summary(game=None):
    return dict(header=copy.deepcopy(game or event()), boxscore=dict(players=[dict(
        team=dict(id='13', displayName='Los Angeles Lakers'), statistics=[dict(
            labels=['MIN', 'PTS', 'REB', 'AST', '3PT', 'BLK', 'STL', 'TO'],
            athletes=[dict(athlete=dict(id='99', displayName='José Player Jr.'),
                           didNotPlay=False, stats=['35', '25', '8', '7', '3-7', '2', '1', '4']),
                      dict(athlete=dict(id='100', displayName='Ben Bench'),
                           didNotPlay=True, stats=['0', '0', '0', '0', '0-0', '0', '0', '0'])])])]))


def row(**changes):
    return dict(dict(event_id='provider-hash', commence_time='2026-10-22T02:00:00Z',
                     home_team='Los Angeles Lakers', away_team='Los Angeles Clippers',
                     player='Jose Player Jr', market='player_points', side='Over', line=24.5,
                     price=-110, model_player_id=999999), **changes)


class NBAWeeklyResultsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.schedule = dict(events=[event()])
        self.box = summary()
        self.calls = []

    def fetch(self, url):
        self.calls.append(url)
        return copy.deepcopy(self.schedule if '/scoreboard?' in url else self.box)

    def results(self, rows=None, offline=False):
        return fetch_results(DAY, DAY, rows if rows is not None else [row()],
                             self.root, self.fetch, offline)

    def test_game_resolution_uses_eastern_date_and_preserves_provider_identity(self):
        games, players, rows = self.results()
        self.assertEqual(rows[0]['event_id'], 'provider-hash')
        self.assertEqual(rows[0]['nba_game_id'], '401000001')
        self.assertEqual(rows[0]['resolution_status'], 'resolved')
        self.assertIn('dates=20261021', self.calls[0])
        self.assertEqual(get_actual(rows[0], games, players), (25, 'graded'))
        # A saved NBA Stats person ID cannot be confused with an ESPN athlete ID.
        self.assertEqual(rows[0]['model_player_id'], 999999)

    def test_all_supported_markets_and_team_sides(self):
        expected = dict(zip(STATS, [25, 8, 7, 3, 40, 33, 32, 15, 2, 1, 4]))
        rows = [row(market=market) for market in expected]
        rows += [row(market='totals'), row(market='spreads', side='Los Angeles Clippers'),
                 row(market='h2h', side='Los Angeles Lakers')]
        games, players, resolved = self.results(rows)
        self.assertEqual([get_actual(r, games, players)[0] for r in resolved],
                         list(expected.values()) + [223, -7, 7])

    def test_offline_replay_is_identical_and_makes_no_requests(self):
        live = self.results()
        self.calls.clear()
        self.assertEqual(self.results(offline=True), live)
        self.assertEqual(self.calls, [])
        files = list((self.root/'artifacts/nba/results').glob('*.json'))
        self.assertEqual(len(files), 2)
        self.assertEqual(json.loads(files[0].read_text())['source'], 'ESPN')

    def test_empty_archive_or_missing_offline_evidence_is_not_a_recap(self):
        self.assertEqual(self.results([]), ({}, {}, []))
        self.assertEqual(self.calls, [])
        games, players, rows = self.results(offline=True)
        self.assertFalse(games or players)
        self.assertEqual(get_actual(rows[0], games, players), (None, 'official_matchup_unavailable'))

    def test_wrong_date_swapped_teams_and_ambiguous_matchup_stay_unresolved(self):
        games, players, rows = self.results([
            row(commence_time='2026-10-21T02:00:00Z'),
            row(home_team='Los Angeles Clippers', away_team='Los Angeles Lakers')])
        self.assertFalse(games or players)
        self.assertTrue(all(r['resolution_status'] == 'official_matchup_unavailable' for r in rows))
        self.schedule['events'].append(event('401000002'))
        self.assertEqual(self.results()[2][0]['resolution_status'], 'official_matchup_ambiguous')

    def test_incorrect_given_event_id_is_not_silently_reassigned(self):
        games, players, rows = self.results([row(nba_game_id='bad-id')])
        self.assertFalse(games or players)
        self.assertNotIn('nba_game_id', rows[0])
        self.assertEqual(rows[0]['resolution_status'], 'official_id_mismatch')

    def test_nonfinal_scoreboard_does_not_grade_even_with_scores(self):
        self.schedule = dict(events=[event(final=False)])
        games, players, rows = self.results()
        self.assertFalse(games or players)
        self.assertEqual(get_actual(rows[0], games, players), (None, 'official_result_unavailable'))
        self.assertEqual(len(self.calls), 1)

    def test_dnp_and_zero_minutes_cannot_create_winning_unders(self):
        games, players, rows = self.results([row(player='Ben Bench', side='Under')])
        self.assertEqual(get_actual(rows[0], games, players), (None, 'participation_unconfirmed'))
        athlete = self.box['boxscore']['players'][0]['statistics'][0]['athletes'][0]
        athlete['stats'][0] = '0:00'
        games, players, rows = self.results()
        self.assertEqual(get_actual(rows[0], games, players), (None, 'participation_unconfirmed'))
        athlete['stats'][0] = '0:05'
        athlete['stats'][1] = '0'
        games, players, rows = self.results()
        self.assertEqual(get_actual(rows[0], games, players), (0, 'graded'))

    def test_missing_stats_and_duplicate_names_are_unresolved(self):
        group = self.box['boxscore']['players'][0]['statistics'][0]
        group['athletes'][0]['stats'][3] = '--'
        games, players, rows = self.results([row(market='player_points_rebounds_assists')])
        self.assertEqual(get_actual(rows[0], games, players), (None, 'stat_or_market_unavailable'))
        duplicate = copy.deepcopy(group['athletes'][0])
        duplicate['athlete']['id'] = '101'
        group['athletes'].append(duplicate)
        games, players, rows = self.results()
        self.assertEqual(get_actual(rows[0], games, players), (None, 'player_identity_ambiguous'))

    def test_wrong_boxscore_does_not_contaminate_results(self):
        self.box['header']['id'] = 'wrong-game'
        games, players, rows = self.results([row(), row(market='totals')])
        self.assertFalse(players)
        self.assertEqual(get_actual(rows[0], games, players), (None, 'official_player_results_unavailable'))
        self.assertEqual(get_actual(rows[1], games, players), (223, 'graded'))

    def test_summary_failure_retains_verified_cache_but_schedule_failure_propagates(self):
        original = self.results()
        def failed_box(url):
            if '/summary?' in url:
                raise requests.Timeout('feed unavailable')
            return self.schedule
        cached = fetch_results(DAY, DAY, [row()], self.root, failed_box)
        self.assertEqual(cached, original)
        with self.assertRaises(requests.Timeout):
            fetch_results(DAY, DAY, [row()], self.root,
                          lambda _: (_ for _ in ()).throw(requests.Timeout()))


if __name__ == '__main__':
    unittest.main()
