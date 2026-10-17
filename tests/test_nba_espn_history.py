"""ESPN history: season types, DNP and missing-stat handling, stable rewrites, loader filters."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nba import espn_history as h

KEYS = ['minutes', 'points', 'fieldGoalsMade-fieldGoalsAttempted',
        'threePointFieldGoalsMade-threePointFieldGoalsAttempted', 'rebounds', 'assists', 'turnovers']


def competitor(side, tid, name, abbr, score):
    return dict(homeAway=side, score=str(score), linescores=[{}] * 4,
                team=dict(id=tid, displayName=name, abbreviation=abbr))


def fixture(kind=2, competition_type='STD', home_id='2', away_id='18'):
    competition = dict(id='9', date='2025-01-16T00:00Z', type=dict(abbreviation=competition_type),
                       status=dict(type=dict(completed=True, state='post')),
                       competitors=[competitor('home', home_id, 'Boston Celtics', 'BOS', 110),
                                    competitor('away', away_id, 'New York Knicks', 'NY', 100)])
    event = dict(id='9', season=dict(type=kind), competitions=[competition])
    athletes = [
        dict(athlete=dict(id='1', displayName='Starter One', position=dict(abbreviation='G')), starter=True,
             didNotPlay=False, stats=['36', '25', '9-18', '3-7', '5', '6', '2']),
        dict(athlete=dict(id='2', displayName='Bench Two'), didNotPlay=True, reason="COACH'S DECISION", stats=[]),
        dict(athlete=dict(id='3', displayName='Odd Three'), didNotPlay=False, stats=['20', '--', '2-5', '0-1', '4', '1', '0']),
        dict(athlete=dict(id='4', displayName='Inactive Four'), active=False, didNotPlay=False,
             stats=['--', '0', '0-0', '0-0', '0', '0', '0']),
    ]
    summary = dict(header=dict(id='9', competitions=[copy.deepcopy(competition)]),
                   boxscore=dict(teams=[dict(team=dict(id=home_id), statistics=[
                       dict(name='fieldGoalsMade-fieldGoalsAttempted', displayValue='40-85'),
                       dict(name='totalTurnovers', displayValue='14')])],
                       players=[dict(team=dict(id=home_id), statistics=[dict(keys=KEYS, athletes=athletes)])]))
    return event, summary


class EspnHistory(unittest.TestCase):
    def test_parse_keeps_dnp_without_zeros_and_missing_stats_absent(self):
        game, teams, players = h.parse_game(*fixture())
        self.assertEqual((game['SEASON_TYPE'], game['GAME_DATE'], game['PERIODS']), ('regular', '2025-01-15', 4))
        home = next(t for t in teams if t['HOME'])
        self.assertEqual((home['MATCHUP'], home['PTS'], home['OPP_PTS'], home['FGA'], home['TOV'], home['WL']),
                         ('BOS vs. NY', 110, 100, 85, 14, 'W'))
        self.assertNotIn('FTA', home, 'a stat ESPN did not report stays absent')
        one, two, three, four = players
        self.assertTrue(four['DNP'])
        self.assertEqual(four['DNP_REASON'], 'INACTIVE')
        self.assertNotIn('PTS', four, 'inactive filler zeros are not a stat line')
        self.assertEqual((one['MIN'], one['PTS'], one['FG3M'], one['FG3A'], one['POSITION'], one['STARTER']),
                         (36.0, 25, 3, 7, 'G', True))
        self.assertTrue(two['DNP'])
        self.assertNotIn('MIN', two)
        self.assertNotIn('PTS', two)
        self.assertEqual(two['DNP_REASON'], "COACH'S DECISION")
        self.assertNotIn('PTS', three, 'an unreadable value is not zero')
        self.assertEqual(three['REB'], 4)

    def test_season_types_and_exhibitions(self):
        self.assertEqual(h.parse_game(*fixture(competition_type='CC'))[0]['SEASON_TYPE'], 'cup_final')
        self.assertEqual(h.parse_game(*fixture(kind=5))[0]['SEASON_TYPE'], 'play_in')
        self.assertEqual(h.parse_game(*fixture(kind=3))[0]['SEASON_TYPE'], 'playoffs')
        self.assertIsNone(h.parse_game(*fixture(kind=1)), 'preseason is dropped')
        self.assertIsNone(h.parse_game(*fixture(home_id='31', away_id='32')), 'All-Star teams are not franchises')

    def test_summary_must_confirm_the_final_score(self):
        event, summary = fixture()
        summary['header']['competitions'][0]['competitors'][0]['score'] = '111'
        self.assertIsNone(h.parse_game(event, summary))

    def test_duplicate_athlete_block_is_dropped(self):
        event, summary = fixture()
        group = summary['boxscore']['players'][0]['statistics'][0]
        group['athletes'].append(copy.deepcopy(group['athletes'][0]))
        self.assertNotIn('1', [p['PLAYER_ID'] for p in h.parse_game(event, summary)[2]])

    def test_season_dates(self):
        self.assertEqual(h.season_of(h.date(2026, 10, 20)), '2026-27')
        self.assertEqual(h.season_of(h.date(2026, 6, 10)), '2025-26')
        self.assertIsNone(h.season_of(h.date(2026, 8, 1)))
        with self.assertRaises(ValueError):
            h.season_start('2025-27')

    def test_write_is_stable_and_loader_filters_types(self):
        with tempfile.TemporaryDirectory() as root, patch.object(h, 'HISTORY', Path(root)), patch.object(h, 'ROOT', Path(root)):
            regular, cup = h.parse_game(*fixture()), h.parse_game(*fixture(competition_type='CC'))
            cup[0]['GAME_ID'] = '10'
            for row in cup[1] + cup[2]:
                row['GAME_ID'] = '10'
            first = h.write('2024-25', {'9': regular, '10': cup}, ['11'], [])
            before = h.path_for('2024-25').read_bytes()
            again = h.write('2024-25', {}, [], [], previous=first)
            self.assertIs(again, first, 'no new games: file is not rewritten')
            self.assertEqual(h.path_for('2024-25').read_bytes(), before)
            with gzip.open(h.path_for('2024-25'), 'rt') as handle:
                self.assertEqual(json.load(handle)['missing_games'], ['11'])
            loaded = h.load_all()
            self.assertEqual({r['GAME_ID'] for r in loaded['players']}, {'9'}, 'Cup final is not regular season')
            self.assertEqual(len(h.load_all(season_types=None)['games']), 2)
            manifest = h.write_manifest()
            self.assertEqual(manifest['seasons']['2024-25']['games_by_type'], dict(regular=1, cup_final=1))

    def test_pipeline_positions_from_latest_game(self):
        from nba import pipeline
        history = dict(players=[dict(PLAYER_ID='1', PLAYER_NAME='A', POSITION='F', GAME_DATE='2025-01-01'),
                                dict(PLAYER_ID='1', PLAYER_NAME='A', POSITION='G', GAME_DATE='2025-02-01'),
                                dict(PLAYER_ID='2', PLAYER_NAME='B', POSITION=None, GAME_DATE='2025-02-01')],
                       teams=[], games=[], seasons=['2024-25'])
        with patch('nba.espn_history.load_all', return_value=history):
            self.assertEqual(pipeline.espn_history()['positions'], [dict(id='1', name='A', position='G')])


if __name__ == '__main__':
    unittest.main()
