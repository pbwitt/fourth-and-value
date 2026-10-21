"""Tests for prospective grading of the NBA v1 shadow model: snapshot choice, settlement, scores and the card pool."""
import copy
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nba import grade_shadow as g  # noqa: E402

START = '2026-10-22T02:00:00Z'   # Wednesday, Oct 21, 10 PM ET
HOME, AWAY = 'Los Angeles Lakers', 'Los Angeles Clippers'
NOW = datetime.fromisoformat('2026-10-23T12:00:00+00:00')


def event(final=True):
    return dict(id='401000001', date=START, competitions=[dict(
        id='401000001', date=START, status=dict(type=dict(completed=final, state='post' if final else 'in')),
        competitors=[dict(homeAway='home', score='115', team=dict(id='13', displayName=HOME)),
                     dict(homeAway='away', score='108', team=dict(id='12', displayName='LA Clippers'))])])


def summary():
    # José Player Jr.: 25 points, 8 rebounds, 7 assists, 3 threes. Ben Bench did not play. Final 115-108.
    athletes = [dict(athlete=dict(id='99', displayName='José Player Jr.'), stats=['35', '25', '8', '7', '3-7', '2', '1', '4']),
                dict(athlete=dict(id='100', displayName='Ben Bench'), didNotPlay=True, stats=[])]
    return dict(header=copy.deepcopy(event()), boxscore=dict(players=[dict(
        team=dict(id='13', displayName=HOME), statistics=[dict(
            labels=['MIN', 'PTS', 'REB', 'AST', '3PT', 'BLK', 'STL', 'TO'], athletes=athletes)])]))


def row(market, side, line, price, model, consensus, book='a', player='', pid=None, books=2, push=0.0,
        version='nba-v1.0', start=START):
    r = dict(event_id='evt1', commence_time=start, quoted_at=None, home_team=HOME, away_team=AWAY,
             game=f'{AWAY} @ {HOME}', market=market, side=side, line=line, price=price, book=book, player=player,
             model_probability=model, model_push_probability=push, model_version=version,
             consensus_probability=consensus, paired_books=books)
    if pid:
        r['espn_player_id'] = pid
    return r


def prop(line, over, under, model_over, consensus_over, book='a', books=2, market='player_points',
         player='Jose Player Jr', pid='99'):
    return [row(market, 'Over', line, over, model_over, consensus_over, book, player, pid, books),
            row(market, 'Under', line, under, 1 - model_over, 1 - consensus_over, book, player, pid, books)]


EARLY = prop(24.5, -110, -110, 0.60, 0.50)
LATE = (prop(24.5, -110, -110, 0.60, 0.55) + prop(24.5, 120, -150, 0.60, 0.55, book='b')
        # An alternate line one book posts: graded, but never on the card (two books must post it).
        + prop(26.5, 150, -190, 0.45, 0.38, books=1)
        + prop(6.5, -110, -110, 0.55, 0.50, market='player_rebounds', player='Ben Bench', pid='100')
        # A whole-number total that lands exactly: a push.
        + [row('totals', 'Over', 223.0, -110, 0.48, 0.50, push=0.04), row('totals', 'Under', 223.0, -110, 0.48, 0.50, push=0.04)]
        + [row('spreads', HOME, -6.5, -110, 0.55, 0.50), row('spreads', AWAY, 6.5, -110, 0.45, 0.50)]
        + [row('h2h', HOME, None, -250, 0.75, 0.70), row('h2h', AWAY, None, 200, 0.25, 0.30)]
        + [row('player_assists', 'Over', 6.5, -110, 0.5, 0.5, player='Jose Player Jr', pid='99', version='nba-v0')])
AFTER_TIP = prop(24.5, -110, -110, 0.10, 0.90)


class HelperTests(unittest.TestCase):
    def test_model_probability_is_conditional_on_no_push(self):
        self.assertAlmostEqual(g.conditional(0.48, 0.04), 0.5)
        self.assertIsNone(g.conditional(None, 0.0))
        self.assertIsNone(g.conditional(0.5, 1.0))

    def test_spreads_are_keyed_on_the_home_handicap(self):
        self.assertEqual(g.unit_key(row('spreads', AWAY, 6.5, -110, 0.45, 0.5)), ('spreads', '', -6.5))
        self.assertEqual(g.unit_key(row('spreads', HOME, -6.5, -110, 0.55, 0.5)), ('spreads', '', -6.5))
        self.assertEqual(g.unit_key(row('h2h', AWAY, None, 200, 0.25, 0.3)), ('h2h', '', None))
        self.assertEqual(g.unit_key(row('player_points', 'Under', 24.5, -110, 0.4, 0.5, pid='99'))[:2], ('player_points', '99'))

    def test_both_sides_and_books_form_one_unit_from_the_reference_side(self):
        units = g.snapshot_units(prop(24.5, -110, -110, 0.6, 0.55) + prop(24.5, 120, -150, 0.6, 0.55, book='b', books=3))
        unit = units[('player_points', '99', 24.5)]
        self.assertEqual((round(unit['model'], 6), round(unit['market_p'], 6), unit['books'], len(unit['offers'])), (0.6, 0.55, 3, 4))
        away_only = g.snapshot_units([row('h2h', AWAY, None, 200, 0.25, 0.30)])[('h2h', '', None)]
        self.assertEqual((round(away_only['model'], 6), round(away_only['market_p'], 6)), (0.75, 0.7))

    def test_main_line_is_the_one_most_books_pair(self):
        units = g.snapshot_units(LATE[:6])  # Points at 24.5 from two books and at 26.5 from one.
        self.assertEqual(g.main_lines(units), {('player_points', '99', 24.5)})

    def test_no_interval_from_a_handful_of_games(self):
        rng = np.random.default_rng(0)
        self.assertIsNone(g.interval([0.1, -0.1, 0.2], ['a', 'b', 'c'], rng)['ci95'])
        games = [str(i) for i in range(g.MIN_CI_GAMES)]
        low, high = g.interval(np.linspace(-1, 1, len(games)), games, rng)['ci95']
        self.assertLess(low, high)

    def test_card_bet_needs_two_books_and_tags_where_the_value_comes_from(self):
        unit = g.snapshot_units(prop(24.5, -110, -110, 0.60, 0.55) + prop(24.5, 120, -150, 0.60, 0.55, book='b'))[('player_points', '99', 24.5)]
        bet = g.card_bet(unit, 1, 'evt1')
        self.assertEqual((bet['book'], bet['price'], bet['source'], bet['result']), ('b', 120.0, 'price', 'win'))
        self.assertAlmostEqual(bet['profit'], 1.2)
        # Only the model's disagreement clears the floor at fair prices: tagged 'model'.
        lean = g.snapshot_units(prop(24.5, 100, -100, 0.70, 0.50))[('player_points', '99', 24.5)]
        self.assertEqual(g.card_bet(lean, 0, 'evt1')['source'], 'model')
        self.assertEqual(g.card_bet(lean, 0, 'evt1')['profit'], -1.0)
        thin = g.snapshot_units(prop(24.5, 120, -150, 0.6, 0.55, books=1))[('player_points', '99', 24.5)]
        self.assertIsNone(g.card_bet(thin, 1, 'evt1'))


class GradeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runs = self.root / 'artifacts/nba/runs'
        self.runs.mkdir(parents=True)
        self.save('20261021T110000Z', '2026-10-21T11:00:00Z', EARLY)
        self.save('20261021T210000Z', '2026-10-21T21:00:00Z', LATE)
        self.save('20261022T030000Z', '2026-10-22T03:00:00Z', AFTER_TIP)
        self.calls = []

    def save(self, name, checked_at, rows):
        with gzip.open(self.runs / f'{name}.json.gz', 'wt') as f:
            json.dump(dict(checked_at=checked_at, rows=rows), f)

    def fetch(self, url):
        self.calls.append(url)
        return copy.deepcopy(dict(events=[event()]) if '/scoreboard?' in url else summary())

    def grade(self, **kw):
        return g.grade(2026, self.runs, self.root, kw.pop('fetch', self.fetch), now=NOW, version='nba-v1.0', **kw)

    def test_end_to_end(self):
        payload = self.grade()
        c = payload['coverage']
        self.assertEqual((c['games_tipped'], c['games_graded']), (1, 1))
        self.assertEqual(c['other_model_versions_excluded'], {'nba-v0': 1})
        self.assertEqual(c['settlement']['participation_unconfirmed'], 1, 'a player who did not play is void, not zero')
        self.assertEqual(c['settlement']['push'], 1)
        self.assertEqual(payload['through'], '2026-10-21')
        points = payload['by_market']['player_points']
        self.assertEqual((points['main_lines']['units'], points['main_lines']['outcome_rate']), (1, 1.0))
        self.assertEqual(points['all_lines']['units'], 2, 'the alternate line is graded too')
        self.assertEqual(payload['groups']['props']['alternate_lines']['units'], 1)
        self.assertEqual(payload['by_market']['spreads']['main_lines']['outcome_rate'], 1.0, 'home -6.5 covered by 7')
        self.assertEqual(payload['by_market']['h2h']['main_lines']['outcome_rate'], 1.0)
        self.assertNotIn('totals', payload['by_market'], 'a push is not scored')
        self.assertNotIn('player_rebounds', payload['by_market'])
        self.assertNotIn('player_assists', payload['by_market'], 'other model versions are not graded')

    def test_last_pregame_snapshot_is_scored_and_movement_measured(self):
        payload = self.grade()
        points = payload['by_market']['player_points']['main_lines']
        self.assertEqual((points['model_mean'], points['market_mean']), (0.6, 0.55))
        move = payload['groups']['props']['movement']
        self.assertEqual((move['units'], move['toward'], move['points']['mean']), (1, 1.0, 5.0))
        pool = payload['groups']['props']['card_pool']
        self.assertEqual((pool['bets'], pool['wins'], pool['units']), (1, 1, 1.2))
        self.assertEqual((pool['by_source']['price']['bets'], pool['by_source']['model']['bets']), (1, 0))
        self.assertEqual(payload['groups']['game_lines']['card_pool']['bets'], 0)

    def test_readiness_waits_for_the_sample(self):
        payload = self.grade()
        self.assertEqual(payload['groups']['props']['readiness'],
                         dict(status='insufficient_sample', games=1, games_needed=g.MIN_GAMES))

    def test_unfinished_game_is_not_graded_but_still_counts_movement(self):
        payload = self.grade(fetch=lambda url: copy.deepcopy(dict(events=[event(final=False)]) if '/scoreboard?' in url else summary()))
        self.assertEqual(payload['coverage']['games_graded'], 0)
        self.assertEqual(payload['coverage']['settlement'], {'result_pending_or_unmatched': 1})
        self.assertEqual(payload['groups']['props']['movement']['units'], 1)

    def test_offline_never_fetches(self):
        self.assertEqual(self.grade(offline=True)['coverage']['games_graded'], 0)
        self.assertEqual(self.calls, [])

    def test_writes_json_and_an_eastern_time_summary(self):
        payload = self.grade()
        out = self.root / 'evaluation'
        g.write(payload, out)
        self.assertEqual(json.loads((out / 'prospective.json').read_text())['model_version'], 'nba-v1.0')
        text = (out / 'PROSPECTIVE.md').read_text()
        self.assertIn('Fri, Oct 23, 2026, 8:00 AM EDT', text)
        self.assertIn('| Points | 1 |', text)
        self.assertIn('Intervals appear once', text)
        self.assertNotIn('UTC', text)


if __name__ == '__main__':
    unittest.main()
