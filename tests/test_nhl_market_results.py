"""Tests for the NHL Market Results ledger: pregame snapshot choice, side pairing and settlement."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import build_market_results as mr

GAME = dict(game_id=2026020001, home_id=12, away_id=13, home_team='Carolina Hurricanes', away_team='Florida Panthers',
            home_score=2, away_score=3)
LABEL = {12: 'CAR', 13: 'FLA'}


def quote(market, side, line, price, book='a', player='', player_id=None, start='2026-09-29T23:00:00Z', quoted=None):
    return dict(nhl_game_id=GAME['game_id'], commence_time=start, quoted_at=quoted, home_team='Carolina Hurricanes',
                away_team='Florida Panthers', market=market, side=side, line=line, price=price, book=book,
                player=player, player_id=player_id)


def run(checked_at, quotes, snapshot_id=None):
    return dict(snapshot=dict(snapshot_id=snapshot_id or checked_at, checked_at=checked_at, rows=quotes))


def skater(pid, name, **stats):
    return {(GAME['game_id'], pid): dict(dict(shots=0, goals=0, assists=0, points=0), player_id=pid, player=name, **stats)}


class PregameSnapshotTests(unittest.TestCase):
    def test_uses_last_snapshot_before_puck_drop(self):
        early = run('2026-09-29T11:00:00Z', [quote('totals', 'Over', 6.5, -110)])
        late = run('2026-09-29T12:30:00Z', [quote('totals', 'Over', 5.5, -110)])
        live = run('2026-09-29T23:30:00Z', [quote('totals', 'Over', 4.5, -110)])
        at, quotes = mr.nhl_pregame_quotes([early, late, live])[GAME['game_id']]
        self.assertEqual(str(at), '2026-09-29 12:30:00+00:00')
        self.assertEqual(quotes[0]['line'], 5.5)

    def test_quote_timed_after_puck_drop_is_ignored(self):
        late = run('2026-09-29T22:59:00Z', [quote('totals', 'Over', 6.5, -110, quoted='2026-09-29T23:01:00Z')])
        self.assertEqual(mr.nhl_pregame_quotes([late]), {})


class GameRowTests(unittest.TestCase):
    def rows(self, quotes, players=None):
        return {r['market']: r for r in mr.nhl_game_rows(GAME['game_id'], quotes, GAME, players or {}, LABEL, 0)}

    def test_total_uses_main_line_and_final_score(self):
        r = self.rows([quote('totals', 'Over', 6.5, -110), quote('totals', 'Under', 6.5, -110),
                       quote('totals', 'Over', 6.5, -120, 'b'), quote('totals', 'Under', 6.5, 100, 'b'),
                       quote('totals', 'Over', 5.5, -150, 'c'), quote('totals', 'Under', 5.5, 130, 'c')])['totals']
        self.assertEqual((r['line'], r['books'], r['actual'], r['label']), (6.5, 2, 5.0, 'FLA @ CAR'))
        self.assertAlmostEqual(r['p_over'], (0.5 + mr.no_vig(-120, 100)) / 2)

    def test_puck_line_is_from_the_favorite_side(self):
        r = self.rows([quote('spreads', 'Carolina Hurricanes', -1.5, 180), quote('spreads', 'Florida Panthers', 1.5, -220)])
        r = r['spreads']
        self.assertEqual((r['label'], r['line'], r['actual']), ('CAR -1.5', 1.5, -1.0))
        self.assertAlmostEqual(r['p_over'], mr.no_vig(180, -220))

    def test_puck_line_needs_matching_lines_at_one_book(self):
        r = self.rows([quote('spreads', 'Carolina Hurricanes', -1.5, 180), quote('spreads', 'Florida Panthers', 2.5, -400)])
        self.assertNotIn('spreads', r)

    def test_moneyline_favorite_and_margin(self):
        r = self.rows([quote('h2h', 'Carolina Hurricanes', None, 140), quote('h2h', 'Florida Panthers', None, -160)])['h2h']
        self.assertEqual((r['label'], r['line'], r['actual']), ('FLA', 0.0, 1.0))
        self.assertGreater(r['p_over'], 0.5)
        self.assertAlmostEqual(r['dec_over'], mr.decimal(-160))

    def test_props_settle_by_player_and_skip_missing_appearance(self):
        players = skater(1, 'Seth Jarvis', shots=4)
        r = self.rows([quote('player_shots_on_goal', 'Over', 2.5, -115, player='Seth Jarvis', player_id=1),
                       quote('player_shots_on_goal', 'Under', 2.5, -105, player='Seth Jarvis', player_id=1),
                       quote('player_points', 'Over', 0.5, -110, player='Scratched Player', player_id=2),
                       quote('player_points', 'Under', 0.5, -110, player='Scratched Player', player_id=2)], players)
        self.assertEqual((r['player_shots_on_goal']['actual'], r['player_shots_on_goal']['label']), (4.0, 'Seth Jarvis'))
        self.assertNotIn('player_points', r)

    def test_missing_player_id_resolves_only_by_unique_name_in_game(self):
        players = {**skater(1, 'Sebastian Aho', points=2), **skater(2, 'Elias Pettersson'), **skater(3, 'Elias Pettersson')}
        rows = mr.nhl_game_rows(GAME['game_id'], [
            quote('player_points', 'Over', 0.5, -110, player='Sebastián Aho'),
            quote('player_points', 'Under', 0.5, -110, player='Sebastián Aho'),
            quote('player_points', 'Over', 1.5, 200, player='Elias Pettersson'),
            quote('player_points', 'Under', 1.5, -250, player='Elias Pettersson')], GAME, players, LABEL, 0)
        self.assertEqual([(r['label'], r['actual']) for r in rows], [('Sebastián Aho', 2.0)])


class ArchiveTests(unittest.TestCase):
    def test_committed_archive_builds_consistent_rows(self):
        payload = mr.build_nhl(2026)
        if not payload['periods']:
            self.skipTest('No settled NHL games in the archive yet')
        rows = payload['rows']
        totals = {r[3]: r for r in rows['totals']}
        self.assertEqual(len(totals), sum(p['games'] for p in payload['periods']))
        for market in ('spreads', 'h2h'):
            for r in rows[market]:
                self.assertGreaterEqual(r[5], 0.0)
                self.assertLessEqual(r[5], 1.0)
                self.assertIn(r[3], totals)
        for r in rows['h2h']:
            self.assertNotEqual(r[6], 0)  # NHL games are never tied once the shootout is counted.
        self.assertTrue(payload['through'])


if __name__ == '__main__':
    unittest.main()
