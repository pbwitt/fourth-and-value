from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nhl.v2.arbitrage import find_arbitrage, find_incoherent, render, report

BASE = dict(event_id='e1', commence_time='2026-10-10T23:00:00Z', home_team='Carolina Hurricanes',
            away_team='Florida Panthers', game='Florida Panthers @ Carolina Hurricanes', player='',
            quoted_at='2026-10-10T18:00:00Z', settlement_profile='nhl_full_game_ot_so', settlement_verified=True)


def quote(book, market, side, price, line=None, **extra):
    return {**BASE, 'book': book, 'market': market, 'side': side, 'price': price, 'line': line, **extra}


class NHLArbitrageTests(unittest.TestCase):
    def test_cross_book_total_under_100_is_locked_with_equal_payout(self):
        rows = [quote('draftkings', 'totals', 'Over', 110, 6.5), quote('draftkings', 'totals', 'Under', -130, 6.5),
                quote('fanduel', 'totals', 'Over', -135, 6.5), quote('fanduel', 'totals', 'Under', 105, 6.5)]
        found, _ = find_arbitrage(rows)
        self.assertEqual([f['status'] for f in found], ['arbitrage'])
        arb = found[0]
        self.assertEqual({l['book'] for l in arb['legs']}, {'draftkings', 'fanduel'})
        self.assertAlmostEqual(sum(l['stake'] for l in arb['legs']), 100)
        for leg in arb['legs']:
            dec = 1 + leg['price'] / 100 if leg['price'] > 0 else 1 + 100 / abs(leg['price'])
            self.assertAlmostEqual(leg['stake'] * dec, arb['payout'])
        self.assertGreater(arb['locked_return_pct'], 0)
        self.assertFalse(arb['push_possible'])

    def test_unverified_settlement_and_stale_quotes_are_not_counted(self):
        rows = [quote('draftkings', 'h2h', 'Florida Panthers', 110),
                quote('bovada', 'h2h', 'Carolina Hurricanes', 105, settlement_verified=False,
                      settlement_profile='unverified:bovada:h2h')]
        self.assertEqual(find_arbitrage(rows)[0][0]['status'], 'check_settlement_rules')
        rows[1] = quote('fanduel', 'h2h', 'Carolina Hurricanes', 105, quoted_at='2026-10-10T18:20:00Z')
        self.assertEqual(find_arbitrage(rows)[0][0]['status'], 'stale_pair')

    def test_puck_line_sides_pair_by_signed_line_and_whole_lines_can_push(self):
        rows = [quote('draftkings', 'spreads', 'Carolina Hurricanes', 180, -1.5),
                quote('fanduel', 'spreads', 'Florida Panthers', -150, 1.5),
                quote('fanduel', 'spreads', 'Florida Panthers', -400, -1.5)]
        found, _ = find_arbitrage(rows)
        self.assertEqual(len(found), 1)
        rows = [quote('draftkings', 'totals', 'Over', 105, 6.0), quote('fanduel', 'totals', 'Under', 105, 6.0)]
        self.assertTrue(find_arbitrage(rows)[0][0]['push_possible'])

    def test_one_sided_and_normal_markets_produce_nothing(self):
        rows = [quote('draftkings', 'player_goals', 'Over', 250, 0.5, player='A Skater'),
                quote('draftkings', 'totals', 'Over', -110, 6.5), quote('fanduel', 'totals', 'Under', -110, 6.5)]
        self.assertEqual(find_arbitrage(rows), ([], []))

    def test_near_miss_is_context_not_arbitrage(self):
        rows = [quote('draftkings', 'totals', 'Over', 100, 6.5), quote('fanduel', 'totals', 'Under', -102, 6.5)]
        found, close = find_arbitrage(rows)
        self.assertEqual(found, [])
        self.assertEqual(close[0]['status'], 'near_miss')

    def test_same_book_containment_contradictions(self):
        rows = [quote('draftkings', 'player_points', 'Over', 150, 0.5, player='A Skater'),
                quote('draftkings', 'player_goals', 'Over', 120, 0.5, player='A Skater'),
                quote('draftkings', 'player_assists', 'Over', 200, 0.5, player='A Skater'),
                quote('fanduel', 'h2h', 'Carolina Hurricanes', -150),
                quote('fanduel', 'spreads', 'Carolina Hurricanes', -160, -1.5)]
        found = find_incoherent(rows)
        self.assertEqual({(f['subject'], f['narrow']['market']) for f in found},
                         {('A Skater', 'player_goals'), ('Carolina Hurricanes', 'spreads')})

    def test_report_renders_empty_and_populated_states(self):
        empty = render(report(dict(status='ready', rows=[], last_success_at=None)))
        self.assertIn('No locked arbitrage', empty)
        rows = [quote('draftkings', 'totals', 'Over', 110, 6.5), quote('fanduel', 'totals', 'Under', 105, 6.5)]
        page = render(report(dict(status='feed_error', rows=rows, last_success_at='2026-10-10T18:01:00Z')))
        self.assertIn('Locked arbitrage', page)
        self.assertIn('Feed is not current', page)


if __name__ == '__main__':
    unittest.main()
