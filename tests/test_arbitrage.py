"""Arbitrage section: generated files are current, the rules config is consistent, and the shared
engine (docs/assets/arbitrage.js) locks every pair the NHL page's Python scanner locks."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/arbitrage'))
import build  # noqa: E402
from nhl.v2.arbitrage import find_arbitrage  # noqa: E402

BASE = dict(event_id='e1', commence_time='2026-10-10T23:00:00Z', home_team='Carolina Hurricanes',
            away_team='Florida Panthers', game='Florida Panthers @ Carolina Hurricanes', player='',
            quoted_at='2026-10-10T18:00:00Z')
NHL = json.loads((ROOT / 'config/nhl_settlement.json').read_text())


def quote(book, market, side, price, line=None, **extra):
    """An NHL snapshot row, with the settlement fields the NHL pipeline derives from its config."""
    group = 'game' if market in ('h2h', 'spreads', 'totals') else 'player'
    profile = NHL['books'].get(book, {}).get(group)
    return {**BASE, 'book': book, 'market': market, 'side': side, 'price': price, 'line': line,
            'settlement_profile': profile or f'unverified:{book}:{market}', 'settlement_verified': bool(profile), **extra}


def engine(rows, now='2026-10-10T18:05:00Z'):
    script = ("const A=require('./docs/assets/arbitrage.js');const fs=require('fs');"
              "const {rows,rules,now}=JSON.parse(fs.readFileSync(0,'utf8'));"
              "process.stdout.write(JSON.stringify(A.scan(A.fromSnapshot(rows,'NHL'),{rules,sport:'NHL',now:Date.parse(now)})));")
    out = subprocess.run(['node', '-e', script], cwd=ROOT, input=json.dumps(dict(rows=rows, rules=build.rules(), now=now)),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def legs(pair):
    return sorted((leg['book'], leg['side'], leg['price']) for leg in pair['legs'])


class GeneratedFiles(unittest.TestCase):
    def test_rules_and_plan_are_current(self):
        for path, text in build.outputs().items():
            self.assertEqual(path.read_text(), text, f'{path.relative_to(ROOT)} is stale; run python scripts/arbitrage/build.py')

    def test_every_profile_and_assumption_is_defined(self):
        r = build.rules()
        for sport, s in r['sports'].items():
            for book, rule in s['books'].items():
                groups = {k for k in rule if k in ('game', 'player', 'batter', 'pitcher')}
                for group in groups:
                    self.assertIn(rule[group], s['profiles'], f'{sport} {book} {group}')
                self.assertLessEqual(set(rule.get('assumed', [])), groups, f'{sport} {book}')
        self.assertEqual({b['kind'] for b in r['books'].values()} - {None}, {'licensed', 'offshore', 'sweepstakes'})
        self.assertTrue(r['books']['bovada']['kind'] == 'offshore' and not r['books']['bovada']['licensed'])
        self.assertEqual(r['sports']['NHL']['books']['draftkings']['game'], NHL['books']['draftkings']['game'])


@unittest.skipUnless(shutil.which('node'), 'node is required for the shared engine')
class NHLParity(unittest.TestCase):
    def assert_superset(self, rows):
        python = [a for a in find_arbitrage(rows)[0] if a['status'] == 'arbitrage']
        js = engine(rows)['locked']
        for arb in python:
            match = [p for p in js if legs(p) == legs(arb)]
            self.assertTrue(match, f'NHL lock missing from the shared engine: {legs(arb)}')
            self.assertAlmostEqual(match[0]['return_pct'], arb['locked_return_pct'])
        return python, js

    def test_same_locks_on_game_lines_and_props(self):
        rows = [quote('draftkings', 'totals', 'Over', 110, 6.5), quote('fanduel', 'totals', 'Under', 105, 6.5),
                quote('draftkings', 'player_points', 'Over', 105, 0.5, player='A Skater'),
                quote('betmgm', 'player_points', 'Under', 105, 0.5, player='A Skater'),
                quote('draftkings', 'spreads', 'Carolina Hurricanes', 180, -1.5), quote('fanduel', 'spreads', 'Florida Panthers', -150, 1.5)]
        python, js = self.assert_superset(rows)
        self.assertEqual(len(python), 3)
        self.assertEqual(len(js), 3)

    def test_unverified_and_stale_pairs_are_not_locked_by_either(self):
        rows = [quote('draftkings', 'h2h', 'Florida Panthers', 110), quote('bovada', 'h2h', 'Carolina Hurricanes', 105),
                quote('draftkings', 'totals', 'Over', 110, 6.5), quote('fanduel', 'totals', 'Under', 105, 6.5, quoted_at='2026-10-10T18:20:00Z')]
        python, js = self.assert_superset(rows)
        self.assertEqual(python, [])
        self.assertEqual(js, [])
        found = engine(rows, now='2026-10-10T18:21:00Z')
        self.assertEqual(sorted(r['code'] for p in found['check'] for r in p['reasons']), ['quotes_apart', 'rules_unknown'])

    def test_the_engine_finds_a_lock_behind_an_unverified_best_price(self):
        # Python pairs only the best price per side (Bovada's game line is unverified); the engine also tries DraftKings.
        rows = [quote('fanduel', 'h2h', 'Florida Panthers', 115), quote('bovada', 'h2h', 'Carolina Hurricanes', 105),
                quote('draftkings', 'h2h', 'Carolina Hurricanes', 100)]
        python, js = self.assert_superset(rows)
        self.assertEqual(python, [])
        self.assertEqual([legs(p) for p in js], [[('draftkings', 'Carolina Hurricanes', 100), ('fanduel', 'Florida Panthers', 115)]])


if __name__ == '__main__':
    unittest.main()
