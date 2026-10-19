"""Published picks are registered for closing prices with Bet Tracker's ticket fields."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pick_closes

ROOT = Path(__file__).resolve().parents[1]
CARD = json.loads((ROOT / 'docs/briefing/cards/2026-10-19-e1c7b2d3922a050ece98f327.json').read_text())
ENV = dict(SUPABASE_URL='https://db.example.supabase.co/', SUPABASE_SERVICE_ROLE_KEY='service')


def response(status=201):
    return Mock(status_code=status, ok=200 <= status < 300)


class PickClosesTests(unittest.TestCase):
    def test_rows_use_ticket_fields_and_the_odds_api_event(self):
        rows = pick_closes.rows(CARD)
        self.assertEqual([(r['pick_index'], r['league'], r['market_type'], r['side'], r['line'], r['odds']) for r in rows],
                         [(0, 'NHL', 'assists', 'over', 0.5, 165), (1, 'NFL', 'pass_attempts', 'under', 30.5, -104)])
        nhl = rows[0]
        self.assertEqual((nhl['player'], nhl['book'], nhl['game_date'], nhl['team_home'], nhl['team_away']),
                         ('Evgeni Malkin', 'draftkings', '2026-10-19', 'Utah Mammoth', 'Pittsburgh Penguins'))
        self.assertEqual(nhl['event_id'], '0e2b3099b25e9827dc4a6abd7fabfa51')
        self.assertAlmostEqual(nhl['decision_market_prob'], CARD['rows'][0]['blend']['market'])
        self.assertEqual((nhl['edition_id'], nhl['edition_kind'], nhl['decision_date']), (CARD['edition_id'], 'morning', '2026-10-19'))
        # A game id that is not an Odds API event id is left for the closing job to match.
        other = dict(CARD, rows=[dict(CARD['rows'][0], event_id='2026020140')])
        self.assertIsNone(pick_closes.rows(other)[0]['event_id'])

    def test_register_upserts_each_pick_and_drops_a_superseded_editions_open_picks(self):
        http = Mock()
        http.post.return_value, http.delete.return_value = response(201), response(204)
        self.assertEqual(pick_closes.register(CARD, ENV, http), 2)
        post = http.post.call_args
        self.assertEqual(post.args[0], 'https://db.example.supabase.co/rest/v1/pick_closes')
        self.assertEqual(post.kwargs['params'], {'on_conflict': 'edition_id,pick_index'})
        self.assertEqual(post.kwargs['headers']['Prefer'], 'resolution=merge-duplicates,return=minimal')
        self.assertEqual(len(post.kwargs['json']), 2)
        self.assertEqual(http.delete.call_args.kwargs['params'], {'decision_date': 'eq.2026-10-19', 'edition_kind': 'eq.morning',
                         'edition_id': 'neq.' + CARD['edition_id'], 'clv_status': 'is.null'})
        http.post.return_value = response(500)
        with self.assertRaises(RuntimeError):
            pick_closes.register(CARD, ENV, http)

    def test_main_registers_only_the_verified_edition_and_is_quiet_without_settings(self):
        with patch('sys.argv', ['pick_closes', '--kind', 'morning', '--edition', 'another']), \
             patch.object(pick_closes.Path, 'read_text', return_value=json.dumps(CARD)):
            with self.assertRaises(SystemExit):
                pick_closes.main()
        with patch('sys.argv', ['pick_closes', '--kind', 'morning', '--edition', CARD['edition_id']]), \
             patch.object(pick_closes.Path, 'read_text', return_value=json.dumps(CARD)), patch.dict('os.environ', {}, clear=True), \
             patch.object(pick_closes, 'register') as register:
            self.assertEqual(pick_closes.main(), 0)
            register.assert_not_called()


if __name__ == '__main__':
    unittest.main()
