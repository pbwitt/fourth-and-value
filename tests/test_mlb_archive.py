import gzip
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mlb.archive import compact, save_run  # noqa: E402

NOW = datetime(2026, 10, 6, 20, 32, tzinfo=timezone.utc)
STATE = dict(checked_at='2026-10-06T20:32:10Z', model_version='mlb-v1',
    events=[dict(mlb_game_id=849819, commence_time='2026-10-06T22:00:00Z', home_team='Atlanta Braves',
                 away_team='Los Angeles Dodgers', venue='Truist Park',
                 home_pitcher={'id': 519242, 'fullName': 'Chris Sale', 'link': '/api/v1/people/519242'})],
    rows=[dict(event_id='e1', mlb_game_id=849819, game='Los Angeles Dodgers @ Atlanta Braves', market='pitcher_outs',
               player='Chris Sale', side='Under', line=18.0, price=100.0, book='betonlineag',
               quoted_at='2026-10-06T20:31:06Z', model_probability=0.536, model_push_probability=0.148,
               model_mean=16.94, player_context={'games': ['large']}, book_probability=0.5, stat_context=None)])


class MLBArchiveTests(unittest.TestCase):
    def test_compact_keeps_quotes_and_forecasts_not_display_context(self):
        payload = compact(STATE)
        row = payload['rows'][0]
        self.assertEqual((row['line'], row['price'], row['model_probability'], row['model_mean']), (18.0, 100.0, 0.536, 16.94))
        self.assertNotIn('player_context', row)
        self.assertNotIn('book_probability', row)
        self.assertEqual(payload['events'][0]['home_pitcher'], {'id': 519242, 'fullName': 'Chris Sale'})
        self.assertNotIn('venue', payload['events'][0])
        self.assertEqual((payload['schema'], payload['sport'], payload['model_version']), (1, 'MLB', 'mlb-v1'))

    def test_save_run_writes_once_and_skips_empty_refreshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_run(STATE, NOW, root=tmp)
            self.assertEqual(path.name, '20261006T203200Z.json.gz')
            with gzip.open(path, 'rt') as f:
                self.assertEqual(json.load(f)['rows'][0]['player'], 'Chris Sale')
            self.assertIsNone(save_run(dict(STATE, rows=[dict(player='Changed')]), NOW, root=tmp), 'never overwritten')
            self.assertIsNone(save_run(dict(STATE, rows=[]), datetime(2026, 10, 7, tzinfo=timezone.utc), root=tmp))
            self.assertEqual(len(list(Path(tmp).glob('*.json.gz'))), 1)


if __name__ == '__main__':
    unittest.main()
