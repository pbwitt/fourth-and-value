import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from recap_archive import save_nba_run


class RecapArchiveTests(unittest.TestCase):
    def test_archive_preserves_snapshot_and_never_overwrites(self):
        now=datetime(2026,10,7,tzinfo=timezone.utc)
        state=dict(status='ready',checked_at=now.isoformat(),rows=[dict(price=-110,line=20.5)])
        with tempfile.TemporaryDirectory() as root:
            path=save_nba_run(state,now,root)
            with gzip.open(path,'rt') as f:self.assertEqual(json.load(f),state)
            self.assertIsNone(save_nba_run(dict(state,rows=[dict(price=200)]),now,root))
            with gzip.open(path,'rt') as f:self.assertEqual(json.load(f)['rows'][0]['price'],-110)
            self.assertIsNone(save_nba_run(dict(state,status='feed_error'),now,root))
            self.assertIsNone(save_nba_run(dict(state,rows=[]),now,root))


if __name__=='__main__':unittest.main()
