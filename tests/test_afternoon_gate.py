"""Afternoon NHL/MLB refresh gate: on-time Supabase starts refresh, late GitHub backups skip."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import afternoon_gate as gate


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class AfternoonGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def feeds(self, nhl, mlb):
        for sport, stamp in [('nhl', nhl), ('mlb', mlb)]:
            path = self.root / gate.FEEDS[sport]
            path.parent.mkdir(parents=True, exist_ok=True)
            if stamp is not None:
                path.write_text(json.dumps(dict(status='ready', last_success_at=stamp)))

    def test_supabase_start_refreshes_after_the_morning_run(self):
        self.feeds('2026-10-05T11:06:27.688810Z', '2026-10-05T11:07:05Z')   # 7:06 a.m. EDT
        self.assertEqual(gate.decide(self.root, utc('2026-10-05T20:30:05'), 'supabase'), dict(nhl=True, mlb=True))

    def test_late_backup_skips_a_sport_already_refreshed_this_afternoon(self):
        # NHL published at 4:41 p.m.; MLB's on-time refresh failed and still shows the morning.
        self.feeds('2026-10-05T20:41:00Z', '2026-10-05T11:07:05Z')
        self.assertEqual(gate.decide(self.root, utc('2026-10-05T23:21:00'), 'github-schedule'), dict(nhl=False, mlb=True))

    def test_backup_after_midnight_does_nothing(self):
        self.feeds('2026-10-05T11:06:00Z', '2026-10-05T11:07:00Z')
        self.assertEqual(gate.decide(self.root, utc('2026-10-06T04:30:00'), 'github-schedule'), dict(nhl=False, mlb=False))

    def test_operator_always_refreshes(self):
        self.feeds('2026-10-05T20:41:00Z', '2026-10-05T20:42:00Z')
        self.assertEqual(gate.decide(self.root, utc('2026-10-05T21:00:00'), 'operator'), dict(nhl=True, mlb=True))

    def test_standard_time_and_unreadable_feeds(self):
        # 4:10 p.m. EST is 21:10 UTC; 20:50 UTC is 3:50 p.m. EST, before the afternoon window.
        self.feeds('2026-12-01T20:50:00Z', None)
        (self.root / gate.FEEDS['mlb']).write_text('{not json')
        self.assertEqual(gate.decide(self.root, utc('2026-12-01T21:40:00'), 'supabase'), dict(nhl=True, mlb=True))
        self.feeds('2026-12-01T21:10:00Z', '2026-12-01T21:12:00')   # MLB stamp without a zone is unusable
        self.assertEqual(gate.decide(self.root, utc('2026-12-01T21:40:00'), 'supabase'), dict(nhl=False, mlb=True))

    def test_writes_workflow_outputs(self):
        out = self.root / 'github-output.txt'
        result = subprocess.run([sys.executable, str(Path(gate.__file__)), '--source', 'operator'],
                                env=dict(os.environ, GITHUB_OUTPUT=str(out)), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(out.read_text(), 'nhl=true\nmlb=true\n')
        self.assertIn('NHL: refresh (started by operator', result.stdout)


if __name__ == '__main__':
    unittest.main()
