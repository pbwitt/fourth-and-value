"""NBA refresh gate: Supabase and operator starts refresh, late GitHub backups skip a published slot."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nba import schedule_gate as gate


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class NBAScheduleGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def feed(self, stamp):
        path = self.root / gate.FEED
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(status='ready', last_success_at=stamp)))

    def test_dispatched_starts_always_refresh(self):
        self.feed('2026-10-20T10:46:00Z')   # 6:46 a.m. EDT
        self.assertTrue(gate.decide(self.root, utc('2026-10-20T10:50:00'), 'workflow_dispatch'))

    def test_late_morning_backup_skips_after_the_timer_published(self):
        self.feed('2026-10-20T10:47:12.5Z')   # 6:47 a.m. EDT
        self.assertFalse(gate.decide(self.root, utc('2026-10-20T14:10:00'), 'schedule'))   # 10:10 a.m.

    def test_backup_refreshes_when_the_timer_run_did_not_publish(self):
        self.feed('2026-10-20T00:02:50Z')   # 8:02 p.m. EDT the night before
        self.assertTrue(gate.decide(self.root, utc('2026-10-20T14:10:00'), 'schedule'))

    def test_evening_backup_needs_a_publication_after_5_pm(self):
        self.feed('2026-10-20T10:47:00Z')   # morning slot only
        self.assertTrue(gate.decide(self.root, utc('2026-10-20T23:30:00'), 'schedule'))   # 7:30 p.m.
        self.feed('2026-10-20T21:01:30Z')   # 5:01 p.m. EDT
        self.assertFalse(gate.decide(self.root, utc('2026-10-20T23:30:00'), 'schedule'))

    def test_backup_after_midnight_uses_the_previous_evening_slot(self):
        self.feed('2026-10-20T21:01:30Z')   # 5:01 p.m. EDT October 20
        self.assertFalse(gate.decide(self.root, utc('2026-10-21T04:30:00'), 'schedule'))   # 12:30 a.m.
        self.assertEqual(gate.latest_slot(utc('2026-10-21T04:30:00')).isoformat(), '2026-10-20T17:00:00-04:00')

    def test_slots_stay_eastern_in_standard_time(self):
        self.assertEqual(gate.latest_slot(utc('2026-12-01T11:50:00')).isoformat(), '2026-12-01T06:45:00-05:00')
        self.assertEqual(gate.latest_slot(utc('2026-12-01T11:40:00')).isoformat(), '2026-11-30T17:00:00-05:00')

    def test_missing_or_unreadable_feed_refreshes(self):
        self.assertTrue(gate.decide(self.root, utc('2026-10-20T14:10:00'), 'schedule'))
        (self.root / gate.FEED).parent.mkdir(parents=True, exist_ok=True)
        (self.root / gate.FEED).write_text('{not json')
        self.assertTrue(gate.decide(self.root, utc('2026-10-20T14:10:00'), 'schedule'))

    def test_cli_writes_the_step_output(self):
        out = Path(self.tmp.name) / 'output'
        env = {**os.environ, 'GITHUB_OUTPUT': str(out)}
        env.pop('GITHUB_STEP_SUMMARY', None)
        script = Path(__file__).resolve().parents[1] / 'scripts/nba/schedule_gate.py'
        result = subprocess.run([sys.executable, str(script), '--event', 'workflow_dispatch'],
                                env=env, capture_output=True, text=True, check=True)
        self.assertIn('NBA: refresh', result.stdout)
        self.assertEqual(out.read_text(), 'refresh=true\n')


if __name__ == '__main__':
    unittest.main()
