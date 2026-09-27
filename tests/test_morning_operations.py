"""Recovery starts cost nothing after completion; delivery must match the edition."""
from datetime import datetime, timezone, timedelta
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock
from urllib.error import URLError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import morning_operations as ops
from nhl.v2.data import write_json

NOW=datetime(2026,9,28,11,5,tzinfo=timezone.utc)


class MorningOperationsTests(unittest.TestCase):
    def value(self):
        return dict(schema_version=1,status='published',kind='morning',rows=[],edition_id='test-edition',decision_date='2026-09-28',published_at=NOW.isoformat())

    def test_gate_recovers_missing_and_incomplete_but_not_completed_editions(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            self.assertTrue(ops.gate(root,NOW)['refresh'])
            write_json(root/'docs/briefing/morning-card.json',self.value())
            self.assertFalse(ops.gate(root,NOW+timedelta(minutes=30))['refresh'])
            self.assertTrue(ops.gate(root,NOW,replace_card=True)['refresh'])
            write_json(root/'docs/briefing/morning-card.json',dict(self.value(),status='research_incomplete'))
            self.assertTrue(ops.gate(root,NOW+timedelta(minutes=30))['refresh'])

    def test_late_schedule_fails_before_feeds_but_completed_card_can_verify(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            with self.assertRaises(RuntimeError):ops.gate(root,NOW+timedelta(hours=5))
            with self.assertRaises(RuntimeError):ops.gate(root,NOW-timedelta(hours=1))
            self.assertTrue(ops.gate(root,NOW+timedelta(hours=5),test_edition=True)['refresh'])
            write_json(root/'docs/briefing/morning-card.json',self.value())
            self.assertFalse(ops.gate(root,NOW+timedelta(hours=5))['refresh'])

    def test_eastern_winter_window_and_previous_day(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            self.assertTrue(ops.gate(root,datetime(2026,12,1,12,5,tzinfo=timezone.utc))['refresh'])
            with self.assertRaises(RuntimeError):ops.gate(root,datetime(2026,12,1,11,5,tzinfo=timezone.utc))
            write_json(root/'docs/briefing/morning-card.json',self.value())
            self.assertTrue(ops.gate(root,NOW+timedelta(days=1))['refresh'])

    def test_live_poll_recovers_network_and_stale_cache_then_matches_exact_id(self):
        with TemporaryDirectory() as td:
            root=Path(td);value=self.value();write_json(root/'docs/briefing/morning-card.json',value)
            old=dict(value,edition_id='old');fetch=Mock(side_effect=[URLError('down'),BytesIO(json.dumps(old).encode()),BytesIO(json.dumps(value).encode())]);sleep=Mock()
            ops.verify_live('test-edition',root=root,attempts=3,interval=1,fetch=fetch,sleep=sleep)
            self.assertEqual(sleep.call_count,2)
            urls=[c.args[0].full_url for c in fetch.call_args_list]
            self.assertEqual(len(set(urls)),3)
            self.assertTrue(all('edition_id=test-edition' in u for u in urls))

    def test_wrong_edition_or_date_fail_instead_of_green_build_request(self):
        with TemporaryDirectory() as td:
            root=Path(td);value=self.value();write_json(root/'docs/briefing/morning-card.json',value)
            for changes in [dict(edition_id='old'),dict(decision_date='2026-09-27'),dict(kind='test'),dict(status='research_incomplete')]:
                fetch=Mock(return_value=BytesIO(json.dumps(dict(value,**changes)).encode()))
                with self.assertRaises(RuntimeError):ops.verify_live('test-edition',root=root,attempts=1,fetch=fetch)
            fetch=Mock()
            with self.assertRaises(RuntimeError):ops.verify_live('changed-locally',root=root,fetch=fetch)
            fetch.assert_not_called()

if __name__=='__main__':unittest.main()
