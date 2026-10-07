from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from recap_schedule import expected_end, should_run, completed, run

class RecapScheduleTests(unittest.TestCase):
    def test_delivery_days_and_catchup_are_eastern(self):
        now=datetime(2026,10,8,2,tzinfo=timezone.utc) # Wednesday evening ET
        self.assertEqual(str(expected_end('mlb',now)),'2026-10-04')
        self.assertEqual(str(expected_end('nhl',now)),'2026-10-05')
        self.assertEqual(str(expected_end('nba',now)),'2026-09-30')
        self.assertEqual(str(expected_end('nba',datetime(2026,10,8,12,tzinfo=timezone.utc))),'2026-10-07')
    def test_success_requires_html_and_matching_data(self):
        now=datetime(2026,10,7,12,tzinfo=timezone.utc);end=expected_end('mlb',now)
        with tempfile.TemporaryDirectory() as root:
            stem=Path(root)/f'docs/blog/mlb-recap-{end}';stem.parent.mkdir(parents=True)
            stem.with_suffix('.html').write_text('article')
            self.assertTrue(should_run('mlb',end,now,{},root))
            report=dict(status='published',sport='mlb',start='2026-09-28',end=str(end),generated_at=now.isoformat(),tickets={'unresolved':0},board={})
            stem.with_suffix('.json').write_text(json.dumps({'summary':report}))
            self.assertIsNotNone(completed('mlb',end,root))
            self.assertFalse(should_run('mlb',end,now,{},root))
    def test_failed_unresolved_retry_is_not_repeated_same_eastern_day(self):
        now=datetime(2026,10,6,15,tzinfo=timezone.utc);end=expected_end('mlb',now)
        with tempfile.TemporaryDirectory() as root:
            stem=Path(root)/f'docs/blog/mlb-recap-{end}';stem.parent.mkdir(parents=True)
            stem.with_suffix('.html').write_text('article')
            report=dict(status='published',sport='mlb',start='2026-09-28',end=str(end),generated_at='2026-10-05T15:00:00Z',tickets={'unresolved':1},board={})
            stem.with_suffix('.json').write_text(json.dumps({'summary':report}))
            previous=dict(end=str(end),checked_at='2026-10-06T11:30:00Z',status='failed')
            self.assertFalse(should_run('mlb',end,now,previous,root))
    def test_waiting_retries_with_backoff(self):
        now=datetime(2026,10,7,12,tzinfo=timezone.utc);end=expected_end('nhl',now)
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(should_run('nhl',end,now,dict(end=str(end),checked_at='2026-10-07T11:05:00Z'),root))
            self.assertTrue(should_run('nhl',end,now,dict(end=str(end),checked_at='2026-10-07T09:05:00Z'),root))

    def test_skipped_runs_preserve_failed_attempt_and_published_artifact(self):
        first=datetime(2026,10,6,11,30,tzinfo=timezone.utc);end=expected_end('mlb',first)
        with tempfile.TemporaryDirectory() as root:
            stem=Path(root)/f'docs/blog/mlb-recap-{end}';stem.parent.mkdir(parents=True)
            stem.with_suffix('.html').write_text('article')
            report=dict(status='published',sport='mlb',start='2026-09-28',end=str(end),generated_at='2026-10-05T15:00:00Z',tickets={'unresolved':1},board={})
            stem.with_suffix('.json').write_text(json.dumps({'summary':report}))
            status_path=Path(root)/'docs/recaps/status.json'
            with patch.dict('recap_schedule.DAYS',{'mlb':0},clear=True), \
                    patch('sport_weekly_review.review',side_effect=RuntimeError('feed unavailable')) as review, \
                    patch('sport_weekly_review.rebuild_home') as rebuild, patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError,'feed unavailable'):
                    run(first,root)
                failed=json.loads(status_path.read_text())['sports'][0]
                self.assertEqual(failed['status'],'failed')
                self.assertEqual(failed['checked_at'],first.isoformat())
                self.assertEqual(failed['url'],f'/blog/mlb-recap-{end}.html')
                for minute in (5,35):
                    skipped=run(datetime(2026,10,6,12,minute,tzinfo=timezone.utc),root)
                    self.assertEqual(skipped['sports'][0],failed)
                review.assert_called_once()
                rebuild.assert_not_called()
                with self.assertRaisesRegex(RuntimeError,'feed unavailable'):
                    run(datetime(2026,10,7,11,30,tzinfo=timezone.utc),root)
                self.assertEqual(review.call_count,2)

if __name__=='__main__':unittest.main()
