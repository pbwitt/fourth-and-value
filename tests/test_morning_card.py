"""Publication preserves reviewed decisions without spending on a repeat run."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import morning_card as card
import analyst_review as review

NOW=datetime(2026,9,27,12,tzinfo=timezone.utc)

class MorningCardTests(unittest.TestCase):
    def test_real_selector_archives_empty_edition_and_versions(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            first=card.publish_card({}, {'budget':{'limit_usd':2.75}}, NOW, root=root)
            self.assertEqual(first['rows'],[])
            self.assertEqual(first['status'],'no_reviewed_candidates')
            self.assertTrue(card.existing_today(root,NOW))
            archive=root/'docs'/first['archive_url'].lstrip('/')
            original=archive.read_bytes()
            second=card.publish_card({}, {}, NOW+timedelta(hours=8),root=root,kind='test')
            self.assertNotEqual(first['edition_id'],second['edition_id'])
            self.assertEqual(archive.read_bytes(),original)
            self.assertEqual(json.loads((root/'docs/briefing/morning-card.json').read_text())['kind'],'test')
            self.assertFalse(card.existing_today(root,NOW+timedelta(days=1)))

    def test_repeat_and_outside_window_stop_before_feeds_or_paid_requests(self):
        with TemporaryDirectory() as td:
            root=Path(td);cfg=root/'config.json'
            cfg.write_text(json.dumps({'sessions':{'morning':[7,12]}}))
            with patch.object(review,'ROOT',root),patch.object(review,'CONFIG',cfg),patch.object(review,'load_feeds') as feeds,patch.object(review,'prepare') as paid,patch.object(review,'datetime') as clock,patch.object(card,'existing_today',return_value=True):
                clock.now.return_value=NOW
                with patch('sys.argv',['review','--astra','--publish-card']):review.main()
                clock.now.return_value=NOW+timedelta(hours=8)
                with patch('sys.argv',['review','--astra','--publish-card','--replace-card']):review.main()
                feeds.assert_not_called();paid.assert_not_called()

    def test_test_edition_is_explicit_and_preserves_budget_config(self):
        with TemporaryDirectory() as td:
            root=Path(td);cfg=root/'config.json'
            cfg.write_text(json.dumps({'sessions':{'morning':[7,12]},'daily_budget_usd':2.75}))
            with patch.object(review,'ROOT',root),patch.object(review,'CONFIG',cfg),patch.object(review,'load_feeds',return_value={}),patch.object(review,'prepare',return_value={'sports':{}}) as paid,patch.object(review,'datetime') as clock,patch.object(card,'existing_today',return_value=True),patch.object(card,'publish_card') as publish:
                clock.now.return_value=NOW+timedelta(hours=8)
                with patch('sys.argv',['review','--astra','--publish-card','--test-edition','--replace-card']):review.main()
                self.assertEqual(paid.call_args.args[2]['daily_budget_usd'],2.75)
                self.assertEqual(publish.call_args.kwargs['kind'],'test')

    def test_schedules_have_one_research_entry_and_no_refresh_trigger(self):
        workflow=card.ROOT/'.github/workflows'
        morning=(workflow/'morning-picks.yml').read_text()
        self.assertIn("cron: '0 7 * * *'",morning)
        self.assertIn('needs: [nfl, mlb, nhl]',morning)
        for sport in ('nhl','mlb'):
            source=(workflow/f'{sport}-daily.yml').read_text()
            self.assertIn("cron: '30 16 * * *'",source)
            self.assertIn('timezone: America/New_York',source)
            self.assertNotIn('steps.window',source)
        research=(workflow/'analyst-daily.yml').read_text()
        self.assertNotIn('workflow_run:',research)
        self.assertNotIn('cron:',research)

if __name__=='__main__':unittest.main()
