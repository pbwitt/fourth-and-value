"""State-aware scheduling regressions for the daily editorial workflow."""
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_schedule as sched


class EditorialScheduleTests(unittest.TestCase):
    def make_root(self):
        td=TemporaryDirectory();root=Path(td.name)
        (root/'config').mkdir();(root/'docs/editorial/runs').mkdir(parents=True)
        (root/'docs/briefing').mkdir(parents=True);(root/'docs/mlb/data').mkdir(parents=True)
        (root/'config/editorial.json').write_text(json.dumps({
            'writing_enabled':True,'writer':{'daily_story_limit':2}
        }))
        (root/'docs/editorial/published.json').write_text('[]')
        return td,root

    def write_state(self,root,day,state):
        (root/'docs/editorial/runs'/f'{day}.json').write_text(json.dumps(state))

    def test_hourly_run_after_five_can_rescue_unattempted_slot_without_cron_match(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,24,12,0,tzinfo=timezone.utc)  # 8 AM Eastern
        self.write_state(root,'2026-09-24',{'date':'2026-09-24','allocation':[['NFL','news-market'],['MLB','news-market']],'slots':{}})
        result=sched.plan(root,now,event_name='schedule',event_schedule='17 * * * *')
        self.assertTrue(result['writer_eligible'])
        self.assertEqual(result['mode'],'catch-up')
        self.assertIn('unattempted',result['writer_reason'])

    def test_hourly_run_before_five_does_not_start_paid_writer(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,24,8,30,tzinfo=timezone.utc)  # 4:30 AM Eastern
        result=sched.plan(root,now,event_name='schedule',event_schedule='17 * * * *')
        self.assertFalse(result['writer_eligible'])

    def test_waiting_for_data_remains_retryable_but_started_slot_does_not(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,24,13,0,tzinfo=timezone.utc)
        allocation=[['NFL','news-market'],['MLB','news-market']]
        self.write_state(root,'2026-09-24',{'allocation':allocation,'slots':{
            '0-nfl':{'status':'published'},'1-mlb':{'status':'waiting_for_data'}}})
        waiting=sched.plan(root,now,event_name='schedule')
        self.assertTrue(waiting['writer_eligible'])
        self.write_state(root,'2026-09-24',{'allocation':allocation,'slots':{
            '0-nfl':{'status':'published'},'1-mlb':{'status':'started'}}})
        started=sched.plan(root,now,event_name='schedule')
        self.assertFalse(started['writer_eligible'])
        self.assertEqual(started['writer_reason'],'uncertain_started_slot')

    def test_daily_limit_and_funding_stop_automatic_writer(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,24,13,0,tzinfo=timezone.utc)
        catalog=[
            {'date':'2026-09-24','kind':'Analysis','url':'/a'},
            {'date':'2026-09-24','kind':'Analysis','url':'/b'}]
        (root/'docs/editorial/published.json').write_text(json.dumps(catalog))
        self.assertFalse(sched.plan(root,now,event_name='schedule')['writer_eligible'])
        (root/'docs/editorial/published.json').write_text('[]')
        self.write_state(root,'2026-09-24',{'funding_required':True})
        result=sched.plan(root,now,event_name='schedule')
        self.assertFalse(result['writer_eligible'])
        self.assertEqual(result['writer_reason'],'funding_required')

    def test_mlb_refresh_is_state_and_freshness_driven(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,24,13,0,tzinfo=timezone.utc)
        self.write_state(root,'2026-09-24',{'allocation':[['MLB','news-market']],'slots':{
            '0-mlb':{'status':'waiting_for_data'}}})
        stale=sched.plan(root,now,event_name='schedule')
        self.assertTrue(stale['refresh_mlb'])
        (root/'docs/mlb/data/latest.json').write_text(json.dumps({
            'status':'ready','model_checked_at':'2026-09-24T12:30:00Z'}))
        fresh=sched.plan(root,now,event_name='schedule')
        self.assertFalse(fresh['refresh_mlb'])
        self.assertTrue(fresh['mlb_board_fresh'])

    def test_manual_refresh_can_run_before_morning_window(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,24,7,0,tzinfo=timezone.utc)
        result=sched.plan(root,now,event_name='workflow_dispatch',manual_refresh=True)
        self.assertTrue(result['writer_eligible'])
        self.assertTrue(result['refresh_briefing'])
        self.assertTrue(result['refresh_mlb'])
        self.assertEqual(result['mode'],'manual')

    def test_manual_refresh_respects_writer_kill_switch(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        (root/'config/editorial.json').write_text(json.dumps({
            'writing_enabled':False,'writer':{'daily_story_limit':2}
        }))
        now=datetime(2026,9,24,13,0,tzinfo=timezone.utc)
        result=sched.plan(root,now,event_name='workflow_dispatch',manual_refresh=True)
        self.assertFalse(result['writer_eligible'])
        self.assertTrue(result['refresh_briefing'])
        self.assertTrue(result['refresh_mlb'])
        self.assertEqual(result['writer_reason'],'writing_disabled')

    def test_expected_writer_requires_recent_completed_marker(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,24,13,0,tzinfo=timezone.utc)
        with self.assertRaises(SystemExit):
            sched.verify_writer(root,now,expected=True)
        self.write_state(root,'2026-09-24',{'last_writer_check':{
            'at':'2026-09-24T12:59:00+00:00','status':'started'}})
        with self.assertRaises(SystemExit):
            sched.verify_writer(root,now,expected=True)
        self.write_state(root,'2026-09-24',{'last_writer_check':{
            'at':'2026-09-24T12:59:00+00:00','status':'completed','counts':{'published':0,'waiting_for_data':1}}})
        sched.verify_writer(root,now,expected=True)

    def test_delivery_target_is_630_eastern_in_summer_and_winter(self):
        for month,utc_hour in ((9,10),(12,11)):
            with self.subTest(month=month):
                before=datetime(2026,month,26,utc_hour,29,59,tzinfo=timezone.utc)
                target=datetime(2026,month,26,utc_hour,30,tzinfo=timezone.utc)
                self.assertFalse(sched.delivery_due(before))
                self.assertTrue(sched.delivery_due(target))
                self.assertEqual(sched.delivery_target(target),target)

    def test_delivery_fails_after_deadline_even_if_writer_completed(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,26,12,0,tzinfo=timezone.utc)
        self.write_state(root,'2026-09-26',{'last_writer_check':{
            'at':now.isoformat(),'status':'completed','counts':{'published':1}}})
        with self.assertRaisesRegex(SystemExit,'0/2'):
            sched.verify_delivery(root,now)

    def test_delivery_counts_unique_existing_public_articles(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,26,12,0,tzinfo=timezone.utc)
        articles=root/'docs/editorial/articles';articles.mkdir()
        rows=[{'date':'2026-09-26','kind':'Analysis','url':f'/editorial/articles/{name}.html'} for name in ('a','b')]
        (root/'docs/editorial/published.json').write_text(json.dumps(rows+[rows[0]]))
        (articles/'a.html').write_text('published')
        with self.assertRaisesRegex(SystemExit,'1/2'):
            sched.verify_delivery(root,now)
        (articles/'b.html').write_text('published')
        self.assertEqual(sched.verify_delivery(root,now)['status'],'complete')

    def test_before_deadline_missing_delivery_is_pending(self):
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,26,10,0,tzinfo=timezone.utc)
        self.assertEqual(sched.verify_delivery(root,now)['status'],'pending')

    def test_live_delivery_retries_propagation_and_checks_article_title(self):
        from io import BytesIO
        from urllib.error import URLError
        from unittest.mock import patch
        td,root=self.make_root();self.addCleanup(td.cleanup)
        now=datetime(2026,9,26,12,tzinfo=timezone.utc)
        (root/'docs/editorial/published.json').write_text(json.dumps([
            {'date':'2026-09-26','kind':'Analysis','title':'Actual story',
             'url':'/editorial/articles/today.html'}]))
        with patch.object(sched,'urlopen',side_effect=[URLError('not deployed'),BytesIO(b'<title>Actual story | Fourth &amp; Value</title>')]) as fetch,patch.object(sched.time,'sleep'):
            sched.verify_live(root,now,attempts=2)
        self.assertEqual(fetch.call_count,2)
        with patch.object(sched,'urlopen',return_value=BytesIO(b'<title>Not found</title>')):
            with self.assertRaisesRegex(SystemExit,'deployment not verified'):
                sched.verify_live(root,now,attempts=1)


if __name__=='__main__':
    unittest.main()
