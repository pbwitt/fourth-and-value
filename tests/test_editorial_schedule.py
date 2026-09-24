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


if __name__=='__main__':
    unittest.main()
