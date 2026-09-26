import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch, Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_diagnostics as diag

class DiagnosticTests(unittest.TestCase):
    now=datetime(2026,9,26,12,tzinfo=timezone.utc)
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.write('config/editorial.json',{'writing_enabled':True,'writer':{'daily_story_limit':2}})
    def write(self,path,data):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data));return p
    def test_no_run_or_data_is_overdue_not_green(self):
        r=diag.build_report(self.root,self.now)
        self.assertEqual(r['status'],'overdue');self.assertEqual(r['saved'],0)
        self.assertEqual(r['live_check'],'not_checked')
        self.assertTrue(all(d['prices']=='stale' for d in r['data']))
        self.assertTrue(r['recovery']['eligible'])
    def test_two_files_without_live_verification_are_not_delivered(self):
        rows=[{'date':'2026-09-26','kind':'Analysis','url':f'/editorial/articles/{n}.html'} for n in ('a','b')]
        self.write('docs/editorial/published.json',rows)
        for row in rows:self.write('docs'+row['url'],{})
        r=diag.build_report(self.root,self.now)
        self.assertEqual(r['status'],'saved_not_verified')
        r=diag.build_report(self.root,self.now,run={'stages':{'live_delivery':{'status':'success'}}})
        self.assertEqual(r['status'],'delivered')
        self.assertFalse(r['recovery']['eligible'])
    def test_odds_and_models_are_separate_and_expired_games_do_not_count(self):
        self.write('docs/briefing/latest.json',{'generated_at':'2026-09-26T11:30:00Z','coverage':{'MLB':'1 upcoming game'},'games':[{'sport':'MLB','commence_time':'2026-09-26T10:00:00Z'}]})
        self.write('docs/mlb/data/latest.json',{'model_checked_at':'2026-09-26T11:30:00Z','model_summary':{'history_through':'2026-09-24'}})
        mlb=next(d for d in diag.build_report(self.root,self.now)['data'] if d['sport']=='MLB')
        self.assertEqual(mlb['prices'],'no_markets');self.assertEqual(mlb['model'],'stale')
    def test_private_text_and_api_usage_never_enter_report(self):
        self.write('docs/editorial/runs/2026-09-26.json',{'allocation':[['NFL','idea:private-uuid']],
            'slots':{'0-nfl':{'status':'skipped','reason':'Private idea SECRET SUBJECT','usage':{'secret':'TOKEN'},'audit_reason':'SECRET DRAFT'}},
            'data_skips':{'NFL':'private idea SECRET SUBJECT'}})
        payload=json.dumps(diag.build_report(self.root,self.now))
        for value in ('SECRET','TOKEN','private-uuid','audit_reason','usage'):self.assertNotIn(value,payload)
    def test_fallback_unknown_is_not_reported_as_success(self):
        self.write('docs/editorial/runs/2026-09-26.json',{'slots':{'0-nfl':{'status':'published'}}})
        self.assertIsNone(diag.build_report(self.root,self.now)['articles'][0]['source_fallback'])
        self.write('docs/editorial/runs/2026-09-26.json',{'slots':{'0-nfl':{'status':'published','source_check':{'fallback_used':True,'hosts':['www.nfl.com','www.cbssports.com']}}}})
        self.assertTrue(diag.build_report(self.root,self.now)['articles'][0]['source_fallback'])
    def test_store_uses_run_attempt_and_phase_for_idempotency(self):
        report=diag.build_report(self.root,self.now)
        with patch.dict(os.environ,{'SUPABASE_URL':'https://example.supabase.co','SUPABASE_SERVICE_ROLE_KEY':'secret','GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'2'}),patch.object(diag.requests,'post',return_value=Mock(ok=True,status_code=201)) as post:
            diag.store_report(report)
        payload=post.call_args.kwargs['json'];self.assertEqual(payload['id'],'123-2-finish')
        self.assertNotIn('secret',json.dumps(payload))
    def test_missing_schema_has_actionable_error_without_response_body(self):
        with patch.dict(os.environ,{'SUPABASE_URL':'https://example.supabase.co','SUPABASE_SERVICE_ROLE_KEY':'secret'}),patch.object(diag.requests,'post',return_value=Mock(ok=False,status_code=404)):
            with self.assertRaisesRegex(RuntimeError,'one-time'):diag.store_report(diag.build_report(self.root,self.now))

if __name__=='__main__':unittest.main()
