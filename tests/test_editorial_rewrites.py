from datetime import datetime,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_requests as requests
import editorial_ideas as ideas
import editorial_writer as writer

class RewriteTests(unittest.TestCase):
    now=datetime(2026,9,26,14,tzinfo=timezone.utc)
    identifier='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
    def idea(self):
        return {'id':self.identifier,'sport':'NFL','kind':'analysis','status':'submitted','owner_idea':True,
            'write_now_requested_at':'2026-09-26T14:00:00Z','write_now_publish':True,
            'research_requested_at':'2026-09-26T14:00:00Z','idea':'Falcons matchup. Changes for the next draft: be concise.',
            'title':'Existing title','body':'Existing draft kept until replacement passes.','updated_at':'v2'}
    def prior(self,status='review'):
        return {'date':'2026-09-26','allocation':[['NFL','idea:'+self.identifier]],'last_writer_check':{'at':'2026-09-26T13:00:00Z','status':'completed'},
            'slots':{'0-nfl':{'status':status}}}
    def test_new_explicit_request_gets_distinct_budget_key_without_draft_history(self):
        state=requests.prepare(self.prior(),self.idea(),self.now)
        self.assertEqual(state['slots'],{})
        self.assertTrue(state['reservation_key'].startswith('requested-'+self.identifier+'-'))
        self.assertEqual(state['request_history'][0]['statuses'],{'0-nfl':'review'})
        self.assertNotIn('Existing draft',json.dumps(state))
    def test_duplicate_request_cannot_reset_completed_attempt(self):
        state=requests.prepare(self.prior(),self.idea(),self.now);key=state['reservation_key']
        state['slots']={'0-nfl':{'status':'review'}}
        duplicate=requests.prepare(state,self.idea(),self.now)
        self.assertEqual(duplicate['reservation_key'],key)
        self.assertEqual(duplicate['slots']['0-nfl']['status'],'review')
    def test_uncertain_attempt_cannot_be_silently_repeated(self):
        with self.assertRaisesRegex(ValueError,'uncertain'):requests.prepare(self.prior('started'),self.idea(),self.now)
    def test_legacy_same_request_does_not_reset_paid_state(self):
        row=self.idea();row['write_now_requested_at']='2026-09-26T12:00:00Z'
        state=requests.prepare(self.prior('skipped'),row,self.now)
        self.assertEqual(state['slots']['0-nfl']['status'],'skipped')
        self.assertNotIn('reservation_key',state)
    def test_failed_rewrite_keeps_existing_body_and_requires_review(self):
        with patch.object(ideas,'request',return_value=[{}]) as update:
            ideas.fail(self.idea(),'Factual review failed')
        payload=update.call_args.kwargs['json']
        self.assertEqual(payload['status'],'review');self.assertNotIn('body',payload)
        self.assertIn('current draft is still available',payload['research_error'])
    def test_rewrite_uses_current_draft_and_never_auto_publishes(self):
        row=self.idea();article={'publish':True,'title':'Replacement draft','excerpt':'Replacement summary','market_ids':['q1'],'sections':[{'heading':'Context','text':'New draft text'}],'sources':[]}
        response={'status':'completed','usage':{'input_tokens':1,'output_tokens':1}}
        with TemporaryDirectory() as td:
            root=Path(td);docs=root/'docs';state=docs/'editorial/runs';state.mkdir(parents=True)
            path=state/('requested-'+row['id']+'.json');writer.ed.write_json(path,self.prior())
            packet={'markets':[{'id':'q1'}],'model_rows':[],'data_readiness':{'ready':True}}
            with patch.dict(writer.ed.CFG,{'writing_enabled':True}),patch.object(writer.ed,'DOCS',docs),patch.object(writer.ed,'ROOT',root),patch.object(writer,'STATE',state),patch.object(writer.budget,'PATH',root/'budget.json'),patch.object(writer.budget,'checkpoint'),patch.object(writer.ed,'render_home'),patch.object(writer,'evidence',return_value=packet),patch.object(writer,'compact',side_effect=lambda p:p),patch.object(writer,'select_target',return_value=None),patch.object(writer.reporting,'collect',return_value=[{'title':'News'}]),patch.object(writer.ideas,'get',return_value=row),patch.object(writer.ideas,'claim',return_value=True),patch.object(writer.ideas,'save_draft') as save,patch.object(writer.ideas,'finish') as finish,patch.object(writer,'call_api',return_value=response) as api,patch.object(writer,'response_text',side_effect=[json.dumps(article),json.dumps({'pass':True})]),patch.object(writer,'validate',return_value=600):
                writer.run(self.now,idea_id=row['id'],publish_own=True)
            save.assert_called_once();finish.assert_not_called()
            request=json.loads(api.call_args_list[0].args[0]['input'])
            self.assertEqual(request['current_draft']['body'],row['body'])
            self.assertEqual(request['requested_angle'],row['idea'])
            self.assertEqual(json.loads(path.read_text())['slots']['0-nfl']['status'],'review')
            self.assertFalse((docs/'editorial/published.json').exists())

if __name__=='__main__':unittest.main()
