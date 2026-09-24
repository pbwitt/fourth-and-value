"""Private suggestions use existing budget and cannot bypass reader approval."""
from datetime import datetime,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch,Mock
import sys,unittest,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_ideas as ideas
import editorial_writer as w

class IdeaTests(unittest.TestCase):
    now=datetime(2026,9,25,10,tzinfo=timezone.utc)
    def row(self,**updates):
        return dict(id='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',user_id='reader',sport='NFL',kind='analysis',
            idea='Investigate Falcons injuries',body='',status='submitted',created_at='2026-09-24T10:00:00Z',updated_at='v1',**updates)
    def test_readers_require_research_acceptance_and_opinion_never_auto_writes(self):
        external=self.row(requires_review=True)
        with patch.object(ideas,'configured',return_value=True),patch.object(ideas,'request',return_value=[external]),patch.object(ideas,'is_editor',return_value=False):
            self.assertEqual(ideas.pending(self.now),[])
            external['research_requested_at']='2026-09-24T11:00:00Z'
            self.assertEqual(len(ideas.pending(self.now)),1)
            external['kind']='opinion'
            self.assertEqual(ideas.pending(self.now),[])
    def test_owner_idea_needs_only_topic_and_sport(self):
        with patch.object(ideas,'configured',return_value=True),patch.object(ideas,'request',return_value=[self.row()]),patch.object(ideas,'is_editor',return_value=True):
            self.assertTrue(ideas.pending(self.now)[0]['owner_idea'])
    def test_reader_origin_still_requires_review_after_role_upgrade(self):
        with patch.object(ideas,'configured',return_value=True),patch.object(ideas,'request',return_value=[self.row(requires_review=True)]),patch.object(ideas,'is_editor',return_value=True):
            self.assertEqual(ideas.pending(self.now),[])
    def test_topic_matches_only_relevant_games(self):
        games=[{'id':'a','game':'Atlanta Falcons @ Green Bay Packers'},{'id':'b','game':'Dallas Cowboys @ New York Giants'}]
        selected,terms=ideas.context(self.row(),{'markets':games})
        self.assertEqual(selected,[games[0]]);self.assertEqual(terms,('falcons','packers'))
    def test_generated_reader_draft_is_saved_as_review_not_approved(self):
        article={'title':'Title','sections':[{'heading':'Context','text':'Verified facts'}],'sources':[{'url':'https://example.com/source'}]}
        with patch.object(ideas,'request',return_value=[{}]) as req:
            ideas.save_draft(self.row(),article)
            data=req.call_args.kwargs['json']
            self.assertEqual(data['status'],'review')
            self.assertEqual(data['byline'],'Fourth & Value')
            self.assertNotIn('approved_by',data)
    def test_email_does_not_expose_private_content_and_marks_delivery(self):
        row=dict(self.row(),requires_review=True,notification_sent_at=None)
        env={'RESEND_API_KEY':'test','EDITORIAL_NOTIFY_FROM':'desk@example.com','EDITORIAL_NOTIFY_EMAIL':'owner@example.com'}
        with patch.object(ideas,'configured',return_value=True),patch.object(ideas,'request',side_effect=[[row],[]]) as db,patch.dict(ideas.os.environ,env),patch.object(ideas.requests,'post',return_value=Mock(ok=True)) as post:
            ideas.notify(self.now)
            self.assertNotIn(row['idea'],json.dumps(post.call_args.kwargs))
            self.assertIn('Idempotency-Key',post.call_args.kwargs['headers'])
            self.assertIn('notification_sent_at',db.call_args.kwargs['json'])
    def test_email_failure_does_not_mark_sent(self):
        row=dict(self.row(),requires_review=True,notification_sent_at=None)
        env={'RESEND_API_KEY':'test','EDITORIAL_NOTIFY_FROM':'desk@example.com','EDITORIAL_NOTIFY_EMAIL':'owner@example.com'}
        with patch.object(ideas,'configured',return_value=True),patch.object(ideas,'request',return_value=[row]) as db,patch.dict(ideas.os.environ,env),patch.object(ideas.requests,'post',return_value=Mock(ok=False,status_code=500)):
            with self.assertRaises(RuntimeError):ideas.notify(self.now)
            self.assertEqual(db.call_count,1)
    def test_reader_draft_never_writes_public_article(self):
        row=dict(self.row(),owner_idea=False,research_requested_at='2026-09-24T11:00:00Z')
        article={'publish':True,'title':'Private draft','excerpt':'Private summary','market_ids':['q1'],'sections':[{'heading':'Context','text':'x'}],'sources':[]}
        response={'status':'completed','usage':{'input_tokens':1,'output_tokens':1}}
        with TemporaryDirectory() as td:
            root=Path(td);docs=root/'docs';state=docs/'editorial/runs';state.mkdir(parents=True)
            w.ed.write_json(state/'2026-09-25.json',{'allocation':[['NFL','idea:'+row['id']],['MLB','news-market']],'slots':{'1-mlb':{'status':'skipped'}}})
            packet={'markets':[{'id':'q1'}],'model_rows':[],'data_readiness':{'ready':True}}
            with patch.dict(w.ed.CFG,{'writing_enabled':True}),patch.object(w.ed,'DOCS',docs),patch.object(w.ed,'ROOT',root),patch.object(w,'STATE',state),patch.object(w.budget,'PATH',root/'budget.json'),patch.object(w.budget,'checkpoint'),patch.object(w.ed,'render_home'),patch.object(w,'evidence',return_value=packet),patch.object(w,'compact',side_effect=lambda p:p),patch.object(w,'select_target',return_value=None),patch.object(w.reporting,'collect',return_value=[{'title':'News'}]),patch.object(w.ideas,'get',return_value=row),patch.object(w.ideas,'claim',return_value=True),patch.object(w.ideas,'save_draft') as save,patch.object(w.ideas,'finish') as finish,patch.object(w,'call_api',return_value=response),patch.object(w,'response_text',side_effect=[json.dumps(article),json.dumps({'pass':True})]),patch.object(w,'validate',return_value=600):
                w.run(self.now)
            save.assert_called_once();finish.assert_not_called()
            self.assertFalse((docs/'editorial/published.json').exists())
            self.assertFalse((docs/'editorial/articles').exists())
            ledger=json.loads((state/'2026-09-25.json').read_text())
            self.assertEqual(ledger['slots']['0-nfl']['status'],'review')
            self.assertNotIn('Private draft',json.dumps(ledger))

    def test_owner_idea_publishes_without_draft_approval(self):
        row=dict(self.row(),owner_idea=True,research_requested_at='2026-09-24T11:00:00Z')
        article={'publish':True,'title':'Private draft','excerpt':'Private summary','market_ids':[],'sections':[{'heading':'Context','text':'x'}],'sources':[]}
        response={'status':'completed','usage':{'input_tokens':1,'output_tokens':1}}
        with TemporaryDirectory() as td:
            root=Path(td);docs=root/'docs';state=docs/'editorial/runs';state.mkdir(parents=True)
            w.ed.write_json(state/'2026-09-25.json',{'allocation':[['NFL','idea:'+row['id']],['MLB','news-market']],'slots':{'1-mlb':{'status':'skipped'}}})
            packet={'markets':[{'id':'q1'}],'model_rows':[],'data_readiness':{'ready':True}}
            with patch.dict(w.ed.CFG,{'writing_enabled':True}),patch.object(w.ed,'DOCS',docs),patch.object(w.ed,'ROOT',root),patch.object(w,'STATE',state),patch.object(w.budget,'PATH',root/'budget.json'),patch.object(w.budget,'checkpoint'),patch.object(w.ed,'render_home'),patch.object(w,'evidence',return_value=packet),patch.object(w,'compact',side_effect=lambda p:p),patch.object(w,'select_target',return_value=None),patch.object(w.reporting,'collect',return_value=[{'title':'News'}]),patch.object(w.ideas,'get',return_value=row),patch.object(w.ideas,'claim',return_value=True),patch.object(w.ideas,'save_draft') as save,patch.object(w.ideas,'finish') as finish,patch.object(w,'call_api',return_value=response),patch.object(w,'response_text',side_effect=[json.dumps(article),json.dumps({'pass':True})]),patch.object(w,'validate',return_value=600):
                w.run(self.now)
            save.assert_not_called();finish.assert_called_once()
            self.assertTrue((docs/'editorial/published.json').exists())
            self.assertTrue((docs/'editorial/articles').exists())
            ledger=json.loads((state/'2026-09-25.json').read_text())
            self.assertEqual(ledger['slots']['0-nfl']['status'],'published')
            self.assertNotIn('Private draft',json.dumps(ledger))

if __name__=='__main__':unittest.main()
