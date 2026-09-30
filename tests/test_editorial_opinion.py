"""Opinion generation must share private review without a market-data prerequisite."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import editorial_writer as w
import editorial_schedule as sched


class OpinionGenerationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.row = dict(id='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', user_id='reader',
                        requires_review=True, owner_idea=False, kind='opinion', sport='NFL',
                        status='submitted', idea='Compare the Browns and Steelers in the trenches.',
                        title='', body='', byline='Fourth & Value contributor', updated_at='v1',
                        created_at=self.now.isoformat(), research_requested_at=self.now.isoformat(),
                        write_now_requested_at=self.now.isoformat(), write_now_publish=True)
        date = self.now.date().isoformat()
        self.sources = [dict(id=key, title='Browns and Steelers reporting', url=url,
                             published_at=date, excerpt='Verified reporting. ' * 60)
                        for key, url in [('a', 'https://www.nfl.com/news/trenches'),
                                         ('b', 'https://www.cbssports.com/nfl/news/trenches/')]]
        self.article = dict(publish=True, title='The line play that shapes a rivalry',
                            excerpt='A sourced argument about what line play means for this rivalry.',
                            sections=[dict(heading=heading, text='Supported argument. ' * 75,
                                           source_ids=['a', 'b'])
                                      for heading in ['Thesis', 'Evidence', 'Countercase', 'Conclusion']],
                            sources=[{k: v for k, v in s.items() if k != 'excerpt'} for s in self.sources],
                            market_ids=[])

    def packet(self):
        return dict(w.opinion.packet('NFL', self.now), reporting=copy.deepcopy(self.sources))

    def test_opinion_uses_verified_sources_without_market_references(self):
        self.assertEqual(w.validate(copy.deepcopy(self.article), {}, self.packet(), self.now), 600)
        analysis = dict(self.packet(), article_kind='analysis')
        with self.assertRaisesRegex(ValueError, 'current market/model evidence'):
            w.validate(copy.deepcopy(self.article), {}, analysis, self.now)

    def test_opinion_rejects_invented_sources_dates_citations_and_market_ids(self):
        for change in ['source_url', 'source_id', 'date', 'citations', 'market']:
            with self.subTest(change=change):
                article = copy.deepcopy(self.article)
                if change == 'source_url': article['sources'][0]['url'] = 'https://example.com/invented'
                if change == 'source_id': article['sources'][0]['id'] = 'fake'
                if change == 'date': article['sources'][0]['published_at'] = '2099-01-01'
                if change == 'citations': article['sections'][0]['source_ids'] = []
                if change == 'market': article['market_ids'] = ['fake-market']
                with self.assertRaises(ValueError): w.validate(article, {}, self.packet(), self.now)

    def run_writer(self, *, sources=True, audit=True, rewrite=False, daily=False):
        row = dict(self.row)
        if rewrite: row.update(title='Existing opinion', body='Keep this draft until the replacement passes.')
        if daily: row['write_now_requested_at'] = None
        response = dict(status='completed', usage=dict(input_tokens=1, output_tokens=1))
        with TemporaryDirectory() as directory:
            root = Path(directory); docs = root / 'docs'; state = docs / 'editorial/runs'
            state.mkdir(parents=True)
            with patch.dict(w.ed.CFG, {'writing_enabled': True}), \
                 patch.object(w.ed, 'ROOT', root), patch.object(w.ed, 'DOCS', docs), \
                 patch.object(w, 'STATE', state), patch.object(w.budget, 'PATH', root / 'budget.json'), \
                 patch.object(w.budget, 'checkpoint'), patch.object(w.ed, 'render_home'), \
                 patch.object(w, 'slots', return_value=[('NFL', 'news-market')]), \
                 patch.object(w, 'evidence') as market_data, \
                 patch.object(w.ideas, 'pending', return_value=[row]), \
                 patch.object(w.ideas, 'get', return_value=row), \
                 patch.object(w.ideas, 'claim', return_value=True) as claim, \
                 patch.object(w.ideas, 'save_draft') as save, patch.object(w.ideas, 'waiting') as waiting, \
                 patch.object(w.ideas, 'fail') as fail, \
                 patch.object(w.reporting, 'collect', return_value=self.sources if sources else []), \
                 patch.object(w, 'call_api', return_value=response) as api, \
                 patch.object(w, 'response_text', side_effect=[json.dumps(self.article), json.dumps({'pass': audit})]):
                w.run(self.now, limit=1, idea_id=None if daily else row['id'], publish_own=True)
            market_data.assert_not_called()
            self.assertFalse((docs / 'editorial/articles').exists())
            self.assertFalse((docs / 'editorial/published.json').exists())
            if sources and audit:
                self.assertEqual(api.call_count, 2); save.assert_called_once(); fail.assert_not_called()
                self.assertFalse(save.call_args.kwargs['market_snapshot'])
                self.assertEqual(save.call_args.args[0]['kind'], 'opinion')
                request = api.call_args_list[0].args[0]
                self.assertIn('opinion editor', request['instructions'])
                assignment = json.loads(request['input'])
                self.assertEqual('current_draft' in assignment, rewrite)
            elif not sources:
                api.assert_not_called(); claim.assert_not_called(); save.assert_not_called(); waiting.assert_called_once()
                self.assertFalse((root / 'budget.json').exists())
            else:
                self.assertEqual(api.call_count, 2); save.assert_not_called(); fail.assert_called_once()
                self.assertEqual(fail.call_args.args[0]['body'], row['body'])

    def test_reader_opinion_returns_private_draft_despite_legacy_publish_flag(self):
        self.run_writer()

    def test_opinion_rewrite_uses_existing_draft_and_keeps_manual_review(self):
        self.run_writer(rewrite=True)

    def test_failed_audit_preserves_existing_opinion(self):
        self.run_writer(rewrite=True, audit=False)

    def test_missing_sources_prevents_paid_calls(self):
        self.run_writer(sources=False)

    def test_daily_accepted_opinion_is_drafted_privately_without_model_board(self):
        self.run_writer(daily=True)

    def test_opinion_pending_requires_editor_acceptance_for_reader(self):
        row = dict(self.row, write_now_requested_at=None, research_requested_at=None)
        with patch.object(w.ideas, 'configured', return_value=True), \
             patch.object(w.ideas, 'request', return_value=[row]), \
             patch.object(w.ideas, 'is_editor', return_value=False):
            self.assertEqual(w.ideas.pending(self.now), [])
            row['research_requested_at'] = self.now.isoformat()
            self.assertEqual(len(w.ideas.pending(self.now)), 1)

    def test_generated_rewrite_preserves_the_saved_byline(self):
        with patch.object(w.ideas, 'request', return_value=[{}]) as request:
            w.ideas.save_draft(self.row, self.article, self.now, market_snapshot=False)
        saved = request.call_args.kwargs['json']
        self.assertEqual(saved['byline'], 'Fourth & Value contributor')
        self.assertEqual(saved['status'], 'review')
        self.assertNotIn('kind', saved)

    def test_opinion_planner_requests_writer_without_sports_model_refresh(self):
        with patch.object(sys, 'argv', ['planner', '--event-name', 'workflow_dispatch', '--idea-id', self.row['id']]), \
             patch.object(sched, 'plan', return_value={}), \
             patch.object(sched, 'config', return_value={'writing_enabled': True}), \
             patch.object(w.ideas, 'get', return_value=self.row), \
             patch.object(sched, 'write_outputs') as outputs, patch('builtins.print'):
            sched.main()
        result = outputs.call_args.args[0]
        self.assertEqual(result['mode'], 'requested-idea'); self.assertTrue(result['writer_eligible'])
        self.assertFalse(result['refresh_mlb']); self.assertFalse(result['refresh_nfl']); self.assertFalse(result['refresh_briefing'])


if __name__ == '__main__': unittest.main()
