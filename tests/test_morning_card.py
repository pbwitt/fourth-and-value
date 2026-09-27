"""Recovery distinguishes no-pick days, failures and immutable published editions."""
from copy import deepcopy
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
from nhl.v2.data import iso, write_json

NOW=datetime(2026,9,27,12,tzinfo=timezone.utc)


def empty_feeds(now=NOW):
    return {s:dict(status='waiting_for_markets',rows=[],last_success_at=iso(now),generated_at=iso(now)) for s in ('NFL','MLB','NHL')}


def empty_reviews():
    return dict(discovery_status='not_requested',sports={s:dict(review_status='no_candidates',candidates=[],candidate_count=0,reviewed_count=0) for s in ('NFL','MLB','NHL')})


class MorningCardTests(unittest.TestCase):
    def test_real_selector_archives_completed_empty_edition_and_versions(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            first=card.publish_card(empty_feeds(), empty_reviews(), NOW, root=root)
            self.assertEqual(first['rows'],[])
            self.assertEqual(first['status'],'no_reviewed_candidates')
            self.assertTrue(card.existing_today(root,NOW))
            self.assertTrue(first['research']['completed'])
            archive=root/'docs'/first['archive_url'].lstrip('/')
            original=archive.read_bytes()
            second=card.publish_card({}, {}, NOW+timedelta(hours=8),root=root,kind='test')
            self.assertEqual(second['status'],'research_incomplete')
            self.assertFalse(card.existing_today(root,NOW+timedelta(hours=8)))
            self.assertNotEqual(first['edition_id'],second['edition_id'])
            self.assertEqual(archive.read_bytes(),original)
            self.assertFalse(card.existing_today(root,NOW+timedelta(days=1)))

    def test_failures_record_sport_counts_and_allow_recovery(self):
        for status in ['api_key_unavailable','review_unavailable','expired_during_research','budget_exhausted']:
            with self.subTest(status=status),TemporaryDirectory() as td:
                root=Path(td); reviews=empty_reviews()
                reviews['sports']['NFL'].update(review_status=status,candidate_count=3,reviewed_count=0)
                value=card.publish_card(empty_feeds(),reviews,NOW,root=root)
                self.assertEqual(value['status'],'research_incomplete')
                self.assertEqual(value['research']['sports']['NFL']['candidate_count'],3)
                self.assertEqual(value['research']['sports']['NFL']['pending_count'],3)
                self.assertFalse(card.existing_today(root,NOW))
                fixed=card.publish_card(empty_feeds(),empty_reviews(),NOW+timedelta(minutes=1),root=root)
                self.assertEqual(fixed['status'],'no_reviewed_candidates')
                self.assertTrue(card.existing_today(root,NOW+timedelta(minutes=1)))

    def test_completed_pass_reviews_and_bounded_partial_coverage_are_legitimate(self):
        with TemporaryDirectory() as td:
            reviews=empty_reviews()
            reviews['sports']['NFL'].update(review_status='review_limit_reached',candidate_count=100,reviewed_count=24,batch_statuses=['completed'])
            value=card.publish_card(empty_feeds(),reviews,NOW,root=Path(td))
            self.assertEqual(value['status'],'no_reviewed_candidates')
            self.assertEqual(value['research']['sports']['NFL']['pending_count'],76)
            reviews['sports']['NFL']['batch_statuses'].append('review_unavailable')
            failed=card.publish_card(empty_feeds(),reviews,NOW,root=Path(td))
            self.assertEqual(failed['status'],'research_incomplete')

    def test_failed_refresh_and_discovery_not_hidden_by_empty_selection(self):
        with TemporaryDirectory() as td:
            reviews=empty_reviews();reviews['discovery_status']='discovery_unavailable'
            value=card.publish_card(empty_feeds(),reviews,NOW,root=Path(td),refresh_results={'MLB':'failure'})
            self.assertEqual(value['status'],'research_incomplete')
            self.assertIn('MLB: refresh failed or did not finish',value['research']['issues'])
            self.assertIn('Independent discovery did not complete',value['research']['issues'])

    def test_every_api_timeout_publishes_failure_and_keeps_normal_budget(self):
        from test_analyst_review import feeds, CONFIG, NOW as asof
        from nhl.v2 import astra, evidence
        config=deepcopy(CONFIG);config['discovery_enabled']=False
        with TemporaryDirectory() as td:
            root=Path(td);archive=root/'archive';public=root/'reviews.json'
            original=feeds();original['NHL']=empty_feeds(asof)['NHL'];frozen={}
            with patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic-not-a-key'}),patch.object(astra,'checkpoint'),patch.object(astra,'call_api',side_effect=TimeoutError) as call,patch.object(evidence,'collect',return_value=([],{'status':'no_usable_reporting'})):
                result=review.prepare(original,asof,config,run_review=True,archive=archive,public=public,clock=lambda:asof,publication_feeds=frozen)
            self.assertGreater(call.call_count,0)
            self.assertLessEqual(result['budget']['charged_or_reserved_usd'],2.75)
            value=card.publish_card(frozen,result,asof,root=root)
            self.assertEqual(value['status'],'research_incomplete')
            self.assertEqual(value['research']['reviewed_count'],0)
            self.assertFalse(card.existing_today(root,asof))
            # Mid-run source mutations cannot change the frozen publication input.
            original['NFL']['rows'][0]['point']=999
            self.assertNotEqual(frozen['NFL']['rows'][0]['point'],999)

    def test_completed_assessments_publish_from_the_reviewed_snapshot(self):
        from test_analyst_review import feeds, response, CONFIG, NOW as asof
        from nhl.v2 import astra, evidence
        config=deepcopy(CONFIG);config['discovery_enabled']=False
        def completed(request):
            result=response(json.loads(request['input']))
            content=result['output'][0]['content'][0]
            body=json.loads(content['text']);q=body['reviews'][0]
            q['evidence']=[];q['assessment'].update(verdict='consider',blocking_checks=[])
            content['text']=json.dumps(body)
            return result
        with TemporaryDirectory() as td:
            root=Path(td);original=feeds();original['NHL']=empty_feeds(asof)['NHL'];frozen={}
            with patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic-not-a-key'}),patch.object(astra,'checkpoint'),patch.object(astra,'call_api',side_effect=completed),patch.object(evidence,'collect',return_value=([],{'status':'no_usable_reporting'})):
                result=review.prepare(original,asof,config,run_review=True,archive=root/'archive',public=root/'reviews.json',clock=lambda:asof,publication_feeds=frozen)
            value=card.publish_card(frozen,result,asof,root=root)
            self.assertEqual(value['status'],'published')
            self.assertEqual(len(value['rows']),2)
            self.assertEqual(value['research']['reviewed_count'],2)
            self.assertTrue(card.existing_today(root,asof))

    def test_exhausted_discovery_retry_cannot_become_completed_empty_day(self):
        for status in ('budget_exhausted','already_attempted'):
            with self.subTest(status=status),TemporaryDirectory() as td:
                reviews=empty_reviews()
                reviews.update(discovery_status=status,discovery_coverage={'slate_games':12,'submitted_games':[]})
                value=card.publish_card(empty_feeds(),reviews,NOW,root=Path(td))
                self.assertEqual(value['status'],'research_incomplete')
                reviews['discovery_coverage']['submitted_games']=[['MLB','1']]
                prior_completed=card.publish_card(empty_feeds(),reviews,NOW,root=Path(td))
                self.assertEqual(prior_completed['status'],'no_reviewed_candidates')

    def test_invalid_future_or_incomplete_cards_do_not_block_recovery(self):
        with TemporaryDirectory() as td:
            root=Path(td);original=card.publish_card(empty_feeds(),empty_reviews(),NOW,root=root)
            for changes in [dict(status='research_incomplete'),dict(schema_version=2),dict(edition_id=''),dict(rows=None),dict(published_at=iso(NOW+timedelta(minutes=1)))]:
                write_json(root/'docs/briefing/morning-card.json',dict(original,**changes))
                self.assertFalse(card.existing_today(root,NOW))

    def test_repeat_and_outside_window_stop_before_feeds_or_paid_requests(self):
        with TemporaryDirectory() as td:
            root=Path(td);cfg=root/'config.json'
            cfg.write_text(json.dumps({'sessions':{'morning':[7,12]}}))
            card.publish_card(empty_feeds(),empty_reviews(),NOW,root=root)
            with patch.object(review,'ROOT',root),patch.object(review,'CONFIG',cfg),patch.object(review,'load_feeds') as feeds,patch.object(review,'prepare') as paid,patch.object(review,'datetime') as clock:
                clock.now.return_value=NOW
                with patch('sys.argv',['review','--astra','--publish-card']):self.assertIsNone(review.main())
                clock.now.return_value=NOW+timedelta(hours=8)
                with patch('sys.argv',['review','--astra','--publish-card','--replace-card']):self.assertEqual(review.main(),1)
                feeds.assert_not_called();paid.assert_not_called()

    def test_test_edition_preserves_budget_and_incomplete_run_exits_nonzero(self):
        with TemporaryDirectory() as td:
            root=Path(td);cfg=root/'config.json'
            cfg.write_text(json.dumps({'sessions':{'morning':[7,12]},'daily_budget_usd':2.75}))
            with patch.object(review,'ROOT',root),patch.object(review,'CONFIG',cfg),patch.object(review,'load_feeds',return_value={}),patch.object(review,'prepare',return_value={'sports':{}}) as paid,patch.object(review,'datetime') as clock:
                clock.now.return_value=NOW+timedelta(hours=8)
                with patch('sys.argv',['review','--astra','--publish-card','--test-edition','--replace-card']):self.assertEqual(review.main(),1)
                self.assertEqual(paid.call_args.args[2]['daily_budget_usd'],2.75)
                value=json.loads((root/'docs/briefing/morning-card.json').read_text())
                self.assertEqual(value['kind'],'test')
                self.assertEqual(value['status'],'research_incomplete')

    def test_schedules_gate_feeds_allow_recovery_and_propagate_results(self):
        workflow=card.ROOT/'.github/workflows'
        morning=(workflow/'morning-picks.yml').read_text()
        self.assertIn("cron: '5,35 7,8 * * *'",morning)
        self.assertIn('needs: [gate, nfl, mlb, nhl]',morning)
        self.assertEqual(morning.count("if: needs.gate.outputs.refresh == 'true'"),3)
        self.assertIn("needs.gate.result == 'success'",morning)
        self.assertIn('needs.mlb.result',morning)
        for sport in ('nhl','mlb'):
            source=(workflow/f'{sport}-daily.yml').read_text()
            self.assertIn("cron: '30 16 * * *'",source)
            self.assertIn('timezone: America/New_York',source)
        research=(workflow/'analyst-daily.yml').read_text()
        self.assertNotIn('workflow_run:',research)
        self.assertNotIn('cron:',research)
        self.assertIn('continue-on-error: true',research)
        self.assertIn('steps.research.outcome',research)
        self.assertIn('--verify-live "$EXPECTED_EDITION"',research)

if __name__=='__main__':unittest.main()
