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

    def test_one_dropped_review_is_counted_without_failing_the_edition(self):
        with TemporaryDirectory() as td:
            reviews=empty_reviews()
            reviews['sports']['NHL'].update(review_status='partially_reviewed',candidate_count=3,reviewed_count=2,
                batch_statuses=['completed_with_rejections'],
                candidates=[dict(review_rejected=dict(reason="assessment.model_case: percent figure '±10%'"))])
            value=card.publish_card(empty_feeds(),reviews,NOW,root=Path(td))
            self.assertEqual(value['status'],'no_reviewed_candidates')
            self.assertEqual(value['research']['sports']['NHL']['rejected_count'],1)
            self.assertEqual(value['research']['issues'],[])

    def test_failed_refresh_and_discovery_not_hidden_by_empty_selection(self):
        with TemporaryDirectory() as td:
            reviews=empty_reviews();reviews['discovery_status']='discovery_unavailable'
            value=card.publish_card(empty_feeds(),reviews,NOW,root=Path(td),refresh_results={'MLB':'failure'})
            self.assertEqual(value['status'],'research_incomplete')
            self.assertIn('MLB: refresh failed or did not finish',value['research']['issues'])
            self.assertIn('Independent discovery did not complete',value['research']['issues'])

    def test_reused_fresh_nfl_refresh_is_not_a_failure(self):
        # A recovery start reuses a recent NFL refresh instead of buying it again.
        with TemporaryDirectory() as td:
            value=card.publish_card(empty_feeds(),empty_reviews(),NOW,root=Path(td),refresh_results={'NFL':'reused_fresh','MLB':'success','NHL':'success'})
            self.assertNotIn('NFL: refresh failed or did not finish',value['research']['issues'])
            self.assertEqual(value['research']['sports']['NFL']['refresh_status'],'reused_fresh')

    def test_every_api_timeout_publishes_failure_and_keeps_normal_budget(self):
        from test_analyst_review import feeds, CONFIG, NOW as asof
        from nhl.v2 import astra, evidence
        config=deepcopy(CONFIG);config['discovery_enabled']=False
        with TemporaryDirectory() as td:
            root=Path(td);archive=root/'archive';public=root/'reviews.json'
            original=feeds();original['NHL']=empty_feeds(asof)['NHL'];frozen={}
            with patch.dict('os.environ',{review.model_api.key_name(review.astra.MODEL):'synthetic-not-a-key'}),patch.object(astra,'checkpoint'),patch.object(astra,'call_api',side_effect=TimeoutError) as call,patch.object(evidence,'collect',return_value=([],{'status':'no_usable_reporting'})):
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
            with patch.dict('os.environ',{review.model_api.key_name(review.astra.MODEL):'synthetic-not-a-key'}),patch.object(astra,'checkpoint'),patch.object(astra,'call_api',side_effect=completed),patch.object(evidence,'collect',return_value=([],{'status':'no_usable_reporting'})):
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

    def test_normal_run_replaces_test_then_skips_repeats_preserving_archive_and_budget(self):
        with TemporaryDirectory() as td:
            root=Path(td);cfg=root/'config.json'
            config={'sessions':{'morning':[7,12]},'daily_budget_usd':2.75}
            cfg.write_text(json.dumps(config))
            previous=NOW-timedelta(hours=2)  # Test completed at 6 AM Eastern.
            test=card.publish_card(empty_feeds(previous),empty_reviews(),previous,root=root,kind='test')
            self.assertFalse(card.existing_today(root,NOW))
            self.assertTrue(card.existing_today(root,NOW,include_test=True))
            archive=root/'docs'/test['archive_url'].lstrip('/');archived=archive.read_bytes()
            ledger=root/'artifacts/analyst/daily-budget.json'
            write_json(ledger,{'version':2,'entries':[{'key':'earlier-test','at':iso(previous),
                'day':'2026-09-27','status':'settled','charge_usd':.75,'reserved_usd':1.0}]})
            spent=ledger.read_bytes()
            def prepare(feeds,now,config,**kwargs):
                kwargs['publication_feeds'].update(feeds)
                return dict(empty_reviews(),budget={'limit_usd':2.75,'charged_or_reserved_usd':.75})
            with patch.object(review,'ROOT',root),patch.object(review,'CONFIG',cfg),patch.object(review,'load_feeds',return_value=empty_feeds()) as feeds,patch.object(review,'prepare',side_effect=prepare) as paid,patch.object(review,'datetime') as clock:
                clock.now.return_value=NOW
                with patch('sys.argv',['review','--astra','--publish-card']):self.assertEqual(review.main(),0)
                published=json.loads((root/'docs/briefing/morning-card.json').read_text())
                self.assertEqual(published['kind'],'morning')
                self.assertEqual(published['status'],'no_reviewed_candidates')
                self.assertNotEqual(published['edition_id'],test['edition_id'])
                self.assertEqual(paid.call_args.args[2],config)
                self.assertEqual(archive.read_bytes(),archived)
                self.assertEqual(ledger.read_bytes(),spent)
                self.assertTrue(card.existing_today(root,NOW))
                with patch('sys.argv',['review','--astra','--publish-card']):self.assertIsNone(review.main())
                feeds.assert_called_once();paid.assert_called_once()

    def test_repeated_operator_tests_preserve_completed_test_or_morning_editions(self):
        for kind in ('morning','test'):
            with self.subTest(kind=kind),TemporaryDirectory() as td:
                root=Path(td);cfg=root/'config.json'
                cfg.write_text(json.dumps({'sessions':{'morning':[7,12]}}))
                card.publish_card(empty_feeds(),empty_reviews(),NOW,root=root,kind=kind)
                path=root/'docs/briefing/morning-card.json';original=path.read_bytes()
                with patch.object(review,'ROOT',root),patch.object(review,'CONFIG',cfg),patch.object(review,'load_feeds') as feeds,patch.object(review,'prepare') as paid,patch.object(review,'datetime') as clock:
                    clock.now.return_value=NOW
                    with patch('sys.argv',['review','--astra','--publish-card','--test-edition']):self.assertIsNone(review.main())
                    feeds.assert_not_called();paid.assert_not_called()
                self.assertEqual(path.read_bytes(),original)

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
        # The 4:30 PM refresh is started by the Supabase timer through Afternoon Market Refresh;
        # its GitHub schedule is a later, gated backup, and the sport workflows carry none.
        afternoon=(workflow/'afternoon-refresh.yml').read_text()
        self.assertIn("cron: '45 16 * * *'\n      timezone: America/New_York",afternoon)
        for sport in ('nhl','mlb'):
            source=(workflow/f'{sport}-daily.yml').read_text()
            self.assertNotIn('cron:',source)
            self.assertIn('workflow_call:',source)
            self.assertIn(f"if: needs.gate.outputs.{sport} == 'true'\n    uses: ./.github/workflows/{sport}-daily.yml",afternoon)
        research=(workflow/'analyst-daily.yml').read_text()
        self.assertNotIn('workflow_run:',research)
        self.assertNotIn('cron:',research)
        self.assertIn('continue-on-error: true',research)
        self.assertIn('steps.research.outcome',research)
        self.assertIn('--verify-live "$EXPECTED_EDITION"',research)

class AfternoonEditionTests(unittest.TestCase):
    AFTER=datetime(2026,9,27,20,45,tzinfo=timezone.utc)   # 4:45 p.m. Eastern

    def afternoon(self,morning=None):
        return {'morning_edition':morning,'reviewed':[]}

    def test_afternoon_edition_is_separate_and_ignores_sports_without_an_afternoon_refresh(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            morning=card.publish_card(empty_feeds(),empty_reviews(),NOW,root=root)
            feeds=empty_feeds(self.AFTER);feeds['NFL']=None   # NFL's morning prices have expired
            later=card.publish_card(feeds,empty_reviews(),self.AFTER,root=root,kind='afternoon',
                refresh_results={'NHL':'success','MLB':'skipped'},afternoon=self.afternoon(morning))
            self.assertEqual((later['kind'],later['policy_version'],later['status']),('afternoon','afternoon-edition-2','no_reviewed_candidates'))
            self.assertEqual((later['morning_edition_id'],later['price_check'],later['discovery_status']),(morning['edition_id'],[],'not_run'))
            self.assertTrue(card.existing_today(root,self.AFTER,kind='afternoon'))
            self.assertTrue((root/'docs'/later['archive_url'].lstrip('/')).exists())
            # The morning edition is never rewritten, and still counts as today's.
            self.assertEqual(json.loads((root/'docs/briefing/morning-card.json').read_text())['edition_id'],morning['edition_id'])
            self.assertTrue(card.existing_today(root,self.AFTER))
            self.assertFalse(card.existing_today(root,self.AFTER+timedelta(days=1),kind='afternoon'))

    def test_an_afternoon_without_allowance_is_a_quiet_no_pick_edition(self):
        with TemporaryDirectory() as td:
            reviews=empty_reviews();reviews['sports']['MLB'].update(review_status='budget_exhausted',candidate_count=4,reviewed_count=0)
            later=card.publish_card(empty_feeds(self.AFTER),reviews,self.AFTER,root=Path(td),kind='afternoon',afternoon=self.afternoon())
            self.assertEqual(later['status'],'no_reviewed_candidates')
            self.assertIn('MLB: no daily research allowance left',later['research']['notes'][0])
            # A failed review is still a failure in the afternoon.
            reviews['sports']['MLB']['review_status']='review_unavailable'
            failed=card.publish_card(empty_feeds(self.AFTER),reviews,self.AFTER+timedelta(minutes=1),root=Path(td),kind='afternoon',afternoon=self.afternoon())
            self.assertEqual(failed['status'],'research_incomplete')

    def test_afternoon_input_is_todays_morning_edition_and_review_verdicts(self):
        with TemporaryDirectory() as td:
            root=Path(td)
            morning=card.publish_card(empty_feeds(),empty_reviews(),NOW,root=root)
            q=lambda v:{'assessment':{'verdict':v}}
            write_json(root/'docs/briefing/reviews.json',{'sports':{
                'NHL':{'decision_date':'2026-09-27','candidates':[dict(sport='NHL',game_id='g',player='P',market='player_points',side='Over',qualitative_review=q('wait')),
                                                               dict(sport='NHL',game_id='g',player='Q',market='player_points',side='Under')]},
                'MLB':{'decision_date':'2026-09-26','candidates':[dict(sport='MLB',game_id='m',player='R',market='batter_hits',side='Over',qualitative_review=q('pass'))]}}})
            value=review.afternoon_input(self.AFTER,root)
            self.assertEqual(value['morning_edition']['edition_id'],morning['edition_id'])
            self.assertEqual([(r['player'],r['verdict']) for r in value['reviewed']],[('P','wait')])
            self.assertIsNone(review.afternoon_input(self.AFTER+timedelta(days=1),root)['morning_edition'])

    def test_afternoon_runs_no_discovery_search(self):
        from test_analyst_review import CONFIG
        config=deepcopy(CONFIG);config['sessions']={'afternoon':[0,24]}
        with TemporaryDirectory() as td,patch.object(review.research_discovery,'run') as discover:
            result=review.prepare(empty_feeds(),NOW,config,run_review=True,archive=Path(td)/'archive',public=Path(td)/'reviews.json',
                clock=lambda:NOW,afternoon=self.afternoon())
            discover.assert_not_called()
            self.assertEqual({b['session'] for b in result['sports'].values()},{'afternoon'})

    def test_afternoon_run_skips_late_starts_and_repeats_and_makes_no_targeted_searches(self):
        with TemporaryDirectory() as td:
            root=Path(td);cfg=root/'config.json'
            cfg.write_text(json.dumps({'sessions':{'morning':[7,12],'afternoon':[16,19]},'verification_candidates':6}))
            published={'edition_id':'afternoon-e','status':'no_reviewed_candidates'}
            with patch.object(review,'ROOT',root),patch.object(review,'CONFIG',cfg),patch.object(review,'load_feeds',return_value={}),\
                 patch.object(review,'prepare',return_value={'sports':{}}) as paid,patch.object(card,'publish_card',return_value=published) as publish,\
                 patch.object(review,'datetime') as clock:
                clock.now.return_value=self.AFTER+timedelta(hours=3)   # 7:45 p.m.: too late, and not a failure
                with patch('sys.argv',['review','--astra','--afternoon-edition']):self.assertEqual(review.main(),0)
                paid.assert_not_called()
                clock.now.return_value=self.AFTER
                with patch('sys.argv',['review','--astra','--afternoon-edition']):self.assertEqual(review.main(),0)
                self.assertEqual(paid.call_args.kwargs['afternoon'],self.afternoon())
                self.assertEqual(paid.call_args.args[2]['verification_candidates'],0)
                self.assertEqual(publish.call_args.kwargs['kind'],'afternoon')
                # A morning card cannot be published in the afternoon window.
                with patch('sys.argv',['review','--astra','--publish-card']):self.assertEqual(review.main(),1)
                self.assertEqual(paid.call_count,1)


if __name__=='__main__':unittest.main()
