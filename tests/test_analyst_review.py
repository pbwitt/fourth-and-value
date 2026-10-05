"""MLB/NFL research uses real screening, point-in-time sources and bounded spend."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import analyst_review as analyst
from nhl.v2 import astra, evidence
from nhl.v2.data import ROOT, iso, write_json

NOW = datetime(2026, 9, 27, 14, tzinfo=timezone.utc)
CONFIG = json.loads(analyst.CONFIG.read_text())
# Baseline fixtures must not inherit a live, date-specific operator exception.
CONFIG.pop('test_budget_override', None)


def feeds():
    base = dict(game='Boston Red Sox @ New York Yankees', home_team='New York Yankees',
        away_team='Boston Red Sox', commence_time=iso(NOW+timedelta(hours=5)),
        player='Synthetic Player', side='Over', line=1.5, price=110, book='test', book_label='Test Book',
        market='batter_hits', market_label='Hits', quoted_at=iso(NOW-timedelta(minutes=2)))
    mlb = dict(base, event_id='mlb1', mlb_game_id=123, is_model_pick=True,
        model_probability=.55, model_push_probability=.01, model_ev_pct=16.5,
        other_book_probability=.5, other_books=3,
        model_inputs={'Lineup slot': 2}, lineup_status='Both batting orders published')
    nfl = dict(base, game_id='nfl1', game='Dallas Cowboys @ New York Giants',
        home_team='New York Giants', away_team='Dallas Cowboys', bookmaker='test',
        market_std='receptions', market_label='Receptions', name='over', point=1.5,
        model_prob=.55, push_prob=.01, ev_per_100=15, edge_bps=100, model_status='Calibration fitted',
        consensus_prob=.5, book_count=3,
        last_update=base['quoted_at'])
    return dict(MLB=dict(status='ready', last_success_at=iso(NOW), model_checked_at=iso(NOW), rows=[mlb]),
        NFL=dict(schema_version=1, status='ready', generated_at=iso(NOW), rows=[nfl]))


def board(sport='MLB'):
    return dict(sport=sport, board_id='synthetic', generated_at=iso(NOW), decision_date='2026-09-27',
        session='morning', candidates=[analyst.normalized(r) for r in analyst.selected(feeds(), NOW)['selected'] if r['sport'] == sport])


def source(b):
    return dict(source_id='s1', url='https://www.espn.com/'+b['sport'].lower()+'/story/synthetic',
        title=b['candidates'][0]['game']+' practice report', published_at=iso(NOW-timedelta(hours=1)),
        retrieved_at=iso(NOW), candidate_ids=[b['candidates'][0]['candidate_id']],
        excerpt='The starting lineup has not been announced yet. This synthetic report is used only for automated tests.')


def response(b):
    value = dict(candidate_id=b['candidates'][0]['candidate_id'], status='needs_information',
        assessment=dict(verdict='wait', reason='The expected role needs verification.',
            model_case='Opportunity drives this estimate.', price_case='The quote passed the price screen.',
            context_case='Participation remains unverified.', blocking_checks=['Confirm the expected role.']),
        countercase='A lineup change could invalidate the projected opportunity.',
        open_checks=['Verify participation and role before deciding.'], evidence=[dict(source_id='s1',
        excerpt='The starting lineup has not been announced yet', interpretation='Verify the current role.',
        kind='deployment', direction='context', represented_in='model_features')])
    return dict(status='completed', output=[dict(type='message', content=[dict(type='output_text',
        text=json.dumps({'reviews': [value]}))])], usage=dict(input_tokens=1000, output_tokens=300))


class ScreeningTests(unittest.TestCase):
    def test_exact_browser_policy_and_probability_semantics(self):
        f = feeds(); before = deepcopy(f)
        rows = analyst.selected(f, NOW)['selected']
        self.assertEqual({r['sport'] for r in rows}, {'MLB', 'NFL'})
        self.assertEqual(f, before)
        nfl, mlb = [analyst.normalized(next(r for r in rows if r['sport'] == s)) for s in ('NFL','MLB')]
        self.assertIsNone(nfl['independent_probability'])
        self.assertEqual(nfl['final_probability'], .55)
        self.assertEqual(nfl['probability_basis'], 'conditional_on_nonpush_outcome_calibrated')
        self.assertEqual(mlb['independent_probability'], .55)
        self.assertEqual(mlb['push_probability'], .01)
        self.assertEqual(mlb['quoted_at'], f['MLB']['rows'][0]['quoted_at'])
        f['MLB']['rows'][0]['is_model_pick'] = False
        f['NFL']['rows'][0]['model_prob'] = None
        self.assertEqual(analyst.selected(f, NOW)['selected'], [])

    def test_nhl_discovery_rows_receive_candidate_identity(self):
        shortlisted = dict(sport='NHL', offer_id='o1', forecast_id='f1', candidate_id='c'*24, review_key='["a"]')
        self.assertEqual(analyst.normalized(shortlisted)['candidate_id'], 'c'*24)
        # Independent discovery copies an offer-feed row: offer/forecast IDs, no candidate_id.
        discovered = dict(sport='NHL', offer_id='o2', forecast_id='f2', review_key='["b"]',
            discovery_origin='independent_research')
        row = analyst.normalized(discovered)
        self.assertEqual(len(row['candidate_id']), 24)
        self.assertNotIn('candidate_id', discovered)
        self.assertNotEqual(row['candidate_id'], analyst.normalized(dict(discovered, review_key='["c"]'))['candidate_id'])

    def test_stale_failed_started_and_future_quotes_not_reviewed(self):
        f = feeds(); f['MLB']['model_checked_at'] = iso(NOW-timedelta(minutes=91))
        f['NFL']['rows'][0]['commence_time'] = iso(NOW)
        self.assertEqual(analyst.selected(f, NOW)['selected'], [])
        f = feeds(); f['MLB']['model_error'] = 'failed'
        f['NFL']['rows'][0]['last_update'] = iso(NOW+timedelta(minutes=1))
        self.assertEqual(analyst.selected(f, NOW)['selected'], [])

    def test_review_identity_changes_with_price_quote_forecast_and_line(self):
        r = analyst.selected(feeds(), NOW)['selected'][0]
        for field, value in [('price', 120), ('last_update', iso(NOW)), ('point', 2.5), ('model_prob', .6)]:
            f = feeds(); f['NFL']['rows'][0][field] = value
            self.assertNotEqual(analyst.selected(f, NOW)['selected'][0]['review_key'], r['review_key'])
        f = feeds(); f['NFL']['generated_at'] = iso(NOW-timedelta(seconds=1))
        self.assertNotEqual(analyst.selected(f, NOW)['selected'][0]['review_key'], r['review_key'])

    def test_eastern_sessions_and_dst(self):
        self.assertEqual(analyst.session_at(NOW, CONFIG), 'morning')
        self.assertIsNone(analyst.session_at(NOW+timedelta(hours=3), CONFIG))
        self.assertIsNone(analyst.session_at(datetime(2026,9,27,10,59,tzinfo=timezone.utc), CONFIG))
        self.assertEqual(analyst.session_at(datetime(2026,9,27,11,tzinfo=timezone.utc), CONFIG), 'morning')
        self.assertIsNone(analyst.session_at(NOW+timedelta(hours=12), CONFIG))
        self.assertEqual(analyst.session_at(datetime(2026,11,1,15,tzinfo=timezone.utc),CONFIG),'morning')


class ResearchTests(unittest.TestCase):
    def test_official_team_reporting_is_allowlisted_and_retrieved(self):
        b=board('NFL'); row=b['candidates'][0]
        url='https://www.giants.com/news/synthetic-player-starter'
        self.assertTrue(evidence.trusted(url))
        self.assertFalse(evidence.trusted('https://www.giants.com.evil.test/news/report'))
        article=dict({'@type':'NewsArticle'},headline='Synthetic Player returns as Giants starter',
            datePublished=iso(NOW-timedelta(hours=1)),articleBody=('Synthetic Player is returning as the starting player for the New York Giants. '*12))
        def fetch(u):
            if u==url:return '<script type="application/ld+json">'+json.dumps(article)+'</script>'
            if u=='https://www.giants.com/news/':return '<a href="/news/synthetic-player-starter">Report</a>'
            return '<rss></rss>'
        with patch.object(evidence,'fetch',side_effect=fetch):
            sources,status=evidence.collect([row],lambda:NOW,sport='NFL')
        self.assertEqual(len(sources),1)
        self.assertEqual(sources[0]['url'],url)
        self.assertTrue(evidence.usable(sources[0],row,NOW))

    def test_request_compaction_preserves_candidate_numbers_and_original_sources(self):
        b=board('NFL');s=source(b);s['excerpt']='The starting lineup has not been announced yet. '*32
        b['candidates']=[dict(b['candidates'][0],candidate_id=str(i)) for i in range(4)]
        trace=json.loads((ROOT/'reports/nfl-model-diagnostics/2026-09-27-murray.json').read_text())['model_diagnostics']
        for r in b['candidates']:
            r['model_diagnostics']=deepcopy(trace)
        sources=[dict(s,source_id=str(i),candidate_ids=[str(i//2)]) for i in range(8)]
        before=deepcopy((b,sources))
        request=analyst.review_payload(b,sources,NOW,CONFIG)
        self.assertLessEqual(len(json.dumps(request,ensure_ascii=False).encode()),26000)
        self.assertEqual((b,sources),before)
        packet=json.loads(request['input'])
        self.assertEqual(len(packet['candidates']),4)
        self.assertTrue(all(r['final_probability']==.55 for r in packet['candidates']))
        self.assertEqual({cid for s in packet['sources'] for cid in s['candidate_ids']},{str(i) for i in range(4)})

    def test_source_priority_prefers_role_reporting_over_betting_picks(self):
        row=board('NFL')['candidates'][0]
        story=lambda title:dict(title=title,url='https://www.nfl.com/news/test')
        self.assertGreater(evidence.reporting_priority(row,story('Giants injury practice report')),
                           evidence.reporting_priority(row,story('Giants week in review')))
        self.assertEqual(evidence.reporting_priority(row,story('Expert picks odds best bets')),-1)

    def test_compacted_injury_rows_preserve_subject_and_teammates(self):
        b=board('NFL'); s=source(b)
        trace=json.loads((ROOT/'reports/nfl-model-diagnostics/2026-09-27-murray.json').read_text())['model_diagnostics']
        b['candidates']=[dict(b['candidates'][0],candidate_id=str(i),model_diagnostics=deepcopy(trace)) for i in range(4)]
        excerpt='Injury table excerpt; additional players may be listed in the full report.\n'
        excerpt+='Giants: Synthetic Player; game status: QUESTIONABLE.\nGiants: Example Receiver; game status: OUT.\n'
        excerpt+='\n'.join(f'Giants: Other Player {i}; injury: Hamstring; game status: OUT; practice FRI: DNP.' for i in range(15))
        sources=[dict(s,source_id=str(i),candidate_ids=[str(i//2)],source_kind='official_injury_report',
                      updated_at=iso(NOW),excerpt=excerpt[:1400]) for i in range(8)]
        before=deepcopy(sources)
        request=analyst.review_payload(b,sources,NOW,CONFIG)
        packet=json.loads(request['input'])
        self.assertEqual(sources,before)
        self.assertLessEqual(len(json.dumps(request,ensure_ascii=False).encode()),26000)
        self.assertEqual({cid for s in packet['sources'] for cid in s['candidate_ids']},{str(i) for i in range(4)})
        for s in packet['sources']:
            self.assertIn('Synthetic Player; game status: QUESTIONABLE.',s['excerpt'])
            self.assertIn('Example Receiver; game status: OUT.',s['excerpt'])
            self.assertTrue(s['excerpt'].endswith('.'))
            self.assertEqual(s['updated_at'],iso(NOW))

    def test_sport_source_pairing_timestamps_and_schema(self):
        for sport in ('MLB','NFL'):
            b=board(sport); s=source(b)
            self.assertTrue(evidence.usable(s,b['candidates'][0],NOW))
            q=astra.parse_response(response(b),b,[s],NOW,schema=analyst.SCHEMA,prompt_version=analyst.PROMPT_VERSION)[0]
            self.assertIsNone(q['probability_adjustment'])
            for bad in [dict(s,published_at=iso(NOW+timedelta(seconds=1))), dict(s,candidate_ids=['wrong']),dict(s,url='https://espn.com.evil.test/news')]:
                with self.assertRaises(ValueError):
                    astra.parse_response(response(b),b,[bad],NOW,schema=analyst.SCHEMA)
            raw=response(b); value=json.loads(raw['output'][0]['content'][0]['text'])
            value['reviews'][0]['probability']=.8
            raw['output'][0]['content'][0]['text']=json.dumps(value)
            with self.assertRaises(ValueError):
                astra.parse_response(raw,b,[s],NOW,schema=analyst.SCHEMA)

    def test_payload_model_explanation_and_hard_bounds(self):
        b=board('NFL'); p=astra.payload(b,[source(b)],NOW,CONFIG,instructions=analyst.INSTRUCTIONS,
            schema=analyst.SCHEMA,prompt_version=analyst.PROMPT_VERSION,extra_fields=('probability_basis','model_limitations'))
        self.assertEqual(p['model'],'gpt-6.1-sol'); self.assertNotIn('tools',p)
        self.assertIn('NOT current market consensus',p['instructions'])
        self.assertIn('historical game outcomes',p['instructions'])
        self.assertEqual(analyst.PROMPT_VERSION,'sports-research-6')
        self.assertLess(astra.bounds(p,CONFIG),1)
        self.assertIsNone(json.loads(p['input'])['candidates'][0].get('independent_probability'))

    def test_success_preserves_forecast_archives_usage_and_deduplicates(self):
        b=board(); before=deepcopy(b['candidates'][0])
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}), \
             patch.object(evidence,'collect',return_value=([source(b)],{})), patch.object(astra,'checkpoint') as checkpoint, \
             patch.object(astra,'call_api',return_value=response(b)) as call:
            archive=Path(td)
            result=analyst.review(b,feeds(),CONFIG,archive,lambda:NOW)
            self.assertEqual(result['review_status'],'completed')
            for k,v in before.items(): self.assertEqual(result['candidates'][0][k],v)
            checkpoint.assert_called_once(); call.assert_called_once()
            ledger=json.loads((archive/'daily-budget.json').read_text())
            self.assertEqual([e for e in ledger['entries'] if not e.get('legacy')][0]['status'],'settled')
            self.assertTrue(list((archive/'responses').glob('*.json')))
            self.assertEqual(analyst.review(board(),feeds(),CONFIG,archive,lambda:NOW)['review_status'],'already_attempted')
            call.assert_called_once()

    def test_missing_key_window_and_empty_slate_never_call(self):
        with tempfile.TemporaryDirectory() as td, patch.object(astra,'call_api') as call, patch.object(evidence,'collect',return_value=([],{})):
            with patch.dict('os.environ',{'OPENAI_API_KEY':''}):
                self.assertEqual(analyst.review(board(),feeds(),CONFIG,Path(td),lambda:NOW)['review_status'],'api_key_unavailable')
            with patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}):
                for changes,status in [(dict(candidates=[]),'no_candidates'),(dict(session=None),'outside_review_window')]:
                    self.assertEqual(analyst.review(dict(board(),**changes),feeds(),CONFIG,Path(td),lambda:NOW)['review_status'],status)
            call.assert_not_called()

    def test_no_reporting_still_assesses_quantitative_case_and_revision_is_bounded(self):
        b=board(); raw=response(b); value=json.loads(raw['output'][0]['content'][0]['text'])
        q=value['reviews'][0]; q['evidence']=[]
        q['assessment'].update(verdict='consider', blocking_checks=[])
        raw['output'][0]['content'][0]['text']=json.dumps(value)
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}), \
             patch.object(evidence,'collect',return_value=([],{'status':'no_usable_reporting'})), \
             patch.object(astra,'checkpoint'), patch.object(astra,'call_api',return_value=raw) as call:
            archive=Path(td)
            for revised in (False,True):
                result=analyst.review(board(),feeds(),CONFIG,archive,lambda:NOW,assessment_update=revised)
                self.assertEqual(result['review_status'],'completed')
                self.assertEqual(result['candidates'][0]['qualitative_review']['assessment']['verdict'],'consider')
                self.assertEqual(analyst.review(board(),feeds(),CONFIG,archive,lambda:NOW,assessment_update=revised)['review_status'],'already_attempted')
            self.assertEqual(call.call_count,2)
            self.assertEqual(len([e for e in json.loads((archive/'daily-budget.json').read_text())['entries'] if not e.get('legacy')]),2)

    def test_assessment_consistency_and_no_invented_numeric_confidence(self):
        b=board()
        for update in [dict(verdict='consider'),dict(blocking_checks=[]),dict(reason='Confidence is 80%.'),dict(adjusted_probability=.8)]:
            raw=response(b); value=json.loads(raw['output'][0]['content'][0]['text'])
            value['reviews'][0]['assessment'].update(update)
            raw['output'][0]['content'][0]['text']=json.dumps(value)
            with self.assertRaises(ValueError): astra.parse_response(raw,b,[source(b)],NOW,schema=analyst.SCHEMA)

    def test_packet_includes_model_projection_and_exact_market_comparison(self):
        f=feeds(); f['NFL']['rows'][0].update(mu=4.2,consensus_line=2.5,consensus_prob=.52,book_count=3)
        row=analyst.normalized(next(r for r in analyst.selected(f,NOW)['selected'] if r['sport']=='NFL'))
        p=astra.payload(dict(board('NFL'),candidates=[row]),[],NOW,CONFIG)
        context=json.loads(p['input'])['candidates'][0]['review_context']
        self.assertEqual(context['projected_quantity'],4.2)
        self.assertEqual(context['market_median_line'],2.5)
        self.assertEqual(context['market'],.52)
        self.assertEqual(context['books'],3)
        self.assertAlmostEqual(context['breakEven'],1/2.1)
        self.assertIsNone(context['calibration_sample_size'])

    def test_expiry_during_collection_and_checkpoint_failure_prevent_payment(self):
        for elapsed, checkpoint_error in [(timedelta(hours=2),None),(timedelta(),RuntimeError('checkpoint'))]:
            b=board()
            with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}), \
                 patch.object(evidence,'collect',return_value=([source(b)],{})), \
                 patch.object(astra,'checkpoint',side_effect=checkpoint_error), patch.object(astra,'call_api') as call:
                if checkpoint_error:
                    with self.assertRaises(RuntimeError): analyst.review(b,feeds(),CONFIG,Path(td),lambda:NOW)
                else:
                    self.assertEqual(analyst.review(b,feeds(),CONFIG,Path(td),lambda:NOW+elapsed)['review_status'],'expired_during_research')
                call.assert_not_called()

    def test_shared_cap_and_timeout_reservation_are_conservative(self):
        for spent,expected in [(5,'budget_exhausted'),(0,'review_unavailable')]:
            b=board()
            with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}), \
                 patch.object(evidence,'collect',return_value=([source(b)],{})), patch.object(analyst.daily_budget,'CAP',2.75), \
                 patch.object(astra,'checkpoint'), patch.object(astra,'call_api',side_effect=RuntimeError('timeout')) as call:
                archive=Path(td)
                if spent: write_json(archive/'daily-budget.json',dict(version=2,entries=[dict(key='spent',at=iso(NOW),day='2026-09-27',charge_usd=2.75)]))
                self.assertEqual(analyst.review(b,feeds(),CONFIG,archive,lambda:NOW)['review_status'],expected)
                if spent: call.assert_not_called()
                else:
                    entry=[e for e in json.loads((archive/'daily-budget.json').read_text())['entries'] if not e.get('legacy')][0]
                    self.assertEqual(entry['status'],'uncertain_reservation_retained')
                    self.assertEqual(entry['charge_usd'],entry['reserved_usd'])

    def test_review_queue_stops_at_bounded_batch_limit(self):
        b=board();b['candidates']=[dict(b['candidates'][0],candidate_id=str(i)) for i in range(30)]
        b['_research_queue']=dict(pending=list(b['candidates']),sources=[],diagnostics={},statuses=[])
        def completed(batch,*args,**kwargs):
            for row in batch['candidates']:row['qualitative_review']={'assessment':{'verdict':'consider'}}
            return dict(batch,review_status='completed')
        with patch.object(analyst,'review',side_effect=completed) as call:
            analyst.run_queue([b],feeds(),dict(CONFIG,max_review_batches=2),Path('/unused'),lambda:NOW)
        self.assertEqual(call.call_count,2)
        self.assertEqual(b['reviewed_count'],6)
        self.assertEqual(b['pending_count'],24)
        self.assertEqual(b['review_status'],'review_limit_reached')

    def test_authorized_test_allowance_reaches_review_without_erasing_spend(self):
        b=board()
        cfg=dict(CONFIG,test_budget_override=dict(date='2026-09-27',limit_usd=10,reason='Owner test'))
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}), \
             patch.object(evidence,'collect',return_value=([source(b)],{})), \
             patch.object(astra,'checkpoint'), patch.object(astra,'call_api',return_value=response(b)) as call:
            archive=Path(td)
            write_json(archive/'daily-budget.json',dict(version=2,entries=[dict(key='spent',at=iso(NOW),day='2026-09-27',charge_usd=2.75)]))
            self.assertEqual(analyst.review(b,feeds(),cfg,archive,lambda:NOW)['review_status'],'completed')
            call.assert_called_once()
            entries=json.loads((archive/'daily-budget.json').read_text())['entries']
            self.assertEqual(entries[0]['charge_usd'],2.75)
            self.assertEqual(entries[-1]['daily_limit_usd'],10)

    def test_preserve_same_day_context_without_relabelling_a_changed_offer(self):
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/'archive'; public=Path(td)/'reviews.json'
            b=board(); b['sources']=[source(b)]
            b['candidates'][0]['qualitative_review']={'status':'needs_information'}
            write_json(public,dict(sports={'MLB':b}))
            f=feeds(); f['MLB']['rows'][0]['price']=120
            out=analyst.prepare(f,NOW,CONFIG,archive=archive,public=public)
            self.assertEqual(out['sports']['MLB']['candidates'][0]['price'],110)
            self.assertEqual(out['sports']['MLB']['review_status'],'not_requested')
            next_day=analyst.prepare(f,NOW+timedelta(days=1),CONFIG,archive=archive,public=public)
            self.assertEqual(next_day['sports']['MLB']['candidates'],[])

    def test_failed_feeds_publish_empty_current_board_not_stale_picks(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            result=analyst.prepare({},NOW,CONFIG,archive=p/'archive',public=p/'public.json')
            self.assertTrue(all(not b['candidates'] and not b['coverage']['available'] for b in result['sports'].values()))


if __name__=='__main__': unittest.main()
