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


def feeds():
    base = dict(game='Boston Red Sox @ New York Yankees', home_team='New York Yankees',
        away_team='Boston Red Sox', commence_time=iso(NOW+timedelta(hours=5)),
        player='Synthetic Player', side='Over', line=1.5, price=110, book='test', book_label='Test Book',
        market='batter_hits', market_label='Hits', quoted_at=iso(NOW-timedelta(minutes=2)))
    mlb = dict(base, event_id='mlb1', mlb_game_id=123, is_model_pick=True,
        model_probability=.55, model_push_probability=.01, model_ev_pct=16.5,
        model_inputs={'Lineup slot': 2}, lineup_status='Both batting orders published')
    nfl = dict(base, game_id='nfl1', game='Dallas Cowboys @ New York Giants',
        home_team='New York Giants', away_team='Dallas Cowboys', bookmaker='test',
        market_std='receptions', market_label='Receptions', name='over', point=1.5,
        model_prob=.55, push_prob=.01, ev_per_100=15, edge_bps=100, model_status='Calibration fitted',
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
        self.assertIn('conditional_on_nonpush', nfl['probability_basis'])
        self.assertEqual(mlb['independent_probability'], .55)
        self.assertEqual(mlb['push_probability'], .01)
        self.assertEqual(mlb['quoted_at'], f['MLB']['rows'][0]['quoted_at'])
        f['MLB']['rows'][0]['is_model_pick'] = False
        f['NFL']['rows'][0]['model_prob'] = None
        self.assertEqual(analyst.selected(f, NOW)['selected'], [])

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
        self.assertEqual(analyst.session_at(NOW+timedelta(hours=3), CONFIG), 'later')
        self.assertIsNone(analyst.session_at(NOW+timedelta(hours=10), CONFIG))
        self.assertEqual(analyst.session_at(datetime(2026,11,1,15,tzinfo=timezone.utc),CONFIG),'morning')


class ResearchTests(unittest.TestCase):
    def test_source_priority_prefers_role_reporting_over_betting_picks(self):
        row=board('NFL')['candidates'][0]
        story=lambda title:dict(title=title,url='https://www.nfl.com/news/test')
        self.assertGreater(evidence.reporting_priority(row,story('Giants injury practice report')),
                           evidence.reporting_priority(row,story('Giants week in review')))
        self.assertEqual(evidence.reporting_priority(row,story('Expert picks odds best bets')),-1)

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
        self.assertEqual(p['model'],'gpt-6-astra'); self.assertNotIn('tools',p)
        self.assertIn('NOT independent',p['instructions'])
        self.assertLess(astra.bounds(p,CONFIG),1)
        self.assertIsNone(json.loads(p['input'])['candidates'][0]['independent_probability'])

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
            ledger=json.loads((archive/'budget.json').read_text())
            self.assertEqual(ledger['entries'][0]['status'],'settled')
            self.assertTrue(list((archive/'responses').glob('*.json')))
            self.assertEqual(analyst.review(board(),feeds(),CONFIG,archive,lambda:NOW)['review_status'],'already_attempted_this_session')
            call.assert_called_once()

    def test_missing_reporting_key_window_and_empty_slate_never_call(self):
        with tempfile.TemporaryDirectory() as td, patch.object(astra,'call_api') as call, patch.object(evidence,'collect',return_value=([],{})):
            with patch.dict('os.environ',{'OPENAI_API_KEY':''}):
                self.assertEqual(analyst.review(board(),feeds(),CONFIG,Path(td),lambda:NOW)['review_status'],'api_key_unavailable')
            with patch.dict('os.environ',{'OPENAI_API_KEY':'synthetic'}):
                for changes,status in [(dict(candidates=[]),'no_candidates'),(dict(session=None),'outside_review_window'),({},'no_usable_reporting')]:
                    self.assertEqual(analyst.review(dict(board(),**changes),feeds(),CONFIG,Path(td),lambda:NOW)['review_status'],status)
            call.assert_not_called()

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
                 patch.object(evidence,'collect',return_value=([source(b)],{})), patch.object(astra,'recent_spend',return_value=spent), \
                 patch.object(astra,'checkpoint'), patch.object(astra,'call_api',side_effect=RuntimeError('timeout')) as call:
                archive=Path(td)
                self.assertEqual(analyst.review(b,feeds(),CONFIG,archive,lambda:NOW)['review_status'],expected)
                if spent: call.assert_not_called()
                else:
                    entry=json.loads((archive/'budget.json').read_text())['entries'][0]
                    self.assertEqual(entry['status'],'uncertain_reservation_retained')
                    self.assertEqual(entry['charge_usd'],entry['reserved_usd'])

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
