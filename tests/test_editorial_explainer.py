"""Today's picks, explained: packets, checks, publication and scheduling."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import editorial_explainer as ex
import editorial_schedule as sched
from site_metadata import seo_title

NOW = datetime(2026, 10, 23, 11, 20, tzinfo=timezone.utc)  # 7:20 AM EDT


def mlb_row(**changes):
    row = dict(sport='MLB', event_id='e1', game='Tampa Bay Rays @ Milwaukee Brewers', home_team='Milwaukee Brewers',
               away_team='Tampa Bay Rays', commence_time='2026-10-24T00:00:00Z', venue='American Family Field',
               market='pitcher_outs', market_label='Pitcher outs', player='Nick Martinez', side='Over', line=13.5, price=108,
               book_label='FanDuel', quoted_at='2026-10-23T11:08:06Z', model_probability=0.6214991, consensus_probability=0.4711,
               paired_books=5, model_mean=14.1037, model_median=14, model_mean_label='Projected pitcher outs',
               model_inputs={'Starter outs/start, last five': 14.14, 'model_player_id': 607259},
               player_context={'stat_label': 'outs', 'sample_label': 'starts', 'recent': [{'games': 5, 'mean': 13.6}],
                               'games': [{'date': '2026-10-17', 'opp': '@ CLE', 'ip': '5⅓'}] * 6,
                               'opponent': {'team': 'MIL', 'label': 'Opposing lineup',
                                            'items': [{'label': 'Strikeout rate', 'value': 23.1, 'unit': '%', 'rank': '13th'}]}},
               qualitative_review={'assessment': {'verdict': 'consider', 'reason': 'Narrow case.', 'model_case': 'The blend shows a slim edge.',
                                                  'price_case': 'Fair.', 'context_case': 'Start sourced.'},
                                   'countercase': 'A quick hook decides it.', 'open_checks': ['Reconfirm +108'],
                                   'evidence': [{'source_id': 's1', 'excerpt': 'Tampa Bay will send Nick Martinez', 'interpretation': 'He starts.'}]},
               review_sources=[{'source_id': 's1', 'title': 'Rays tab Martinez', 'url': 'https://www.mlb.com/news/rays-martinez',
                                'published_at': '2026-10-22T22:45:00Z'},
                               {'source_id': 's2', 'title': 'Unused', 'url': 'https://www.espn.com/x', 'published_at': '2026-10-22T20:00:00Z'}])
    row.update(changes)
    return row


def card(kind='morning', rows=None, published='2026-10-23T11:16:00Z'):
    return dict(schema_version=1, kind=kind, status='published', decision_date='2026-10-23', published_at=published,
                edition_id=kind + '-edition', rows=[mlb_row()] if rows is None else rows)


def section(pick_id, text=None):
    return dict(pick_id=pick_id, heading='Martinez has to get 14 outs', source_ids=['s1'], text=text or (
        'Nick Martinez over 13.5 pitcher outs at +108 on FanDuel is the first idea. Our experimental model projects about '
        '14.1 outs, a hair above the line, and gives the over 62.1% against a 48.1% break-even. ' + 'Recent form matters here. ' * 12 +
        'The case against: a quick hook decides it, and the market sits at 47.1%. Check the price before first pitch.'))


def article(*ids, **changes):
    value = dict(publish=True, reason='ok', hook="Martinez's leash and a long night for the Brewers' bats",
                 seo_hook='Martinez and the Brewers', excerpt='One World Series starter prop: what we like, why, and what could go wrong before Game 1 tonight.',
                 intro='One pick made the card today. It comes from our experimental model, which has not been validated against betting prices, so treat it as an idea to look at.',
                 picks=[section(i) for i in (ids or ('pick-1',))], closing='Prices and lineups move, so recheck before acting.')
    value.update(changes)
    return value


def response(payload, model='claude-sonnet-5-5'):
    return {'status': 'completed', 'usage': {'input_tokens': 1000, 'output_tokens': 500},
            'output': [{'content': [{'type': 'output_text', 'text': json.dumps(payload)}]}]}


class PacketTests(unittest.TestCase):
    def test_pick_record_states_the_offer_and_softens_research_notes(self):
        pick = ex.pick_record(mlb_row(), 1)
        self.assertEqual(pick['pick']['label'], 'Nick Martinez over 13.5 pitcher outs')
        self.assertEqual((pick['pick']['price'], pick['pick']['line_text'], pick['pick']['sportsbook']), ('+108', '13.5', 'FanDuel'))
        self.assertEqual(pick['chances'], {'model_pct': 62.1, 'break_even_pct': 48.1, 'market_no_vig_pct': 47.1, 'market_books': 5})
        self.assertEqual(pick['player_team'], 'Tampa Bay Rays')
        self.assertEqual(pick['opponent']['team'], 'Milwaukee Brewers')
        self.assertNotIn('model_player_id', pick['model_inputs'])
        self.assertIn('slim margin', pick['review']['model_case'])
        self.assertEqual([s['id'] for s in pick['sources']], ['s1'])  # only cited reporting
        self.assertEqual(len(pick['recent_form']['last_games']), 5)

    def test_labels_cover_game_lines_and_nfl_rows(self):
        base = dict(game='A @ B', home_team='B', away_team='A', price=-120)
        self.assertEqual(ex.pick_label(dict(base, market='h2h', side='A')), 'A moneyline')
        self.assertEqual(ex.pick_label(dict(base, market='spreads', market_label='Run line', side='B', line=-1.5)), 'B -1.5 run line')
        self.assertEqual(ex.pick_label(dict(base, market='totals', market_label='Game total', side='Under', line=6.5)), 'Under 6.5 game total')
        nfl = dict(base, player='Jayden Daniels', market_std='pass_attempts', market_label='Passing attempts', name='under', point=30.5)
        self.assertEqual(ex.pick_label(nfl), 'Jayden Daniels under 30.5 passing attempts')
        record = ex.pick_record(dict(nfl, sport='NFL', model_prob=0.636, consensus_prob=0.502, book_count=4, mu=24.95,
                                     last_update='2026-10-19T11:07:26Z', book_label='DraftKings', commence_time='2026-10-20T00:15:00Z'), 2)
        self.assertEqual(record['chances']['model_pct'], 63.6)
        self.assertEqual(record['pick']['quoted_at'], '2026-10-19T11:07:26Z')

    def test_team_is_only_named_when_the_abbreviation_is_unambiguous(self):
        self.assertEqual(ex.team_named('STL', {'home_team': 'St Louis Blues', 'away_team': 'Vegas Golden Knights'}), 'St Louis Blues')
        self.assertIsNone(ex.team_named('MTL', {'home_team': 'Chicago Blackhawks', 'away_team': 'Montréal Canadiens'}))

    def test_large_cards_are_trimmed_to_fit(self):
        rows = [mlb_row(player=f'Pitcher {i}') for i in range(10)]
        evidence = ex.packet(card(rows=rows), rows, 'morning', 'full', NOW)
        request = ex.fit(ex.instructions(), {'evidence': evidence}, 'picks')
        self.assertLessEqual(len(request['instructions'].encode()) + len(request['input'].encode()), ex.budget.LIMITS['picks'][0])


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.evidence = ex.packet(card(), [mlb_row()], 'morning', 'full', NOW)

    def check(self, value):
        return ex.validate(copy.deepcopy(value), self.evidence)

    def test_valid_article_passes(self):
        self.assertGreater(self.check(article()), 100)

    def test_rejections_name_the_problem(self):
        cases = {
            'edge': article(intro='Our experimental model sees an edge here. ' * 5),
            'do not match': article('pick-1', 'pick-1'),
            'does not state its line': article(picks=[section('pick-1', 'Our experimental model likes Martinez. ' * 15)]),
            'experimental': article(intro='One pick made the card today. ' * 5, picks=[section('pick-1', 'Nick Martinez over 13.5 pitcher outs at +108 on FanDuel. ' * 8)]),
            'ISO timestamp': article(closing='Prices as of 2026-10-23T11:08:06Z.'),
            'not listed': article(picks=[dict(section('pick-1'), source_ids=['s2'])]),
            'hook is': article(hook='Short'),
        }
        for expected, value in cases.items():
            with self.subTest(expected), self.assertRaisesRegex(ValueError, expected):
                self.check(value)

    def test_even_money_may_be_written_out(self):
        evidence = ex.packet(card(), [mlb_row(price=100)], 'morning', 'full', NOW)
        text = section('pick-1')['text'].replace('+108', 'even money')
        self.assertGreater(ex.validate(article(picks=[section('pick-1', text)]), evidence), 100)

    def test_every_check_is_stated_to_the_writer(self):
        prompt = ex.instructions()
        for phrase in ('skips or repeats a pick', str(ex.MIN_PICK_WORDS), 'line and price exactly', 'seo_hook', 'source_ids',
                       'ISO form', 'edge, value', 'experimental', 'player_team'):
            self.assertIn(phrase, prompt)


class RunTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(); self.addCleanup(directory.cleanup)
        self.root = Path(directory.name); docs = self.root / 'docs'
        (docs / 'briefing').mkdir(parents=True); (docs / 'editorial/runs').mkdir(parents=True)
        (docs / 'editorial/published.json').write_text('[]')
        for target in (patch.object(ex.ed, 'ROOT', self.root), patch.object(ex.ed, 'DOCS', docs),
                       patch.object(ex.writer, 'STATE', docs / 'editorial/runs'), patch.object(ex.budget, 'PATH', self.root / 'budget.json'),
                       patch.object(ex.budget, 'checkpoint'), patch.object(ex.ed, 'render_home'), patch('builtins.print'),
                       patch.dict(ex.ed.CFG, {'writing_enabled': True})):
            target.start(); self.addCleanup(target.stop)
        self.docs = docs

    def write_card(self, value, kind='morning'):
        (self.docs / 'briefing' / sched.CARD_FILES[kind]).write_text(json.dumps(value))

    def state(self):
        return json.loads((self.docs / 'editorial/runs/2026-10-23.json').read_text())

    def test_morning_article_then_afternoon_addition_on_one_page(self):
        self.write_card(card())
        with patch.object(ex.writer, 'call_api', side_effect=[response(article()), response({'pass': True, 'reason': 'Supported.'})]) as api:
            ex.run(NOW)
        self.assertEqual(api.call_count, 2)
        page = (self.docs / 'editorial/articles/2026-10-23-todays-picks.html').read_text()
        self.assertIn("Today&#39;s picks, explained: Martinez&#39;s leash", page)
        self.assertIn('<title>Oct 23 Picks Explained: Martinez and the Brewers | Fourth &amp; Value</title>', page)
        self.assertIn('Nick Martinez over 13.5 pitcher outs</strong> · +108 at FanDuel (7:08 AM EDT)', page)
        self.assertIn('fv-betting-notice', page)
        entry = json.loads((self.docs / 'editorial/published.json').read_text())[0]
        self.assertEqual((entry['kind'], entry['sport'], entry['featured_until']), ('Analysis', 'MLB', '2026-10-24T00:00:00Z'))
        self.assertEqual(self.state()['slots']['picks-morning']['status'], 'published')
        self.assertEqual(json.loads((self.root / 'budget.json').read_text())['entries'][0]['status'], 'settled')

        later = datetime(2026, 10, 23, 20, 45, tzinfo=timezone.utc)
        self.write_card(card('afternoon', [mlb_row(player='Christian Yelich', market_label='Total bases', line=1.5, price=185,
                                                   side='Over', book_label='Bovada', quoted_at='2026-10-23T20:32:00Z',
                                                   player_context={}, review_sources=[])], '2026-10-23T20:41:00Z'), 'afternoon')
        addition = article(hook='', seo_hook='', excerpt='', intro='One more idea from the afternoon prices, from the same experimental model.',
                           picks=[dict(section('pick-1', section('pick-1')['text'].replace('Nick Martinez over 13.5 pitcher outs at +108',
                                       'Christian Yelich over 1.5 total bases at +185')), source_ids=[])])
        with patch.object(ex.writer, 'call_api', side_effect=[response(addition), response({'pass': True, 'reason': 'ok'})]):
            ex.run(later)
        page = (self.docs / 'editorial/articles/2026-10-23-todays-picks.html').read_text()
        self.assertIn('Added this afternoon', page)
        self.assertLess(page.index('Nick Martinez over 13.5'), page.index('Christian Yelich over 1.5 total bases</strong>'))
        entries = json.loads((self.docs / 'editorial/published.json').read_text())
        self.assertEqual(len(entries), 1)
        self.assertIn('modified_at', entries[0])
        with patch.object(ex.writer, 'call_api') as api:
            ex.run(later)  # both sessions are terminal: no paid call
        api.assert_not_called()

    def test_failed_check_gets_one_revision(self):
        self.write_card(card())
        bad = article(intro='Our experimental model has an edge today. ' * 5)
        calls = [response(bad), response(article()), response({'pass': True, 'reason': 'ok'})]
        with patch.object(ex.writer, 'call_api', side_effect=calls) as api:
            ex.run(NOW)
        self.assertEqual(api.call_count, 3)
        self.assertIn('edge', self.state()['slots']['picks-morning']['first_check'])
        self.assertEqual(self.state()['slots']['picks-morning']['status'], 'published')

    def test_second_failed_audit_skips_without_publishing(self):
        self.write_card(card())
        calls = [response(article()), response({'pass': False, 'reason': 'Wrong price.'}),
                 response(article()), response({'pass': False, 'reason': 'Still wrong.'})]
        with patch.object(ex.writer, 'call_api', side_effect=calls):
            ex.run(NOW)
        slot = self.state()['slots']['picks-morning']
        self.assertEqual((slot['status'], slot['first_audit_reason']), ('skipped', 'Wrong price.'))
        self.assertFalse((self.docs / 'editorial/articles/2026-10-23-todays-picks.html').exists())

    def test_no_pick_day_and_test_editions_make_no_paid_call(self):
        self.write_card(dict(card(rows=[]), status='no_reviewed_candidates'))
        with patch.object(ex.writer, 'call_api') as api:
            ex.run(NOW)
        api.assert_not_called()
        self.assertEqual(self.state()['slots']['picks-morning']['status'], 'no_picks')
        self.write_card(dict(card(), kind='test'))
        self.assertIsNone(sched.today_card(self.root, NOW, 'morning'))

    def test_unreadable_card_row_is_skipped_without_a_paid_call(self):
        broken = mlb_row(); broken.pop('price')
        self.write_card(card(rows=[broken]))
        with patch.object(ex.writer, 'call_api') as api:
            ex.run(NOW)
        api.assert_not_called()
        slot = self.state()['slots']['picks-morning']
        self.assertEqual(slot['status'], 'skipped')
        self.assertIn('no paid request', slot['reason'])

    def test_seo_title_is_used_only_when_present(self):
        item = dict(title='Today’s picks, explained: a long and specific headline about the day', excerpt='x' * 80,
                    url='/editorial/articles/a.html', date='2026-10-23')
        self.assertIn(f"<title>{seo_title(item['title'])}</title>", ex.seo.metadata(item))
        self.assertTrue(seo_title(item['title']).endswith('…'))
        self.assertIn('<title>Oct 23 Picks Explained: Short | Fourth &amp; Value</title>',
                      ex.seo.metadata(dict(item, seo_title='Oct 23 Picks Explained: Short')))


class PicksScheduleTests(unittest.TestCase):
    def make_root(self, card_value=None):
        directory = TemporaryDirectory(); root = Path(directory.name)
        (root / 'config').mkdir(); (root / 'docs/editorial/runs').mkdir(parents=True); (root / 'docs/briefing').mkdir(parents=True)
        (root / 'config/editorial.json').write_text(json.dumps({'writing_enabled': True, 'writer': {'mode': 'picks'}}))
        (root / 'docs/editorial/published.json').write_text('[]')
        if card_value is not None:
            (root / 'docs/briefing/morning-card.json').write_text(json.dumps(card_value))
        return directory, root

    def test_planner_waits_for_the_card_then_writes_once(self):
        directory, root = self.make_root(); self.addCleanup(directory.cleanup)
        plan = sched.plan(root, NOW, event_name='schedule')
        self.assertFalse(plan['writer_eligible']); self.assertEqual(plan['writer_reason'], 'waiting_for_morning_card')
        (root / 'docs/briefing/morning-card.json').write_text(json.dumps(card()))
        plan = sched.plan(root, NOW, event_name='schedule')
        self.assertTrue(plan['writer_eligible']); self.assertEqual(plan['writer_reason'], 'morning_picks_unexplained')
        (root / 'docs/editorial/runs/2026-10-23.json').write_text(json.dumps({'slots': {'picks-morning': {'status': 'published'}}}))
        self.assertFalse(sched.plan(root, NOW, event_name='schedule')['writer_eligible'])

    def test_picks_mode_never_forces_a_paid_briefing_refresh(self):
        directory, root = self.make_root(card()); self.addCleanup(directory.cleanup)
        (root / 'docs/briefing/latest.json').write_text(json.dumps({'generated_at': '2026-10-23T10:00:00Z'}))
        plan = sched.plan(root, NOW, event_name='schedule')
        self.assertTrue(plan['writer_eligible'])
        self.assertFalse(plan['refresh_briefing'])

    def test_delivery_statuses(self):
        due = datetime(2026, 10, 23, 12, 45, tzinfo=timezone.utc)  # 8:45 AM EDT
        directory, root = self.make_root(card(rows=[])); self.addCleanup(directory.cleanup)
        with patch('builtins.print'):
            self.assertEqual(sched.verify_delivery(root, due)['status'], 'no_picks')
            (root / 'docs/briefing/morning-card.json').write_text(json.dumps(card()))
            self.assertEqual(sched.verify_delivery(root, NOW)['status'], 'pending')
            with self.assertRaises(SystemExit):
                sched.verify_delivery(root, due)
            url = sched.picks_url('2026-10-23')
            (root / 'docs' / url.lstrip('/')).parent.mkdir(parents=True)
            (root / 'docs' / url.lstrip('/')).write_text('<html></html>')
            (root / 'docs/editorial/published.json').write_text(json.dumps([{'url': url, 'date': '2026-10-23', 'kind': 'Analysis'}]))
            self.assertEqual(sched.verify_delivery(root, due)['status'], 'complete')
            (root / 'docs/briefing/morning-card.json').unlink()
            (root / 'docs/editorial/published.json').write_text('[]')
            self.assertEqual(sched.verify_delivery(root, due)['status'], 'waiting_for_card')


if __name__ == '__main__':
    unittest.main()
