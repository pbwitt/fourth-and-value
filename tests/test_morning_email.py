"""The owner's morning email: one per finished edition, exact card prices, Eastern times,
and an unsubscribe that always holds."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import morning_email as email

NOW = datetime(2026, 10, 18, 11, 25, tzinfo=timezone.utc)
CONFIG = json.loads((Path(__file__).resolve().parents[1] / 'config/morning_email.json').read_text())
ENV = dict(RESEND_API_KEY='key', EDITORIAL_NOTIFY_FROM='Fourth & Value <picks@example.com>',
           EDITORIAL_NOTIFY_EMAIL='Owner@Example.com', SUPABASE_URL='https://db.example.supabase.co/',
           SUPABASE_SERVICE_ROLE_KEY='service')
TOKEN = '3f2b8c1e-9d4a-4e6b-8f1a-2c5d7e9b0a14'
UNSUBSCRIBE = 'https://fourthandvalue.com/email/unsubscribe.html?token=' + TOKEN


def row(**extra):
    base = dict(sport='NHL', game='Columbus Blue Jackets @ Minnesota Wild', game_id='g1',
                commence_time='2026-10-18T22:10:00Z', quoted_at='2026-10-18T11:06:09Z',
                player='Sean Monahan', side='Over', line=0.5, market='player_assists', market_label='Assists',
                price=310, book='draftkings', book_label='DraftKings', review='Consider',
                exposure_group='NHL:g1',
                blend=dict(final=0.2735840927564265, ev=0.1216947803013484, threshold=0.01, confidence='high'),
                qualitative_review=dict(open_checks=['Confirm the <b>starting</b> goalie', 'Two', 'Three', 'Four']))
    return {**base, **extra}


def card(**extra):
    base = dict(schema_version=1, kind='morning', status='published', edition_id='edition-a',
                decision_date='2026-10-18', published_at='2026-10-18T11:19:52Z', rows=[row()])
    return {**base, **extra}


def response(status=200, body=None):
    return Mock(status_code=status, ok=200 <= status < 300, json=Mock(return_value=body))


class MorningEmailTests(unittest.TestCase):
    def test_lowest_price_is_where_the_pick_falls_below_the_floor(self):
        r = row(price=145, blend=dict(final=0.41538674109994234, ev=0.01769751569485889, threshold=0.01))
        self.assertEqual(email.lowest_price(r), 144)
        ev = lambda price: 0.41538674109994234 * email.decimal(price) - 1
        self.assertGreaterEqual(ev(144), 0.01)
        self.assertLess(ev(143), 0.01)
        self.assertEqual(email.lowest_price(row()), 270)
        # Pushes: the no-push share comes from the card's own EV at the offered price.
        push = 0.1
        final, price = 0.6, -110
        r = row(price=price, blend=dict(final=final, ev=(1 - push) * (final * email.decimal(price) - 1), threshold=0.01))
        lowest = email.lowest_price(r)
        self.assertLess(lowest, -100)
        self.assertGreaterEqual((1 - push) * (final * email.decimal(lowest) - 1), 0.01 - 1e-12)
        self.assertLess((1 - push) * (final * email.decimal(lowest - 1) - 1), 0.01)
        for broken in (dict(), dict(final=0.5, ev=None, threshold=0.01), dict(final=1.2, ev=0.1, threshold=0.01)):
            self.assertIsNone(email.lowest_price(row(blend=broken)))

    def test_stake_risks_at_most_a_unit_and_wins_at_most_about_a_unit(self):
        self.assertEqual(email.stake(-135, CONFIG)[0], 5)
        self.assertEqual(email.stake(100, CONFIG)[0], 5)
        self.assertEqual(email.stake(150, CONFIG)[0], 3.5)
        amount, win = email.stake(310, CONFIG)
        self.assertEqual(amount, 1.5)
        self.assertAlmostEqual(win, 4.65)
        self.assertEqual(email.stake(2800, CONFIG)[0], 1, 'never below the minimum stake')

    def test_message_shows_exact_card_prices_in_eastern_time_and_escapes_text(self):
        mail = email.message(card(rows=[row(), row(player='Blake Corum', sport='NFL', game_id='g2', exposure_group='NFL:g2',
                                                   book='betus', book_label='BetUS', commence_time='2026-10-18T20:05:00Z')]),
                             CONFIG, UNSUBSCRIBE)
        self.assertEqual(mail['subject'], 'Morning Card · Sun, Oct 18 · 2 picks')
        self.assertIn('+310 at DraftKings (quoted 7:06 AM EDT) · min price +270 or better', mail['text'])
        self.assertIn('starts 6:10 PM EDT', mail['text'])
        self.assertIn('Suggested stake $1.50 to win $4.65', mail['text'])
        self.assertIn('2 picks · first start 4:05 PM EDT · $3.00 total stake', mail['text'])
        self.assertIn('Open DraftKings: https://sportsbook.draftkings.com/', mail['text'])
        self.assertIn('https://fourthandvalue.com/briefing/?edition=edition-a&pick=1&stake=1.50#daily-picks', mail['text'])
        self.assertNotIn('Open BetUS', mail['text'], 'books without a known site get no button')
        self.assertEqual(mail['text'].count('Before you bet:'), 6, 'three open checks per pick')
        self.assertIn('&lt;b&gt;starting&lt;/b&gt;', mail['html'])
        self.assertNotIn('<b>starting', mail['html'])
        self.assertNotRegex(mail['text'] + mail['html'], r'UTC|\dZ\b|T\d\d:\d\d')
        self.assertIn('Analysis, not a guarantee. Bet responsibly.', mail['text'])
        self.assertIn('1-800-MY-RESET', mail['text'] + mail['html'])

    def test_every_email_is_branded_and_carries_a_working_unsubscribe(self):
        mail = email.message(card(), CONFIG, UNSUBSCRIBE)
        self.assertIn(f'Unsubscribe: {UNSUBSCRIBE}', mail['text'])
        self.assertIn(f'href="{UNSUBSCRIBE}"', mail['html'])
        self.assertEqual(mail['headers'], {'List-Unsubscribe': f'<{UNSUBSCRIBE}>'})
        self.assertIn('Fourth <span style="color:#4c6fff">&amp;</span> <span style="color:#22c55e">Value</span>', mail['html'])
        self.assertIn('>FV</td>', mail['html'])
        self.assertIn('1 pick · first start 6:10 PM EDT · $1.50 total stake', mail['html'], 'inbox preview line')
        self.assertEqual(email.unsubscribe_url(CONFIG, TOKEN), UNSUBSCRIBE)

    def test_book_button_opens_the_bet_when_the_book_sent_a_link(self):
        slip, game = 'https://sportsbook.draftkings.com/event/123?outcomes=0QA1', 'https://sportsbook.draftkings.com/event/123'
        mail = email.message(card(rows=[row(bet_link=slip, bet_link_level='bet')]), CONFIG, UNSUBSCRIBE)
        self.assertIn(f'Open bet at DraftKings: {slip}', mail['text'])
        self.assertIn(f'href="{slip}"', mail['html'])
        mail = email.message(card(rows=[row(bet_link=game, bet_link_level='event')]), CONFIG, UNSUBSCRIBE)
        self.assertIn(f'Open game at DraftKings: {game}', mail['text'])
        # State-specific books: the reader's home state (MD) first, their other states offered.
        mgm = 'https://sports.{state}.betmgm.com/en/sports/events/456?options=789'
        r = row(book='betmgm', book_label='BetMGM', bet_link=mgm, bet_link_level='bet')
        mail = email.message(card(rows=[r]), CONFIG, UNSUBSCRIBE)
        self.assertIn('Open bet at BetMGM: https://sports.md.betmgm.com/en/sports/events/456?options=789', mail['text'])
        self.assertIn('DC link: https://sports.dc.betmgm.com/en/sports/events/456?options=789', mail['text'])
        self.assertIn('Betting from DC?', mail['html'])
        self.assertIn('href="https://sports.dc.betmgm.com/en/sports/events/456?options=789"', mail['html'])
        self.assertNotIn('{state}', mail['text'] + mail['html'])
        self.assertNotIn('Betting from', email.message(card(rows=[row(bet_link=slip, bet_link_level='bet')]), CONFIG, UNSUBSCRIBE)['html'])
        # Weekdays default to DC and offer MD; the card's Eastern date decides.
        monday = email.message(card(rows=[r], decision_date='2026-10-19'), CONFIG, UNSUBSCRIBE)
        self.assertIn('Open bet at BetMGM: https://sports.dc.betmgm.com/en/sports/events/456?options=789', monday['text'])
        self.assertIn('MD link: https://sports.md.betmgm.com/en/sports/events/456?options=789', monday['text'])
        self.assertIn('Betting from MD?', monday['html'])
        self.assertEqual(email.reader_states(CONFIG['states'], '2026-10-23'), ['dc', 'md'], 'Friday')
        self.assertEqual(email.reader_states(CONFIG['states'], '2026-10-24'), ['md', 'dc'], 'Saturday')
        self.assertEqual(email.reader_states(['md'], '2026-10-19'), ['md'], 'a plain list works for any day')
        self.assertEqual(email.reader_states(None, '2026-10-19'), [])
        # A reader without a state gets the book's site, never a broken template link.
        link = email.picks(card(rows=[r]), CONFIG, states=[])[0]['link']
        self.assertEqual((link['url'], link['label'], link['alternates']), ('https://sports.betmgm.com/', 'Open BetMGM', []))
        entries = email.picks(card(rows=[row(bet_link=slip, bet_link_level='bet'), r, row(book='betus', book_label='BetUS')]), CONFIG)
        self.assertEqual(email.link_summary(entries), 'Book buttons: bet slip 2, no link 1')

    def test_shared_game_empty_cards_and_test_sends_say_so(self):
        mail = email.message(card(rows=[row(), row(player='Valeri Nichushkin')]), CONFIG, UNSUBSCRIBE)
        self.assertEqual(mail['text'].count('Shares a game with another pick'), 2)
        empty = email.message(card(rows=[], status='no_reviewed_candidates'), CONFIG, UNSUBSCRIBE)
        self.assertEqual(empty['subject'], 'Morning Card · Sun, Oct 18 · No picks today')
        self.assertIn('no reviewed offer cleared the card', empty['text'])
        test = email.message(card(), CONFIG, UNSUBSCRIBE, test=True)
        self.assertTrue(test['subject'].startswith('[Test] '))
        self.assertIn('Test send of the Sun, Oct 18 card.', test['html'])

    def test_only_todays_finished_morning_edition_is_sent(self):
        self.assertIsNone(email.skip_reason(card(), NOW))
        self.assertIsNone(email.skip_reason(card(rows=[], status='no_reviewed_candidates'), NOW))
        self.assertEqual(email.skip_reason(card(status='research_incomplete'), NOW), 'edition_not_final')
        self.assertEqual(email.skip_reason(card(kind='test'), NOW), 'not_a_morning_edition')
        self.assertEqual(email.skip_reason(card(decision_date='2026-10-17'), NOW), 'not_today')
        self.assertEqual(email.skip_reason(card(), NOW, expected='edition-b'), 'edition_mismatch')
        self.assertEqual(email.skip_reason(card(edition_id=''), NOW), 'no_edition_id')
        # 11:25 p.m. Eastern on Oct 18 is already Oct 19 in UTC; the card is still today's.
        late = datetime(2026, 10, 19, 3, 25, tzinfo=timezone.utc)
        self.assertIsNone(email.skip_reason(card(), late))

    def test_subscription_row_is_read_or_created_with_the_service_role(self):
        http = Mock()
        http.get.return_value = response(200, [{'token': TOKEN, 'subscribed': True}])
        self.assertEqual(email.subscription(ENV, ENV['EDITORIAL_NOTIFY_EMAIL'], http), {'token': TOKEN, 'subscribed': True})
        args = http.get.call_args
        self.assertEqual(args.args[0], 'https://db.example.supabase.co/rest/v1/email_subscriptions')
        self.assertEqual(args.kwargs['params'], {'list': 'eq.morning_card', 'email': 'eq.owner@example.com', 'select': 'token,subscribed'})
        self.assertEqual(args.kwargs['headers'], {'apikey': 'service', 'Authorization': 'Bearer service'})
        http.post.assert_not_called()
        # First send: no row yet, so one is created subscribed.
        http = Mock()
        http.get.return_value = response(200, [])
        http.post.return_value = response(201, [{'token': TOKEN, 'subscribed': True, 'email': 'owner@example.com'}])
        self.assertTrue(email.subscription(ENV, 'owner@example.com', http)['subscribed'])
        self.assertEqual(http.post.call_args.kwargs['json'], {'list': 'morning_card', 'email': 'owner@example.com'})
        # Created concurrently by another start: read it again.
        http = Mock()
        http.get.side_effect = [response(200, []), response(200, [{'token': TOKEN, 'subscribed': False}])]
        http.post.return_value = response(409)
        self.assertFalse(email.subscription(ENV, 'owner@example.com', http)['subscribed'])
        http = Mock()
        http.get.return_value = response(404)
        with self.assertRaises(RuntimeError):
            email.subscription(ENV, 'owner@example.com', http)

    def test_send_once_per_edition_without_logging_the_recipient(self):
        post = Mock(return_value=response(200))
        mail = email.message(card(), CONFIG, UNSUBSCRIBE)
        self.assertEqual(email.send(mail, 'morning-card-edition-a', ENV, post), 'sent')
        args = post.call_args
        self.assertEqual(args.args[0], 'https://api.resend.com/emails')
        self.assertEqual(args.kwargs['headers']['Idempotency-Key'], 'morning-card-edition-a')
        self.assertEqual(args.kwargs['json']['to'], ['Owner@Example.com'])
        self.assertEqual(set(args.kwargs['json']), {'from', 'to', 'subject', 'html', 'text', 'headers'})
        post.return_value = response(409)
        self.assertEqual(email.send(mail, 'morning-card-edition-a', ENV, post), 'already_sent')
        post.return_value = response(500)
        with self.assertRaises(RuntimeError):
            email.send(mail, 'morning-card-edition-a', ENV, post)
        self.assertNotIn('example.com', ' '.join(email.MESSAGES.values()))

    def test_run_honors_settings_and_unsubscribes_before_sending(self):
        with TemporaryDirectory() as td:
            path = Path(td) / 'card.json'
            path.write_text(json.dumps(card()))
            post, http = Mock(return_value=response(200)), Mock()
            http.get.return_value = response(200, [{'token': TOKEN, 'subscribed': True}])
            go = lambda **kw: email.run(NOW, card_path=path, env=kw.pop('env', ENV), post=post, http=http, **kw)
            for name in email.SETTINGS:
                self.assertEqual(go(env=dict(ENV, **{name: ''})), 'not_configured')
            post.assert_not_called()
            http.get.assert_not_called()
            self.assertEqual(go(expected='edition-a'), 'sent')
            sent = post.call_args.kwargs
            self.assertEqual(sent['headers']['Idempotency-Key'], 'morning-card-edition-a')
            self.assertIn(UNSUBSCRIBE, sent['json']['text'])
            # Unsubscribed, or a row that cannot be read: nothing is sent.
            calls = post.call_count
            http.get.return_value = response(200, [{'token': TOKEN, 'subscribed': False}])
            self.assertEqual(go(), 'unsubscribed')
            http.get.return_value = response(503)
            with self.assertRaises(RuntimeError):
                go()
            self.assertEqual(post.call_count, calls)
            # Test sends: any card, labeled, keyed by minute, still subscription-checked.
            http.get.return_value = response(200, [{'token': TOKEN, 'subscribed': True}])
            path.write_text(json.dumps(card(kind='test', decision_date='2026-10-01')))
            self.assertEqual(go(), 'not_a_morning_edition')
            self.assertEqual(go(test=True), 'sent')
            self.assertEqual(post.call_args.kwargs['headers']['Idempotency-Key'], 'morning-card-test-edition-a-202610181125')
            self.assertTrue(post.call_args.kwargs['json']['subject'].startswith('[Test] '))
            # Preview: written locally, no settings needed, nothing sent or looked up.
            preview = Path(td) / 'preview.html'
            path.write_text(json.dumps(card()))
            calls, lookups = post.call_count, http.get.call_count
            self.assertEqual(email.run(NOW, card_path=path, env={}, post=post, http=http, preview=preview), 'previewed')
            self.assertEqual((post.call_count, http.get.call_count), (calls, lookups))
            self.assertIn('Sean Monahan', preview.read_text())

    def test_afternoon_email_lists_new_picks_and_the_morning_price_check(self):
        after = datetime(2026, 10, 18, 20, 50, tzinfo=timezone.utc)
        check = dict(key='k', status='below_threshold', price=135, book_label='DraftKings', ev=-0.004, min_price=141,
                     pick=dict(sport='NHL', game='g', player='Sam Reinhart', side='Over', line=0.5, market='player_assists',
                               market_label='Assists', price=160, book_label='DraftKings'))
        later = card(kind='afternoon', edition_id='edition-p', published_at='2026-10-18T20:47:00Z', price_check=[check])
        mail = email.message(later, CONFIG, UNSUBSCRIBE)
        self.assertEqual(mail['subject'], 'Afternoon Picks · Sun, Oct 18 · 1 new pick')
        self.assertIn('MORNING PICKS AT 4:30 P.M.', mail['text'])
        self.assertIn('- Sam Reinhart Over 0.5 assists (NHL): No longer a pick: +135 at DraftKings gives -0.4% after blending; '
                      'needs +141 or better.', mail['text'])
        self.assertIn('AFTERNOON PICKS', mail['html'])
        self.assertIn('Morning picks at 4:30 p.m.', mail['html'])
        self.assertIn('edition=edition-p', mail['text'])
        # Nothing new and every morning pick still qualifies: no email.
        quiet = dict(later, rows=[], price_check=[dict(check, status='qualifies', ev=0.03)])
        self.assertEqual(email.skip_reason(quiet, after, kind='afternoon'), 'nothing_new')
        self.assertEqual(email.message(dict(later, rows=[]), CONFIG, UNSUBSCRIBE)['subject'], 'Afternoon Picks · Sun, Oct 18 · Morning price check')
        self.assertIsNone(email.skip_reason(dict(later, rows=[]), after, kind='afternoon'))
        self.assertEqual(email.skip_reason(card(), after, kind='afternoon'), 'not_an_afternoon_edition')
        self.assertEqual(email.skip_reason(later, after), 'not_a_morning_edition')
        with TemporaryDirectory() as td:
            path = Path(td) / 'afternoon.json'
            path.write_text(json.dumps(later))
            post, http = Mock(return_value=response(200)), Mock()
            http.get.return_value = response(200, [{'token': TOKEN, 'subscribed': True}])
            self.assertEqual(email.run(after, card_path=path, env=ENV, post=post, http=http, kind='afternoon'), 'sent')
            self.assertEqual(post.call_args.kwargs['headers']['Idempotency-Key'], 'afternoon-picks-edition-p')
            # One subscription covers both emails.
            self.assertEqual(http.get.call_args.kwargs['params']['list'], 'eq.morning_card')


if __name__ == '__main__':
    unittest.main()
