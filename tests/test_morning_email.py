"""The owner's morning email: one per finished edition, exact card prices, Eastern times."""
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
           EDITORIAL_NOTIFY_EMAIL='owner@example.com')


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
        mail = email.message(card(rows=[row(), row(player='Blake Corum', sport='NFL', game_id='g2',
                                                   exposure_group='NFL:g2', book='betus', book_label='BetUS')]), CONFIG)
        self.assertEqual(mail['subject'], "Today's picks · Sunday, Oct 18: 2 picks")
        self.assertIn('+310 at DraftKings (quoted 7:06 AM EDT) · still a pick at +270 or better', mail['text'])
        self.assertIn('starts 6:10 PM EDT', mail['text'])
        self.assertIn('Suggested stake $1.50 to win $4.65', mail['text'])
        self.assertIn('Open DraftKings: https://sportsbook.draftkings.com/', mail['text'])
        self.assertIn('https://fourthandvalue.com/briefing/?edition=edition-a&pick=1&stake=1.50#daily-picks', mail['text'])
        self.assertNotIn('Open BetUS', mail['text'], 'books without a known site get no button')
        self.assertEqual(mail['text'].count('Before you bet:'), 6, 'three open checks per pick')
        self.assertIn('&lt;b&gt;starting&lt;/b&gt;', mail['html'])
        self.assertNotIn('<b>starting', mail['html'])
        self.assertNotRegex(mail['text'] + mail['html'], r'UTC|Z\b|T\d\d:\d\d')
        self.assertIn('Analysis, not a guarantee. Bet responsibly.', mail['text'])

    def test_shared_game_and_empty_cards_say_so(self):
        mail = email.message(card(rows=[row(), row(player='Valeri Nichushkin')]), CONFIG)
        self.assertEqual(mail['text'].count('Shares a game with another pick'), 2)
        empty = email.message(card(rows=[], status='no_reviewed_candidates'), CONFIG)
        self.assertEqual(empty['subject'], "Today's card · Sunday, Oct 18: no picks qualified")
        self.assertIn('no reviewed offer cleared the card', empty['text'])

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

    def test_send_once_per_edition_without_logging_the_recipient(self):
        post = Mock(return_value=Mock(status_code=200, ok=True))
        mail = email.message(card(), CONFIG)
        self.assertEqual(email.send(mail, 'morning-card-edition-a', ENV, post), 'sent')
        args = post.call_args
        self.assertEqual(args.args[0], 'https://api.resend.com/emails')
        self.assertEqual(args.kwargs['headers']['Idempotency-Key'], 'morning-card-edition-a')
        self.assertEqual(args.kwargs['json']['to'], ['owner@example.com'])
        self.assertEqual(set(args.kwargs['json']), {'from', 'to', 'subject', 'html', 'text'})
        post.return_value = Mock(status_code=409, ok=False)
        self.assertEqual(email.send(mail, 'morning-card-edition-a', ENV, post), 'already_sent')
        post.return_value = Mock(status_code=500, ok=False)
        with self.assertRaises(RuntimeError):
            email.send(mail, 'morning-card-edition-a', ENV, post)
        unused = Mock()
        self.assertEqual(email.send(mail, 'k', dict(ENV, RESEND_API_KEY=''), unused), 'not_configured')
        unused.assert_not_called()
        self.assertNotIn('example.com', ' '.join(email.MESSAGES.values()))

    def test_run_keys_by_edition_and_test_sends_are_labeled(self):
        with TemporaryDirectory() as td:
            path = Path(td) / 'card.json'
            path.write_text(json.dumps(card()))
            post = Mock(return_value=Mock(status_code=200, ok=True))
            self.assertEqual(email.run(NOW, card_path=path, env=ENV, post=post, expected='edition-a'), 'sent')
            self.assertEqual(post.call_args.kwargs['headers']['Idempotency-Key'], 'morning-card-edition-a')
            path.write_text(json.dumps(card(kind='test', decision_date='2026-10-01')))
            self.assertEqual(email.run(NOW, card_path=path, env=ENV, post=post), 'not_a_morning_edition')
            self.assertEqual(email.run(NOW, card_path=path, env=ENV, post=post, test=True), 'sent')
            self.assertEqual(post.call_args.kwargs['headers']['Idempotency-Key'], 'morning-card-test-edition-a-202610181125')
            self.assertTrue(post.call_args.kwargs['json']['subject'].startswith('Test · '))
            preview = Path(td) / 'preview.html'
            path.write_text(json.dumps(card()))
            calls = post.call_count
            self.assertEqual(email.run(NOW, card_path=path, env=ENV, post=post, preview=preview), 'previewed')
            self.assertEqual(post.call_count, calls)
            self.assertIn('Sean Monahan', preview.read_text())


if __name__ == '__main__':
    unittest.main()
