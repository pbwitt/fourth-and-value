"""X post drafts: exact offered prices, X's 280 limit, Eastern times and no repeats."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import social_drafts as sd

NOV1_CARD = ROOT / 'docs/briefing/cards/2026-11-01-80e8259b511e54a2c72405c9.json'
AT = datetime(2026, 11, 1, 12, 45, tzinfo=timezone.utc)  # 7:45 AM EST, the day DST ended


def row(**kw):
    base = dict(sport='NHL', player='Aliaksei Protas', side='Over', line=0.5, market='player_assists',
                market_label='Assists', price=275, book='draftkings', book_label='DraftKings',
                commence_time='2026-11-01T20:40:00Z', game='Minnesota Wild @ Washington Capitals')
    base.update(kw)
    return base


def card(**kw):
    base = dict(schema_version=1, kind='morning', status='published', decision_date='2026-11-01',
                published_at='2026-11-01T12:21:18Z', edition_id='e1', rows=[row()])
    base.update(kw)
    return base


def article(n=0, published_at='2026-11-01T12:14:59Z', **kw):
    base = dict(title='Cardinals–Cowboys: A 54.5 total meets a lower experimental estimate', sport='NFL',
                excerpt='All 11 tracked books sit at 54.5. The experimental estimate is lower, but that gap is not a proven edge.',
                url=f'/editorial/articles/2026-11-01-{n}-nfl.html', published_at=published_at)
    base.update(kw)
    return base


def rows_as_published(c, now):
    return c['rows']


class LengthTests(unittest.TestCase):
    def test_links_emoji_and_typography_weigh_like_x(self):
        self.assertEqual(sd.weighted_length('https://fourthandvalue.com/editorial/articles/2026-11-01-0-nfl.html'), 23)
        self.assertEqual(sd.weighted_length('🏒'), 2)
        self.assertEqual(sd.weighted_length('Rays–Bills’ · ok'), 16)
        self.assertGreaterEqual(sd.weighted_length('BetOnline.ag'), 23)
        self.assertEqual(sd.weighted_length('Oct. 26 at 7 p.m. vs. U.S.'), 26)


class PickLineTests(unittest.TestCase):
    def test_player_props_keep_the_offered_line_price_and_book(self):
        self.assertEqual(sd.pick_line(row()), '🏒 Aliaksei Protas o0.5 ast +275 DK')
        nfl = row(sport='NFL', player='Daniel Jones', side='over', line=31.5, market=None,
                  market_label='Passing attempts', price=-104)
        self.assertEqual(sd.pick_line(nfl), '🏈 Daniel Jones o31.5 pass att -104 DK')
        nba = row(sport='NBA', player='Jalen Brunson', side='Under', line=27.0, market='player_points',
                  market_label='Points', price=-115, book='fanduel', book_label='FanDuel')
        self.assertEqual(sd.pick_line(nba), '🏀 Jalen Brunson u27 pts -115 FD')
        mlb = row(sport='MLB', player='Tarik Skubal', side='Over', line=7.5, market='pitcher_strikeouts',
                  market_label='Pitcher strikeouts', price=110, book='betonlineag', book_label='BetOnline.ag')
        self.assertEqual(sd.pick_line(mlb), '⚾ Tarik Skubal o7.5 K +110 BetOnline')

    def test_game_markets_name_the_side(self):
        ml = row(sport='MLB', player=None, side='Chicago Cubs', line=None, market='h2h', market_label='Moneyline', price=-120)
        self.assertEqual(sd.pick_line(ml), '⚾ Chicago Cubs ML -120 DK')
        rl = row(sport='MLB', player=None, side='Los Angeles Dodgers', line=-1.5, market='spreads',
                 market_label='Run line', price=140, book='williamhill_us', book_label='Caesars')
        self.assertEqual(sd.pick_line(rl), '⚾ Los Angeles Dodgers -1.5 run line +140 Caesars')
        pl = row(player=None, side='Vancouver Canucks', line=1.5, market='spreads', market_label='Puck line', price=-180)
        self.assertEqual(sd.pick_line(pl), '🏒 Vancouver Canucks +1.5 puck line -180 DK')
        total = row(sport='MLB', player=None, side='Under', line=7.5, market='totals', market_label='Game total',
                    price=-105, game='Houston Astros @ Tampa Bay Rays')
        self.assertEqual(sd.pick_line(total), '⚾ Houston Astros @ Tampa Bay Rays u7.5 total -105 DK')


class DraftTests(unittest.TestCase):
    def test_card_lists_every_pick_that_fits_and_counts_the_rest(self):
        text, shown = sd.card_draft(card(), [row()])
        self.assertEqual(shown, 1)
        self.assertIn('Today’s picks · Sun, Nov 1', text)
        self.assertIn('Odds as posted 7:21 AM EST; lines move.', text)
        self.assertTrue(text.endswith('https://fourthandvalue.com/briefing/'))
        many = [row(player=f'Charle-Edouard D’Astous {i}', line=1.5, market_label='Shots on goal', side='Under')
                for i in range(10)]
        text, shown = sd.card_draft(card(), many)
        self.assertLessEqual(sd.weighted_length(text), sd.LIMIT)
        self.assertLess(shown, 10)
        self.assertIn(f'+{10 - shown} more on the card', text)

    def test_daylight_time_is_labeled_edt(self):
        text, _ = sd.card_draft(card(decision_date='2026-10-31', published_at='2026-10-31T11:24:00Z'), [row()])
        self.assertIn('Sat, Oct 31', text)
        self.assertIn('7:24 AM EDT', text)

    def test_article_keeps_whole_sentences_and_the_link(self):
        long = article(excerpt='First sentence is short. ' + 'Second sentence is far too long to fit ' * 8 + 'here.')
        text = sd.article_draft(long)
        self.assertLessEqual(sd.weighted_length(text), sd.LIMIT)
        self.assertIn('First sentence is short.', text)
        self.assertNotIn('Second', text)
        self.assertTrue(text.startswith('🏈 Cardinals–Cowboys'))
        self.assertTrue(text.endswith('https://fourthandvalue.com/editorial/articles/2026-11-01-0-nfl.html'))
        self.assertIn('not a proven edge.', sd.article_draft(article()))

    def test_intent_link_round_trips_the_exact_text(self):
        text, _ = sd.card_draft(card(), [row()])
        url = urlparse(sd.intent_url(text))
        self.assertEqual(f'{url.scheme}://{url.netloc}{url.path}', 'https://x.com/intent/tweet')
        self.assertEqual(parse_qs(url.query)['text'], [text])


class CollectTests(unittest.TestCase):
    def collect(self, c, articles=(), now=AT, sent=(), **kw):
        kw.setdefault('rows_for', rows_as_published)
        return sd.collect(c, list(articles), now, set(sent), **kw)

    def test_waits_for_todays_finished_card_then_sends_articles_alone(self):
        for pending in (card(status='research_incomplete'), card(decision_date='2026-10-31'), card(kind='test')):
            drafts, handled, message = self.collect(pending, [article()])
            self.assertEqual((drafts, handled), ([], []))
            self.assertIn('Waiting', message)
        late = datetime(2026, 11, 1, 14, 50, tzinfo=timezone.utc)  # 9:50 AM EST
        drafts, handled, _ = self.collect(card(status='research_incomplete'), [article()], now=late)
        self.assertEqual(handled, ['article:/editorial/articles/2026-11-01-0-nfl.html'])

    def test_card_and_todays_articles_arrive_once(self):
        old = article(1, published_at='2026-10-31T11:15:00Z')
        future = article(2, published_at='2026-11-01T15:00:00Z')
        drafts, handled, _ = self.collect(card(), [article(), old, future])
        self.assertEqual(handled, ['card:e1', 'article:/editorial/articles/2026-11-01-0-nfl.html'])
        self.assertIn('first game 3:40 PM EST', drafts[0]['note'])
        drafts, handled, message = self.collect(card(), [article()], sent=handled)
        self.assertEqual((drafts, handled, message), ([], [], 'Nothing new to draft.'))

    def test_replacement_card_is_labeled_and_empty_card_releases_the_hold(self):
        drafts, _, _ = self.collect(card(edition_id='e2'), sent=['card:e1'])
        self.assertIn('replaces the card', drafts[0]['note'])
        drafts, handled, _ = self.collect(card(status='no_reviewed_candidates', rows=[]), [article()])
        self.assertEqual([d['id'] for d in drafts], ['article:/editorial/articles/2026-11-01-0-nfl.html'])
        self.assertIn('card:e1', handled)

    def test_started_games_are_dropped(self):
        started = row(player='Mark Jankowski', commence_time='2026-11-01T12:30:00Z')
        drafts, _, _ = self.collect(card(rows=[row(), started]))
        self.assertNotIn('Jankowski', drafts[0]['text'])
        drafts, handled, _ = self.collect(card(rows=[started]))
        self.assertEqual((drafts, handled), ([], ['card:e1']))

    def test_articles_stop_once_their_game_starts(self):
        kickoff = article(featured_until='2026-11-01T18:00:00Z')
        self.assertEqual(len(self.collect(card(), [kickoff])[0]), 2)
        evening = datetime(2026, 11, 2, 1, 55, tzinfo=timezone.utc)  # 8:55 PM EST, Nov 1
        self.assertEqual(self.collect(card(status='no_reviewed_candidates'), [kickoff], now=evening)[0], [])

    def test_items_wait_until_the_site_serves_them(self):
        drafts, handled, _ = self.collect(card(), [article()], is_live=lambda path, check=None: False)
        self.assertEqual((drafts, handled), ([], []))

    def test_shared_selector_supplies_the_rows_today_s_picks_shows(self):
        if not shutil.which('node'):
            self.skipTest('node is required for the shared selector')
        published = json.loads(NOV1_CARD.read_text())
        rows = sd.displayed_rows(published, AT)
        self.assertTrue(rows)
        offered = {(r['player'], r['line'], r['price'], r['book']) for r in published['rows']}
        self.assertTrue({(r['player'], r['line'], r['price'], r['book']) for r in rows} <= offered)
        drafts, _, _ = sd.collect(published, [], AT, set())
        self.assertLessEqual(sd.weighted_length(drafts[0]['text']), sd.LIMIT)


class LedgerTests(unittest.TestCase):
    def run_main(self, *args, now='2026-11-01T12:45:00Z'):
        with mock.patch.object(sd, 'CARD_FILE', self.dir / 'card.json'), \
                mock.patch.object(sd, 'ARTICLES_FILE', self.dir / 'articles.json'), \
                mock.patch.object(sd, 'displayed_rows', rows_as_published), \
                mock.patch.object(sd, 'send') as send, mock.patch('builtins.print'):
            sd.main(['--ledger', str(self.ledger), '--now', now, *args])
        return send

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        (self.dir / 'card.json').write_text(json.dumps(card()))
        (self.dir / 'articles.json').write_text(json.dumps([article()]))
        self.ledger = self.dir / 'ledger.json'

    def test_dry_run_records_nothing_and_send_records_once(self):
        self.assertFalse(self.run_main().called)
        self.assertFalse(self.ledger.exists())
        self.assertTrue(self.run_main('--send').called)
        self.assertEqual(json.loads(self.ledger.read_text())['sent'],
                         ['article:/editorial/articles/2026-11-01-0-nfl.html', 'card:e1'])
        self.assertFalse(self.run_main('--send').called)

    def test_resend_today_keeps_the_ledger_and_a_new_day_starts_fresh(self):
        self.ledger.write_text(json.dumps({'date': '2026-11-01', 'sent': ['article:/older.html', 'card:e1']}))
        self.assertTrue(self.run_main('--send', '--resend-today').called)
        self.assertIn('article:/older.html', json.loads(self.ledger.read_text())['sent'])
        self.assertEqual(sd.load_ledger(self.ledger, datetime(2026, 11, 2, 13, tzinfo=timezone.utc)), set())


if __name__ == '__main__':
    unittest.main()
