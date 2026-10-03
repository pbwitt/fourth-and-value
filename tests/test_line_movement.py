from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import line_movement as lm

NOW = datetime(2026, 10, 3, 20, tzinfo=timezone.utc)
START = (NOW+timedelta(hours=2)).isoformat()


def card(rows, published=NOW-timedelta(hours=8)):
    return dict(edition_id='ed1', published_at=published.isoformat(), rows=rows)


def pick(**changes):
    row = dict(sport='MLB', event_id='e1', game='A @ B', commence_time=START, market='pitcher_strikeouts',
               market_label='Pitcher strikeouts', player='Hagen Smith', side='Over', line=1.5, book='betonlineag',
               book_label='BetOnline.ag', price=-169, quoted_at=(NOW-timedelta(hours=9)).isoformat(), offer_id='o1')
    return {**row, **changes}


def quote(book, line, price, fair, side='Over'):
    return dict(event_id='e1', market='pitcher_strikeouts', player='Hagen Smith', side=side, line=line,
                book=book, price=price, fair_probability=fair, quoted_at=NOW.isoformat())


class LineMovementTests(unittest.TestCase):
    def run_update(self, rows, feed_rows, checked=NOW, ledger=None, published=NOW-timedelta(hours=8)):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'card.json').write_text(json.dumps(card(rows, published)))
            feed = dict(status='ready', checked_at=checked.isoformat(), rows=feed_rows)
            return lm.update(ledger or {}, 'mlb', feed, NOW, Path(tmp))

    def test_book_line_move_is_not_a_same_line_probability(self):
        ledger = self.run_update([pick()], [quote('betonlineag', 2.5, 136, .41)])
        entry = ledger['entries']['ed1:o1']['last_pregame']
        self.assertEqual((entry['book_line'], entry['line_move'], entry['line_move_favorable']), (2.5, 1, True))
        self.assertIsNone(entry['other_fair'])
        self.assertNotIn('probability_move', entry)
        self.assertEqual(ledger['summary']['line_moves_favorable'], 1)

    def test_same_line_other_books_measure_the_move(self):
        ledger = self.run_update([pick()], [quote('betonlineag', 1.5, -200, .66), quote('fanduel', 1.5, -250, .70),
                                            quote('draftkings', 1.5, -240, .68), quote('bovada', 2.5, 130, .41)])
        entry = ledger['entries']['ed1:o1']['last_pregame']
        self.assertAlmostEqual(entry['other_fair'], .69)
        self.assertEqual(entry['other_books'], 2)
        self.assertAlmostEqual(entry['probability_move'], .69-169/269)
        self.assertEqual(ledger['summary']['same_line_beat'], 1)

    def test_only_snapshots_after_publication_and_before_start(self):
        rows = [quote('betonlineag', 2.5, 136, .41)]
        # The card's own snapshot (ingested before publication) is not movement.
        self.assertNotIn('last_pregame', self.run_update([pick()], rows, published=NOW+timedelta(minutes=5))['entries']['ed1:o1'])
        late = self.run_update([pick(commence_time=(NOW-timedelta(hours=1)).isoformat())], rows)
        self.assertEqual(late['entries'], {})
        ledger = self.run_update([pick()], rows)
        kept = self.run_update([pick()], [quote('betonlineag', 3.5, 150, .3)], checked=NOW-timedelta(hours=1), ledger=ledger)
        self.assertEqual(kept['entries']['ed1:o1']['last_pregame']['book_line'], 2.5)

    def test_other_sports_and_failed_feeds_are_ignored(self):
        self.assertEqual(self.run_update([pick(sport='NHL')], [quote('betonlineag', 2.5, 136, .41)])['entries'], {})
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'card.json').write_text(json.dumps(card([pick()])))
            ledger = lm.update({}, 'mlb', dict(status='feed_error', checked_at=NOW.isoformat(), rows=[quote('betonlineag', 2.5, 136, .41)]), NOW, Path(tmp))
        self.assertNotIn('last_pregame', ledger['entries']['ed1:o1'])


if __name__ == '__main__':
    unittest.main()
