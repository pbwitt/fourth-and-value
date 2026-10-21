"""Pinnacle as the sharp reference, recorded by every core odds refresh at no extra cost.

The Odds API bills every group of ten named bookmakers as one region. The probe in
scripts/pinnacle_snapshot.py confirmed it on Oct 15-17, 2026: ten named books and one
market cost one credit. So the core refreshes name ten books instead of `regions=us`:
the nine US books our feeds use most, plus Pinnacle. Before 2026-10-21 the US region
returned eleven books; BetUS and LowVig (a mirror of BetOnline's lines) are the two
dropped.

Pinnacle is split off each response before any other code sees it, so it never enters
the US median, a best price, a pick or a public page. Its quotes are recorded under
artifacts/pinnacle/reference/<sport>/<Eastern date>.json for the Step 2c market-blend
refit and closing-line work (docs/model-improvement-plan.md), and
scripts/check_sharp_reference.py raises an alert when a refreshed sport has none.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from display_time import EASTERN  # noqa: E402

REFERENCE_BOOK = 'pinnacle'
US_BOOKS = ('draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'betrivers',
            'betonlineag', 'bovada', 'fanatics', 'mybookieag')
REQUEST_BOOKS = US_BOOKS + (REFERENCE_BOOK,)
BOOKS_PER_REGION = 10
assert len(REQUEST_BOOKS) <= BOOKS_PER_REGION, 'more than ten named books costs a second region'

OUT_DIR = ROOT / 'artifacts/pinnacle/reference'
SPORTS = {'americanfootball_nfl': 'nfl', 'icehockey_nhl': 'nhl', 'baseball_mlb': 'mlb',
          'basketball_nba': 'nba'}
FIELDS = ['event_id', 'commence_time', 'market', 'player', 'side', 'line', 'price', 'last_update',
          'captured_at']


def label(sport):
    """'nhl' for 'icehockey_nhl' (or for 'NHL')."""
    return SPORTS.get(sport, str(sport).lower())


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def odds_params(params):
    """Swap `regions=us` for the ten named books; any other request is left unchanged."""
    if 'bookmakers' in params or str(params.get('regions', '')).strip().lower() != 'us':
        return params
    swapped = {k: v for k, v in params.items() if k != 'regions'}
    swapped['bookmakers'] = ','.join(REQUEST_BOOKS)
    return swapped


def quotes_from(event, captured_at):
    """Pinnacle's outcomes in one event as rows of FIELDS."""
    rows = []
    for book in event.get('bookmakers') or []:
        if book.get('key') != REFERENCE_BOOK:
            continue
        for market in book.get('markets') or []:
            for outcome in market.get('outcomes') or []:
                rows.append(dict(event_id=event.get('id'), commence_time=event.get('commence_time'),
                                 market=market.get('key'), player=outcome.get('description') or '',
                                 side=outcome.get('name'), line=outcome.get('point'), price=outcome.get('price'),
                                 last_update=market.get('last_update') or book.get('last_update'),
                                 captured_at=captured_at))
    return rows


def _strip(event):
    books = event.get('bookmakers')
    if not isinstance(books, list):
        return event
    return {**event, 'bookmakers': [b for b in books if b.get('key') != REFERENCE_BOOK]}


def split(payload, captured_at=None):
    """(payload without Pinnacle, Pinnacle quotes) for an odds response: a list of events
    or one event. Anything else comes back unchanged with no quotes."""
    captured_at = captured_at or iso(datetime.now(timezone.utc))
    if isinstance(payload, list):
        events = [e for e in payload if isinstance(e, dict)]
        if len(events) != len(payload):
            return payload, []
        quotes = [q for e in events for q in quotes_from(e, captured_at)]
        return [_strip(e) for e in events], quotes
    if isinstance(payload, dict) and isinstance(payload.get('bookmakers'), list):
        return _strip(payload), quotes_from(payload, captured_at)
    return payload, []


class Recorder:
    """Collects the Pinnacle quotes and request counts of one refresh run."""

    def __init__(self, sport):
        self.sport = label(sport)
        self.quotes = []
        self.odds_requests = 0
        self.events_with_reference = set()

    def take(self, payload, captured_at=None):
        """Strip Pinnacle from an odds response, keep its quotes, return the rest."""
        self.odds_requests += 1
        stripped, quotes = split(payload, captured_at)
        self.quotes.extend(quotes)
        self.events_with_reference.update(q['event_id'] for q in quotes)
        return stripped

    def save(self, now, root=None):
        """Merge this run into today's file (Eastern date). Never raises: the reference is
        research data and must not cost a published refresh. Returns the path or None."""
        try:
            return save(self.sport, self.quotes, now, run=dict(
                started_at=iso(now), odds_requests=self.odds_requests,
                pinnacle_quotes=len(self.quotes), pinnacle_events=len(self.events_with_reference),
                books_requested=list(REQUEST_BOOKS)), root=root)
        except Exception as error:  # noqa: BLE001
            print(f'Pinnacle reference not saved for {self.sport} ({type(error).__name__}: {error})',
                  file=sys.stderr)
            return None


def path_for(sport, now, root=None):
    base = (Path(root) / 'artifacts/pinnacle/reference') if root else OUT_DIR
    return base / label(sport) / f'{now.astimezone(EASTERN).date().isoformat()}.json'


def load(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get('fields') == FIELDS else None


def save(sport, quotes, now, run=None, root=None):
    """Append one run's quotes to the sport's file for this Eastern date. A quote already
    recorded with the same price and update time keeps its first capture time."""
    if not run and not quotes:
        return None
    path = path_for(sport, now, root)
    data = load(path) or dict(schema_version=1, sport=label(sport), date=path.stem,
                              evaluation_status='record_only', reference_book=REFERENCE_BOOK,
                              fields=FIELDS, runs=[], quotes=[])
    seen = {tuple(q[:-1]) for q in data['quotes']}
    for q in quotes:
        row = [q.get(k) for k in FIELDS]
        if tuple(row[:-1]) not in seen:
            seen.add(tuple(row[:-1]))
            data['quotes'].append(row)
    if run:
        data['runs'].append(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    temp.replace(path)
    return path
