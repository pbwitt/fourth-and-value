"""Record-only Pinnacle snapshot, taken after the morning sports refreshes.

Our feeds request regions=us, so the market blend has never seen Pinnacle. Each
morning this records Pinnacle's game lines and the props our feeds request, for the
events in our own feeds that start within 36 hours, so the split between Pinnacle and
the US-book median can be fitted against graded results (docs/model-improvement-plan.md,
Step 2c). A one-market probe naming ten bookmakers records whether The Odds API bills
them as one region. Nothing recorded here enters a model or a pick. From October 24, 2026 a small
public file (docs/briefing/pinnacle.json) gives Pinnacle's no-vig chance for the lines in our feeds,
shown beside morning picks as a sharp reference only; it never selects, ranks or reprices a pick.
Paid requests go through the shared credit budget as an optional job (scripts/odds_budget.py),
leaving the sports refreshes their reserve, and the run stops at --max-credits. One snapshot per Eastern date; later starts skip.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_markets import ALIAS_MAP
from display_time import EASTERN, eastern_time
from odds_budget import OPTIONAL, CreditBudget, CreditFloorReached, estimate_cost, requests_preflight

API = 'https://api.the-odds-api.com/v4/sports'
GAME = ['h2h', 'spreads', 'totals']
WINDOW = timedelta(hours=36)
MAX_EVENTS = 16
# The prop markets each sport's refresh requests (NFL: the modeled markets only).
# NBA has no model yet; its Pinnacle coverage is recorded for Market Watch.
SPORTS = {
    'NHL': dict(key='icehockey_nhl', feeds=['docs/nhl/data/latest.json'],
                props=['player_shots_on_goal', 'player_goals', 'player_assists', 'player_points']),
    'NFL': dict(key='americanfootball_nfl', feeds=['docs/nfl/data/quotes.json', 'docs/props/top-picks.json'],
                props=['player_receptions', 'player_reception_yds', 'player_rush_yds', 'player_rush_attempts',
                       'player_pass_yds', 'player_pass_tds', 'player_pass_attempts', 'player_pass_completions']),
    'MLB': dict(key='baseball_mlb', feeds=['docs/mlb/data/latest.json'],
                props=['pitcher_strikeouts', 'pitcher_outs', 'batter_hits', 'batter_total_bases',
                       'batter_home_runs', 'batter_rbis']),
    'NBA': dict(key='basketball_nba', feeds=['docs/nba/data/latest.json'],
                props=['player_points', 'player_rebounds', 'player_assists', 'player_threes',
                       'player_points_rebounds_assists', 'player_points_rebounds', 'player_points_assists',
                       'player_rebounds_assists', 'player_blocks', 'player_steals', 'player_turnovers']),
}
# Pinnacle plus the nine US books our feeds return most: if ten named bookmakers bill as
# one region, Pinnacle can later join the regular pulls at no extra cost.
PROBE_BOOKS = ['pinnacle', 'draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'betrivers',
               'betonlineag', 'bovada', 'fanatics', 'mybookieag']
FIELDS = ['sport', 'event_id', 'commence_time', 'market', 'player', 'side', 'line', 'price', 'last_update']
ODDS_KEY = {v: k for k, v in ALIAS_MAP.items() if k.startswith('player_')}


def stamp(value):
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def identity(q):
    return (q['event_id'], q['market'], q['player'].casefold(), str(q['side']).casefold(), q['line'])


def feed_outcomes(root, sport):
    """Unique outcomes (any US book) in our latest committed feeds."""
    found = {}
    for path in SPORTS[sport]['feeds']:
        try:
            rows = json.loads((root / path).read_text()).get('rows', [])
        except (OSError, ValueError):
            continue
        for r in rows:
            market = r.get('market') or ODDS_KEY.get(r.get('market_std'), r.get('market_std'))
            q = dict(event_id=r.get('event_id') or r.get('game_id'), commence_time=r.get('commence_time'),
                     market=market, player=r.get('player') or '', side=r.get('side') or r.get('name'),
                     line=r['line'] if r.get('line') is not None else r.get('point'))
            if q['event_id'] and q['market'] and q['side'] and q['commence_time']:
                found.setdefault(identity(q), q)
    return list(found.values())


def flatten(event, sport):
    quotes = []
    for book in event.get('bookmakers', []):
        if book.get('key') != 'pinnacle':
            continue
        for m in book.get('markets', []):
            for o in m.get('outcomes', []):
                quotes.append(dict(sport=sport, event_id=event['id'], commence_time=event['commence_time'],
                                   market=m['key'], player=o.get('description') or '', side=o['name'],
                                   line=o.get('point'), price=o['price'],
                                   last_update=m.get('last_update') or book.get('last_update')))
    return quotes


class Client:
    def __init__(self, key, budget, session=None):
        self.key, self.budget, self.session = key, budget, session or requests
        self.log = []

    def get(self, path, **params):
        cost = estimate_cost(path, params)
        entry = dict(path=path, markets=params.get('markets'), estimated_cost=cost)
        self.log.append(entry)
        try:
            self.budget.ensure(cost, preflight=requests_preflight(self.key, self.session))
        except CreditFloorReached as error:
            entry['error'] = str(error)
            return None
        try:
            response = self.session.get(f'{API}/{path}', params=dict(params, apiKey=self.key), timeout=30)
        except requests.RequestException as error:   # the message can carry the keyed URL
            entry['error'] = type(error).__name__
            return None
        self.budget.observe(response.headers)
        entry.update(status=response.status_code, charged=response.headers.get('x-requests-last'))
        if response.status_code != 200:
            entry['error'] = response.text[:300]
            return None
        return response.json()


def window(now):
    return dict(commenceTimeFrom=now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                commenceTimeTo=(now + WINDOW).strftime('%Y-%m-%dT%H:%M:%SZ'))


def probe(client, sport, now):
    """One market, ten named bookmakers: does the provider bill it as one region?"""
    data = client.get(f"{SPORTS[sport]['key']}/odds", bookmakers=','.join(PROBE_BOOKS), markets='h2h',
                      oddsFormat='american', **window(now))
    entry = client.log[-1]
    try:
        charged = float(entry.get('charged'))
    except (TypeError, ValueError):
        charged = None
    posted = len({b['key'] for e in data or [] for b in e.get('bookmakers', [])})
    return dict(sport=sport, bookmakers_named=len(PROBE_BOOKS), bookmakers_returned=posted, markets=1,
                charged=charged, billed_as_one_region=None if charged is None or not data else charged <= 1)


def run(client, root, now, sports):
    """Pinnacle quotes plus what was actually checked: sports whose game lines and events whose
    props came back, so a line is reported as not posted only when Pinnacle was asked."""
    quotes, ours, asked, probed = [], {}, {}, None
    checked = dict(games=[], props=[])
    for sport in sports:
        cfg = SPORTS[sport]
        upcoming = [q for q in feed_outcomes(root, sport) if now < stamp(q['commence_time']) <= now + WINDOW]
        ours[sport] = upcoming
        if not upcoming:
            asked[sport] = []
            continue
        if probed is None:
            probed = probe(client, sport, now)
        games = client.get(f"{cfg['key']}/odds", bookmakers='pinnacle', markets=','.join(GAME),
                           oddsFormat='american', **window(now))
        if isinstance(games, list):
            checked['games'].append(sport)
        for event in games or []:
            quotes += flatten(event, sport)
        # Props for our own feed's events, earliest first, preferring events with prop rows.
        starts = {q['event_id']: q['commence_time'] for q in upcoming}
        with_props = {q['event_id'] for q in upcoming if q['market'] not in GAME}
        asked[sport] = sorted(starts, key=lambda e: (e not in with_props, starts[e]))[:MAX_EVENTS]
        for event_id in asked[sport]:
            event = client.get(f"{cfg['key']}/events/{event_id}/odds", bookmakers='pinnacle',
                               markets=','.join(cfg['props']), oddsFormat='american')
            if isinstance(event, dict) and event.get('id') == event_id:
                checked['props'].append(event_id)
                quotes += flatten(event, sport)
    return quotes, ours, asked, probed, checked


def coverage(quotes, ours, asked):
    rows = []
    for sport, outcomes in ours.items():
        posted = {identity(q) for q in quotes if q['sport'] == sport}
        for market in GAME + SPORTS[sport]['props']:
            # Game lines: our feed's events in the window. Props: the events requested.
            events = {q['event_id'] for q in outcomes if q['market'] == market} if market in GAME else set(asked[sport])
            mine = [q for q in outcomes if q['market'] == market and q['event_id'] in events]
            theirs = [q for q in quotes if q['sport'] == sport and q['market'] == market and q['event_id'] in events]
            if not mine and not theirs:
                continue
            rows.append(dict(sport=sport, market=market, events_requested=len(events),
                             pinnacle_events=len({q['event_id'] for q in theirs}), pinnacle_outcomes=len(theirs),
                             our_outcomes=len(mine), exact_line_matches=sum(identity(q) in posted for q in mine)))
    return rows


def report(result):
    probe_result = result['billing_probe'] or {}
    billed = {True: 'yes', False: 'no', None: 'not established'}[probe_result.get('billed_as_one_region')]
    lines = [f"## Pinnacle snapshot, {eastern_time(result['generated_at'])}", '',
             f"Record only. Credits charged: {result['credits_spent']:.0f}; balance after: {result['credits_remaining']}. "
             f"Ten named bookmakers billed as one region: {billed}.", '',
             '| Sport | Market | Events | Pinnacle events | Pinnacle outcomes | Our outcomes | Same exact line |',
             '|---|---|---|---|---|---|---|']
    for r in result['coverage']:
        share = f" ({r['exact_line_matches'] / r['our_outcomes']:.0%})" if r['our_outcomes'] else ''
        lines.append(f"| {r['sport']} | {r['market']} | {r['events_requested']} | {r['pinnacle_events']} | "
                     f"{r['pinnacle_outcomes']} | {r['our_outcomes']} | {r['exact_line_matches']}{share} |")
    failed = [r for r in result['requests'] if r.get('error')]
    if failed:
        lines += ['', 'Requests without data:'] + [f"- {r['path']} ({r.get('status', 'not sent')}): {r['error'][:160]}" for r in failed]
    return '\n'.join(lines)


def no_vig(quotes):
    """Pinnacle's no-vig chance per outcome, from its own two-sided prices (multiplicative)."""
    pairs = {}
    for q in quotes:
        line = abs(q['line']) if q['market'] == 'spreads' and q['line'] is not None else q['line']
        pairs.setdefault((q['event_id'], q['market'], q['player'].casefold(), line), []).append(q)
    fair = {}
    for pair in pairs.values():
        if len(pair) != 2 or any(not isinstance(q['price'], (int, float)) or q['price'] == 0 for q in pair):
            continue
        implied = [1 / (1 + (q['price'] / 100 if q['price'] > 0 else 100 / -q['price'])) for q in pair]
        for q, chance in zip(pair, implied):
            fair[identity(q)] = chance / sum(implied)
    return fair


def public_reference(quotes, ours, checked, now):
    """The public sharp reference: Pinnacle's no-vig chance for the outcomes in our own feeds."""
    fair = no_vig(quotes)
    wanted = {identity(q) for outcomes in ours.values() for q in outcomes}
    lines = [dict(sport=q['sport'], event_id=q['event_id'], market=q['market'],
                  market_std=ALIAS_MAP.get(q['market']), player=q['player'], side=q['side'], line=q['line'],
                  price=q['price'], probability=round(fair[identity(q)], 4), last_update=q['last_update'])
             for q in quotes if identity(q) in fair and identity(q) in wanted]
    checked_any = bool(checked['games'] or checked['props'])
    return dict(schema_version=1, generated_at=now.isoformat(), decision_date=now.astimezone(EASTERN).date().isoformat(),
                status='recorded' if checked_any else 'unavailable', use='reference_only',
                basis='Pinnacle two-sided prices at the same line, vig removed proportionally',
                checked_games=sorted(checked['games']), checked_events=sorted(set(checked['props'])), lines=lines)


def snapshot(client, budget, root, now, sports):
    """The record-only research snapshot, and the public sharp reference built from it."""
    quotes, ours, asked, probed, checked = run(client, root, now, sports)
    public = public_reference(quotes, ours, checked, now)
    return dict(schema_version=1, generated_at=now.isoformat(), decision_date=now.astimezone(EASTERN).date().isoformat(),
                evaluation_status='record_only', window_hours=WINDOW.total_seconds() / 3600,
                credits_spent=budget.spent, credits_remaining=budget.remaining, billing_probe=probed,
                requests=client.log, events_requested=asked, coverage=coverage(quotes, ours, asked),
                checked=dict(games=sorted(checked['games']), props=sorted(set(checked['props']))),
                fields=FIELDS, quotes=[[q[k] for k in FIELDS] for q in quotes]), public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='Checkout whose docs/ feeds define the events')
    parser.add_argument('--sports', default=','.join(SPORTS))
    parser.add_argument('--max-credits', type=int, default=250)
    parser.add_argument('--out-dir', default='artifacts/pinnacle/snapshots')
    parser.add_argument('--public', default='docs/briefing/pinnacle.json', help='Public sharp-reference file')
    parser.add_argument('--force', action='store_true', help='Replace an existing snapshot for this Eastern date')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    out = Path(args.out_dir) / f'{now.astimezone(EASTERN).date().isoformat()}.json'
    if out.exists() and not args.force:
        print(f'{out} already recorded; skipping.')
        return
    key = os.environ.get('ODDS_API_KEY')
    if not key:
        sys.exit('ODDS_API_KEY is not set')
    budget = CreditBudget(run_cap=args.max_credits, label='Pinnacle snapshot', tier=OPTIONAL)
    client = Client(key, budget)
    result, public = snapshot(client, budget, Path(args.root), now, [s.strip() for s in args.sports.split(',') if s.strip()])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    Path(args.public).parent.mkdir(parents=True, exist_ok=True)
    Path(args.public).write_text(json.dumps(public, separators=(',', ':')) + '\n')
    text = report(result)
    print(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as handle:
            handle.write(text + '\n')


if __name__ == '__main__':
    main()
