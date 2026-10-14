"""One-off check: what Pinnacle posts through The Odds API for our sports and markets.

Our feeds request regions=us only, so the market blend has never seen Pinnacle. This
asks for bookmakers=pinnacle (billed as one region) for each sport's game lines and the
props our feeds pull, then compares every Pinnacle outcome with the exact lines in our
latest US feeds and today's card. Every paid request goes through the shared credit
floor (scripts/odds_budget.py) and the run stops at --max-credits. Nothing it fetches
feeds a model, a pick or a public page.
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_markets import ALIAS_MAP
from display_time import eastern_time
from odds_budget import CreditBudget, CreditFloorReached, estimate_cost, requests_preflight

API = 'https://api.the-odds-api.com/v4/sports'
GAME = ['h2h', 'spreads', 'totals']
# The prop markets each sport's refresh requests (NFL: the modeled markets only).
SPORTS = {
    'NFL': dict(key='americanfootball_nfl', days=7, events=3,
                feeds=['docs/nfl/data/quotes.json', 'docs/props/top-picks.json'],
                props=['player_receptions', 'player_reception_yds', 'player_rush_yds', 'player_rush_attempts',
                       'player_pass_yds', 'player_pass_tds', 'player_pass_attempts', 'player_pass_completions']),
    'MLB': dict(key='baseball_mlb', days=2, events=2, feeds=['docs/mlb/data/latest.json'],
                props=['pitcher_strikeouts', 'pitcher_outs', 'batter_hits', 'batter_total_bases',
                       'batter_home_runs', 'batter_rbis']),
    'NHL': dict(key='icehockey_nhl', days=2, events=4, feeds=['docs/nhl/data/latest.json'],
                props=['player_shots_on_goal', 'player_goals', 'player_assists', 'player_points']),
    'NBA': dict(key='basketball_nba', days=7, events=1, feeds=['docs/nba/data/latest.json'],
                props=['player_points', 'player_rebounds', 'player_assists', 'player_threes',
                       'player_points_rebounds_assists', 'player_points_rebounds', 'player_points_assists',
                       'player_rebounds_assists', 'player_blocks', 'player_steals', 'player_turnovers']),
}
ODDS_KEY = {v: k for k, v in ALIAS_MAP.items() if k.startswith('player_')}


def stamp(value):
    return datetime.fromisoformat(str(value).replace('Z', '+00:00'))


def decimal(price):
    return 1 + price / 100 if price > 0 else 1 - 100 / price


def logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


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
            q = dict(event_id=r.get('event_id') or r.get('game_id'), game=r.get('game'),
                     commence_time=r.get('commence_time'), market=market, player=r.get('player') or '',
                     side=r.get('side') or r.get('name'),
                     line=r['line'] if r.get('line') is not None else r.get('point'))
            if q['event_id'] and q['market'] and q['side'] and q['commence_time']:
                found.setdefault(identity(q), q)
    return list(found.values())


def flatten(event, sport):
    quotes = []
    for book in event.get('bookmakers', []):
        for m in book.get('markets', []):
            for o in m.get('outcomes', []):
                quotes.append(dict(sport=sport, event_id=event['id'], game=f"{event['away_team']} @ {event['home_team']}",
                                   commence_time=event['commence_time'], book=book['key'], market=m['key'],
                                   player=o.get('description') or '', side=o['name'], line=o.get('point'),
                                   price=o['price'], last_update=m.get('last_update') or book.get('last_update')))
    return quotes


def opposite(q, quotes):
    for o in quotes:
        if o is q or (o['event_id'], o['market'], o['player']) != (q['event_id'], q['market'], q['player']):
            continue
        if o['side'] == q['side']:
            continue
        if q['market'] == 'h2h' or o['line'] == (-q['line'] if q['market'] == 'spreads' else q['line']):
            return o
    return None


def no_vig(q, quotes):
    o = opposite(q, quotes)
    if not o:
        return None
    a, b = 1 / decimal(q['price']), 1 / decimal(o['price'])
    return a / (a + b)


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


def run(client, root, now, sports):
    quotes, ours, asked = [], {}, {}
    for sport in sports:
        cfg = SPORTS[sport]
        horizon = now + timedelta(days=cfg['days'])
        upcoming = [q for q in feed_outcomes(root, sport) if now < stamp(q['commence_time']) <= horizon]
        ours[sport] = upcoming
        window = dict(commenceTimeFrom=now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                      commenceTimeTo=horizon.strftime('%Y-%m-%dT%H:%M:%SZ'))
        games = client.get(f"{cfg['key']}/odds", bookmakers='pinnacle', markets=','.join(GAME),
                           oddsFormat='american', **window)
        for event in games or []:
            quotes += flatten(event, sport)
        # Props for the earliest events in our own feed, preferring events with prop rows.
        starts = {q['event_id']: q['commence_time'] for q in upcoming}
        with_props = {q['event_id'] for q in upcoming if q['market'] not in GAME}
        asked[sport] = sorted(starts, key=lambda e: (e not in with_props, starts[e]))[:cfg['events']]
        for event_id in asked[sport]:
            event = client.get(f"{cfg['key']}/events/{event_id}/odds", bookmakers='pinnacle',
                               markets=','.join(cfg['props']), oddsFormat='american')
            if isinstance(event, dict) and event.get('id') == event_id:
                quotes += flatten(event, sport)
    return quotes, ours, asked


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


def card_check(root, quotes):
    """Today's card picks with Pinnacle as the market, at the price we published."""
    try:
        card = json.loads((root / 'docs/briefing/morning-card.json').read_text())
    except (OSError, ValueError):
        return []
    result = []
    for r in card.get('rows', []):
        market = r.get('market') if r.get('sport') != 'NFL' else ODDS_KEY.get(r.get('market_std'), r.get('market_std'))
        mine = dict(event_id=r.get('event_id') or r.get('game_id'), market=market, player=r.get('player') or '',
                    side=r.get('side'), line=r.get('line'))
        same = [q for q in quotes if (q['event_id'], q['market'], q['player'].casefold(), str(q['side']).casefold()) ==
                (mine['event_id'], market, mine['player'].casefold(), str(mine['side']).casefold())]
        exact = next((q for q in same if q['line'] == mine['line']), None)
        blend = r.get('blend') or {}
        entry = dict(sport=r['sport'], bet=' '.join(str(x) for x in (r.get('player') or r.get('game'), r.get('side'), r.get('line'), market) if x not in (None, '')),
                     book=r.get('book'), price=r.get('price'), quoted_at=r.get('quoted_at'),
                     model=blend.get('model'), us_market=blend.get('market'), us_blend_ev=blend.get('ev'),
                     pinnacle_lines=sorted({q['line'] for q in same if q['line'] is not None}))
        if exact:
            p = no_vig(exact, quotes)
            entry.update(pinnacle_price=exact['price'], pinnacle_updated=exact['last_update'], pinnacle_no_vig=p)
            if p is not None and isinstance(blend.get('model'), (int, float)):
                final = 1 / (1 + math.exp(-(0.25 * logit(blend['model']) + 0.75 * logit(p))))
                entry.update(pinnacle_blend=final, pinnacle_blend_ev=final * decimal(r['price']) - 1)
        result.append(entry)
    return result


def pct(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def report(result):
    lines = [f"## Pinnacle coverage, {eastern_time(result['generated_at'])}", '',
             f"Credits charged: {result['credits_spent']:.0f}; balance after: {result['credits_remaining']}", '',
             '| Sport | Market | Events | Pinnacle events | Pinnacle outcomes | Our outcomes | Same exact line |',
             '|---|---|---|---|---|---|---|']
    for r in result['coverage']:
        share = f" ({r['exact_line_matches'] / r['our_outcomes']:.0%})" if r['our_outcomes'] else ''
        lines.append(f"| {r['sport']} | {r['market']} | {r['events_requested']} | {r['pinnacle_events']} | "
                     f"{r['pinnacle_outcomes']} | {r['our_outcomes']} | {r['exact_line_matches']}{share} |")
    lines += ['', '| Card pick | Our price | Model | US market | US blend EV | Pinnacle | Pinnacle no-vig | Blend EV with Pinnacle |',
              '|---|---|---|---|---|---|---|---|']
    for c in result['card']:
        pinnacle = (f"{c['pinnacle_price']:+.0f}" if c.get('pinnacle_price') is not None else
                    'lines ' + ', '.join(map(str, c['pinnacle_lines'])) if c['pinnacle_lines'] else 'not posted')
        lines.append(f"| {c['sport']} {c['bet']} | {c['price']:+.0f} {c['book']} | {pct(c['model'])} | {pct(c['us_market'])} | "
                     f"{pct(c['us_blend_ev'])} | {pinnacle} | {pct(c.get('pinnacle_no_vig'))} | {pct(c.get('pinnacle_blend_ev'))} |")
    times = sorted(c['quoted_at'] for c in result['card'] if c.get('quoted_at'))
    if times:
        lines += ['', f"Card prices are as published (quoted from {eastern_time(times[0])}); Pinnacle prices are as of this run."]
    failed = [r for r in result['requests'] if r.get('error')]
    if failed:
        lines += ['', 'Requests without data:'] + [f"- {r['path']} ({r.get('status', 'not sent')}): {r['error'][:160]}" for r in failed]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='Checkout whose docs/ feeds are compared')
    parser.add_argument('--sports', default='NFL,MLB,NHL,NBA')
    parser.add_argument('--max-credits', type=int, default=120)
    parser.add_argument('--out', default='artifacts/pinnacle/coverage.json')
    args = parser.parse_args()
    key = os.environ.get('ODDS_API_KEY')
    if not key:
        sys.exit('ODDS_API_KEY is not set')
    now = datetime.now(timezone.utc)
    budget = CreditBudget(run_cap=args.max_credits, label='Pinnacle coverage')
    client = Client(key, budget)
    root = Path(args.root)
    quotes, ours, asked = run(client, root, now, [s.strip() for s in args.sports.split(',') if s.strip()])
    result = dict(generated_at=now.isoformat(), credits_spent=budget.spent, credits_remaining=budget.remaining,
                  requests=client.log, events_requested=asked, coverage=coverage(quotes, ours, asked),
                  card=card_check(root, quotes), quotes=quotes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    text = report(result)
    print(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as handle:
            handle.write(text + '\n')


if __name__ == '__main__':
    main()
