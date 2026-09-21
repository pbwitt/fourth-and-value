#!/usr/bin/env python3
"""NBA market refresh. Empty markets are valid; failed fetches are not empty markets."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys
import unicodedata

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from market_math import implied_probability

UTC = timezone.utc
SPORT = 'basketball_nba'
MARKETS = {
    'player_points': 'Points', 'player_rebounds': 'Rebounds',
    'player_assists': 'Assists', 'player_threes': 'Three-pointers made',
    'player_points_rebounds_assists': 'Points + rebounds + assists',
    'player_points_rebounds': 'Points + rebounds',
    'player_points_assists': 'Points + assists',
    'player_rebounds_assists': 'Rebounds + assists',
    'player_blocks': 'Blocks', 'player_steals': 'Steals',
    'player_turnovers': 'Turnovers',
    'totals': 'Game total', 'spreads': 'Spread', 'h2h': 'Moneyline',
}
PROP_MARKETS = list(MARKETS)[:11]
STATS = {
    'player_points': ['PTS'], 'player_rebounds': ['REB'],
    'player_assists': ['AST'], 'player_threes': ['FG3M'],
    'player_points_rebounds_assists': ['PTS', 'REB', 'AST'],
    'player_points_rebounds': ['PTS', 'REB'], 'player_points_assists': ['PTS', 'AST'],
    'player_rebounds_assists': ['REB', 'AST'], 'player_blocks': ['BLK'],
    'player_steals': ['STL'], 'player_turnovers': ['TOV'],
}


def iso(value):
    return value.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def timestamp(value):
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result if result.tzinfo else result.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


class FeedError(RuntimeError):
    pass


class OddsClient:
    def __init__(self, key, sport=SPORT):
        if not key:
            raise FeedError('ODDS_API_KEY is missing')
        self.key = key
        self.sport = sport
        self.quota_remaining = None
        self.requests = 0

    def get(self, suffix, **params):
        try:
            response = requests.get(f'https://api.the-odds-api.com/v4/sports/{self.sport}/{suffix}',
                                    params=dict(params, apiKey=self.key), timeout=25)
            self.requests += 1
            self.quota_remaining = response.headers.get('x-requests-remaining')
            if not response.ok:
                raise FeedError(f'Odds provider returned HTTP {response.status_code}')
            return response.json()
        except (requests.RequestException, ValueError) as error:
            # Never include the request URL: it contains credentials.
            raise FeedError(f'Odds provider request failed ({type(error).__name__})') from None


def flatten(event, now, sport=SPORT, markets=MARKETS, prop_markets=PROP_MARKETS):
    rows = []
    kickoff = timestamp(event.get('commence_time'))
    if not kickoff or kickoff <= now or event.get('sport_key', sport) != sport:
        return rows
    for book in event.get('bookmakers', []):
        for market in book.get('markets', []):
            key = market.get('key')
            if key not in markets:
                continue
            updated = timestamp(market.get('last_update') or book.get('last_update'))
            if not updated or not timedelta(minutes=-5) <= now - updated <= timedelta(hours=24):
                continue
            for outcome in market.get('outcomes', []):
                probability = implied_probability(outcome.get('price'))
                if not math.isfinite(probability):
                    continue
                side = str(outcome.get('name', ''))
                player = outcome.get('description', '') if key in prop_markets else ''
                point = outcome.get('point')
                if key != 'h2h':
                    try:
                        point = float(point)
                    except (ValueError, TypeError):
                        continue
                    if not math.isfinite(point):
                        continue
                if (key in prop_markets and not player) or (key in list(prop_markets) + ['totals'] and side not in ['Over', 'Under']):
                    continue
                if key in ['spreads', 'h2h'] and side not in [event['home_team'], event['away_team']]:
                    continue
                rows.append(dict(event_id=event['id'], commence_time=event['commence_time'],
                    home_team=event['home_team'], away_team=event['away_team'],
                    game=f"{event['away_team']} @ {event['home_team']}",
                    book=book['key'], book_label=book['title'], market=key,
                    market_label=markets[key], player=player, side=side, line=point,
                    price=float(outcome['price']), book_probability=probability,
                    quoted_at=iso(updated)))
    return rows


def comparison_key(row):
    # Spread pair: home -3.5 must pair with away +3.5, never away -3.5.
    line = row['line']
    if row['market'] == 'spreads' and row['side'] == row['away_team']:
        line = -line
    return row['event_id'], row['player'], row['market'], line


def compare(rows):
    unique = {}
    conflicts = set()
    for row in rows:
        identity = (*comparison_key(row), row['book'], row['side'])
        if identity in unique and unique[identity]['price'] != row['price']:
            conflicts.add(identity)
        unique[identity] = row
    rows = [r for k, r in unique.items() if k not in conflicts]
    pairs = defaultdict(list)
    for row in rows:
        pairs[(*comparison_key(row), row['book'])].append(row)
    for pair in pairs.values():
        expected = {pair[0]['home_team'], pair[0]['away_team']} if pair[0]['market'] in ['spreads', 'h2h'] else {'Over', 'Under'}
        total = sum(r['book_probability'] for r in pair)
        valid = len(pair) == 2 and {r['side'] for r in pair} == expected
        for row in pair:
            row['fair_probability'] = row['book_probability'] / total if valid else None
    groups = defaultdict(list)
    for row in rows:
        groups[(*comparison_key(row), row['side'])].append(row)
    for quotes in groups.values():
        values = [r['fair_probability'] for r in quotes if r['fair_probability'] is not None]
        best = min(r['book_probability'] for r in quotes)
        for row in quotes:
            row.update(consensus_probability=statistics.median(values) if values else None,
                       paired_books=len(values), best_price=row['book_probability'] == best)
            # Leave this book out so its outlier price cannot endorse itself.
            other = [r['fair_probability'] for r in quotes if r['book'] != row['book'] and r['fair_probability'] is not None]
            row['other_book_probability'] = statistics.median(other) if other else None
            row['other_books'] = len(other)
            row['consensus_ev'] = 100 * (statistics.median(other) / row['book_probability'] - 1) if len(other) >= 3 else None
    return rows


def normal_name(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(value)).lower() if c.isalnum())


def add_baselines(rows, history, now):
    """Historical reference, not a calibrated or injury-adjusted forecast.

    Use only completed earlier regular-season games; combos use joint game
    observations rather than falsely independent marginal distributions.
    """
    players = defaultdict(list)
    team_games = defaultdict(list)
    seen = set()
    for game in history.get('players', []):
        dt = timestamp(game.get('GAME_DATE'))
        identity = game.get('GAME_ID'), game.get('PLAYER_ID')
        if not dt or not now - timedelta(days=370) < dt < now or identity in seen:
            continue
        if float(game.get('MIN') or 0) <= 0:
            continue
        seen.add(identity)
        players[normal_name(game.get('PLAYER_NAME'))].append(game)
    for game in history.get('teams', []):
        dt = timestamp(game.get('GAME_DATE'))
        if dt and now - timedelta(days=370) < dt < now:
            team_games[game.get('TEAM_NAME')].append(game)
    for row in rows:
        row.update(baseline_mean=None, baseline_probability=None, baseline_games=0,
                   baseline_last_game=None, baseline_push=None,
                   model_status='Awaiting sufficient player history', model_probability=None)
        if row['market'] not in STATS:
            row['model_status'] = 'NBA game model not yet validated'
            continue
        games = sorted(players.get(normal_name(row['player']), []), key=lambda g:g['GAME_DATE'])[-30:]
        # Ambiguous name matches are unavailable, not guessed player identities.
        if len({g['PLAYER_ID'] for g in games}) != 1 or len(games) < 20:
            continue
        vals = []
        for game in games:
            if all(game.get(k) is not None for k in STATS[row['market']]):
                vals.append(sum(float(game[k]) for k in STATS[row['market']]))
        if len(vals) < 20:
            continue
        wins = sum(v > row['line'] if row['side'] == 'Over' else v < row['line'] for v in vals)
        pushes = sum(v == row['line'] for v in vals)
        row.update(baseline_mean=statistics.mean(vals), baseline_games=len(vals),
                   baseline_probability=(wins + 1) / (len(vals) - pushes + 2),
                   baseline_last_game=games[-1]['GAME_DATE'], baseline_push=pushes / len(vals),
                   model_status='Historical baseline only; minutes and injuries not adjusted')
    return rows


def fetch_history(season):
    """NBA Stats game logs: native player IDs, no preseason mixed into training."""
    result = dict(season=season, fetched_at=iso(datetime.now(UTC)), players=[], teams=[])
    for kind, destination in [('P', 'players'), ('T', 'teams')]:
        try:
            response = requests.get('https://stats.nba.com/stats/leaguegamelog', timeout=35,
                headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nba.com/'},
                params=dict(Counter=0, Direction='DESC', LeagueID='00', PlayerOrTeam=kind,
                            Season=season, SeasonType='Regular Season', Sorter='DATE'))
            if not response.ok:
                raise FeedError(f'NBA Stats returned HTTP {response.status_code}')
            sets = response.json()['resultSets']
            result[destination] = [dict(zip(sets[0]['headers'], values)) for values in sets[0]['rowSet']]
        except (requests.RequestException, KeyError, ValueError, IndexError) as error:
            raise FeedError(f'NBA Stats unavailable ({type(error).__name__})') from None
    return result


def refresh(client, now, history, previous=None):
    events = client.get('events', dateFormat='iso')
    if not isinstance(events, list):
        raise FeedError('Events response has an unexpected format')
    events = [e for e in events if timestamp(e.get('commence_time')) and timestamp(e['commence_time']) > now]
    events.sort(key=lambda e:e['commence_time'])
    near = [e for e in events if timestamp(e['commence_time']) <= now + timedelta(days=45)]
    rows = []
    if near:
        games = client.get('odds', regions='us', markets='h2h,spreads,totals', oddsFormat='american',
                           commenceTimeFrom=iso(now.replace(microsecond=0)),
                           commenceTimeTo=iso((now + timedelta(days=45)).replace(microsecond=0)))
        if not isinstance(games, list):
            raise FeedError('Game odds response has an unexpected format')
        for event in games:
            rows.extend(flatten(event, now))
    props_events = [e for e in near if timestamp(e['commence_time']) <= now + timedelta(hours=48)]
    for event in props_events[:16]:
        data = client.get(f"events/{event['id']}/odds", regions='us',
                          markets=','.join(PROP_MARKETS), oddsFormat='american')
        if not isinstance(data, dict) or data.get('id') != event['id']:
            raise FeedError('Prop odds response has an unexpected format')
        rows.extend(flatten(data, now))
    rows = add_baselines(compare(rows), history, now)
    return dict(sport=SPORT, checked_at=iso(now), last_success_at=iso(now),
                status='ready' if rows else 'waiting_for_markets', events=events, rows=rows,
                history_players=len(history.get('players', [])),
                history_updated_at=history.get('fetched_at'),
                model_status='Historical baselines; NBA predictions are not validated',
                requests=client.requests, quota_remaining=client.quota_remaining,
                props_events_skipped=max(0, len(props_events)-16))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Build from saved public snapshot; do not call APIs')
    parser.add_argument('--history-season', help='Fetch NBA Stats regular-season game logs, e.g. 2025-26')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    public = ROOT / 'docs/nba/data/latest.json'
    history_dir = ROOT / 'data/nba/history'
    now = datetime.now(UTC)
    state = read_json(public, dict(status='not_checked', checked_at=None, last_success_at=None,
                                  events=[], rows=[], history_players=0))
    if args.history_season:
        try:
            save_json(history_dir / f'{args.history_season}.json', fetch_history(args.history_season))
        except FeedError as error:
            print(str(error), file=sys.stderr)
            return 1
    history = dict(players=[], teams=[])
    for path in sorted(history_dir.glob('*.json')):
        data = read_json(path, {})
        history['players'].extend(data.get('players', []))
        history['teams'].extend(data.get('teams', []))
        history['fetched_at'] = data.get('fetched_at')
    exit_code = 0
    if not args.offline:
        try:
            client = OddsClient(os.getenv('ODDS_API_KEY'))
            state = refresh(client, now, history, state)
            archive = ROOT / 'data/nba/snapshots' / f"{now.strftime('%Y%m%dT%H%M%SZ')}.json"
            save_json(archive, state)
        except FeedError as error:
            # Keep saved evidence but UI hides it on failure; do not label it fresh.
            state.update(status='feed_error', checked_at=iso(now), error=str(error))
            print(str(error), file=sys.stderr)
            exit_code = 1
        save_json(public, state)
    from nba.site import build
    build(state)
    print(f"NBA: {state['status']}; {len(state['events'])} upcoming provider events; {len(state['rows'])} saved quotes")
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
