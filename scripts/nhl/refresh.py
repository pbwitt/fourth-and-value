#!/usr/bin/env python3
"""Regular-season NHL refresh; old predictions are never fallback inputs."""
import argparse
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys

import requests
from dotenv import load_dotenv
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from nba.pipeline import OddsClient, FeedError, compare, flatten, iso, normal_name, read_json, save_json, timestamp

UTC = timezone.utc
SPORT = 'icehockey_nhl'
PROPS = {'player_shots_on_goal': 'Shots on goal', 'player_goals': 'Goals',
         'player_assists': 'Assists', 'player_points': 'Points'}
MARKETS = {**PROPS, 'totals': 'Game total', 'spreads': 'Puck line', 'h2h': 'Moneyline'}
STAT_KEYS = dict(zip(PROPS, ['shots', 'goals', 'assists', 'points']))


def season_for(now):
    # September preseason and opening night belong to the upcoming winter.
    year = now.year if now.month >= 9 else now.year - 1
    return year * 10000 + year + 1


def official_json(url, **params):
    try:
        response = requests.get(url, params=params, timeout=25)
        if not response.ok:
            raise FeedError(f'NHL feed returned HTTP {response.status_code}')
        return response.json()
    except (requests.RequestException, ValueError):
        raise FeedError('NHL official feed unavailable') from None


def team_name(team):
    return (team['placeName']['default'] + ' ' + team['commonName']['default']).strip()


def schedule(now):
    games = {}
    for days in range(0, 46, 7):
        payload = official_json('https://api-web.nhle.com/v1/schedule/' + (now + timedelta(days=days)).date().isoformat())
        if not isinstance(payload, dict) or not isinstance(payload.get('gameWeek'), list):
            raise FeedError('NHL schedule response has an unexpected format')
        for day in payload['gameWeek']:
            for game in day.get('games', []):
                start = timestamp(game.get('startTimeUTC'))
                if (game.get('gameType') != 2 or game.get('season') != season_for(now)
                    or game.get('gameState') not in ['FUT', 'PRE'] or game.get('gameScheduleState') != 'OK'
                    or not start or not now < start <= now + timedelta(days=45)):
                    continue
                games[game['id']] = dict(nhl_game_id=game['id'], season=game['season'], game_type=2,
                    commence_time=iso(start), home_team=team_name(game['homeTeam']),
                    away_team=team_name(game['awayTeam']))
    return sorted(games.values(), key=lambda g:g['commence_time'])


def regular_events(events, games):
    result = []
    for event in events:
        start = timestamp(event.get('commence_time'))
        if not start:
            continue
        matches = [g for g in games if normal_name(g['home_team']) == normal_name(event.get('home_team'))
                   and normal_name(g['away_team']) == normal_name(event.get('away_team'))
                   and abs((timestamp(g['commence_time']) - start).total_seconds()) <= 600]
        if len(matches) == 1:
            result.append({**event, 'nhl_game_id': matches[0]['nhl_game_id'], 'game_type': 2})
    return result


def stats_rows(kind, season, now):
    rows = []
    for start in range(0, 10000, 100):
        payload = official_json(f'https://api.nhle.com/stats/rest/en/{kind}/summary',
            isAggregate='false', isGame='false', start=start, limit=100,
            sort='[{"property":"' + ('playerId' if kind == 'skater' else 'teamId') + '","direction":"ASC"}]',
            cayenneExp=f'seasonId={season} and gameTypeId=2 and gameDate<"{now.date().isoformat()}"')
        if not isinstance(payload, dict) or not isinstance(payload.get('data'), list) or 'total' not in payload:
            raise FeedError('NHL stats response has an unexpected format')
        rows.extend(payload['data'])
        if len(rows) >= int(payload['total']):
            return rows
        if not payload['data']:
            raise FeedError('NHL stats pagination ended before the reported total')
    raise FeedError('NHL stats pagination exceeded the safety limit')


def load_history(now, offline=False):
    """Refresh both seasons atomically; yesterday cutoff prevents partial live stats."""
    path = ROOT / 'data/nhl/history/current.json'
    saved = read_json(path, {})
    checked = timestamp(saved.get('fetched_at'))
    if offline or (checked and timedelta(0) <= now - checked < timedelta(hours=12)):
        return saved
    try:
        current = season_for(now)
        history = dict(fetched_at=iso(now), through_date=(now - timedelta(days=1)).date().isoformat(),
                       current_season=current, seasons={})
        for season in [current - 10001, current]:
            history['seasons'][str(season)] = dict(players=stats_rows('skater', season, now),
                                                  teams=stats_rows('team', season, now))
        save_json(path, history)
        return history
    except FeedError as error:
        print(f'History unavailable: {error}', file=sys.stderr)
        # Saved history is retained for audit but never silently treated as current.
        return {**saved, 'error': str(error)}


def historical_rates(history, kind, now):
    checked = timestamp(history.get('fetched_at'))
    if history.get('error') or not checked or not timedelta(0) <= now - checked < timedelta(hours=36):
        return {}
    groups = defaultdict(list)
    for season, data in history.get('seasons', {}).items():
        for item in data.get(kind, []):
            name = item.get('skaterFullName') if kind == 'players' else item.get('teamFullName')
            if name and item.get('gamesPlayed', 0) > 0:
                groups[normal_name(name)].append((int(season), item))
    output = {}
    for name, values in groups.items():
        identities = {r.get('playerId' if kind == 'players' else 'teamId') for _, r in values}
        if len(identities) != 1 or None in identities or len({s for s, _ in values}) != len(values):
            continue
        values.sort(reverse=True, key=lambda v:v[0])
        current = next((r for s, r in values if s == season_for(now)), None)
        prior = next((r for s, r in values if s == season_for(now) - 10001), None)
        if current and current['gamesPlayed'] >= 20:
            selected, source = current, str(season_for(now))
        elif prior and prior['gamesPlayed'] >= 20:
            selected, source = prior, str(season_for(now) - 10001)
        else:
            continue
        output[name] = (selected, source)
    return output


def baselines(rows, history, now):
    players, teams = historical_rates(history, 'players', now), historical_rates(history, 'teams', now)
    for row in rows:
        row.update(baseline_mean=None, model_probability=None, model_status='Historical reference only; not a validated prediction')
        source = players.get(normal_name(row['player']))
        if row['market'] in PROPS and source:
            record, season = source
            value = record.get(STAT_KEYS[row['market']])
            if value is not None:
                mean = float(value) / record['gamesPlayed']
                over = float(poisson.sf(row['line'], mean))
                under = float(poisson.cdf(math.ceil(row['line']) - 1, mean))
                push = float(poisson.pmf(row['line'], mean)) if row['line'].is_integer() else 0.0
                row.update(baseline_mean=mean, baseline_probability=over if row['side'] == 'Over' else under,
                           baseline_push=push, baseline_games=record['gamesPlayed'], baseline_season=season,
                           baseline_source='Season-average Poisson reference')
        elif row['market'] == 'totals':
            home, away = teams.get(normal_name(row['home_team'])), teams.get(normal_name(row['away_team']))
            if home and away:
                h, hs = home
                a, ass = away
                if hs == ass and all(k in r for r in [h, a] for k in ['goalsForPerGame', 'goalsAgainstPerGame']):
                    row.update(baseline_mean=(h['goalsForPerGame'] + a['goalsAgainstPerGame'] + a['goalsForPerGame'] + h['goalsAgainstPerGame']) / 2,
                               baseline_games=min(h['gamesPlayed'], a['gamesPlayed']), baseline_season=hs,
                               baseline_source='Team scoring/conceding reference', baseline_probability=None)
    return rows


def refresh(client, now, games, history):
    events = client.get('events', dateFormat='iso')
    if not isinstance(events, list):
        raise FeedError('NHL odds event response has an unexpected format')
    accepted = regular_events(events, games)
    allowed = {e['id'] for e in accepted}
    rows = []
    if accepted:
        odds = client.get('odds', regions='us', markets='h2h,spreads,totals', oddsFormat='american',
                          commenceTimeFrom=iso(now.replace(microsecond=0)),
                          commenceTimeTo=iso((now + timedelta(days=45)).replace(microsecond=0)))
        if not isinstance(odds, list):
            raise FeedError('NHL game odds response has an unexpected format')
        for event in regular_events(odds, games):
            if event['id'] in allowed:
                rows.extend(flatten(event, now, SPORT, MARKETS, list(PROPS)))
    near = [e for e in accepted if timestamp(e['commence_time']) <= now + timedelta(hours=48)]
    for event in near[:16]:
        prop = client.get(f"events/{event['id']}/odds", regions='us', markets=','.join(PROPS), oddsFormat='american')
        if not isinstance(prop, dict) or prop.get('id') != event['id'] or not regular_events([prop], games):
            raise FeedError('NHL prop odds response did not match the regular-season event')
        rows.extend(flatten(prop, now, SPORT, MARKETS, list(PROPS)))
    rows = baselines(compare(rows), history, now)
    return dict(sport=SPORT, season=season_for(now), status='ready' if rows else 'waiting_for_markets',
                checked_at=iso(now), last_success_at=iso(now), events=games, rows=rows,
                matched_events=len(accepted), excluded_events=len(events)-len(accepted),
                props_events_skipped=max(0, len(near)-16), history_checked_at=history.get('fetched_at'),
                history_through_date=history.get('through_date'), history_error=history.get('error'),
                history_player_count=sum(len(s.get('players', [])) for s in history.get('seasons', {}).values()),
                requests=client.requests, quota_remaining=client.quota_remaining,
                model_status='Historical references only; legacy models withheld pending leakage correction and validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Build saved pages without network calls')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    now = datetime.now(UTC)
    public = ROOT / 'docs/nhl/data/latest.json'
    state = read_json(public, dict(status='not_checked', events=[], rows=[], last_success_at=None, season=season_for(now)))
    code = 0
    if not args.offline:
        try:
            games = schedule(now)
            history = load_history(now)
            client = OddsClient(os.getenv('NHL_ODDS_API_KEY') or os.getenv('ODDS_API_KEY'), SPORT)
            state = refresh(client, now, games, history)
            save_json(ROOT / 'data/nhl/snapshots' / (now.strftime('%Y%m%dT%H%M%SZ') + '.json'), state)
        except FeedError as error:
            state.update(status='feed_error', error=str(error), checked_at=iso(now))
            print(str(error), file=sys.stderr)
            code = 1
        save_json(public, state)
    from nhl.site import build
    build(state)
    print(f"NHL: {state['status']}; {len(state['events'])} regular-season games; {len(state['rows'])} saved quotes")
    return code


if __name__ == '__main__':
    raise SystemExit(main())
