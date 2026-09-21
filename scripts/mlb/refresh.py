#!/usr/bin/env python3
"""MLB regular/postseason markets with official fixture and pitcher context."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from nba.pipeline import OddsClient, FeedError, compare, flatten, iso, normal_name, read_json, save_json, timestamp

UTC = timezone.utc
ET = ZoneInfo('America/New_York')
SPORT = 'baseball_mlb'
PROPS = {'pitcher_strikeouts': 'Pitcher strikeouts', 'pitcher_outs': 'Pitcher outs',
         'batter_hits': 'Batter hits', 'batter_total_bases': 'Batter total bases',
         'batter_home_runs': 'Batter home runs', 'batter_rbis': 'Batter RBIs'}
MARKETS = {**PROPS, 'totals': 'Game total', 'spreads': 'Run line', 'h2h': 'Moneyline'}
PHASES = {'R': 'Regular season', 'F': 'Wild Card', 'D': 'Division Series',
          'L': 'Championship Series', 'W': 'World Series'}


def official_json(endpoint, **params):
    try:
        response = requests.get('https://statsapi.mlb.com/api/v1/' + endpoint, params=params, timeout=25)
        if not response.ok:
            raise FeedError(f'MLB official feed returned HTTP {response.status_code}')
        return response.json()
    except (requests.RequestException, ValueError):
        raise FeedError('MLB official feed unavailable') from None


def normalize_team(name):
    value = normal_name(name)
    return 'athletics' if value in ['oaklandathletics', 'sacramentoathletics'] else value


def schedule(now):
    payload = official_json('schedule', sportId=1, startDate=now.astimezone(ET).date().isoformat(),
        endDate=(now + timedelta(days=45)).date().isoformat(), hydrate='probablePitcher,team')
    if not isinstance(payload, dict) or not isinstance(payload.get('dates'), list):
        raise FeedError('MLB schedule response has an unexpected format')
    games = {}
    for day in payload['dates']:
        for g in day.get('games', []):
            start, status = timestamp(g.get('gameDate')), g.get('status', {})
            if (g.get('gameType') not in PHASES or str(g.get('season')) != str(now.year)
                or status.get('abstractGameState') != 'Preview' or status.get('startTimeTBD', False)
                or g.get('resumeDate') or g.get('resumedFrom')
                or status.get('detailedState') not in ['Scheduled', 'Pre-Game']
                or not start or not now < start <= now + timedelta(days=45)):
                continue
            sides = g.get('teams', {})
            if any(not sides.get(s, {}).get('team', {}).get('id') or not sides[s]['team'].get('name') for s in ['home', 'away']):
                continue
            home, away = sides['home'], sides['away']
            games[g['gamePk']] = dict(mlb_game_id=g['gamePk'], season=int(g['season']), game_type=g['gameType'],
                phase=PHASES[g['gameType']], commence_time=iso(start),
                home_team=home['team']['name'], away_team=away['team']['name'],
                home_team_id=home['team']['id'], away_team_id=away['team']['id'],
                home_pitcher=home.get('probablePitcher'), away_pitcher=away.get('probablePitcher'),
                venue=g.get('venue', {}).get('name'), game_number=g.get('gameNumber', 1),
                doubleheader=g.get('doubleHeader', 'N') != 'N',
                series_game=g.get('seriesGameNumber'), if_necessary=g.get('ifNecessary') == 'Y',
                lineup_status='Batting lineups not verified')
    return sorted(games.values(), key=lambda g:g['commence_time'])


def match_events(events, games):
    matched = []
    for event in events:
        start = timestamp(event.get('commence_time'))
        if not start:
            continue
        candidates = [g for g in games if normalize_team(g['home_team']) == normalize_team(event.get('home_team'))
            and normalize_team(g['away_team']) == normalize_team(event.get('away_team'))
            and abs((timestamp(g['commence_time']) - start).total_seconds()) <= 600]
        if len(candidates) == 1:
            matched.append((event, candidates[0]))
    # Do not count duplicate provider events as separate games or books.
    counts = defaultdict(int)
    for _, game in matched:
        counts[game['mlb_game_id']] += 1
    return [(e,g) for e,g in matched if counts[g['mlb_game_id']] == 1]


def load_history(now):
    path = ROOT / 'data/mlb/history/current.json'
    saved = read_json(path, {})
    checked = timestamp(saved.get('fetched_at'))
    through = (now.astimezone(ET).date() - timedelta(days=1)).isoformat()
    if checked and timedelta(0) <= now - checked < timedelta(hours=12) and saved.get('through_date') == through:
        return saved
    try:
        history = dict(season=now.year, fetched_at=iso(now), through_date=through, groups={})
        for group in ['hitting', 'pitching']:
            data = official_json('stats', stats='byDateRange', group=group, sportIds=1, season=now.year,
                startDate=f'{now.year}-01-01', endDate=through, gameType='R', playerPool='ALL', limit=10000)
            blocks = [s for s in data.get('stats', []) if s.get('group', {}).get('displayName') == group]
            if len(blocks) != 1 or not isinstance(blocks[0].get('splits'), list):
                raise FeedError('MLB statistics response has an unexpected format')
            splits = blocks[0]['splits']
            if len(splits) != blocks[0].get('totalSplits'):
                raise FeedError('MLB statistics response is incomplete')
            history['groups'][group] = [dict(player_id=r['player']['id'], name=r['player']['fullName'],
                team_id=r.get('team', {}).get('id'), stat=r['stat']) for r in splits if r.get('sport', {}).get('id', 1) == 1]
        save_json(path, history)
        return history
    except (FeedError, KeyError, TypeError) as error:
        message = str(error) if isinstance(error, FeedError) else 'MLB statistics response is incomplete'
        print(message, file=sys.stderr)
        return {**saved, 'error': message}


def history_lookup(history, now):
    checked = timestamp(history.get('fetched_at'))
    if (history.get('error') or history.get('season') != now.year or not checked
        or not timedelta(0) <= now - checked < timedelta(hours=36)):
        return {}
    result = {}
    for group, rows in history.get('groups', {}).items():
        names = defaultdict(list)
        for row in rows:
            names[normal_name(row['name'])].append(row)
        # Ambiguous identities and multiple team splits stay unavailable.
        result[group] = {name: records[0] for name, records in names.items() if len(records) == 1}
    return result


def context(row, game, history):
    row.update(market_family='props' if row['market'] in PROPS else 'lines',
               mlb_game_id=game['mlb_game_id'], game_type=game['game_type'], phase=game['phase'],
               home_pitcher=game['home_pitcher'], away_pitcher=game['away_pitcher'],
               game_number=game['game_number'], doubleheader=game['doubleheader'],
               if_necessary=game.get('if_necessary', False), series_game=game.get('series_game'),
               venue=game['venue'], lineup_status=game['lineup_status'],
               model_probability=None, model_status='MLB predictions are not validated', stat_context=None)
    if game['doubleheader']:
        row['game'] += f" · Game {game['game_number']}"
    group = 'pitching' if row['market'].startswith('pitcher_') else 'hitting'
    record = history.get(group, {}).get(normal_name(row['player']))
    if not record:
        return row
    stats = record['stat']
    if group == 'pitching':
        probable = [p['id'] for p in [game['home_pitcher'], game['away_pitcher']] if p and p.get('id')]
        row['starter_status'] = 'Listed probable starter' if record['player_id'] in probable else 'Not listed as a probable starter'
        # Innings such as 5.2 mean 17 outs, not 5.2 decimal innings.
        outs = stats.get('outsPitched', stats.get('outs', 0))
        if outs > 0:
            row['stat_context'] = dict(group=group, player_id=record['player_id'], innings=stats.get('inningsPitched'),
                games=stats.get('gamesPlayed'), starts=stats.get('gamesStarted'), strikeouts=stats.get('strikeOuts'),
                k_per_nine=27*stats.get('strikeOuts', 0)/outs, era=27*stats.get('earnedRuns', 0)/outs)
    elif row['market'] in PROPS and stats.get('plateAppearances', 0) > 0:
        row['stat_context'] = dict(group=group, player_id=record['player_id'], pa=stats['plateAppearances'],
            avg=stats.get('avg'), ops=stats.get('ops'), hits=stats.get('hits'), home_runs=stats.get('homeRuns'),
            total_bases=stats.get('totalBases'), rbis=stats.get('rbi'))
    return row


def refresh(client, now, games, history):
    events = client.get('events', dateFormat='iso')
    if not isinstance(events, list):
        raise FeedError('MLB odds event response has an unexpected format')
    matches = match_events(events, games)
    accepted = {e['id']:g for e,g in matches}
    rows = []
    lookup = history_lookup(history, now)
    def add(event, game):
        for row in flatten(event, now, SPORT, MARKETS, list(PROPS)):
            if now - timestamp(row['quoted_at']) <= timedelta(hours=12):
                rows.append(context(row, game, lookup))
    if matches:
        odds = client.get('odds', regions='us', markets='h2h,spreads,totals', oddsFormat='american',
            commenceTimeFrom=iso(now.replace(microsecond=0)), commenceTimeTo=iso((now+timedelta(days=45)).replace(microsecond=0)))
        if not isinstance(odds, list):
            raise FeedError('MLB game odds response has an unexpected format')
        for event, game in match_events(odds, games):
            if event['id'] in accepted:
                add(event, game)
    near = sorted([(e,g) for e,g in matches if timestamp(e['commence_time']) <= now+timedelta(hours=24)], key=lambda v:v[0]['commence_time'])
    for event, game in near[:20]:
        prop = client.get(f"events/{event['id']}/odds", regions='us', markets=','.join(PROPS), oddsFormat='american')
        if not isinstance(prop, dict) or prop.get('id') != event['id'] or not match_events([prop], [game]):
            raise FeedError('MLB props response did not match the scheduled game')
        add(prop, game)
    return dict(sport=SPORT, season=now.year, status='ready' if rows else 'waiting_for_markets',
        checked_at=iso(now), last_success_at=iso(now), rows=compare(rows), events=games,
        matched_events=len(matches), excluded_events=len(events)-len(matches),
        props_events_skipped=max(0,len(near)-20), history_checked_at=history.get('fetched_at'),
        history_through_date=history.get('through_date'), history_error=history.get('error'),
        history_players={k:len(v) for k,v in history.get('groups', {}).items()},
        requests=client.requests, quota_remaining=client.quota_remaining, model_status='No validated MLB predictions')


def validate(state, now):
    errors = []
    checked = timestamp(state.get('last_success_at'))
    if state.get('status') not in ['ready', 'waiting_for_markets'] or not checked or not timedelta(minutes=-5) <= now-checked <= timedelta(hours=12):
        errors.append('MLB snapshot is unavailable or expired')
    if state.get('season') != now.year:
        errors.append('MLB snapshot has the wrong season')
    for row in state.get('rows', []):
        quote, start = timestamp(row.get('quoted_at')), timestamp(row.get('commence_time'))
        if (row.get('game_type') not in PHASES or not quote or not start or start <= now
            or not timedelta(minutes=-5) <= now-quote <= timedelta(hours=12)):
            errors.append('MLB snapshot contains an ineligible or expired market')
            break
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    now = datetime.now(UTC)
    public = ROOT / 'docs/mlb/data/latest.json'
    state = read_json(public, dict(status='not_checked', events=[], rows=[], last_success_at=None, season=now.year))
    if args.validate:
        errors = validate(state, now)
        print('\n'.join(errors) if errors else 'MLB freshness: PASS')
        return int(bool(errors))
    code = 0
    if not args.offline:
        try:
            games = schedule(now)
            history = load_history(now)
            client = OddsClient(os.getenv('MLB_ODDS_API_KEY') or os.getenv('ODDS_API_KEY'), SPORT)
            state = refresh(client, now, games, history)
            # Long prop fetches can cross first pitch; remove those games before publishing.
            finished = datetime.now(UTC)
            state['rows'] = [r for r in state['rows'] if timestamp(r['commence_time']) > finished]
            state['events'] = [g for g in state['events'] if timestamp(g['commence_time']) > finished]
            state['status'] = 'ready' if state['rows'] else 'waiting_for_markets'
            save_json(ROOT/'data/mlb/snapshots'/(now.strftime('%Y%m%dT%H%M%SZ')+'.json'), state)
        except FeedError as error:
            state.update(status='feed_error', checked_at=iso(now), error=str(error))
            print(str(error), file=sys.stderr)
            code = 1
        save_json(public, state)
    from mlb.site import build
    build(state)
    print(f"MLB: {state['status']}; {len(state['events'])} scheduled games; {len(state['rows'])} quotes")
    return code


if __name__ == '__main__':
    raise SystemExit(main())
