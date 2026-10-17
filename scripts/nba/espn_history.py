#!/usr/bin/env python3
"""NBA game, team and player history from ESPN's public site API (no paid calls).

Production history for the NBA pipeline: stats.nba.com refuses GitHub runners, so
the daily refresh reads the compact files this script commits to
``artifacts/nba/history/<season>.json.gz`` (plus ``manifest.json``). Raw ESPN
responses are cached under ``data/nba/espn/`` (gitignored) so a rebuild is free.

Rows use NBA Stats column names so ``pipeline.add_baselines`` and
``player_context.nba_defense`` keep working; IDs are ESPN event/athlete/team IDs.
Every game carries ``SEASON_TYPE``: regular, cup_final (counted by ESPN as regular
season, excluded from NBA regular-season stats), play_in or playoffs. Preseason and
exhibition games are dropped. A missing statistic stays absent, never zero; a
player listed as not playing keeps a row with ``DNP`` true and no minutes.

    python scripts/nba/espn_history.py --season 2023-24 --season 2024-25 --season 2025-26
    python scripts/nba/espn_history.py --recent 4      # daily: add the last 4 ET dates
    python scripts/nba/espn_history.py --season 2025-26 --offline   # rebuild from cache
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from nba_weekly_results import API, ET, _competition, _valid_summary, count, minutes  # noqa: E402

HISTORY = ROOT / 'artifacts/nba/history'
CACHE = ROOT / 'data/nba/espn'
SCHEMA_VERSION = 1
# ESPN NBA franchise IDs; All-Star and international exhibition teams fall outside.
FRANCHISES = {str(i) for i in range(1, 31)}
PLAYER_KEYS = {
    'minutes': 'MIN', 'points': 'PTS', 'rebounds': 'REB', 'assists': 'AST',
    'turnovers': 'TOV', 'steals': 'STL', 'blocks': 'BLK',
    'offensiveRebounds': 'OREB', 'defensiveRebounds': 'DREB', 'fouls': 'PF',
    'plusMinus': 'PLUS_MINUS',
    'fieldGoalsMade-fieldGoalsAttempted': ('FGM', 'FGA'),
    'threePointFieldGoalsMade-threePointFieldGoalsAttempted': ('FG3M', 'FG3A'),
    'freeThrowsMade-freeThrowsAttempted': ('FTM', 'FTA'),
}
TEAM_KEYS = {
    'totalRebounds': 'REB', 'offensiveRebounds': 'OREB', 'defensiveRebounds': 'DREB',
    'assists': 'AST', 'steals': 'STL', 'blocks': 'BLK', 'totalTurnovers': 'TOV',
    'fouls': 'PF', 'fieldGoalsMade-fieldGoalsAttempted': ('FGM', 'FGA'),
    'threePointFieldGoalsMade-threePointFieldGoalsAttempted': ('FG3M', 'FG3A'),
    'freeThrowsMade-freeThrowsAttempted': ('FTM', 'FTA'),
}


def season_name(start_year):
    return f'{start_year}-{(start_year + 1) % 100:02}'


def season_start(name):
    year, suffix = name.split('-')
    if len(suffix) != 2 or (int(year) + 1) % 100 != int(suffix):
        raise ValueError(f'Season must look like 2025-26, not {name}')
    return int(year)


def season_of(day):
    """The season an Eastern calendar date belongs to; July-September belong to none."""
    return season_name(day.year if day.month >= 10 else day.year - 1) if day.month not in (7, 8, 9) else None


def season_type(event):
    competition = event['competitions'][0]
    kind = event.get('season', {}).get('type')
    if kind == 2:
        return 'cup_final' if competition.get('type', {}).get('abbreviation') == 'CC' else 'regular'
    return {3: 'playoffs', 5: 'play_in'}.get(kind)


class Fetcher:
    """Polite cached GETs. A cached final summary is never refetched."""

    def __init__(self, offline=False, pause=0.25):
        import requests
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'fourth-and-value NBA history (research; contact via site)'
        self.offline, self.pause, self.requests = offline, pause, 0

    def get(self, url, path, reuse):
        if path.exists() and (reuse or self.offline):
            return json.loads(path.read_text())
        if self.offline:
            return None
        for attempt in range(4):
            try:
                time.sleep(self.pause * (1 + 3 * attempt))
                response = self.session.get(url, timeout=35)
                self.requests += 1
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                payload = response.json()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload))
                return payload
            except Exception:  # network, HTTP or JSON error: retry, then report missing
                if attempt == 3:
                    return None
        return None


def split(value):
    parts = str(value).split('-')
    return tuple(count(p) for p in parts) if len(parts) == 2 else (None, None)


def read_stats(keys, values, mapping):
    """Map ESPN stat keys to NBA Stats columns; unknown or unreadable values stay absent."""
    out = {}
    if len(keys) != len(values):
        return out
    for key, value in zip(keys, values):
        column = mapping.get(key)
        if isinstance(column, tuple):
            made, attempted = split(value)
            if made is not None and attempted is not None and made <= attempted:
                out[column[0]], out[column[1]] = made, attempted
        elif column == 'MIN':
            out[column] = minutes(value)
        elif column == 'PLUS_MINUS':
            try:
                out[column] = int(str(value).replace('+', ''))
            except ValueError:
                pass
        elif column:
            parsed = count(value)
            if parsed is not None:
                out[column] = parsed
    return {k: v for k, v in out.items() if v is not None}


def matchup(own, other, home):
    return f"{own} vs. {other}" if home else f"{own} @ {other}"


def parse_game(event, summary):
    """(game, team rows, player rows) for one verified final, or None."""
    game = _competition(event)
    kind = season_type(event)
    if not game or not kind or not game['final'] or not _valid_summary(summary, game):
        return None
    if {game['home_team_id'], game['away_team_id']} - FRANCHISES:
        return None
    competition = event['competitions'][0]
    sides = {c['homeAway']: c for c in competition['competitors']}
    abbreviation = {c['team']['id']: c['team'].get('abbreviation') for c in sides.values()}
    periods = max((len(c.get('linescores') or []) for c in sides.values()), default=0)
    day = game['date'].isoformat()
    record = dict(GAME_ID=game['id'], GAME_DATE=day, COMMENCE_TIME=game['commence_time'],
                  SEASON_TYPE=kind, NEUTRAL=competition.get('neutralSite') is True,
                  PERIODS=periods or None, NOTE='; '.join(n.get('headline', '') for n in competition.get('notes', [])) or None,
                  HOME_TEAM_ID=game['home_team_id'], HOME_TEAM_NAME=game['home_team'],
                  AWAY_TEAM_ID=game['away_team_id'], AWAY_TEAM_NAME=game['away_team'],
                  HOME_PTS=game['home_score'], AWAY_PTS=game['away_score'])
    if record['HOME_PTS'] is None or record['AWAY_PTS'] is None:
        return None
    teams, players = [], []
    box_teams = {str(t.get('team', {}).get('id')): t for t in summary.get('boxscore', {}).get('teams', [])}
    for side, other in (('home', 'away'), ('away', 'home')):
        own_id, opp_id = game[f'{side}_team_id'], game[f'{other}_team_id']
        base = dict(GAME_ID=game['id'], GAME_DATE=day, SEASON_TYPE=kind, TEAM_ID=own_id,
                    TEAM_NAME=game[f'{side}_team'], TEAM_ABBREVIATION=abbreviation[own_id],
                    OPP_TEAM_ID=opp_id, OPP_TEAM_NAME=game[f'{other}_team'], HOME=side == 'home',
                    MATCHUP=matchup(abbreviation[own_id], abbreviation[opp_id], side == 'home'))
        stats = box_teams.get(own_id, {}).get('statistics', [])
        row = dict(base, PTS=record[f'{side.upper()}_PTS'], OPP_PTS=record[f'{other.upper()}_PTS'],
                   **read_stats([s.get('name') for s in stats], [s.get('displayValue') for s in stats], TEAM_KEYS))
        row['WL'] = 'W' if row['PTS'] > row['OPP_PTS'] else 'L'
        teams.append(row)
    seen = set()
    for roster in summary.get('boxscore', {}).get('players', []):
        tid = str(roster.get('team', {}).get('id', ''))
        if tid not in (game['home_team_id'], game['away_team_id']):
            continue
        team_row = next(t for t in teams if t['TEAM_ID'] == tid)
        for group in roster.get('statistics', []):
            keys = group.get('keys') or []
            for entry in group.get('athletes', []):
                athlete = entry.get('athlete', {})
                pid, full = athlete.get('id'), athlete.get('displayName') or athlete.get('fullName')
                if pid is None or not full:
                    continue
                if (game['id'], str(pid)) in seen:
                    # A duplicated athlete block cannot establish a stat line; drop both.
                    players = [p for p in players if p['PLAYER_ID'] != str(pid)]
                    continue
                seen.add((game['id'], str(pid)))
                stats = read_stats(keys, entry.get('stats') or [], PLAYER_KEYS)
                # Inactive players arrive with '--' minutes and zero filler: not an appearance.
                inactive = entry.get('active') is False and stats.get('MIN') is None
                dnp = entry.get('didNotPlay') is True or stats.get('MIN') == 0 or inactive
                if dnp:
                    stats = {}
                players.append(dict(
                    {k: team_row[k] for k in ('GAME_ID', 'GAME_DATE', 'SEASON_TYPE', 'TEAM_ID', 'TEAM_NAME',
                                              'TEAM_ABBREVIATION', 'OPP_TEAM_ID', 'HOME', 'MATCHUP')},
                    PLAYER_ID=str(pid), PLAYER_NAME=full,
                    POSITION=(athlete.get('position') or {}).get('abbreviation') or None,
                    STARTER=entry.get('starter') is True, DNP=dnp,
                    DNP_REASON=(entry.get('reason') or ('INACTIVE' if inactive else None)) if dnp else None,
                    EJECTED=entry.get('ejected') is True, **stats))
    if not players:
        return None
    return record, teams, players


def scoreboard(fetcher, day, reuse):
    url = f'{API}/scoreboard?dates={day:%Y%m%d}&limit=100'
    payload = fetcher.get(url, CACHE / 'scoreboard' / f'{day:%Y%m%d}.json', reuse)
    if payload is not None and not isinstance(payload.get('events'), list):
        return None
    return payload


def collect(fetcher, days, workers=4):
    """Events, parsed games and a list of finals that could not be read, for these ET dates."""
    today = datetime.now(ET).date()
    events, unreadable_days = {}, []
    for day in days:
        # Past scoreboards are settled; today's and yesterday's may still change.
        board = scoreboard(fetcher, day, reuse=day < today - timedelta(days=2))
        if board is None:
            unreadable_days.append(day.isoformat())
            continue
        for event in board['events']:
            game = _competition(event)
            # All-Star and other exhibitions are not missing games; never fetch them.
            if (game and game['final'] and season_type(event) and game['date'] == day and
                    {game['home_team_id'], game['away_team_id']} <= FRANCHISES):
                events[game['id']] = event

    def one(item):
        gid, event = item
        summary = fetcher.get(f'{API}/summary?event={gid}', CACHE / 'summary' / f'{gid}.json', reuse=True)
        parsed = parse_game(event, summary) if isinstance(summary, dict) else None
        if parsed is None and not fetcher.offline:
            # A cached summary saved before the box score settled: refetch once.
            summary = fetcher.get(f'{API}/summary?event={gid}', CACHE / 'summary' / f'{gid}.json', reuse=False)
            parsed = parse_game(event, summary) if isinstance(summary, dict) else None
        return gid, parsed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, sorted(events.items())))
    parsed = {gid: p for gid, p in results if p}
    missing = sorted(gid for gid, p in results if not p)
    return parsed, missing, unreadable_days


def path_for(season):
    return HISTORY / f'{season}.json.gz'


def load(season):
    path = path_for(season)
    if not path.exists():
        return None
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def load_all(seasons=None, season_types=('regular',)):
    """Merged rows for the pipeline; ``season_types=None`` keeps every game type."""
    history = dict(players=[], teams=[], games=[], seasons=[], fetched_at=None)
    for path in sorted(HISTORY.glob('*.json.gz')):
        season = path.name.split('.')[0]
        if seasons and season not in seasons:
            continue
        data = load(season)
        keep = (lambda r: True) if season_types is None else (lambda r: r.get('SEASON_TYPE') in season_types)
        for key in ('players', 'teams', 'games'):
            history[key].extend(r for r in data[key] if keep(r))
        history['seasons'].append(season)
        history['fetched_at'] = max(filter(None, [history['fetched_at'], data.get('updated_at')]), default=None)
    return history


def write(season, parsed, missing, unreadable_days, previous=None):
    games = {g['GAME_ID']: g for g in (previous or {}).get('games', [])}
    teams = {(t['GAME_ID'], t['TEAM_ID']): t for t in (previous or {}).get('teams', [])}
    players = {(p['GAME_ID'], p['PLAYER_ID']): p for p in (previous or {}).get('players', [])}
    for gid, (game, team_rows, player_rows) in parsed.items():
        # Replacing a game replaces all of its rows, so a corrected box score never mixes.
        teams = {k: v for k, v in teams.items() if k[0] != gid}
        players = {k: v for k, v in players.items() if k[0] != gid}
        games[gid] = game
        teams.update({(t['GAME_ID'], t['TEAM_ID']): t for t in team_rows})
        players.update({(p['GAME_ID'], p['PLAYER_ID']): p for p in player_rows})
    order = lambda r: (r['GAME_DATE'], r['GAME_ID'], r.get('TEAM_ID', ''), r.get('PLAYER_ID', ''))
    still_missing = sorted(set((previous or {}).get('missing_games', [])) - set(games) | set(missing))
    data = dict(schema_version=SCHEMA_VERSION, season=season, source='ESPN',
                source_api=f'{API}/scoreboard and {API}/summary',
                updated_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                missing_games=still_missing,
                unreadable_dates=sorted(set((previous or {}).get('unreadable_dates', [])) | set(unreadable_days)),
                games=sorted(games.values(), key=order), teams=sorted(teams.values(), key=order),
                players=sorted(players.values(), key=order))
    if previous and all(previous.get(k) == data[k] for k in data if k != 'updated_at'):
        return previous  # Unchanged history: no rewrite, so the daily job makes no empty commit.
    HISTORY.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    path = path_for(season)
    temporary = path.with_suffix('.tmp')
    # mtime=0 keeps the gzip bytes reproducible for unchanged data.
    with open(temporary, 'wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as handle:
        handle.write(blob)
    temporary.replace(path)
    return data


def write_manifest():
    seasons = {}
    for path in sorted(HISTORY.glob('*.json.gz')):
        data = load(path.name.split('.')[0])
        types = {}
        for game in data['games']:
            types[game['SEASON_TYPE']] = types.get(game['SEASON_TYPE'], 0) + 1
        seasons[data['season']] = dict(
            path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            updated_at=data['updated_at'], games=len(data['games']), games_by_type=types,
            team_rows=len(data['teams']), player_rows=len(data['players']),
            first_game=data['games'][0]['GAME_DATE'] if data['games'] else None,
            last_game=data['games'][-1]['GAME_DATE'] if data['games'] else None,
            missing_games=data['missing_games'], unreadable_dates=data['unreadable_dates'])
    manifest = dict(schema_version=SCHEMA_VERSION, source='ESPN site API (public, unauthenticated)',
                    source_api=API, generator='scripts/nba/espn_history.py', seasons=seasons)
    (HISTORY / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def season_days(season):
    year = season_start(season)
    first, last = date(year, 10, 1), min(date(year + 1, 6, 30), datetime.now(ET).date())
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--season', action='append', default=[], help='Season to rebuild, e.g. 2025-26 (repeatable)')
    parser.add_argument('--recent', type=int, help='Add finals from the last N Eastern dates to their season files')
    parser.add_argument('--offline', action='store_true', help='Use cached ESPN responses only')
    args = parser.parse_args(argv)
    if not args.season and not args.recent:
        parser.error('give --season or --recent')
    fetcher = Fetcher(offline=args.offline)
    jobs = {}
    for season in args.season:
        jobs[season] = (season_days(season), None)
    if args.recent:
        today = datetime.now(ET).date()
        for day in (today - timedelta(days=i) for i in range(args.recent, -1, -1)):
            season = season_of(day)
            if season and season not in args.season:
                jobs.setdefault(season, ([], load(season)))[0].append(day)
    for season, (days, previous) in jobs.items():
        parsed, missing, unreadable = collect(fetcher, days)
        if not parsed and previous is None and not path_for(season).exists():
            print(f'NBA history {season}: no finals in {len(days)} dates')
            continue
        data = write(season, parsed, missing, unreadable, previous)
        print(f"NBA history {season}: {len(parsed)} games read now; file has {len(data['games'])} games, "
              f"{len(data['players'])} player rows; {len(data['missing_games'])} unreadable finals")
    if jobs:
        write_manifest()
    print(f'ESPN requests: {fetcher.requests}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
