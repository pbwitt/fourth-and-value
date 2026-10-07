#!/usr/bin/env python3
"""Resolve saved NBA offers against ESPN's public final scores and boxscores.

``fetch_results(start, end, rows, root, fetch=..., offline=False)`` returns
``(games, players, resolved_rows)``; rows retain their original order and provider
event_id. The nba_game_id is the ESPN event ID, not an NBA Stats GAME_ID. ESPN
athlete IDs are similarly separate from saved NBA Stats player IDs.

Only an exact home/away matchup on the same Eastern calendar date can resolve a
game. Raw responses are retained under artifacts/nba/results. Missing finals,
ambiguous identities, DNPs and missing statistics remain unresolved. An empty
archive does not request a feed or manufacture a recap.
"""
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo('America/New_York')
API = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
STATS = {
    'player_points': ('PTS',), 'player_rebounds': ('REB',),
    'player_assists': ('AST',), 'player_threes': ('FG3M',),
    'player_points_rebounds_assists': ('PTS', 'REB', 'AST'),
    'player_points_rebounds': ('PTS', 'REB'),
    'player_points_assists': ('PTS', 'AST'),
    'player_rebounds_assists': ('REB', 'AST'),
    'player_blocks': ('BLK',), 'player_steals': ('STL',),
    'player_turnovers': ('TOV',),
}


def stamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except (TypeError, ValueError):
        return None


def name(value):
    text = unicodedata.normalize('NFKD', str(value or '')).casefold()
    return re.sub(r'[^a-z0-9]', '', text)


def team(value):
    key = name(value)
    return {'laclippers': 'losangelesclippers', 'lalakers': 'losangeleslakers'}.get(key, key)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def count(value):
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
        return int(parsed) if math.isfinite(parsed) and parsed >= 0 and parsed.is_integer() else None
    except (TypeError, ValueError):
        return None


def minutes(value):
    try:
        raw = str(value)
        if ':' in raw:
            mm, ss = raw.split(':')
            if not mm.isdigit() or not ss.isdigit() or not 0 <= int(ss) < 60:
                return None
            parsed = int(mm) + int(ss) / 60
        else:
            parsed = float(raw)
        return parsed if math.isfinite(parsed) and parsed >= 0 else None
    except (TypeError, ValueError):
        return None


def fetch_json(url):
    import requests
    response = requests.get(url, timeout=35)
    response.raise_for_status()
    return response.json()


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def _read(path):
    return json.loads(path.read_text()) if path.exists() else None


def _competition(event):
    competitions = event.get('competitions', [])
    if len(competitions) != 1:
        return None
    competition = competitions[0]
    competitors = competition.get('competitors', [])
    homes = [c for c in competitors if c.get('homeAway') == 'home']
    aways = [c for c in competitors if c.get('homeAway') == 'away']
    start = stamp(competition.get('date') or event.get('date'))
    if len(competitors) != 2 or len(homes) != 1 or len(aways) != 1 or not start:
        return None
    home, away = homes[0], aways[0]
    hn = home.get('team', {}).get('displayName')
    an = away.get('team', {}).get('displayName')
    if (not hn or not an or not event.get('id') or
            str(competition.get('id', event['id'])) != str(event['id'])):
        return None
    status = competition.get('status', event.get('status', {})).get('type', {})
    return dict(id=str(event['id']), home_team=hn, away_team=an,
                home_team_id=str(home.get('team', {}).get('id', '')),
                away_team_id=str(away.get('team', {}).get('id', '')),
                commence_time=start.isoformat(), date=start.astimezone(ET).date(),
                home_score=count(home.get('score')), away_score=count(away.get('score')),
                final=status.get('completed') is True and status.get('state') == 'post')


def _valid_summary(summary, game):
    final = _competition(summary.get('header', {}))
    return bool(final and final['final'] and final['id'] == game['id'] and
                all(final[key] == game[key] for key in ('home_score', 'away_score')) and
                all(team(final[key]) == team(game[key]) for key in ('home_team', 'away_team')))


def _players(summary, game):
    """Read named ESPN columns; never substitute zero for absent statistics."""
    output = {}
    aliases = {
        'min': 'MIN', 'minutes': 'MIN', 'pts': 'PTS', 'points': 'PTS',
        'reb': 'REB', 'rebounds': 'REB', 'ast': 'AST', 'assists': 'AST',
        'stl': 'STL', 'steals': 'STL', 'blk': 'BLK', 'blocks': 'BLK',
        'to': 'TOV', 'tov': 'TOV', 'turnovers': 'TOV', '3pt': 'FG3M',
        'threepointfieldgoalsmadethreepointfieldgoalsattempted': 'FG3M',
    }
    for roster in summary.get('boxscore', {}).get('players', []):
        roster_team = roster.get('team', {})
        if str(roster_team.get('id', '')) not in (game['home_team_id'], game['away_team_id']):
            continue
        for group in roster.get('statistics', []):
            labels = group.get('labels') or group.get('names') or []
            for entry in group.get('athletes', []):
                athlete = entry.get('athlete', {})
                pid = athlete.get('id')
                full_name = athlete.get('displayName') or athlete.get('fullName')
                if pid is None or not full_name:
                    continue
                values = entry.get('stats') or []
                stats = {}
                if len(labels) == len(values):
                    for label, value in zip(labels, values):
                        key = aliases.get(name(label))
                        if key:
                            stats[key] = minutes(value) if key == 'MIN' else count(
                                str(value).split('-')[0] if key == 'FG3M' else value)
                player = dict(player=full_name, espn_player_id=str(pid),
                              names=sorted({v for v in (full_name, athlete.get('fullName')) if v}),
                              team=roster_team.get('displayName'), **stats)
                player['participated'] = (entry.get('didNotPlay') is not True and
                                          number(stats.get('MIN')) and stats['MIN'] > 0)
                key = (game['id'], str(pid))
                if key in output:
                    # A duplicated/conflicting athlete block cannot establish participation.
                    player['participated'] = False
                    player['identity_ambiguous'] = True
                output[key] = player
    return output


def fetch_results(start, end, rows, root=ROOT, fetch=fetch_json, offline=False):
    """Resolve this review's offers and retrieve final results, using no paid APIs.

    Offline mode reads only saved raw responses and never calls fetch. A schedule
    fetch failure propagates to the review's failure status; a boxscore failure
    leaves team scores usable and player offers unresolved (or uses a verified
    cached final summary). Caller must recheck archived timestamps against each
    returned official_commence_time before grading, since tipoff may have moved.
    """
    import requests
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    if start > end:
        raise ValueError('Review start must not follow its end')
    cache = Path(root) / 'artifacts/nba/results'
    resolved = [dict(row) for row in rows]
    days = {at.astimezone(ET).date() for row in resolved
            if (at := stamp(row.get('commence_time'))) and start <= at.astimezone(ET).date() <= end}
    events = {}
    for day in sorted(days):
        path = cache / f'schedule-{day}.json'
        url = f'{API}/scoreboard?dates={day:%Y%m%d}&limit=100'
        if offline:
            record = _read(path)
        else:
            payload = fetch(url)
            if not isinstance(payload, dict) or not isinstance(payload.get('events'), list):
                raise ValueError('Invalid NBA scoreboard response')
            record = dict(source='ESPN', source_url=url, fetched_at=datetime.now(timezone.utc).isoformat(),
                          scoreboard=payload)
            _write(path, record)
        for event in (record or {}).get('scoreboard', {}).get('events', []):
            game = _competition(event)
            if game and start <= game['date'] <= end:
                events[game['id']] = (game, event)
    matched = set()
    for row in resolved:
        given = row.pop('nba_game_id', None)
        row.pop('official_commence_time', None)
        at = stamp(row.get('commence_time'))
        candidates = [game for game, _ in events.values() if at and
                      game['date'] == at.astimezone(ET).date() and
                      team(game['home_team']) == team(row.get('home_team')) and
                      team(game['away_team']) == team(row.get('away_team'))]
        row['resolution_status'] = ('official_matchup_ambiguous' if len(candidates) > 1
                                    else 'official_matchup_unavailable')
        if len(candidates) != 1:
            continue
        game = candidates[0]
        if ((given is not None and str(given) != game['id']) or
                (row.get('espn_event_id') is not None and str(row['espn_event_id']) != game['id'])):
            row['resolution_status'] = 'official_id_mismatch'
            continue
        row.update(nba_game_id=game['id'], espn_event_id=game['id'],
                   official_commence_time=game['commence_time'], result_source='ESPN',
                   official_schedule_path=f'artifacts/nba/results/schedule-{game["date"]}.json',
                   resolution_status='resolved')
        matched.add(game['id'])
    games, players = {}, {}
    for key in sorted(matched):
        game, event = events[key]
        if not game['final'] or game['home_score'] is None or game['away_score'] is None:
            continue
        path = cache / f'{key}.json'
        record = _read(path)
        url = f'{API}/summary?event={key}'
        if not offline:
            try:
                payload = fetch(url)
                if not isinstance(payload, dict) or not _valid_summary(payload, game):
                    raise ValueError('NBA summary does not confirm the final event')
                record = dict(source='ESPN', source_url=url, fetched_at=datetime.now(timezone.utc).isoformat(),
                              event=event, summary=payload)
                _write(path, record)
            except (requests.RequestException, ValueError):
                # Retain an existing verified final; never replace it with a feed error.
                pass
        summary = (record or {}).get('summary', {})
        valid_box = _valid_summary(summary, game)
        games[key] = {k: v for k, v in game.items() if k != 'date'}
        games[key].update(result_source='ESPN', source_url=url,
                          boxscore_status='available' if valid_box else 'unavailable')
        if valid_box:
            players.update(_players(summary, game))
    for row in resolved:
        key = row.get('nba_game_id')
        if key in games:
            row['result_source_url'] = games[key]['source_url']
            if games[key]['boxscore_status'] == 'available':
                row['official_result_path'] = f'artifacts/nba/results/{key}.json'
        candidates = [p for (g, _), p in players.items() if g == key and row.get('player') and
                      name(row['player']) in {name(n) for n in p['names']}]
        if len(candidates) == 1 and row.get('espn_player_id') is None:
            row['espn_player_id'] = candidates[0]['espn_player_id']
    return games, players, resolved


def get_actual(row, games, players):
    """Return (numeric actual, status), with no outcome or staking assumptions."""
    if row.get('resolution_status') not in (None, 'resolved'):
        return None, row['resolution_status']
    gid = str(row.get('nba_game_id', ''))
    game = games.get(gid)
    if not game or not all(number(game.get(k)) for k in ('home_score', 'away_score')):
        return None, 'official_result_unavailable'
    if any(team(row.get(k)) != team(game.get(k)) for k in ('home_team', 'away_team')):
        return None, 'official_matchup_mismatch'
    market = row.get('market')
    if market == 'totals':
        return game['home_score'] + game['away_score'], 'graded'
    if market in ('spreads', 'h2h'):
        side = team(row.get('side'))
        margin = game['home_score'] - game['away_score']
        if side == team(game['home_team']):
            return margin, 'graded'
        if side == team(game['away_team']):
            return -margin, 'graded'
        return None, 'unknown_team_side'
    if market not in STATS:
        return None, 'stat_or_market_unavailable'
    if game.get('boxscore_status') == 'unavailable':
        return None, 'official_player_results_unavailable'
    candidates = [p for (g, _), p in players.items() if g == gid and
                  name(row.get('player')) in {name(n) for n in p.get('names', [p.get('player')])}]
    if row.get('espn_player_id') is not None:
        candidates = [p for p in candidates if p.get('espn_player_id') == str(row['espn_player_id'])]
    if len(candidates) > 1:
        return None, 'player_identity_ambiguous'
    if len(candidates) != 1 or not candidates[0].get('participated'):
        return None, 'participation_unconfirmed'
    player = candidates[0]
    if not all(number(player.get(stat)) for stat in STATS[market]):
        return None, 'stat_or_market_unavailable'
    return sum(player[stat] for stat in STATS[market]), 'graded'
