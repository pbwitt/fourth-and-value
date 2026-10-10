"""Goalie appearances, shrunk save rates and start probabilities (context only).

Built from the official per-game goalie report (api.nhle.com stats `goalie/summary`, the same
source family as the skater history). Nothing here changes a forecast yet: each refresh publishes
the projections as context (`goalie_assumption` on rows, `goalie_projections` on the snapshot)
and the run archive keeps them with their as-of time, so a later model can be backtested only on
what was known at decision time. Confirmed starters need a permitted source (see
docs/model-improvement-plan.md); until then a start is a probability, never a confirmation.
"""
from collections import defaultdict
from datetime import date, timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from .data import digest, iso, stamp, write_json, history_day

LEAGUE_SAVE_PCT = .903     # shrinkage target; recent NHL league averages sit near .900-.905
PRIOR_SHOTS = 1000         # shots of prior weight: a goalie needs ~1,000 faced to count half
START_HALF_LIFE = 10       # team games; recent starts count most
START_WINDOW = 20          # team games considered for the start share
BACK_TO_BACK_FACTOR = .35  # heuristic: yesterday's starter rarely starts the next night
ET = ZoneInfo('America/New_York')
HORIZON = timedelta(hours=48)  # the forecast window; later games get no projection yet
REQUIRED = ('gameId', 'playerId', 'homeRoad', 'gamesStarted', 'shotsAgainst', 'goalsAgainst')


def normalize(rows, games):
    """Per-game goalie appearances joined to the team game history (stable team ids)."""
    if rows and any(k not in rows[0] for k in REQUIRED):
        raise ValueError('Unexpected NHL goalie report schema')
    by_game = {g['game_id']: g for g in games}
    out, seen = [], set()
    for r in rows:
        gid, pid = int(r['gameId']), int(r['playerId'])
        if (gid, pid) in seen:
            raise ValueError(f'Duplicate goalie-game: {gid}/{pid}')
        seen.add((gid, pid))
        g = by_game.get(gid)
        if not g:
            continue   # the team history has not caught up with this game yet
        home = r['homeRoad'] == 'H'
        out.append(dict(game_id=gid, player_id=pid,
                        player=r.get('goalieFullName') or str(pid), season=g['season'],
                        game_date=g['game_date'], available_at=g['available_at'],
                        team_id=g['home_id'] if home else g['away_id'],
                        started=int(r.get('gamesStarted') or 0) == 1,
                        shots_against=int(r.get('shotsAgainst') or 0),
                        goals_against=int(r.get('goalsAgainst') or 0)))
    starters = defaultdict(int)
    for a in out:
        starters[(a['game_id'], a['team_id'])] += a['started']
    if any(n > 1 for n in starters.values()):
        raise ValueError('More than one starting goalie for a team game')
    return sorted(out, key=lambda a: (a['game_date'], a['game_id'], a['player_id']))


def collect(season, root, games, through=None):
    """Fetch, normalize and save one season; the raw pages are kept by fetch_report."""
    from .data import fetch_report
    root = Path(root)
    rows, pages = fetch_report('goalie', season, root, through)
    appearances = normalize(rows, [g for g in games if g['season'] == season])
    if rows and not appearances:
        raise ValueError(f'Goalie report for {season} did not join the team game history')
    manifest = dict(schema_version=1, season=season, through=through, pages=pages,
                    appearances=len(appearances), data_sha256=digest(appearances))
    write_json(root / 'goalies' / f'{season}.json', dict(manifest=manifest, appearances=appearances))
    return appearances


def load(root, seasons):
    out = []
    for season in seasons:
        path = Path(root) / 'goalies' / f'{season}.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if digest(data['appearances']) != data['manifest']['data_sha256']:
            raise ValueError(f'Goalie history checksum mismatch: {path.name}')
        out.extend(data['appearances'])
    return out


def live(root, games, now, season, refresh=False):
    """Refresh with the live board; same-run callers may reuse the current-day cache."""
    root = Path(root)
    previous = season - 10001
    if not (root / 'goalies' / f'{previous}.json').exists():
        collect(previous, root, games)
    check = root / 'goalies' / 'live-check.json'
    last = json.loads(check.read_text()) if check.exists() else {}
    age = now - stamp(last['checked_at']) if last.get('checked_at') else None
    through = history_day(now)
    if (refresh or last.get('season') != season or last.get('through') != through
            or age is None or not timedelta(0) <= age < timedelta(hours=12)):
        collect(season, root, games, through)
        write_json(check, dict(checked_at=iso(now), season=season, through=through))
    return load(root, [previous, season])


def upcoming(events, now):
    return [e for e in events if e.get('nhl_game_id') and e.get('commence_time') and e.get('home_id')
            and e.get('away_id') and stamp(e['commence_time']) - now <= HORIZON]


def rosters(events, now):
    """Goalie ids on each playing team's official current roster; a team that fails is left out.

    Requests are spaced and a 429 or 5xx is retried once. A network failure (timeout, refused
    connection) stops further requests so a slow API cannot stall the refresh.
    """
    import time
    import requests
    out = {}
    for event in upcoming(events, now):
        for side in ('home', 'away'):
            tid, abbrev = event[side + '_id'], event.get(side + '_abbrev')
            if tid in out or not abbrev:
                continue
            for attempt in range(2):
                try:
                    response = requests.get(f'https://api-web.nhle.com/v1/roster/{abbrev}/current', timeout=10)
                except requests.RequestException:
                    return {t: ids for t, ids in out.items() if ids}
                if attempt == 0 and (response.status_code == 429 or response.status_code >= 500):
                    time.sleep(retry_after(response.headers.get('Retry-After')))
                    continue
                break
            try:
                response.raise_for_status()
                out[tid] = {int(g['id']) for g in response.json()['goalies']}
            except (requests.RequestException, ValueError, KeyError, TypeError):
                out[tid] = None
            time.sleep(.15)
    return {tid: ids for tid, ids in out.items() if ids}


def retry_after(value):
    try:
        return min(max(float(value), 0), 5)
    except (TypeError, ValueError):
        return 1


def save_rate(appearances):
    shots = sum(a['shots_against'] for a in appearances)
    saves = shots - sum(a['goals_against'] for a in appearances)
    return (saves + PRIOR_SHOTS * LEAGUE_SAVE_PCT) / (shots + PRIOR_SHOTS), shots


def project(appearances, team_id, game_date, asof, current=None):
    """Start probabilities and shrunk save rates for one team's next game.

    Only appearances available by `asof` and before `game_date` are used. Start shares are
    recency-weighted over the team's last 20 games; the previous night's starter is discounted
    on a back-to-back. A goalie whose latest known box score is for another team has moved and is
    dropped; `current` (the team's goalie ids from the official roster) also drops goalies who left
    before playing elsewhere. Returns None when nothing is known about the team's goalies.
    """
    known = [a for a in appearances if stamp(a['available_at']) <= asof and a['game_date'] < game_date]
    team = [a for a in known if a['team_id'] == team_id]
    games = sorted({(a['game_date'], a['game_id']) for a in team})[-START_WINDOW:]
    if not games:
        return None
    age = {g: len(games) - 1 - i for i, g in enumerate(games)}
    weights, total = defaultdict(float), 0.
    for a in team:
        g = (a['game_date'], a['game_id'])
        if g in age and a['started']:
            weights[a['player_id']] += 2 ** (-age[g] / START_HALF_LIFE)
    total = sum(2 ** (-n / START_HALF_LIFE) for n in age.values())
    latest = {}
    for a in sorted(known, key=lambda a: (a['game_date'], a['game_id'])):
        latest[a['player_id']] = a['team_id']
    share = {pid: w / total for pid, w in weights.items() if latest[pid] == team_id}
    if current is not None:
        share = {pid: s for pid, s in share.items() if pid in current}
    if not share:
        return None
    last_date, last_id = games[-1]
    back_to_back = date.fromisoformat(last_date) == date.fromisoformat(game_date) - timedelta(days=1)
    last_starter = next((a['player_id'] for a in team if a['game_id'] == last_id and a['started']), None)
    if back_to_back and last_starter in share and len(share) > 1:
        share[last_starter] *= BACK_TO_BACK_FACTOR
    norm = sum(share.values())
    by_goalie = defaultdict(list)
    for a in known:
        by_goalie[a['player_id']].append(a)
    goalies = []
    for pid, s in sorted(share.items(), key=lambda kv: (-kv[1], kv[0])):
        mine = by_goalie[pid]
        rate, shots = save_rate(mine)
        goalies.append(dict(player_id=pid, player=mine[-1]['player'], start_probability=round(s / norm, 4),
                            save_pct=round(rate, 4), shots_faced=shots,
                            last_start=max((a['game_date'] for a in mine if a['started']), default=None)))
    seasons = sorted({a['season'] for a in team if (a['game_date'], a['game_id']) in age})
    basis = (f'Recency-weighted starts over the last {len(games)} team games; save rate shrunk toward '
             f'{LEAGUE_SAVE_PCT:.3f} with {PRIOR_SHOTS:,} shots of prior weight. Not a confirmed starter.')
    if current is not None:
        basis += ' Limited to goalies on the current roster.'
    elif len(seasons) > 1:
        basis += ' Includes last season without a current roster: a goalie who moved but has not yet played for his new team may still be listed.'
    return dict(team_id=team_id, as_of=iso(asof), back_to_back=back_to_back, confirmed=False,
                goalies=goalies, expected_save_pct=round(sum(g['start_probability'] * g['save_pct'] for g in goalies), 4),
                basis=basis)


def describe(projection, team):
    if not projection:
        return f'{team}: goalie unknown.'
    top, rest = projection['goalies'][0], projection['goalies'][1:3]
    text = f"{team}: likely {top['player']} {100 * top['start_probability']:.0f}% (sv {top['save_pct']:.3f})"
    if rest:
        text += ', ' + ', '.join(f"{g['player']} {100 * g['start_probability']:.0f}%" for g in rest)
    return text + (', back-to-back.' if projection['back_to_back'] else '.')


def attach(state, appearances, now, rosters=None):
    """Context only: projections per game on the snapshot and a goalie line on each row.

    A skater's line names the goalie he is projected to face; a game line names both. Nothing
    here changes a probability, price or eligibility.
    """
    projections = {}
    for event in upcoming(state.get('events', []), now):
        gid, ids = event['nhl_game_id'], (event['home_id'], event['away_id'])
        day = stamp(event['commence_time']).astimezone(ET).date().isoformat()
        sides = {}
        for side, tid in zip(('home', 'away'), ids):
            sides[side] = project(appearances, tid, day, now, (rosters or {}).get(tid))
        projections[str(gid)] = dict(as_of=iso(now), home_team=event.get('home_team'), away_team=event.get('away_team'),
                                     home_id=ids[0], away_id=ids[1], **sides)
    for row in state.get('rows', []):
        p = projections.get(str(row.get('nhl_game_id')))
        if not p:
            continue
        team = row.get('player_team_id')
        if row.get('player') and team in (p['home_id'], p['away_id']):
            rival = 'away' if team == p['home_id'] else 'home'
            text = 'Opposing goalie, ' + describe(p[rival], p[rival + '_team'])
        else:
            text = ' '.join(describe(p[s], p[s + '_team']) for s in ('home', 'away'))
        # Start chances and shrunk save rates from official box scores; never a confirmed starter.
        row['goalie_assumption'] = 'Projected, not confirmed. ' + text
    state['goalie_projections'] = projections
    return state
