"""Daily NBA v1 forecasts for saved quotes: no training, only committed history and parameters.

Shadow mode: rows gain model_* fields and the feed gains model_version/model_predicted_at,
but never model_checked_at, so editorial and Top Picks keep NBA ineligible until graded
2026-27 evidence supports it. Each unsupported row states why.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from nba import model as M

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / 'artifacts/nba/model/nba-v1.json'
API = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
STATUS = 'Experimental NBA v1 forecast (shadow mode); not yet graded against prices'
SIMULATIONS = 20000


def team_key(name):
    key = re.sub(r'[^a-z0-9]', '', str(name or '').lower())
    return {'laclippers': 'losangelesclippers'}.get(key, key)


def name_key(name):
    import unicodedata
    text = unicodedata.normalize('NFKD', str(name or '')).lower()
    text = re.sub(r'\b(jr|sr|ii|iii|iv)\b\.?', '', text)
    return re.sub(r'[^a-z0-9]', '', text)


def et_day(commence_time):
    from zoneinfo import ZoneInfo
    at = datetime.fromisoformat(str(commence_time).replace('Z', '+00:00'))
    return at.astimezone(ZoneInfo('America/New_York')).date()


def fetch_json(url):
    import requests
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    return response.json()


def rosters(team_ids, fetch=fetch_json):
    """ESPN current rosters: name key -> set of (athlete id, team id). Failures stay empty."""
    found, failed = defaultdict(set), []
    for tid in sorted(team_ids):
        try:
            for athlete in fetch(f'{API}/teams/{tid}/roster').get('athletes', []):
                if athlete.get('id') and athlete.get('displayName'):
                    for label in {athlete['displayName'], athlete.get('fullName')} - {None}:
                        found[name_key(label)].add((str(athlete['id']), str(tid)))
        except Exception:  # network or format: fall back to the last team in history
            failed.append(str(tid))
    return found, failed


def injuries(fetch=fetch_json):
    """ESPN injury report: athlete id -> status and dated note. None when unavailable."""
    try:
        payload = fetch(f'{API}/injuries')
    except Exception:
        return None
    out = {}
    for team in payload.get('injuries', []):
        for item in team.get('injuries', []):
            athlete = item.get('athlete', {})
            link = next((l.get('href', '') for l in athlete.get('links', []) if '/id/' in l.get('href', '')), '')
            found = re.search(r'/id/(\d+)', link)
            if found and item.get('status'):
                out[found.group(1)] = dict(status=item['status'], date=item.get('date'),
                                           note=item.get('shortComment') or None)
    return out


def player_states(history, half_life):
    players = {}
    for row in sorted((r for r in history['players'] if not r.get('DNP') and r.get('MIN')),
                      key=lambda r: (r['GAME_DATE'], r['GAME_ID'])):
        players.setdefault(row['PLAYER_ID'], M.Player((half_life,))).add(row, M.day_of(row['GAME_DATE']))
    return players


def ev_pct(win, push, price):
    decimal = 1 + (price / 100 if price > 0 else 100 / -price)
    return 100 * (win * (decimal - 1) - (1 - win - push))


def unsupported(row, reason):
    row.update(model_probability=None, model_push_probability=None, model_ev_pct=None, model_mean=None,
               model_unsupported_reason=reason)


def add_model(rows, history, now, params=None, fetch=fetch_json):
    """Attach v1 forecasts to rows in place; returns feed-level model metadata."""
    params = params or json.loads(MODEL_PATH.read_text())
    history = M.state_history(history)
    _, _, teams = M.game_table(history, params['team_config'])
    names = {}
    for g in history['games']:
        names[team_key(g['HOME_TEAM_NAME'])] = g['HOME_TEAM_ID']
        names[team_key(g['AWAY_TEAM_NAME'])] = g['AWAY_TEAM_ID']
    pp = params['players']
    players = player_states(history, pp['config']['rate_half_life'])
    through = max((g['GAME_DATE'] for g in history['games']), default=None)
    event_ids = {}
    for row in rows:
        event_ids.setdefault(row['event_id'], row)
    prop_teams = {names.get(team_key(r[k])) for r in rows if r['market'] in M.MARKET_STATS for k in ('home_team', 'away_team')} - {None}
    roster, roster_failed = rosters(prop_teams, fetch) if prop_teams else ({}, [])
    injury = injuries(fetch) if prop_teams else None
    by_name = defaultdict(set)
    for pid, p in players.items():
        by_name[name_key(p_name(history, pid))].add(pid)
    games, draws = {}, {}
    for event_id, sample in event_ids.items():
        home, away = names.get(team_key(sample['home_team'])), names.get(team_key(sample['away_team']))
        f = teams.features(home, away, et_day(sample['commence_time'])) if home and away else None
        games[event_id] = (home, away, f, M.score_matrix(params['game'], f) if f else None)
    for row in rows:
        row['baseline_status'] = row.get('baseline_status') or row.get('model_status')
        row.update(model_version=params['version'], model_status=STATUS, model_input_through=through,
                   model_unsupported_reason=None)
        home, away, f, scored = games[row['event_id']]
        if not scored:
            unsupported(row, 'Team not found in the NBA history')
            continue
        matrix, mean_home, mean_away = scored
        if row['market'] in ('h2h', 'spreads', 'totals'):
            side_home = row['side'] == 'Over' if row['market'] == 'totals' else row['side'] == row['home_team']
            win, push = M.game_market(matrix, row['market'], side_home, row['line'] or 0)
            mean = mean_home + mean_away if row['market'] == 'totals' else (mean_home - mean_away) * (1 if side_home else -1)
            row.update(model_probability=win, model_push_probability=push, model_mean=round(mean, 2),
                       model_ev_pct=round(ev_pct(win, push, row['price']), 2))
            continue
        if row['market'] not in M.MARKET_STATS:
            unsupported(row, 'Market not modeled')
            continue
        key = name_key(row['player'])
        listed = {(pid, tid) for pid, tid in roster.get(key, set()) if tid in (home, away)}
        if len(listed) == 1:
            pid, tid = next(iter(listed))
        elif len(listed) > 1:
            unsupported(row, 'Ambiguous player name on the current rosters')
            continue
        else:
            # Roster unavailable or player not listed: use history only if unambiguous and on a team in this game.
            ids = [pid for pid in by_name.get(key, ()) if players[pid].team in (home, away)]
            if len(ids) != 1:
                unsupported(row, 'Player not on either current roster' if not roster_failed else 'Player not matched (roster unavailable)')
                continue
            pid, tid = ids[0], players[ids[0]].team
        p = players.get(pid)
        if not p or p.n < M.MIN_APPEARANCES:
            unsupported(row, f'Fewer than {M.MIN_APPEARANCES} NBA appearances in the history')
            continue
        status = (injury or {}).get(pid)
        row['injury_status'] = status
        if status and status['status'].lower() == 'out':
            unsupported(row, 'Listed out on the ESPN injury report')
            continue
        cache_key = (row['event_id'], pid)
        if cache_key not in draws:
            is_home = tid == home
            day = et_day(row['commence_time'])
            rest = f['home_rest'] if is_home else f['away_rest']
            s = dict(minutes=M.minute_features(p, tid, day, rest == 1, abs(mean_home - mean_away)),
                     rates={pp['config']['rate_half_life']: {k: v for k, v in p.rates[pp['config']['rate_half_life']].sums.items()}},
                     recent=M.recent_means(p), position=p.position or 'F', pace=f['pace'],
                     own_pace=f['home_pace'] if is_home else f['away_pace'], league_pace=f['league_pace'],
                     allow=f['allow_away'] if is_home else f['allow_home'], league_allow=f['league_allow'])
            mu, sd, _ = M.minute_distribution(pp, s['minutes'])
            rates = M.final_rates(s, pp, mu)
            draws[cache_key] = (M.simulate(pp, rates, mu, sd, n=SIMULATIONS, seed=M.seed_for(row['event_id'], pid))[0], mu)
        sim, minutes = draws[cache_key]
        win, push, mean = M.player_market(sim, row['market'], row['side'], row['line'],
                                           pp.get('calibration', {}).get(row['market'], 1.0))
        row.update(model_probability=win, model_push_probability=push, model_mean=round(mean, 2),
                   model_minutes=round(minutes, 1), espn_player_id=pid,
                   model_ev_pct=round(ev_pct(win, push, row['price']), 2))
    reasons = defaultdict(int)
    for row in rows:
        if row.get('model_unsupported_reason'):
            reasons[row['model_unsupported_reason']] += 1
    return dict(model_version=params['version'], model_status=STATUS, model_trained_at=params['trained_at'],
                model_input_through=through, model_predicted_at=now.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
                model_coverage=dict(rows=len(rows), forecast=sum(r.get('model_probability') is not None for r in rows),
                                    unsupported=dict(reasons)),
                model_roster_failures=roster_failed, model_injuries_available=injury is not None)


_NAMES = {}


def p_name(history, pid):
    if not _NAMES or id(history) != _NAMES.get('_id'):
        _NAMES.clear()
        _NAMES['_id'] = id(history)
        for r in history['players']:
            _NAMES[r['PLAYER_ID']] = r['PLAYER_NAME']
    return _NAMES.get(pid, '')
