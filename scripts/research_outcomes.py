"""Outcome adapters for research grading: {entry_id: outcome} from official final results.

Outcome statuses: final (with value or scores), did_not_participate (void for props
whose settlement voids non-participation), postponed, unknown (stays pending).
A missing record is never a zero and never a loss.

  python scripts/research_outcomes.py --sport nfl --out data/research/outcomes-nfl.json
  python scripts/research_outcomes.py --sport mlb --out data/research/outcomes-mlb.json
  python scripts/research_outcomes.py --sport nhl --out data/research/outcomes-nhl.json
"""
import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from nhl.v2.data import ROOT, stamp, write_json

ET = ZoneInfo('America/New_York')
MLB_STAT = {'batter_hits': ('batters', 'hits'), 'batter_total_bases': ('batters', 'totalBases'),
            'batter_home_runs': ('batters', 'homeRuns'), 'batter_rbis': ('batters', 'rbi'),
            'pitcher_strikeouts': ('pitchers', 'strikeOuts'), 'pitcher_outs': ('pitchers', 'outs')}
NHL_STAT = {'player_shots_on_goal': 'shots', 'player_goals': 'goals', 'player_assists': 'assists', 'player_points': 'points'}


def entries(ledgers=ROOT/'artifacts/research/decisions', sport=None):
    for path in sorted(ledgers.glob('*.json')):
        for e in json.loads(path.read_text())['entries']:
            if sport is None or e['sport'] == sport:
                yield e


def mlb(rows, games):
    """Official MLB box scores (mlb.model_data history), joined by game id and player name."""
    by_id = {str(g['id']): g for g in games}
    out = {}
    for e in rows:
        g = by_id.get(str(e.get('mlb_game_id') or e.get('game_id')))
        if not g:
            out[e['entry_id']] = dict(status='unknown')
            continue
        if e['market'] in ('h2h', 'spreads', 'totals'):
            out[e['entry_id']] = dict(status='final', home_score=g['home_score'], away_score=g['away_score'])
            continue
        group, stat = MLB_STAT[e['market']]
        found = [p for side in ('home', 'away') for p in g['teams'][side][group] if p['name'] == e['player']]
        out[e['entry_id']] = dict(status='final', value=found[0][stat]) if len(found) == 1 else \
            dict(status='did_not_participate') if not found else dict(status='unknown')
    return out


def nhl(rows, games, players):
    """NHL v2 history; a player absent from the game record stays unresolved (pending)."""
    games = {g['game_id']: g for g in games}
    players = {(p['game_id'], p['player_id']): p for p in players}
    out = {}
    for e in rows:
        g = games.get(e.get('nhl_game_id'))
        if not g:
            out[e['entry_id']] = dict(status='unknown')
        elif e['market'] in ('h2h', 'spreads', 'totals'):
            out[e['entry_id']] = dict(status='final', home_score=g['home_score'], away_score=g['away_score'])
        else:
            p = players.get((e.get('nhl_game_id'), e.get('player_id')))
            out[e['entry_id']] = dict(status='final', value=p[NHL_STAT[e['market']]]) if p else dict(status='unknown')
    return out


def nfl(rows, data=ROOT/'data/nfl_validation'):
    """nflverse box scores with snap-count participation (see nfl_validation.outcomes)."""
    import pandas as pd
    import nfl_validation as v
    from make_player_prop_params import TEAM_TO_ABBREV
    schedule = pd.read_parquet(data/'schedule.parquet')
    out, cache = {}, {}
    for e in rows:
        day = stamp(e['commence_time']).astimezone(ET).date().isoformat()
        home, away = TEAM_TO_ABBREV.get(e.get('home_team'), e.get('home_team')), TEAM_TO_ABBREV.get(e.get('away_team'), e.get('away_team'))
        match = schedule[(schedule.gameday == day) & (schedule.home_team.isin([home, 'LA' if home == 'LAR' else home])) &
                         (schedule.away_team.isin([away, 'LA' if away == 'LAR' else away]))]
        if match.empty or pd.isna(match.iloc[0].home_score):
            out[e['entry_id']] = dict(status='unknown')
            continue
        g = match.iloc[0]
        if e['market'] in ('h2h', 'spreads', 'totals'):
            out[e['entry_id']] = dict(status='final', home_score=float(g.home_score), away_score=float(g.away_score))
            continue
        key = (int(g.season), int(g.week))
        if key not in cache:
            cache[key] = v.outcomes(*key, data=data)
        played, values = cache[key]
        team = next((t for t in (g.home_team, g.away_team) if (v.name_key(e['player']), t) in played), None)
        if team is None:
            out[e['entry_id']] = dict(status='did_not_participate')
            continue
        value = v.outcome_value(played, values, e['player'], team, e['market'])
        out[e['entry_id']] = dict(status='final', value=value) if value is not None else dict(status='unknown')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sport', choices=['nfl', 'mlb', 'nhl'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = list(entries(sport=args.sport.upper()))
    if args.sport == 'mlb':
        from mlb.model_data import load
        result = mlb(rows, load()[0])
    elif args.sport == 'nhl':
        import gzip
        from nhl.v2.data import load
        with gzip.open(ROOT/'models/nhl/v2/history.json.gz', 'rt') as f:
            past = json.load(f)
        games, players, _ = load(ROOT/'data/nhl/v2/history')
        result = nhl(rows, past['games']+games, past['players']+players)
    else:
        result = nfl(rows)
    write_json(args.out, result)
    print(json.dumps({k: sum(o['status'] == k for o in result.values()) for k in ('final', 'did_not_participate', 'unknown', 'postponed')}))


if __name__ == '__main__':
    main()
