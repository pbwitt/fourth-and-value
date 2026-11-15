"""Score Pinnacle's morning no-vig chance against the US-book median on graded lines.

Research only (docs/model-improvement-plan.md, Step 2d). Each line pairs Pinnacle's no-vig
chance from the morning snapshot (artifacts/pinnacle/snapshots/) with the US median from the
same morning's published feed, read from git history, and with the last pregame feed of the day:

- NFL props: docs/props/top-picks.json (the model's side as listed on the Top Picks board),
  graded with nflverse weekly player stats (--stats, scripts/fetch_weekly_player_stats.py).
- NHL and NBA: docs/<sport>/data/latest.json, graded by the weekly recap boards
  (docs/blog/<sport>-recap-*.json); one side per line (Over, or the home team).

It reports log loss and Brier score for each source, the game-clustered bootstrap of the
Pinnacle-minus-US log-loss difference, how far the US median moved toward Pinnacle's morning
price by game time (with the model's own gap as a control), and, for NFL, the realized return of
betting the best US price where Pinnacle's chance said it was positive. It needs commit history
back to --since: in a shallow clone run
`git fetch --filter=blob:none --shallow-since=<date> origin main` first.

Usage: python scripts/pinnacle_refit.py --stats data/weekly_player_stats_2026.parquet \
           --out reports/pinnacle-refit/2026-11-15
"""
import argparse
import collections
import glob
import json
import re
import statistics
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAME = ('h2h', 'spreads', 'totals')
NFL_MARKETS = {'player_receptions': 'receptions', 'player_reception_yds': 'recv_yds', 'player_rush_yds': 'rush_yds',
               'player_rush_attempts': 'rush_attempts', 'player_pass_yds': 'pass_yds', 'player_pass_tds': 'pass_tds',
               'player_pass_attempts': 'pass_attempts', 'player_pass_completions': 'pass_completions'}
STAT_COLUMN = {'receptions': 'receptions', 'recv_yds': 'receiving_yards', 'rush_yds': 'rushing_yards',
               'rush_attempts': 'carries', 'pass_yds': 'passing_yards', 'pass_tds': 'passing_tds',
               'pass_attempts': 'attempts', 'pass_completions': 'completions'}
TEAMS = {'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL', 'Buffalo Bills': 'BUF',
         'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI', 'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE',
         'Dallas Cowboys': 'DAL', 'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
         'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX', 'Kansas City Chiefs': 'KC',
         'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC', 'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA',
         'Minnesota Vikings': 'MIN', 'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
         'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT', 'San Francisco 49ers': 'SF',
         'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB', 'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'}
SAME_MORNING = (timedelta(hours=-3), timedelta(minutes=45))   # feed commit relative to the snapshot
RNG = np.random.default_rng(7)
DRAWS = 2000


def ts(s):
    return datetime.fromisoformat(str(s).replace('Z', '+00:00'))


def decimal(a):
    return 1 + (a / 100 if a > 0 else 100 / -a)


def norm(name):
    n = str(name or '').lower().replace('.', '').replace("'", '').replace('-', ' ')
    return ' '.join(re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', n).split())


# ---------------------------------------------------------------- Pinnacle and feed history

def pinnacle(sport, key_of):
    """No-vig chance per side from every snapshot: key -> [(snapshot time, chance, commence)]."""
    out = collections.defaultdict(list)
    for path in sorted(glob.glob(str(ROOT / 'artifacts/pinnacle/snapshots/*.json'))):
        d = json.loads(Path(path).read_text())
        snap, pairs = ts(d['generated_at']), collections.defaultdict(list)
        for s, ev, ct, mkt, player, side, line, price, _ in d.get('quotes') or []:
            if s == sport:
                pairs[(ev, mkt, player.casefold(), abs(line) if mkt == 'spreads' and line is not None else line)].append(
                    (side, line, price, ct))
        for (ev, mkt, player, _), quotes in pairs.items():
            if len(quotes) != 2 or any(not isinstance(p, (int, float)) or p == 0 for _, _, p, _ in quotes):
                continue
            implied = [1 / decimal(p) for _, _, p, _ in quotes]
            for (side, line, _, ct), i in zip(quotes, implied):
                k = key_of(ev, mkt, player, side, line)
                if k:
                    out[k].append((snap, i / sum(implied), ts(ct)))
    return out


class History:
    """Published versions of one feed file, reduced by `reduce` and cached per commit."""

    def __init__(self, path, since, reduce):
        log = subprocess.run(['git', '-C', str(ROOT), 'log', '--format=%H %cI', f'--since={since}', '--', path],
                             capture_output=True, text=True, check=True).stdout.split('\n')
        self.path, self.reduce, self.cache = path, reduce, {}
        self.versions = sorted((ts(l.split()[1]), l.split()[0]) for l in log if l.strip())

    def at(self, commit):
        if commit not in self.cache:
            raw = subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{self.path}'],
                                 capture_output=True, text=True, check=True).stdout
            self.cache[commit] = self.reduce(json.loads(raw))
        return self.cache[commit]

    def morning(self, snap, before):
        """The version closest to the snapshot within the same-morning window, before the game."""
        near = [(abs((t - snap).total_seconds()), t, c) for t, c in self.versions
                if SAME_MORNING[0] <= t - snap <= SAME_MORNING[1] and t < before]
        return min(near)[1:] if near else None

    def last_before(self, after, before):
        later = [(t, c) for t, c in self.versions if after + timedelta(minutes=30) <= t < before]
        return later[-1] if later else None


def pair(snaps, history, k):
    """Latest snapshot before the game with a same-morning feed version that lists the line."""
    commence = snaps[0][2]
    for snap, chance, _ in sorted((s for s in snaps if s[0] < commence), reverse=True):
        found = history.morning(snap, commence)
        if found and history.at(found[1]).get(k):
            late = history.last_before(found[0], commence)
            late_row = history.at(late[1]).get(k) if late else None
            return snap, chance, found[0], history.at(found[1])[k], late_row, commence
    return None


# ---------------------------------------------------------------- NFL props

def nfl_board(d):
    agg = collections.defaultdict(list)
    for r in d.get('rows', []):
        if r.get('model_prob') is not None:
            agg[(r['game_id'], r['market_std'], norm(r['player']), r['name'].lower(), r['point'])].append(r)
    out = {}
    for k, rows in agg.items():
        us = [r['consensus_prob'] for r in rows if r.get('consensus_prob') is not None]
        if us:
            out[k] = dict(us=statistics.median(us), model=rows[0]['model_prob'], player=rows[0]['player'],
                          home=rows[0]['home_team'], away=rows[0]['away_team'],
                          best=max((r['price'] for r in rows), key=decimal))
    return out


def nfl_lines(stats_path, since):
    snaps = pinnacle('NFL', lambda ev, mkt, pl, side, line: (ev, NFL_MARKETS[mkt], norm(pl), side.lower(), line)
                     if mkt in NFL_MARKETS else None)
    history = History('docs/props/top-picks.json', since, nfl_board)
    stats = pd.read_parquet(stats_path) if str(stats_path).endswith('.parquet') else pd.read_csv(stats_path, low_memory=False)
    stats = stats[stats.season_type == 'REG'].copy()
    stats['pnorm'] = stats.player_display_name.fillna('').map(norm)
    index = {(r.week, r.team, r.pnorm): r for r in stats.itertuples()}
    rows = []
    for k, values in snaps.items():
        found = pair(values, history, k)
        if not found:
            continue
        snap, chance, board_at, row, late, commence = found
        week = ((commence - timedelta(hours=5)).date() - date(2026, 9, 8)).days // 7 + 1
        hit = next((index[(week, TEAMS.get(t), k[2])] for t in (row['home'], row['away'])
                    if (week, TEAMS.get(t), k[2]) in index), None)
        if hit is None:
            continue
        actual = float(getattr(hit, STAT_COLUMN[k[1]]))
        if actual == k[4]:
            continue   # push
        rows.append(dict(sport='NFL', event_id=k[0], market=k[1], player=row['player'], side=k[3], line=k[4],
                         commence=commence.isoformat(), snapshot=snap.isoformat(), board_at=board_at.isoformat(),
                         p_pinn=chance, p_us=row['us'], p_model=row['model'], best_price=row['best'],
                         p_us_late=late['us'] if late else None, actual=actual,
                         y=float((actual > k[4]) == (k[3] == 'over'))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- NHL and NBA

def sport_feed(d):
    agg = collections.defaultdict(list)
    for r in d.get('rows', []):
        agg[(r['event_id'], r['market'], (r.get('player') or '').casefold().strip(), str(r['side']).casefold().strip(),
             None if r.get('line') is None else float(r['line']))].append(r)
    out = {}
    for k, rows in agg.items():
        us = [r['consensus_probability'] for r in rows if r.get('consensus_probability') is not None]
        if us:
            out[k] = dict(us=statistics.median(us), model=next((r['model_probability'] for r in rows
                                                                 if r.get('model_probability') is not None), None))
    return out


def board_lines(sport, since):
    outcome, homes = {}, {}
    for path in sorted(glob.glob(str(ROOT / f'docs/blog/{sport.lower()}-recap-*.json'))):
        for r in json.loads(Path(path).read_text()).get('board', []):
            if r.get('outcome') in ('win', 'loss', 'push'):
                k = (r['event_id'], r['market'], (r.get('player') or '').casefold().strip(), str(r['side']).casefold().strip(),
                     None if r.get('line') is None else float(r['line']))
                outcome[k], homes[r['event_id']] = r['outcome'], (r.get('home_team') or '').casefold()

    def key_of(ev, mkt, player, side, line):
        side = side.casefold().strip()
        if ev not in homes or side != (homes[ev] if mkt in ('h2h', 'spreads') else 'over'):
            return None   # one side per line
        return (ev, mkt, player.strip(), side, None if line is None else float(line))

    snaps = pinnacle(sport, key_of)
    history = History(f'docs/{sport.lower()}/data/latest.json', since, sport_feed)
    rows = []
    for k, values in snaps.items():
        if outcome.get(k) not in ('win', 'loss'):
            continue
        found = pair(values, history, k)
        if not found:
            continue
        snap, chance, board_at, row, late, commence = found
        rows.append(dict(sport=sport, event_id=k[0], market=k[1], player=k[2], side=k[3], line=k[4],
                         commence=commence.isoformat(), snapshot=snap.isoformat(), board_at=board_at.isoformat(),
                         p_pinn=chance, p_us=row['us'], p_model=row['model'], best_price=None,
                         p_us_late=late['us'] if late else None, actual=None, y=float(outcome[k] == 'win')))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scoring

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mix(a, b, w):
    return 1 / (1 + np.exp(-(w * logit(a) + (1 - w) * logit(b))))


def resamples(games):
    """Game-clustered bootstrap indices."""
    codes, uniques = pd.factorize(games)
    members = [np.flatnonzero(codes == i) for i in range(len(uniques))]
    for _ in range(DRAWS):
        yield np.concatenate([members[i] for i in RNG.integers(0, len(uniques), len(uniques))])


def interval(values):
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def score(df, name):
    df = df[df.p_us.notna()].reset_index(drop=True)
    y, pp, pu = df.y.values, df.p_pinn.values, df.p_us.values
    diff = log_loss(pp, y) - log_loss(pu, y)
    boot = [diff[i].mean() for i in resamples(df.event_id.values)]
    out = dict(set=name, lines=len(df), games=int(df.event_id.nunique()), mean_gap_pts=round(100 * float(np.abs(pp - pu).mean()), 2),
               log_loss_pinnacle=round(float(log_loss(pp, y).mean()), 4), log_loss_us=round(float(log_loss(pu, y).mean()), 4),
               brier_pinnacle=round(float(((pp - y) ** 2).mean()), 4), brier_us=round(float(((pu - y) ** 2).mean()), 4),
               difference=round(float(diff.mean()), 5), difference_95=interval(boot),
               pinnacle_better_share=round(float(np.mean(np.array(boot) < 0)), 3),
               gain_if_pinnacle_exact=round(float(np.mean(pp * np.log(pp / pu) + (1 - pp) * np.log((1 - pp) / (1 - pu)))), 5))
    late = df[df.p_us_late.notna()].reset_index(drop=True)
    if len(late) > 30:
        gap, move = (late.p_pinn - late.p_us).values, (late.p_us_late - late.p_us).values
        out['movement_lines'] = len(late)
        out['movement_share'] = round(float((gap * move).sum() / (gap * gap).sum()), 3)
        out['movement_share_95'] = interval([(gap[i] * move[i]).sum() / (gap[i] * gap[i]).sum()
                                             for i in resamples(late.event_id.values)])
        both = late[late.p_model.notna()].reset_index(drop=True)
        if len(both) > 30:
            X = np.c_[(both.p_pinn - both.p_us).values, (both.p_model - both.p_us).values]
            d = (both.p_us_late - both.p_us).values
            fits = np.array([np.linalg.lstsq(X[i], d[i], rcond=None)[0] for i in resamples(both.event_id.values)])
            beta = np.linalg.lstsq(X, d, rcond=None)[0]
            out['movement_share_with_model_control'] = round(float(beta[0]), 3)
            out['movement_share_with_model_control_95'] = interval(fits[:, 0])
            out['model_gap_movement'] = round(float(beta[1]), 3)
            out['model_gap_movement_95'] = interval(fits[:, 1])
    return out


def betting(df):
    """NFL: realized return at the best US price, by what Pinnacle's chance said."""
    df = df[df.p_us.notna() & df.best_price.notna()].copy()
    df['dec'] = df.best_price.map(decimal)
    df['ev_pinnacle'] = df.p_pinn * df.dec - 1
    df['ev_blend'] = mix(df.p_model.values, df.p_us.values, .10) * df.dec - 1
    df['ret'] = df.y * df.dec - 1
    out = []
    for name, m in [('All board lines', df.index == df.index), ('Pinnacle EV > 0', df.ev_pinnacle > 0),
                    ('Pinnacle EV > 2%', df.ev_pinnacle > .02), ('Blend EV >= 1% (US median)', df.ev_blend >= .01),
                    ('Blend EV >= 1% and Pinnacle EV > 0', (df.ev_blend >= .01) & (df.ev_pinnacle > 0))]:
        s = df[m].reset_index(drop=True)
        if len(s) >= 10:
            out.append(dict(set=name, lines=len(s), expected_by_pinnacle=round(float(s.ev_pinnacle.mean()), 4),
                            realized=round(float(s.ret.mean()), 4),
                            realized_95=interval([s.ret.values[i].mean() for i in resamples(s.event_id.values)])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stats', required=True, help='nflverse weekly player stats (parquet or csv.gz)')
    parser.add_argument('--since', default='2026-10-14', help='Earliest feed commit to read')
    parser.add_argument('--out', required=True, help='Output folder for lines.csv and results.json')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nfl = nfl_lines(args.stats, args.since)
    nhl, nba = board_lines('NHL', args.since), board_lines('NBA', args.since)
    pd.concat([nfl, nhl, nba]).to_csv(out / 'lines.csv', index=False)
    sets = [('NFL props (model side)', nfl)] + [(f'  NFL {m}', g) for m, g in nfl.groupby('market') if len(g) >= 80]
    for sport, d in (('NHL', nhl), ('NBA', nba)):
        props = d[~d.market.isin(GAME)]
        sets += [(f'{sport} props', props)] + [(f'  {sport} {m}', g) for m, g in props.groupby('market') if len(g) >= 80]
        sets.append((f'{sport} game lines', d[d.market.isin(GAME)]))
    results = dict(generated_at=datetime.now().astimezone().isoformat(), scores=[score(g, n) for n, g in sets],
                   nfl_betting=betting(nfl))
    (out / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    print('| Set | Lines | Games | Log loss Pinnacle / US | Pinnacle better in | US moved toward Pinnacle (95%) |')
    print('|---|---|---|---|---|---|')
    for r in results['scores']:
        move = (f"{r.get('movement_share_with_model_control', r.get('movement_share'))} "
                f"({'–'.join(map(str, r.get('movement_share_with_model_control_95', r.get('movement_share_95', ['', '']))))})"
                if 'movement_share' in r else '—')
        print(f"| {r['set']} | {r['lines']} | {r['games']} | {r['log_loss_pinnacle']} / {r['log_loss_us']} | "
              f"{r['pinnacle_better_share']:.0%} | {move} |")


if __name__ == '__main__':
    main()
