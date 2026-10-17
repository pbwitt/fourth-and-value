#!/usr/bin/env python3
"""Build Market Results data: what the market expected versus what happened.

For every graded player prop and game line we keep one row: the main line
(offered by the most books; ties go to the line nearest the median), the
average of each book's no-vig probability at that line, the average offered
price per side, and the official result. The page aggregates these rows into
season and recent-window views, so it never needs a model or a pick.

NFL sources are the frozen weekly pregame archives graded with nflverse
statistics. Weeks 1-2 predate the immutable archive and are read from their
one-off audit folders. NHL rows are rebuilt from the committed refresh archive
(artifacts/nhl/runs): each game's quotes come from the last snapshot saved before
puck drop, graded with the official NHL statistics pages archived alongside them.
NBA rows come the same way from artifacts/nba/runs, graded with ESPN's final box
scores through the weekly recap's resolver (scripts/nba_weekly_results.py), whose
raw responses are kept in artifacts/nba/results.
"""
import argparse
from collections import defaultdict
import gzip
import json
from html import escape
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from nfl_weekly_review import STATS, TEAM_NAMES, grade_props  # noqa: E402

PAGES = ROOT / 'docs/markets'
OUT = PAGES / 'data'
NHL_ARCHIVE = ROOT / 'artifacts/nhl'
NBA_ARCHIVE = ROOT / 'artifacts/nba'

# Display order is the order readers scan the board: game lines, then by unit.
NFL_MARKETS = [
    dict(key='game_total', label='Game total', group='Game lines', kind='ou', unit='points', bin=4,
         buckets=[[None, 41.5, 'Under 42'], [41.5, 45.5, '42–45.5'], [45.5, 49.5, '46–49.5'], [49.5, None, '50+']]),
    dict(key='spread', label='Spread', group='Game lines', kind='side', unit='points', bin=3,
         buckets=[[None, 3.0, 'Under 3'], [3.0, 4.0, '3–3.5'], [4.0, 7.0, '4–6.5'], [7.0, None, '7+']]),
    dict(key='pass_yds', label='Passing yards', group='Passing', kind='ou', unit='yards', bin=25,
         buckets=[[None, 200, 'Under 200'], [200, 225, '200–224.5'], [225, 250, '225–249.5'], [250, None, '250+']]),
    dict(key='pass_attempts', label='Pass attempts', group='Passing', kind='ou', unit='attempts', bin=3,
         buckets=[[None, 30, 'Under 30'], [30, 34, '30–33.5'], [34, None, '34+']]),
    dict(key='pass_completions', label='Completions', group='Passing', kind='ou', unit='completions', bin=2,
         buckets=[[None, 20, 'Under 20'], [20, 23, '20–22.5'], [23, None, '23+']]),
    dict(key='pass_tds', label='Passing TDs', group='Passing', kind='ou', unit='touchdowns', bin=1,
         buckets=[[None, 1, '0.5'], [1, 2, '1.5'], [2, None, '2.5+']]),
    dict(key='interceptions', label='Interceptions', group='Passing', kind='ou', unit='interceptions', bin=1,
         buckets=[[None, 1, '0.5'], [1, None, '1.5+']]),
    dict(key='rush_yds', label='Rushing yards', group='Rushing', kind='ou', unit='yards', bin=10,
         buckets=[[None, 20, 'Under 20'], [20, 40, '20–39.5'], [40, 60, '40–59.5'], [60, None, '60+']]),
    dict(key='rush_attempts', label='Rush attempts', group='Rushing', kind='ou', unit='attempts', bin=2,
         buckets=[[None, 8, 'Under 8'], [8, 13, '8–12.5'], [13, 17, '13–16.5'], [17, None, '17+']]),
    dict(key='receptions', label='Receptions', group='Receiving', kind='ou', unit='receptions', bin=1,
         buckets=[[None, 2, '1.5'], [2, 3, '2.5'], [3, 4, '3.5'], [4, 5, '4.5'], [5, None, '5.5+']]),
    dict(key='recv_yds', label='Receiving yards', group='Receiving', kind='ou', unit='yards', bin=10,
         buckets=[[None, 20, 'Under 20'], [20, 40, '20–39.5'], [40, 60, '40–59.5'], [60, None, '60+']]),
]

# A moneyline is won, not covered, and has no line to beat.
MONEYLINE_TERMS = dict(a='favorite win', b='underdog win', rate='Favorite win rate', beat='won', missed='lost', vs=None)

NHL_MARKETS = [
    dict(key='totals', label='Game total', group='Game lines', kind='ou', unit='goals', bin=1),
    dict(key='spreads', label='Puck line', group='Game lines', kind='side', unit='goals', bin=1,
         terms=dict(vs='the puck line')),
    dict(key='h2h', label='Moneyline', group='Game lines', kind='side', unit='goals', bin=1, terms=MONEYLINE_TERMS),
    dict(key='player_shots_on_goal', label='Shots on goal', group='Skaters', kind='ou', unit='shots', bin=1),
    dict(key='player_points', label='Points', group='Skaters', kind='ou', unit='points', bin=1),
    dict(key='player_goals', label='Goals', group='Skaters', kind='ou', unit='goals', bin=1),
    dict(key='player_assists', label='Assists', group='Skaters', kind='ou', unit='assists', bin=1),
]

NBA_MARKETS = [
    dict(key='totals', label='Game total', group='Game lines', kind='ou', unit='points', bin=4,
         buckets=[[None, 215, 'Under 215'], [215, 225, '215–224.5'], [225, 235, '225–234.5'], [235, None, '235+']]),
    dict(key='spreads', label='Spread', group='Game lines', kind='side', unit='points', bin=3,
         buckets=[[None, 3, 'Under 3'], [3, 6, '3–5.5'], [6, 10, '6–9.5'], [10, None, '10+']]),
    dict(key='h2h', label='Moneyline', group='Game lines', kind='side', unit='points', bin=3, terms=MONEYLINE_TERMS),
    dict(key='player_points', label='Points', group='Scoring', kind='ou', unit='points', bin=2,
         buckets=[[None, 10, 'Under 10'], [10, 15, '10–14.5'], [15, 20, '15–19.5'], [20, 25, '20–24.5'], [25, None, '25+']]),
    dict(key='player_threes', label='Threes made', group='Scoring', kind='ou', unit='threes', bin=1),
    dict(key='player_rebounds', label='Rebounds', group='Rebounds and assists', kind='ou', unit='rebounds', bin=1,
         buckets=[[None, 4, 'Under 4'], [4, 6, '4–5.5'], [6, 8, '6–7.5'], [8, 10, '8–9.5'], [10, None, '10+']]),
    dict(key='player_assists', label='Assists', group='Rebounds and assists', kind='ou', unit='assists', bin=1,
         buckets=[[None, 3, 'Under 3'], [3, 5, '3–4.5'], [5, 7, '5–6.5'], [7, None, '7+']]),
    dict(key='player_points_rebounds_assists', label='Points + rebounds + assists', group='Combos', kind='ou',
         unit='points + rebounds + assists', bin=3,
         buckets=[[None, 20, 'Under 20'], [20, 25, '20–24.5'], [25, 30, '25–29.5'], [30, 35, '30–34.5'], [35, None, '35+']]),
    dict(key='player_points_rebounds', label='Points + rebounds', group='Combos', kind='ou', unit='points + rebounds', bin=2,
         buckets=[[None, 15, 'Under 15'], [15, 20, '15–19.5'], [20, 25, '20–24.5'], [25, 30, '25–29.5'], [30, None, '30+']]),
    dict(key='player_points_assists', label='Points + assists', group='Combos', kind='ou', unit='points + assists', bin=2,
         buckets=[[None, 15, 'Under 15'], [15, 20, '15–19.5'], [20, 25, '20–24.5'], [25, 30, '25–29.5'], [30, None, '30+']]),
    dict(key='player_rebounds_assists', label='Rebounds + assists', group='Combos', kind='ou', unit='rebounds + assists', bin=1,
         buckets=[[None, 6, 'Under 6'], [6, 8, '6–7.5'], [8, 10, '8–9.5'], [10, 12, '10–11.5'], [12, None, '12+']]),
    dict(key='player_blocks', label='Blocks', group='Defense and turnovers', kind='ou', unit='blocks', bin=1),
    dict(key='player_steals', label='Steals', group='Defense and turnovers', kind='ou', unit='steals', bin=1),
    dict(key='player_turnovers', label='Turnovers', group='Defense and turnovers', kind='ou', unit='turnovers', bin=1),
]

# Odds feed team names (folded) -> the NBA's three-letter codes, for compact game labels.
NBA_CODES = {
    'atlantahawks': 'ATL', 'bostonceltics': 'BOS', 'brooklynnets': 'BKN', 'charlottehornets': 'CHA',
    'chicagobulls': 'CHI', 'clevelandcavaliers': 'CLE', 'dallasmavericks': 'DAL', 'denvernuggets': 'DEN',
    'detroitpistons': 'DET', 'goldenstatewarriors': 'GSW', 'houstonrockets': 'HOU', 'indianapacers': 'IND',
    'losangelesclippers': 'LAC', 'laclippers': 'LAC', 'losangeleslakers': 'LAL', 'lalakers': 'LAL',
    'memphisgrizzlies': 'MEM', 'miamiheat': 'MIA', 'milwaukeebucks': 'MIL', 'minnesotatimberwolves': 'MIN',
    'neworleanspelicans': 'NOP', 'newyorkknicks': 'NYK', 'oklahomacitythunder': 'OKC', 'orlandomagic': 'ORL',
    'philadelphia76ers': 'PHI', 'phoenixsuns': 'PHX', 'portlandtrailblazers': 'POR', 'sacramentokings': 'SAC',
    'sanantoniospurs': 'SAS', 'torontoraptors': 'TOR', 'utahjazz': 'UTA', 'washingtonwizards': 'WAS',
}

ABBR = {'LAR': 'LA', 'WSH': 'WAS', 'JAC': 'JAX', 'ARZ': 'ARI', 'LVR': 'LV', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}


def decimal(price):
    price = float(price)
    return 1 + (price / 100 if price > 0 else 100 / abs(price))


def no_vig(price_a, price_b):
    a, b = 1 / decimal(price_a), 1 / decimal(price_b)
    return a / (a + b)


def kickoffs(schedule):
    when = pd.to_datetime(schedule.gameday + ' ' + schedule.gametime).dt.tz_localize('America/New_York')
    return dict(zip(schedule.game_id, (int(t.timestamp()) for t in when)))


def main_line(group):
    """One row per player/game/market: the books' main line and its average no-vig price."""
    paired = group.dropna(subset=['point', 'p_over', 'over_price', 'under_price'])
    if paired.empty:
        return None
    counts = paired.groupby('point').bookmaker.nunique()
    median = paired.point.median()
    line = sorted(counts.index, key=lambda x: (-counts[x], abs(x - median), x))[0]
    at = paired[paired.point.eq(line)]
    return dict(line=float(line), p_over=float(at.p_over.mean()), books=int(at.bookmaker.nunique()),
                dec_over=float(at.over_price.map(decimal).mean()), dec_under=float(at.under_price.map(decimal).mean()))


def pair_sides(offers):
    """Book-level over/under offers -> one row per book and line with both prices."""
    over = offers[offers.side.eq('over')].rename(columns={'price': 'over_price'})
    under = offers[offers.side.eq('under')].rename(columns={'price': 'under_price'})
    keys = ['stat_game', 'player', 'market_std', 'bookmaker', 'point']
    both = over.drop(columns='side').merge(under[keys + ['under_price']], on=keys, how='inner')
    both = both.drop_duplicates(keys, keep='last')
    both['p_over'] = [no_vig(a, b) for a, b in zip(both.over_price, both.under_price)]
    return both


def week1_offers():
    d = pd.read_csv(ROOT / 'reports/consensus-outliers-week1-2026/central_book_lines.csv')
    d = d.rename(columns={'over': 'over_price', 'under': 'under_price'})
    d['p_over'] = [no_vig(a, b) for a, b in zip(d.over_price, d.under_price)]
    return d


def week2_offers():
    d = pd.read_csv(ROOT / 'reports/week2-2026/offers.csv')
    d = d[d.market_std.isin(STATS) & d.name.isin(['over', 'under'])].rename(columns={'name': 'side'})
    return pair_sides(d)


def archived_offers(season, week, stats, schedule):
    base = ROOT / f'reports/nfl-weekly/{season}/week-{week}/pregame'
    props = pd.read_csv(base / 'props.csv')
    props = props[props.market_std.isin(STATS)].copy()
    official = schedule[(schedule.season == season) & (schedule.week == week) & (schedule.game_type == 'REG')]
    week_stats = stats[(stats.season == season) & (stats.week == week) & (stats.season_type == 'REG')]
    graded = grade_props(props.drop(columns='side', errors='ignore'), week_stats, official, season, week).rename(columns={'name': 'side'})
    return pair_sides(graded[graded.side.isin(['over', 'under'])])


def prop_rows(week, offers, kick):
    rows = []
    for (stat_game, player, market), group in offers.groupby(['stat_game', 'player', 'market_std'], sort=True):
        graded = group[group.grading_status.eq('graded') & group.actual.notna()]
        if graded.empty:
            continue
        line = main_line(graded)
        if line is None:
            continue
        rows.append(dict(market=market, period=week, t=kick.get(stat_game), label=player,
                         game=stat_game.split('_', 2)[2].replace('_', ' @ '), actual=float(graded.actual.iloc[0]), **line))
    return rows


def game_rows(season, week, schedule, kick):
    path = ROOT / f'reports/nfl-weekly/{season}/week-{week}/pregame/totals_spreads.csv'
    if not path.exists():
        return []
    lines = pd.read_csv(path)
    games = schedule[(schedule.season == season) & (schedule.week == week)].set_index('game_id')
    rows = []
    for game, group in lines.groupby('game'):
        away, home = (ABBR.get(x, x) for x in game.split(' @ '))
        gid = f'{season}_{week:02d}_{away}_{home}'
        if gid not in games.index or pd.isna(games.at[gid, 'home_score']):
            continue
        hs, as_ = float(games.at[gid, 'home_score']), float(games.at[gid, 'away_score'])
        tot = group.dropna(subset=['total_over_line', 'total_over_price', 'total_under_price'])
        tot = tot.assign(point=tot.total_over_line, over_price=tot.total_over_price, under_price=tot.total_under_price,
                         bookmaker=tot.book)
        tot['p_over'] = [no_vig(a, b) for a, b in zip(tot.over_price, tot.under_price)]
        total = main_line(tot)
        if total:
            rows.append(dict(market='game_total', period=week, t=kick.get(gid), label=game, game=game, actual=hs + as_, **total))
        spr = group.dropna(subset=['spread_home_line', 'spread_home_price', 'spread_away_price'])
        spr = spr.assign(point=spr.spread_home_line, over_price=spr.spread_home_price, under_price=spr.spread_away_price,
                         bookmaker=spr.book)
        spr['p_over'] = [no_vig(a, b) for a, b in zip(spr.over_price, spr.under_price)]
        home = main_line(spr)
        if not home or home['line'] == 0:
            continue
        # Express the spread from the favorite's side: line = points laid, actual = favorite's margin.
        away_label, home_label = game.split(' @ ')
        if home['line'] < 0:
            margin, laid, fav = hs - as_, -home['line'], home_label
            side = dict(p_over=home['p_over'], dec_over=home['dec_over'], dec_under=home['dec_under'])
        else:
            margin, laid, fav = as_ - hs, home['line'], away_label
            side = dict(p_over=1 - home['p_over'], dec_over=home['dec_under'], dec_under=home['dec_over'])
        rows.append(dict(market='spread', period=week, t=kick.get(gid), label=f'{fav} -{laid:g}', game=game,
                         actual=margin, line=laid, books=home['books'], **side))
    return rows


def capture_note(season, week):
    manifest = ROOT / f'reports/nfl-weekly/{season}/week-{week}/pregame/manifest.json'
    if manifest.exists():
        at = pd.Timestamp(json.loads(manifest.read_text())['snapshot_at']).tz_convert('America/New_York')
        return at.strftime('%a %b %-d, %-I:%M %p ET')
    # Weeks 1-2 predate the immutable archive; their audits document the snapshots used.
    return {(2026, 1): 'Wed Sep 9 board', (2026, 2): 'Thu for DET @ BUF, Sun morning for the rest'}.get((season, week))


def build_nfl(season, stats_path, schedule_path):
    stats = pd.read_parquet(stats_path)
    schedule = pd.read_csv(schedule_path)
    kick = kickoffs(schedule[schedule.season == season])
    completed = schedule[(schedule.season == season) & (schedule.game_type == 'REG')].groupby('week').home_score.apply(
        lambda s: s.notna().all())
    stat_weeks = set(stats[(stats.season == season) & (stats.season_type == 'REG')].week)
    rows, periods = [], []
    for week in sorted(w for w, done in completed.items() if done and w in stat_weeks):
        if season == 2026 and week == 1:
            offers = week1_offers()
        elif season == 2026 and week == 2:
            offers = week2_offers()
        elif (ROOT / f'reports/nfl-weekly/{season}/week-{week}/pregame/manifest.json').exists():
            offers = archived_offers(season, week, stats, schedule)
        else:
            continue
        week_rows = prop_rows(week, offers, kick) + game_rows(season, week, schedule, kick)
        rows += week_rows
        periods.append(dict(key=week, label=f'Week {week}', short=f'Wk {week}', captured=capture_note(season, week),
                            games=len({r['game'] for r in week_rows})))
    return package('nfl', season, NFL_MARKETS, rows, periods,
                   windows=[dict(key='season', label='Season'), dict(key='last2', label='Last 2 weeks', last=2),
                            dict(key='last1', label='Last week', last=1)],
                   notes=dict(source='Frozen pregame snapshots of our sportsbook feed; results from nflverse official statistics.',
                              timing='Prices come from one saved snapshot per week, taken from a few hours to several days before '
                                     'kickoff (each week’s time is listed under How this page works). Lines often move before games '
                                     'start, so this compares results with the market when we saved it, not the closing line.',
                              not_yet='Anytime, first and last touchdown scorers, longest rush and reception, and moneylines '
                                      'are not graded yet. Game lines begin in Week 3, when the weekly archive started saving them.'))


NHL_TIMING = ('Prices come from the last snapshot we saved before each game, normally our morning refresh of the '
              'game day (about 7 to 8:30 AM ET).{lead} Lines often move before games start, especially after starting '
              'goalies are confirmed, so this compares results with that snapshot, not the closing line.')


def nhl_runs(root=NHL_ARCHIVE / 'runs'):
    """Archived refresh runs in capture order; a snapshot archived twice is read once."""
    runs = {}
    for path in sorted(Path(root).glob('*.json.gz')):
        with gzip.open(path, 'rt') as f:
            run = json.load(f)
        runs.setdefault(run['snapshot']['snapshot_id'], run)
    return sorted(runs.values(), key=lambda r: pd.Timestamp(r['snapshot']['checked_at']))


def nhl_results(runs, season, archive=NHL_ARCHIVE):
    """Official games and skater lines from the newest run's archived NHL statistics pages."""
    from nhl.v2.data import digest, normalize
    for run in reversed(runs):
        manifest = next((m for m in run.get('history_manifests') or [] if m['season'] == season), None)
        if manifest:
            break
    else:
        return {}, {}
    stored = {ref['source']: ref['path'] for ref in run['input_objects']}
    teams, skaters = [], []
    for page in manifest['pages']:
        with gzip.open(Path(archive) / stored[f"data/nhl/v2/history/{page['path']}"], 'rt') as f:
            raw = json.load(f)
        if digest(raw['payload']) != page['sha256']:
            raise ValueError(f"Archived NHL page does not match its manifest: {page['path']}")
        (teams if '/team/' in raw['source_url'] else skaters).extend(raw['payload']['data'])
    games, players = normalize(teams, skaters, season)
    if digest([games, players]) != manifest['data_sha256']:
        raise ValueError(f'Archived NHL results do not reproduce the {season} manifest')
    return {g['game_id']: g for g in games}, {(p['game_id'], p['player_id']): p for p in players}


def pregame_quotes(snapshots, game_key):
    """game id -> (captured_at, quotes) from the last snapshot saved before that game started."""
    chosen = {}
    for snap in snapshots:
        at = pd.Timestamp(snap['checked_at'])
        by_game = defaultdict(list)
        for q in snap.get('rows', []):
            if not q.get(game_key) or q.get('price') is None or pd.Timestamp(q['commence_time']) <= at:
                continue
            if q.get('quoted_at') and pd.Timestamp(q['quoted_at']) >= pd.Timestamp(q['commence_time']):
                continue
            by_game[q[game_key]].append(q)
        for gid, quotes in by_game.items():
            chosen[gid] = (at, quotes)
    return chosen


def nhl_pregame_quotes(runs):
    return pregame_quotes([run['snapshot'] for run in runs], 'nhl_game_id')


def fold(name):
    """Compare skater names without case, accents or punctuation."""
    plain = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in plain.lower()).split())


def pair_quotes(d, identity, side_a, side_b, line_b=lambda line: line):
    """Book-level quotes -> one row per book and line with both prices, as pair_sides() does for NFL.

    ``identity`` names the column that tells players apart (empty or 0 for game lines)."""
    keys = ['market', identity, 'book', 'point']
    a = d[d.side.eq(side_a)].assign(point=d.line).rename(columns={'price': 'over_price'})
    b = d[d.side.eq(side_b)].assign(point=d.line.map(line_b)).rename(columns={'price': 'under_price'})
    both = a.merge(b[keys + ['under_price']], on=keys, how='inner').drop_duplicates(keys, keep='last')
    both = both.rename(columns={'book': 'bookmaker'})
    both['p_over'] = [no_vig(x, y) for x, y in zip(both.over_price, both.under_price)]
    return both


def nhl_pairs(quotes, side_a, side_b, line_b=lambda line: line):
    """NHL skaters are told apart by their NHL player id."""
    d = pd.DataFrame(quotes, columns=['market', 'player_id', 'player', 'book', 'side', 'line', 'price'])
    d['player_id'] = d.player_id.fillna(0).astype(int)
    return pair_quotes(d, 'player_id', side_a, side_b, line_b)


def side_rows(quotes, home, away, home_label, away_label, margin, game_name, t):
    """Spread and moneyline rows for one game, expressed from the favorite's side: line = goals or points
    laid (0 on the moneyline), actual = its final margin. ``margin`` is home score minus away score."""
    rows = []
    for market in ('spreads', 'h2h'):
        sides = [q for q in quotes if q['market'] == market and q['side'] in (home, away)]
        if not sides:
            continue
        flip = (lambda line: -line) if market == 'spreads' else (lambda line: line)
        paired = nhl_pairs([dict(q, line=q['line'] if market == 'spreads' else 0.0) for q in sides], home, away, flip)
        line = main_line(paired)
        if not line or (market == 'spreads' and line['line'] == 0):
            continue
        home_fav = line['line'] < 0 if market == 'spreads' else line['p_over'] >= 0.5
        fav = home_label if home_fav else away_label
        side = (dict(p_over=line['p_over'], dec_over=line['dec_over'], dec_under=line['dec_under']) if home_fav else
                dict(p_over=1 - line['p_over'], dec_over=line['dec_under'], dec_under=line['dec_over']))
        laid = abs(line['line'])
        rows.append(dict(market=market, t=t, label=f'{fav} -{laid:g}' if market == 'spreads' else fav, game=game_name,
                         actual=float(margin if home_fav else -margin), line=laid, books=line['books'], **side))
    return rows


def nhl_game_rows(gid, quotes, game, players, label, t):
    """Main-line rows for one settled game: skater props, total, puck line and moneyline."""
    rows, game_name = [], f"{label[game['away_id']]} @ {label[game['home_id']]}"
    home, away = quotes[0]['home_team'], quotes[0]['away_team']
    keys = {m['key'] for m in NHL_MARKETS}
    # The feed leaves player_id empty when a name is ambiguous league-wide (two Sebastian Ahos) or new
    # (rookies); a name that is unique in this game's official box score identifies the skater.
    named = defaultdict(list)
    for (game_id, pid), p in players.items():
        if game_id == gid:
            named[fold(p['player'])].append(pid)
    ou = []
    for q in quotes:
        if q['market'] not in keys or q['side'] not in ('Over', 'Under'):
            continue
        if q['market'] != 'totals' and not q.get('player_id'):
            match = named.get(fold(q['player']), [])
            if len(match) != 1:
                continue
            q = dict(q, player_id=match[0])
        ou.append(q)
    if ou:
        for (market, pid), group in nhl_pairs(ou, 'Over', 'Under').groupby(['market', 'player_id'], sort=True):
            if market == 'totals':
                actual, who = game['home_score'] + game['away_score'], game_name
            else:
                player = players.get((gid, pid))
                if player is None:  # No official appearance: the bet is void, not a zero.
                    continue
                stat = {'player_shots_on_goal': 'shots', 'player_goals': 'goals', 'player_assists': 'assists',
                        'player_points': 'points'}[market]
                actual, who = player[stat], group.player.iloc[0]
            line = main_line(group)
            if line:
                rows.append(dict(market=market, t=t, label=who, game=game_name, actual=float(actual), **line))
    return rows + side_rows(quotes, home, away, label[game['home_id']], label[game['away_id']],
                            game['home_score'] - game['away_score'], game_name, t)


def weekly(games, start_word):
    """Games as (start, captured_at, rows) in start order -> rows numbered by Eastern Monday-to-Sunday
    week from the first game, the weeks, a sentence on how early prices were captured, and the last day."""
    rows, leads, by_week, opener = [], [], defaultdict(list), None
    for start, at, game_rows in games:
        day = start.tz_convert('America/New_York').date()
        opener = opener or day - timedelta(days=day.weekday())
        week = (day - opener).days // 7 + 1
        if not game_rows:
            continue
        lead = (start - at).total_seconds() / 3600
        leads.append(lead)
        by_week[week].append((day, lead))
        rows += [dict(r, period=week) for r in game_rows]
    periods = []
    for week, played in sorted(by_week.items()):
        monday = opener + timedelta(days=7 * (week - 1))
        hours = sorted(lead for _, lead in played)
        periods.append(dict(key=week, label=f'Week of {monday:%b} {monday.day}', short=f'{monday:%b} {monday.day}',
                            games=len(played), captured=f'median {np.median(hours):.1f} hours before {start_word} '
                                                        f'(range {hours[0]:.1f} to {hours[-1]:.1f})'))
    lead = (f' For the games graded so far that was a median of {np.median(leads):.1f} hours before {start_word} '
            f'(range {min(leads):.1f} to {max(leads):.1f}).') if leads else ''
    last = max((day for played in by_week.values() for day, _ in played), default=None)
    return rows, periods, lead, last


SEASON_WINDOWS = [dict(key='season', label='Season'), dict(key='d30', label='Last 30 days', days=30),
                  dict(key='d7', label='Last 7 days', days=7)]


def build_nhl(season, runs_root=NHL_ARCHIVE / 'runs', archive=NHL_ARCHIVE):
    runs = nhl_runs(runs_root)
    games, players = nhl_results(runs, int(f'{season}{season + 1}'), archive) if runs else ({}, {})
    quotes = nhl_pregame_quotes(runs)
    label = {g[f'{s}_id']: g[f'{s}_team'] for g in games.values() for s in ('home', 'away')}
    label.update({p['team_id']: p['team_abbrev'] for p in players.values() if p.get('team_abbrev')})
    graded = []
    for gid in sorted(set(games) & set(quotes), key=lambda g: (quotes[g][1][0]['commence_time'], g)):
        at, game_quotes = quotes[gid]
        start = pd.Timestamp(game_quotes[0]['commence_time'])
        graded.append((start, at, nhl_game_rows(gid, game_quotes, games[gid], players, label, int(start.timestamp()))))
    rows, periods, lead, last = weekly(graded, 'puck drop')
    payload = package('nhl', season, NHL_MARKETS, rows, periods,
                      windows=SEASON_WINDOWS,
                      notes=dict(source='Last pregame snapshot of our sportsbook feed; results from NHL official statistics.',
                                 timing=NHL_TIMING.format(lead=lead),
                                 board_axis='Share of lines that went over (puck line and moneyline: share the favorite '
                                            'covered or won)',
                                 board_axis_short='Share over (game lines: favorite covered or won)',
                                 empty='NHL results are not graded yet. They will appear here once grading starts.'))
    if last:
        payload['through'] = f'{last:%b} {last.day}'
    return payload


NBA_TIMING = ('Prices come from the last snapshot we saved before each game, normally our late-afternoon refresh on '
              'game days (5 PM ET in daylight time, 4 PM in standard time).{lead} Lines often move before tipoff, '
              'especially after injury reports and starting lineups, so this compares results with that snapshot, '
              'not the closing line.')


def nba_runs(root=NBA_ARCHIVE / 'runs'):
    """Archived NBA refreshes (scripts/recap_archive.py) in capture order; a snapshot saved twice is read once."""
    runs = {}
    for path in sorted(Path(root).glob('*.json.gz')):
        with gzip.open(path, 'rt') as f:
            run = json.load(f)
        if run.get('checked_at'):
            runs.setdefault(run['checked_at'], run)
    return sorted(runs.values(), key=lambda r: pd.Timestamp(r['checked_at']))


def nba_pregame_quotes(runs):
    return pregame_quotes(runs, 'event_id')


def nba_results(quotes, root=ROOT, fetch=None, offline=False):
    """ESPN finals and box scores for these games, through the weekly recap's resolver.

    Saved responses in artifacts/nba/results come first; the free feed is asked only for days with a game
    that is not final with a box score there. A feed failure keeps what is saved. Returns
    (games, players, event id -> resolved row)."""
    import requests
    from nba_weekly_results import fetch_json, fetch_results
    probes = [dict(event_id=gid, commence_time=qs[0]['commence_time'], home_team=qs[0]['home_team'],
                   away_team=qs[0]['away_team']) for gid, (_, qs) in quotes.items()]
    if not probes:
        return {}, {}, {}
    day = lambda r: pd.Timestamp(r['commence_time']).tz_convert('America/New_York').date()
    span = lambda rows: (min(map(day, rows)), max(map(day, rows)))
    games, players, resolved = fetch_results(*span(probes), probes, root, offline=True)
    settled = lambda r: games.get(r.get('nba_game_id'), {}).get('boxscore_status') == 'available'
    missing = [probe for probe, r in zip(probes, resolved) if not settled(r)]
    if missing and not offline:
        try:
            more_games, more_players, more = fetch_results(*span(missing), missing, root, fetch=fetch or fetch_json)
        except (requests.RequestException, ValueError) as error:
            print(f'NBA results feed unavailable ({type(error).__name__}); graded saved results only', file=sys.stderr)
        else:
            games.update(more_games)
            players.update(more_players)
            fresh = {r['event_id']: r for r in more}
            resolved = [fresh.get(r['event_id'], r) for r in resolved]
    return games, players, {r['event_id']: r for r in resolved}


def nba_code(team):
    return NBA_CODES.get(''.join(c for c in fold(team) if c.isalnum()), team)


def nba_game_rows(quotes, game, players, t):
    """Main-line rows for one final game: player props, total, spread and moneyline."""
    from nba_weekly_results import get_actual
    home, away = quotes[0]['home_team'], quotes[0]['away_team']
    game_name = f'{nba_code(away)} @ {nba_code(home)}'
    keys = {m['key'] for m in NBA_MARKETS} - {'spreads', 'h2h'}
    ou = [q for q in quotes if q['market'] in keys and q['side'] in ('Over', 'Under') and q.get('line') is not None]
    rows = []
    if ou:
        # The feed has no player ids; names are told apart without case, accents or punctuation.
        d = pd.DataFrame(ou, columns=['market', 'player', 'book', 'side', 'line', 'price'])
        d['who'] = d.player.fillna('').map(fold)
        for (market, _), group in pair_quotes(d, 'who', 'Over', 'Under').groupby(['market', 'who'], sort=True):
            line = main_line(group)
            if not line:
                continue
            if market == 'totals':
                actual, label = game['home_score'] + game['away_score'], game_name
            else:
                label = group.player.iloc[0]
                actual, _ = get_actual(dict(market=market, player=label, home_team=game['home_team'],
                                            away_team=game['away_team'], nba_game_id=game['id'], resolution_status='resolved'),
                                       {game['id']: game}, players)
                if actual is None:  # Did not play, an ambiguous name or a missing stat: void, not a zero.
                    continue
            rows.append(dict(market=market, t=t, label=label, game=game_name, actual=float(actual), **line))
    return rows + side_rows(quotes, home, away, nba_code(home), nba_code(away),
                            game['home_score'] - game['away_score'], game_name, t)


def build_nba(season, runs_root=NBA_ARCHIVE / 'runs', root=ROOT, fetch=None, offline=False, now=None):
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
    first, after = (pd.Timestamp(f'{year}-10-01', tz='America/New_York') for year in (season, season + 1))
    quotes = {gid: v for gid, v in nba_pregame_quotes(nba_runs(runs_root)).items()
              if first <= pd.Timestamp(v[1][0]['commence_time']) < after and pd.Timestamp(v[1][0]['commence_time']) <= now}
    games, players, resolved = nba_results(quotes, root, fetch, offline)
    graded = []
    for gid in sorted(quotes, key=lambda g: (quotes[g][1][0]['commence_time'], g)):
        game = games.get(resolved.get(gid, {}).get('nba_game_id'))
        if not game or not game.get('final'):
            continue
        at, game_quotes = quotes[gid]
        start = pd.Timestamp(game_quotes[0]['commence_time'])
        graded.append((start, at, nba_game_rows(game_quotes, game, players, int(start.timestamp()))))
    rows, periods, lead, last = weekly(graded, 'tipoff')
    payload = package('nba', season, NBA_MARKETS, rows, periods, windows=SEASON_WINDOWS,
                      notes=dict(source='Last pregame snapshot of our sportsbook feed; results from ESPN box scores.',
                                 timing=NBA_TIMING.format(lead=lead),
                                 board_axis='Share of lines that went over (spread and moneyline: share the favorite '
                                            'covered or won)',
                                 board_axis_short='Share over (game lines: favorite covered or won)',
                                 empty='NBA results are not graded yet. They will appear here once the first games are final.'))
    if last:
        payload['through'] = f'{last:%b} {last.day}'
    return payload


def package(sport, season, markets, rows, periods, windows, notes):
    fields = ['period', 't', 'label', 'game', 'line', 'p_over', 'actual', 'books', 'dec_over', 'dec_under']
    by_market = {m['key']: [] for m in markets}
    for r in sorted(rows, key=lambda r: (r['t'] or 0, r['label'])):
        if r['market'] not in by_market:
            continue
        by_market[r['market']].append([r['period'], r['t'], r['label'], r['game'], r['line'], round(r['p_over'], 4),
                                       r['actual'], r['books'], round(r['dec_over'], 3), round(r['dec_under'], 3)])
    return dict(schema=1, sport=sport, season=season, generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                fields=fields, periods=periods, windows=windows, markets=markets, rows=by_market, notes=notes)


def write(payload):
    """Write the page data, leaving the file untouched when only the build time would change."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'{payload["sport"]}.json'
    counts = {k: len(v) for k, v in payload['rows'].items()}
    if path.exists():
        old = json.loads(path.read_text())
        if {**old, 'generated_at': None} == {**payload, 'generated_at': None}:
            print(f'{path.relative_to(ROOT)}: unchanged ({sum(counts.values())} rows)')
            return
    path.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
    print(f'{path.relative_to(ROOT)}: {sum(counts.values())} rows {counts}')


def season_summary(payload):
    """Season totals per market, for the static HTML that search engines and no-JS readers see."""
    out = []
    for m in payload['markets']:
        rows = payload['rows'][m['key']]
        settled = [r for r in rows if r[6] != r[4]]
        a = sum(1 for r in settled if r[6] > r[4])
        expected = sum(r[5] for r in settled)
        spread = sum(r[5] * (1 - r[5]) for r in settled) ** 0.5
        out.append(dict(market=m, lines=len(rows), a=a, b=len(settled) - a, push=len(rows) - len(settled),
                        rate=a / len(settled) if settled else None, implied=expected / len(settled) if settled else None,
                        gap=a - expected, z=(a - expected) / spread if spread else 0.0))
    return out


def status_text(payload, summary):
    total = sum(s['lines'] for s in summary)
    if not total:
        return payload['notes'].get('empty', 'No graded lines yet.')
    games = sum(p.get('games', 0) for p in payload['periods'])
    through = payload.get('through') or payload['periods'][-1]['label']
    return f"Through {through} · {games} games · {total:,} graded lines"


def table_html(summary):
    if not any(s['lines'] for s in summary):
        return '<p class="sub">No graded markets yet.</p>'
    pct = lambda v: f'{100 * v:.1f}%'
    body = ''
    for s in summary:
        if not s['lines']:
            continue
        m = s['market']
        label = f"{m['label']} (favorites vs. underdogs)" if m['kind'] == 'side' else m['label']
        gap = f"{s['gap']:+.1f}".replace('-', '−')
        body += (f"<tr><td>{escape(label)}</td><td>{s['lines']:,}</td><td>{s['a']}–{s['b']}</td>"
                 f"<td>{pct(s['rate'])}</td><td>{pct(s['implied'])}</td><td>{gap}</td></tr>")
    return ('<div class="table-wrap"><table><thead><tr><th>Market</th><th>Lines</th><th>Overs–unders</th>'
            '<th>Over rate</th><th>Prices implied</th><th>Vs. expected</th></tr></thead><tbody>'
            + body + '</tbody></table></div>')


def hub_text(payload, summary):
    live = [s for s in summary if s['lines']]
    if not live:
        return escape(payload['notes'].get('empty', 'Not graded yet.'))
    markets = len(live)
    text = f"{status_text(payload, summary)} across {markets} markets."
    notable = max((s for s in live if s['a'] + s['b'] >= 20), key=lambda s: abs(s['z']), default=None)
    if notable:
        m = notable['market']
        if m['kind'] == 'side':
            lead = 'favorites' if notable['a'] >= notable['b'] else 'underdogs'
        else:
            lead = 'overs' if notable['a'] >= notable['b'] else 'unders'
        hi, lo = max(notable['a'], notable['b']), min(notable['a'], notable['b'])
        text += f" Furthest from its prices: {m['label'].lower()}, {lead} {hi}–{lo}."
    return escape(text)


def replace_between(html, name, content):
    start, end = f'<!-- {name}:start -->', f'<!-- {name}:end -->'
    i, j = html.find(start), html.find(end)
    if i < 0 or j < 0:
        raise ValueError(f'Missing {name} markers')
    return html[:i + len(start)] + content + html[j:]


def publish_static(payload, page=PAGES / 'index.html'):
    """Write this sport's crawlable summary and table into the shared page; touch it only when text changes."""
    summary = season_summary(payload)
    sport = payload['sport']
    old = page.read_text()
    new = replace_between(old, f'market-status:{sport}', hub_text(payload, summary))
    new = replace_between(new, f'market-table:{sport}', table_html(summary))
    if new != old:
        page.write_text(new)
        print(f'{page.relative_to(ROOT)}: {sport} static summary updated')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--stats', default=None, help='nflverse weekly player stats parquet')
    parser.add_argument('--schedule', default=None, help='nflverse games/schedule CSV')
    parser.add_argument('--sport', choices=['nfl', 'nhl', 'nba', 'all'], default='all',
                        help='Each sport workflow rebuilds only its own page')
    parser.add_argument('--offline', action='store_true',
                        help='NBA: grade from saved ESPN responses only (artifacts/nba/results)')
    args = parser.parse_args()
    stats = args.stats or ROOT / f'data/weekly_player_stats_{args.season}.parquet'
    schedule = args.schedule or ROOT / f'data/schedule_{args.season}.csv'
    builders = dict(nfl=lambda: build_nfl(args.season, stats, schedule), nhl=lambda: build_nhl(args.season),
                    nba=lambda: build_nba(args.season, offline=args.offline))
    for sport in (builders if args.sport == 'all' else [args.sport]):
        payload = builders[sport]()
        write(payload)
        publish_static(payload)


if __name__ == '__main__':
    main()
