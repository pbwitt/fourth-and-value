#!/usr/bin/env python3
"""Build Market Results data: what the market expected versus what happened.

For every graded player prop and game line we keep one row: the main line
(offered by the most books; ties go to the line nearest the median), the
average of each book's no-vig probability at that line, the average offered
price per side, and the official result. The page aggregates these rows into
season and recent-window views, so it never needs a model or a pick.

NFL sources are the frozen weekly pregame archives graded with nflverse
statistics. Weeks 1-2 predate the immutable archive and are read from their
one-off audit folders. NHL output is a placeholder until its first settled
night; the NHL ledger is not built yet.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from nfl_weekly_review import STATS, TEAM_NAMES, grade_props  # noqa: E402

OUT = ROOT / 'docs/markets/data'

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

NHL_MARKETS = [
    dict(key='totals', label='Game total', group='Game lines', kind='ou', unit='goals', bin=1),
    dict(key='spreads', label='Puck line', group='Game lines', kind='side', unit='goals', bin=1),
    dict(key='h2h', label='Moneyline', group='Game lines', kind='side', unit='games', bin=1),
    dict(key='player_shots_on_goal', label='Shots on goal', group='Skaters', kind='ou', unit='shots', bin=1),
    dict(key='player_points', label='Points', group='Skaters', kind='ou', unit='points', bin=1),
    dict(key='player_goals', label='Goals', group='Skaters', kind='ou', unit='goals', bin=1),
    dict(key='player_assists', label='Assists', group='Skaters', kind='ou', unit='assists', bin=1),
]

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


def build_nhl(season):
    return package('nhl', season, NHL_MARKETS, [], [],
                   windows=[dict(key='season', label='Season'), dict(key='d30', label='Last 30 days', days=30),
                            dict(key='d7', label='Last 7 days', days=7)],
                   notes=dict(source='Last pregame snapshot of our sportsbook feed; results from NHL official box scores.',
                              timing='Prices come from our morning snapshot (about 7 to 8:30 AM ET), roughly 10 to 11 hours before '
                                     'a 7 PM puck drop. Lines often move before games start, especially after starting goalies are '
                                     'confirmed, so this compares results with the morning market, not the closing line.',
                              empty='NHL results are not graded yet. They will appear here once grading starts.'))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--season', type=int, default=2026)
    parser.add_argument('--stats', default=None, help='nflverse weekly player stats parquet')
    parser.add_argument('--schedule', default=None, help='nflverse games/schedule CSV')
    args = parser.parse_args()
    stats = args.stats or ROOT / f'data/weekly_player_stats_{args.season}.parquet'
    schedule = args.schedule or ROOT / f'data/schedule_{args.season}.csv'
    write(build_nfl(args.season, stats, schedule))
    write(build_nhl(args.season))


if __name__ == '__main__':
    main()
