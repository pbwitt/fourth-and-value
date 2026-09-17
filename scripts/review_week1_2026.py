#!/usr/bin/env python3
"""Audit the actual archived Week 1 pages against saved 2026 results.

No model refit, paid API calls, or inference of missing results as zero.
Run from repository root: .venv/bin/python scripts/review_week1_2026.py
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REF = '6d420cf'
OUT = Path('reports/week1-2026')
STATS = {
    'pass_attempts': 'attempts', 'pass_completions': 'completions',
    'pass_yds': 'passing_yards', 'pass_tds': 'passing_tds',
    'interceptions': 'passing_interceptions', 'receptions': 'receptions',
    'recv_yds': 'receiving_yards', 'rush_attempts': 'carries',
    'rush_yds': 'rushing_yards',
}
TEAMS = dict(zip(
    ['Arizona Cardinals','Atlanta Falcons','Baltimore Ravens','Buffalo Bills',
     'Carolina Panthers','Chicago Bears','Cincinnati Bengals','Cleveland Browns',
     'Dallas Cowboys','Denver Broncos','Detroit Lions','Green Bay Packers',
     'Houston Texans','Indianapolis Colts','Jacksonville Jaguars','Kansas City Chiefs',
     'Las Vegas Raiders','Los Angeles Chargers','Los Angeles Rams','Miami Dolphins',
     'Minnesota Vikings','New England Patriots','New Orleans Saints','New York Giants',
     'New York Jets','Philadelphia Eagles','Pittsburgh Steelers','San Francisco 49ers',
     'Seattle Seahawks','Tampa Bay Buccaneers','Tennessee Titans','Washington Commanders'],
    'ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LV LAC LA MIA MIN NE NO NYG NYJ PHI PIT SF SEA TB TEN WAS'.split()))
KEY = ['game_id', 'player', 'market_std']


def normalize(name):
    return re.sub(r'(?:iii|ii|iv|jr|sr)$', '', re.sub('[^a-z]', '', str(name).lower()))


def archive(page):
    raw = subprocess.check_output(['git', 'show', f'{REF}:docs/props/{page}.html'])
    html = raw.decode()
    payload = json.loads(re.search(r'id="props-data">(.*?)</script>', html, re.S)[1])
    rows = [{k: payload['dictionary'][k][v] if k in payload['dictionary'] else v
             for k, v in zip(payload['fields'], row)} for row in payload['rows']]
    frame = pd.DataFrame(rows)
    built = pd.Timestamp(re.search(r'<time datetime="([^"]+)"', html)[1])
    assert (pd.to_datetime(frame.commence_time, utc=True) > built).all()
    assert (pd.to_datetime(frame.last_update, utc=True) <= built).all()
    return frame, {'git_ref': REF, 'path': f'docs/props/{page}.html',
                   'sha256': hashlib.sha256(raw).hexdigest(), 'built_at': built.isoformat()}


def payout(price):
    return np.where(price > 0, price / 100, 100 / np.abs(price))


def record(frame):
    graded = frame[frame.outcome.notna()]
    w, l, p = [int(graded.outcome.eq(x).sum()) for x in ['win', 'loss', 'push']]
    return dict(selected=len(frame), graded=len(graded), pending=len(frame)-len(graded),
                wins=w, losses=l, pushes=p, win_rate=w/(w+l) if w+l else None,
                units=float(graded.units.sum()), roi=float(graded.units.mean()) if len(graded) else None)


def grade(props, stats, schedule):
    d = props.copy()
    d['stat_game'] = '2026_01_' + d.away_team.map(TEAMS) + '_' + d.home_team.map(TEAMS)
    assert d.stat_game.isin(schedule.game_id).all()
    # Validate event date as well as both teams; UTC dates cross midnight.
    dates = schedule.set_index('game_id').gameday
    assert (pd.to_datetime(d.commence_time, utc=True).dt.tz_convert('America/New_York')
            .dt.strftime('%Y-%m-%d') == d.stat_game.map(dates)).all()
    lookup = {}
    for _, s in stats.iterrows():
        key = (s.game_id, normalize(s.player_display_name))
        if key in lookup:
            raise ValueError(f'Ambiguous player identity: {key}')
        lookup[key] = s
    actual, reason = [], []
    for _, row in d.iterrows():
        s = lookup.get((row.stat_game, normalize(row.player)))
        if row.market_std not in STATS:
            actual.append(np.nan); reason.append('unsupported settlement market'); continue
        if s is None:
            actual.append(np.nan); reason.append('no matching game/player result'); continue
        # Conservative participation evidence: do not grade a zero-usage player
        # without snap/activation and book-specific void information.
        if s[['attempts', 'carries', 'targets', 'receptions']].fillna(0).sum() <= 0:
            actual.append(np.nan); reason.append('no offensive usage evidence'); continue
        value = s[STATS[row.market_std]]
        actual.append(value); reason.append('graded' if pd.notna(value) else 'missing statistic')
    d['actual'] = actual
    d['grading_status'] = reason
    valid = d.actual.notna() & d.point.notna() & d.name.isin(['over', 'under'])
    d['outcome'] = None
    push = d.actual.eq(d.point)
    win = (d.name.eq('over') & d.actual.gt(d.point)) | (d.name.eq('under') & d.actual.lt(d.point))
    d.loc[valid, 'outcome'] = np.where(push[valid], 'push', np.where(win[valid], 'win', 'loss'))
    d['units'] = np.where(d.outcome.eq('win'), payout(d.price),
                          np.where(d.outcome.eq('loss'), -1., np.where(d.outcome.eq('push'), 0., np.nan)))
    return d


def representative(frame):
    # Fixed results-blind descriptive rule: most books at an exact line, then
    # closest to the median of distinct book/line pairs, then lower line.
    # Use an actually offered line, never an artificial median with borrowed odds.
    chosen = []
    for _, group in frame.groupby(KEY, sort=True):
        lines = group[['bookmaker', 'point']].drop_duplicates()
        count = lines.groupby('point').bookmaker.nunique()
        median = lines.point.median()
        line = sorted(count.index, key=lambda x: (-count[x], abs(x-median), x))[0]
        at_line = group[group.point.eq(line)]
        for side, offers in at_line.groupby('name'):
            # Best payout at the SAME line and side, tie broken by book name.
            offers = offers.assign(payout=payout(offers.price))
            chosen.append(offers.sort_values(['payout','bookmaker'], ascending=[False,True]).iloc[0])
    return pd.DataFrame(chosen).reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    board, board_source = archive('index')
    shortlist, top_source = archive('top')
    schedule = pd.read_csv('data/schedule_2026.csv').query('season == 2026 and week == 1').copy()
    stats = pd.read_parquet('data/weekly_player_stats_2026.parquet').query('season == 2026 and week == 1 and season_type == "REG"')
    assert len(schedule) == 16 and schedule.total.notna().all()
    assert (schedule.total == schedule.home_score + schedule.away_score).all()
    board = grade(board, stats, schedule)
    shortlist = grade(shortlist, stats, schedule)
    # One illustrative ticket per player/game/market, chosen only on published EV.
    tickets = shortlist.sort_values(['ev_per_100','bookmaker','point','name'],
                                    ascending=[False,True,True,True]).drop_duplicates(KEY)
    reps = representative(board[board.market_std.isin(STATS) & board.point.notna()])
    overs = reps[reps.name.eq('over')].copy()
    valid = overs[overs.actual.notna() & overs.actual.ne(overs.point)].copy()
    valid['y'] = valid.actual.gt(valid.point).astype(float)
    paired = valid[valid.model_prob.notna() & valid.consensus_prob.notna()].copy()
    paired['model_brier'] = (paired.model_prob-paired.y)**2
    paired['market_brier'] = (paired.consensus_prob-paired.y)**2
    paired['model_logloss'] = -(paired.y*np.log(paired.model_prob.clip(.000001,.999999)) +
                              (1-paired.y)*np.log(1-paired.model_prob.clip(.000001,.999999)))
    paired['market_logloss'] = -(paired.y*np.log(paired.consensus_prob.clip(.000001,.999999)) +
                               (1-paired.y)*np.log(1-paired.consensus_prob.clip(.000001,.999999)))
    paired['model_correct'] = (paired.model_prob.ge(.5) == paired.y.astype(bool))
    paired['bias'] = paired.mu-paired.actual
    paired['model_abs_error'] = paired.bias.abs()
    paired['line_abs_error'] = (paired.point-paired.actual).abs()
    calibration = paired.assign(confidence=np.maximum(paired.model_prob,1-paired.model_prob))
    calibration['bucket'] = pd.cut(calibration.confidence, [.499999,.55,.60,.70,.80,1.0], right=False)
    cal = calibration.groupby('bucket', observed=True).agg(n=('y','size'), predicted=('confidence','mean'), actual=('model_correct','mean')).reset_index()
    cal['bucket'] = cal.bucket.astype(str)
    market = []
    for m, group in overs.groupby('market_std'):
        v = group[group.actual.notna()]
        same = reps[reps.market_std.eq(m)]
        market.append(dict(market=m, player_games=len(group), matched=len(v),
                           over=int(v.actual.gt(v.point).sum()), under=int(v.actual.lt(v.point).sum()),
                           pushes=int(v.actual.eq(v.point).sum()),
                           over_bets=record(same[same.name.eq('over')]),
                           under_bets=record(same[same.name.eq('under')])) )
    metrics = paired.groupby('market_std').agg(n=('y','size'), model_brier=('model_brier','mean'),
        market_brier=('market_brier','mean'),model_logloss=('model_logloss','mean'),market_logloss=('market_logloss','mean'),
        model_direction_accuracy=('model_correct','mean'), model_bias=('bias','mean'),
        model_mae=('model_abs_error','mean'), line_mae=('line_abs_error','mean')).reset_index()
    metrics['brier_skill'] = 1-metrics.model_brier/metrics.market_brier
    # Paired game-block bootstrap: uncertainty respects within-game dependence.
    rng = np.random.default_rng(20260917)
    blocks = [g[['model_brier','market_brier']].to_numpy() for _,g in paired.groupby('game_id')]
    delta = []
    for _ in range(5000):
        sample = np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        delta.append(float((sample[:,0]-sample[:,1]).mean()))
    priced = pd.read_csv('data/nfl/predictions/priced_week1_2026.csv')
    totals = pd.read_csv('data/nfl/consensus/team_totals_week1.csv')
    schedule['game'] = schedule.away_team + ' @ ' + schedule.home_team
    games = priced.merge(schedule, on='game', validate='one_to_one').merge(
        totals[['game','consensus_spread_home','implied_home_total','implied_away_total','quoted_at']], on='game',validate='one_to_one')
    assert len(games) == 16
    assert (pd.to_datetime(games.quoted_at,utc=True) < pd.to_datetime(games.commence_time,utc=True)).all()
    games['raw_error'] = games.model_projection-games.total
    games['calibrated_error'] = games.calibrated_projection-games.total
    games['market_error'] = games.consensus_total-games.total
    games['closing_error'] = games.total_line-games.total
    games['over_outcome'] = np.where(games.total.eq(games.consensus_total),'push',np.where(games.total.gt(games.consensus_total),'win','loss'))
    games['raw_pick'] = np.where(games.claimed_edge>0,'over','under')
    games['raw_pick_result'] = np.where(games.over_outcome.eq('push'),'push',
        np.where(games.raw_pick.eq('over'),games.over_outcome,np.where(games.over_outcome.eq('win'),'loss','win')))
    for side in ['over','under']:
        result = games.over_outcome if side=='over' else games.over_outcome.map({'win':'loss','loss':'win','push':'push'})
        games[f'{side}_units'] = np.where(result.eq('push'),0,np.where(result.eq('win'),payout(games[f'best_{side}_price']),-1))
    games['raw_pick_units'] = np.where(games.raw_pick.eq('over'),games.over_units,games.under_units)
    games['best_ev_units'] = np.where(games.best_side.eq('over'),games.over_units,games.under_units)
    games['favorite_win'] = ((games.home_score-games.away_score)*games.spread_line > 0)
    games['favorite_cover_margin'] = (games.home_score-games.away_score)*np.sign(games.spread_line)-games.spread_line.abs()
    games['closing_over'] = games.total>games.total_line
    games['favorite_ml'] = np.where(games.spread_line>0,games.home_moneyline,games.away_moneyline)
    games['dog_ml'] = np.where(games.spread_line>0,games.away_moneyline,games.home_moneyline)
    games['favorite_ml_units'] = np.where(games.favorite_win,payout(games.favorite_ml),-1)
    games['dog_ml_units'] = np.where(~games.favorite_win,payout(games.dog_ml),-1)
    for side in ['over','under']:
        win = games.total.gt(games.total_line) if side=='over' else games.total.lt(games.total_line)
        games[f'closing_{side}_units'] = np.where(games.total.eq(games.total_line),0,np.where(win,payout(games[f'{side}_odds']),-1))
    fav_spread_odds = np.where(games.spread_line>0,games.home_spread_odds,games.away_spread_odds)
    dog_spread_odds = np.where(games.spread_line>0,games.away_spread_odds,games.home_spread_odds)
    games['favorite_ats_units'] = np.where(games.favorite_cover_margin.eq(0),0,
        np.where(games.favorite_cover_margin>0,payout(fav_spread_odds),-1))
    games['dog_ats_units'] = np.where(games.favorite_cover_margin.eq(0),0,
        np.where(games.favorite_cover_margin<0,payout(dog_spread_odds),-1))
    graded_tickets = tickets[tickets.outcome.notna()].copy()
    graded_tickets['edge_bucket'] = pd.cut(graded_tickets.ev_per_100,[0,5,10,20,100],right=False).astype(str)
    edge_buckets = {bucket:record(group) for bucket,group in graded_tickets.groupby('edge_bucket')}
    price_savings = []
    for _, row in graded_tickets.iterrows():
        same = board[board.game_id.eq(row.game_id) & board.player.eq(row.player) &
                     board.market_std.eq(row.market_std) & board.name.eq(row['name']) & board.point.eq(row.point)]
        median_profit = np.median(payout(same.price)) if row.outcome=='win' else (-1 if row.outcome=='loss' else 0)
        price_savings.append(float(row.units-median_profit))
    for name, frame in [('offers',board),('shortlist',shortlist),('tickets',tickets),('representative_lines',reps),('probability_comparison',paired),('games',games)]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    source_paths = ['data/schedule_2026.csv','data/weekly_player_stats_2026.parquet',
        'data/nfl/predictions/priced_week1_2026.csv','data/nfl/consensus/team_totals_week1.csv']
    summary = dict(season=2026, week=1, dates='September 9–14, 2026', sources=[board_source,top_source]+
        [dict(path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in source_paths],
        board_rows=len(board), model_rows=int(board.model_prob.notna().sum()),
        shortlist=record(shortlist), tickets=record(tickets),
        ticket_markets={m:record(g) for m,g in tickets.groupby('market_std')},
        ticket_sides={m:record(g) for m,g in tickets.groupby('name')},
        ticket_games={m:record(g) for m,g in tickets.groupby('stat_game')},
        ticket_pending=tickets[tickets.outcome.isna()][['player','market_std','grading_status']].to_dict('records'),
        ticket_expected_probability=float(tickets[tickets.outcome.isin(['win','loss'])].model_prob.mean()),
        ticket_expected_units=float(tickets[tickets.outcome.notna()].ev_per_100.sum()/100),
        ticket_edge_buckets=edge_buckets, ticket_price_shopping_units_vs_median=float(sum(price_savings)),
        market_results=market, probability_markets=metrics.to_dict('records'),
        probability_overall=dict(n=len(paired),player_games=len(paired[['game_id','player']].drop_duplicates()),
            model_brier=float(paired.model_brier.mean()),market_brier=float(paired.market_brier.mean()),
            model_logloss=float(paired.model_logloss.mean()),market_logloss=float(paired.market_logloss.mean()),
            brier_difference_ci95=np.quantile(delta,[.025,.975]).tolist()),
        confidence=cal.to_dict('records'),
        total_forecasts={k:dict(mae=float(games[k].abs().mean()),rmse=float(np.sqrt((games[k]**2).mean())),bias=float(games[k].mean()))
                         for k in ['raw_error','calibrated_error','market_error','closing_error']},
        total_points=int(games.total.sum()), snapshot_total_points=float(games.consensus_total.sum()),
        totals_over=int(games.over_outcome.eq('win').sum()),totals_under=int(games.over_outcome.eq('loss').sum()),
        totals_push=int(games.over_outcome.eq('push').sum()),
        totals_over_units=float(games.over_units.sum()),totals_under_units=float(games.under_units.sum()),
        raw_total_pick_results=games.raw_pick_result.value_counts().to_dict(),raw_total_pick_units=float(games.raw_pick_units.sum()),
        positive_ev_totals=games[games.best_ev_per_100.gt(0)][['game','best_side','best_ev_per_100','best_ev_units']].to_dict('records'),
        closing=dict(favorites_su=int(games.favorite_win.sum()),favorite_covers=int(games.favorite_cover_margin.gt(0).sum()),
                     dog_covers=int(games.favorite_cover_margin.lt(0).sum()),spread_pushes=int(games.favorite_cover_margin.eq(0).sum()),
                     overs=int(games.closing_over.sum()),unders=int(games.total.lt(games.total_line).sum()),
                     home_wins=int(games.home_score.gt(games.away_score).sum()),
                     favorite_ml_units=float(games.favorite_ml_units.sum()),dog_ml_units=float(games.dog_ml_units.sum()),
                     favorite_ats_units=float(games.favorite_ats_units.sum()),dog_ats_units=float(games.dog_ats_units.sum()),
                     over_units=float(games.closing_over_units.sum()),under_units=float(games.closing_under_units.sum())))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    # Ship enough row-level evidence to inspect the article without local CSVs.
    public = dict(summary=summary,
        tickets=json.loads(tickets[['stat_game','player','market_std','name','point','price','bookmaker',
            'model_prob','ev_per_100','actual','outcome','units','grading_status']].to_json(orient='records')),
        probability_comparison=json.loads(paired[['stat_game','player','market_std','point','actual','mu',
            'model_prob','consensus_prob','model_brier','market_brier']].to_json(orient='records')),
        games=json.loads(games[['game','away_score','home_score','consensus_total','total_line','model_projection',
            'calibrated_projection','best_over_price','best_under_price','raw_pick','raw_pick_result',
            'raw_pick_units','over_units','under_units','favorite_ats_units','dog_ats_units']].to_json(orient='records')))
    Path('docs/blog/week-1-2026-review-data.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
