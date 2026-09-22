#!/usr/bin/env python3
"""Reproduce every number in the Cardinals-Pirates run-line article.

Reads the saved MLB board snapshot and the cached game files. Writes the
article's data files and a JSON record of all calculations. This scores the
existing model output against saved prices; it does not retrain anything.
"""
import csv
import glob
import json
from pathlib import Path

BOARD = Path('docs/mlb/data/latest.json')
GAMES = Path('data/mlb/model_data')
OUT = Path('docs/blog/pirates-cardinals-2026')
GAME_ID = 823328
PIT, STL = 134, 138
JONES, PALLANTE = 683003, 669467


def american_to_prob(price):
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def decimal(price):
    return price / 100 + 1 if price > 0 else 100 / -price + 1


def load_games():
    out = []
    for f in glob.glob(str(GAMES / '*.json')):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if 'teams' in d and d.get('season') == 2026:
            out.append(d)
    return sorted(out, key=lambda d: d['date'])


def bullpen(games, team_id, last=None):
    """Relief-only RA/9, splitting the listed starter out of each team's line."""
    gs = [d for d in games if d['teams']['home']['id'] == team_id or d['teams']['away']['id'] == team_id]
    if last:
        gs = gs[-last:]
    outs = er = starter_outs = starts = 0
    for d in gs:
        side = 'home' if d['teams']['home']['id'] == team_id else 'away'
        t = d['teams'][side]
        for p in t.get('pitchers', []):
            if p.get('id') == t.get('starter'):
                starter_outs += p['outs']
                starts += 1
            else:
                outs += p['outs']
                er += p['earnedRuns']
    return dict(games=len(gs), relief_outs=outs, relief_er=er,
                ra9=round(er * 27 / outs, 2) if outs else None,
                relief_ip_per_game=round(outs / 3 / len(gs), 2),
                starter_ip_per_game=round(starter_outs / 3 / starts, 2) if starts else None)


def pitcher_log(games, pid):
    rows = []
    for d in games:
        for side in ('home', 'away'):
            t = d['teams'][side]
            for p in t.get('pitchers', []):
                if p.get('id') != pid:
                    continue
                opp = d['teams']['away' if side == 'home' else 'home']['name']
                rows.append(dict(date=d['date'], opponent=opp, started=t.get('starter') == pid,
                                 outs=int(p['outs']), ip=round(p['outs'] / 3, 1),
                                 pitches=int(p['numberOfPitches']), batters_faced=int(p['battersFaced']),
                                 strikeouts=int(p['strikeOuts']), earned_runs=int(p['earnedRuns'])))
    return sorted(rows, key=lambda r: r['date'])


def recalibrate(p, bins):
    """Map a model probability onto the held-out observed rate by interpolation."""
    lo = max((b for b in bins if b['predicted'] <= p), key=lambda b: b['predicted'])
    hi = min((b for b in bins if b['predicted'] > p), key=lambda b: b['predicted'])
    f = (p - lo['predicted']) / (hi['predicted'] - lo['predicted'])
    return lo['observed'] + f * (hi['observed'] - lo['observed']), lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    board = json.load(open(BOARD))
    rows = [r for r in board['rows'] if r['mlb_game_id'] == GAME_ID]
    games = load_games()

    pick = next(r for r in rows if r['market'] == 'spreads' and r['side'] == 'Pittsburgh Pirates'
                and r['book'] == 'bovada')
    p_model = pick['model_probability']
    price = pick['price']
    breakeven = american_to_prob(price)

    # Book comparison, both sides, margin removed at the -1.5 / +1.5 pair.
    books = {}
    for r in rows:
        if r['market'] != 'spreads':
            continue
        b = books.setdefault(r['book_label'], {})
        b['line_pit'] = r['line'] if r['side'] == 'Pittsburgh Pirates' else b.get('line_pit')
        b['pit' if r['side'] == 'Pittsburgh Pirates' else 'stl'] = r['price']
    book_rows = []
    for name, b in sorted(books.items(), key=lambda kv: -(kv[1].get('pit') or -999)):
        if b.get('pit') is None or b.get('stl') is None:
            continue
        ip, is_ = american_to_prob(b['pit']), american_to_prob(b['stl'])
        book_rows.append(dict(book=name, pirates_line=b['line_pit'], pirates_price=int(b['pit']),
                              cardinals_price=int(b['stl']),
                              pirates_no_margin=round(ip / (ip + is_), 4),
                              hold=round(ip + is_ - 1, 4)))

    v = board['model_validation']['regular']['spreads']
    p_cal, lo, hi = recalibrate(p_model, v['calibration_bins'])
    dec = decimal(price)

    scenarios = [
        ('Published model', p_model),
        ('Interpolated between audit bins', p_cal),
        ('Nearest lower audit bin (44.1% predicted)', lo['observed']),
        ('Nearest upper audit bin (55.9% predicted)', hi['observed']),
    ]
    scen_rows = [dict(scenario=n, probability=round(p, 4),
                      edge_pp=round(100 * (p - breakeven), 2),
                      ev_pct=round(100 * (p * dec - 1), 2),
                      profit_per_100=round(100 * (p * dec - 1), 2)) for n, p in scenarios]

    price_rows = [dict(book=r['book'], price=r['pirates_price'],
                       ev_pct=round(100 * (p_cal * decimal(r['pirates_price']) - 1), 2))
                  for r in book_rows if r['pirates_line'] == -1.5]

    bull = {'pittsburgh': {'season': bullpen(games, PIT), 'last30': bullpen(games, PIT, 30)},
            'st_louis': {'season': bullpen(games, STL), 'last30': bullpen(games, STL, 30)}}
    jones, pallante = pitcher_log(games, JONES), pitcher_log(games, PALLANTE)

    analysis = dict(
        game=dict(id=GAME_ID, matchup=pick['game'], venue=pick['venue'],
                  first_pitch_utc=pick['commence_time'], lineup_status=pick['lineup_status']),
        snapshot=dict(board_checked_at=board['checked_at'], quoted_at=pick['quoted_at'],
                      model_version=pick['model_version'], model_inputs_through=pick['model_input_through'],
                      history_through=board['history_through_date']),
        pick=dict(selection='Pittsburgh Pirates -1.5', book=pick['book_label'], price=int(price),
                  model_probability=round(p_model, 4), break_even=round(breakeven, 4),
                  model_edge_pp=round(100 * (p_model - breakeven), 2),
                  model_ev_pct=round(pick['model_ev_pct'], 2),
                  model_fair_price=pick['model_fair_price'],
                  projected_run_margin=round(pick['model_mean'], 3),
                  market_no_margin=round(pick['consensus_probability'], 4),
                  paired_books=pick['paired_books'], best_price=pick['best_price'],
                  displayed_inputs=pick['model_inputs']),
        recalibration=dict(interpolated=round(p_cal, 4),
                           edge_pp=round(100 * (p_cal - breakeven), 2),
                           ev_pct=round(100 * (p_cal * dec - 1), 2),
                           lower_bin=lo, upper_bin=hi,
                           spreads_brier_skill=v['brier_skill'], spreads_ece=v['ece'],
                           spreads_samples=v['samples'], spreads_passed=v['passed'],
                           test_window=[board['model_validation']['test_start'],
                                        board['model_validation']['test_end']]),
        scenarios=scen_rows, books=book_rows, price_sensitivity=price_rows, bullpens=bull,
        starters=dict(jones=dict(season_ip=round(sum(r['outs'] for r in jones) / 3, 1),
                                 starts=sum(1 for r in jones if r['started']),
                                 last_three=[r['outs'] for r in jones[-3:]],
                                 last_three_pitches=[r['pitches'] for r in jones[-3:]],
                                 starts_over_14_5=sum(1 for r in jones if r['outs'] > 14.5)),
                      pallante=dict(season_ip=round(sum(r['outs'] for r in pallante) / 3, 1),
                                    appearances=len(pallante),
                                    ra9=round(sum(r['earned_runs'] for r in pallante) * 27
                                              / sum(r['outs'] for r in pallante), 2),
                                    last_five=[r['outs'] for r in pallante[-5:]])),
        limitations=[
            'Prices are a saved snapshot; sportsbook availability at first pitch is not guaranteed.',
            'Bullpen splits are computed by removing each listed starter, so an opener or a bullpen game shifts innings between the two columns.',
            'The last-30 bullpen window is a small sample and overlaps the model validation window.',
            'The spreads model shows a Brier skill of 0.0018 against its reference distribution.',
            'Recalibration interpolates between two held-out bins whose errors have opposite signs.',
        ])

    (OUT / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    for name, data in [('book-prices.csv', book_rows), ('scenarios.csv', scen_rows),
                       ('jones-2026-game-log.csv', jones), ('pallante-2026-game-log.csv', pallante)]:
        with open(OUT / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    with open(OUT / 'bullpen-splits.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['team', 'window', 'games', 'relief_innings', 'relief_earned_runs', 'ra9',
                    'relief_ip_per_game', 'starter_ip_per_game'])
        for team, key in [('Pittsburgh', 'pittsburgh'), ('St. Louis', 'st_louis')]:
            for win in ('season', 'last30'):
                b = bull[key][win]
                w.writerow([team, win, b['games'], round(b['relief_outs'] / 3, 1), b['relief_er'],
                            b['ra9'], b['relief_ip_per_game'], b['starter_ip_per_game']])

    print(f"pick: {analysis['pick']['selection']} {price:+.0f} at {pick['book_label']}")
    print(f"  model {p_model:.4f} | break-even {breakeven:.4f} | edge {100*(p_model-breakeven):+.2f}pp | EV {100*(p_model*dec-1):+.2f}%")
    print(f"  recalibrated {p_cal:.4f} | edge {100*(p_cal-breakeven):+.2f}pp | EV {100*(p_cal*dec-1):+.2f}%")
    print(f"  bins: {lo['predicted']:.4f}->{lo['observed']:.4f} (n={lo['n']}) | {hi['predicted']:.4f}->{hi['observed']:.4f} (n={hi['n']})")
    for t in ('pittsburgh', 'st_louis'):
        print(f"  {t:12} season RA/9 {bull[t]['season']['ra9']} | last 30 {bull[t]['last30']['ra9']}")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == '__main__':
    main()
