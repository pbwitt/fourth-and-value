#!/usr/bin/env python3
"""Grade the NBA game lines the picks rule would have taken, without ever picking them.

Technical report FV-2026-05 found DraftKings' 2025-26 game lines beat the v1 game model, so
NBA moneylines, spreads and totals are shadow-tracked: shown on the board, never on the card.
This script keeps the evidence that could change that. From the committed run archive
(artifacts/nba/runs), it takes the last pregame snapshot of each game and applies the same
rule docs/assets/briefing-picks.js applies to props: a 10% model / 90% market log-odds blend
against the other books' no-vig price at that exact line, at least two books posting it, the
model at or above the market, and blended expected value of at least 1%. One line per game and
market (the highest blended EV) is kept and graded with ESPN finals (pushes refunded).

Output: artifacts/nba/shadow/game-lines.json (committed by the NBA workflow). Report only;
nothing here feeds a pick. Run: python scripts/nba/shadow_grade.py [--offline]
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

OUT = ROOT / 'artifacts/nba/shadow/game-lines.json'
WEIGHT, MIN_EV, MIN_BOOKS = 0.10, 0.01, 2  # mirrors BLEND in docs/assets/briefing-picks.js
GAME_MARKETS = ('h2h', 'spreads', 'totals')


def decimal(price):
    return 1 + (price / 100 if price > 0 else 100 / -price)


def logit(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def blend(row):
    """(blended conditional probability, EV per unit) or None when the rule would not pick."""
    win, push, market, books = (row.get(k) for k in ('model_probability', 'model_push_probability',
                                                      'other_book_probability', 'other_books'))
    if not all(isinstance(v, (int, float)) for v in (win, push, market, books, row.get('price'))):
        return None
    if not 0 <= push < 1 or not 0 < market < 1 or books + 1 < MIN_BOOKS:
        return None
    model = win / (1 - push)
    if not 0 < model < 1 or model < market:
        return None
    final = 1 / (1 + math.exp(-(WEIGHT * logit(model) + (1 - WEIGHT) * logit(market))))
    ev = (1 - push) * (final * decimal(row['price']) - 1)
    return (final, ev) if ev >= MIN_EV else None


def candidates(quotes):
    """Best qualifying line per game and market from one pregame snapshot."""
    best = {}
    for row in quotes:
        if row.get('market') not in GAME_MARKETS or row.get('model_probability') is None:
            continue
        b = blend(row)
        if not b:
            continue
        key = row['market']
        if key not in best or b[1] > best[key][1][1]:
            best[key] = (row, b)
    return list(best.values())


def outcome(row, game):
    """'win', 'loss' or 'push' for the row's side, from the final score."""
    home, away = game['home_score'], game['away_score']
    if row['market'] == 'totals':
        value = home + away - row['line']
        value = value if row['side'] == 'Over' else -value
    else:
        margin = home - away if row['side'] == row['home_team'] else away - home
        value = margin + ((row['line'] or 0) if row['market'] == 'spreads' else 0)
    return 'push' if value == 0 else 'win' if value > 0 else 'loss'


def summarize(picks):
    graded = [p for p in picks if p['result']]
    units = sum(p['units'] for p in graded)
    risked = sum(1 for p in graded if p['result'] != 'push')
    return dict(picks=len(picks), graded=len(graded), wins=sum(p['result'] == 'win' for p in graded),
                losses=sum(p['result'] == 'loss' for p in graded), pushes=sum(p['result'] == 'push' for p in graded),
                units=round(units, 3), roi=round(units / risked, 4) if risked else None)


def build(runs_root=ROOT / 'artifacts/nba/runs', offline=False, fetch=None, now=None):
    from build_market_results import nba_pregame_quotes, nba_results, nba_runs
    now = now or datetime.now(timezone.utc)
    quotes = {gid: v for gid, v in nba_pregame_quotes(nba_runs(runs_root)).items()
              if any(q.get('market') in GAME_MARKETS and q.get('model_probability') is not None for q in v[1])}
    started = {gid: v for gid, v in quotes.items()
               if datetime.fromisoformat(v[1][0]['commence_time'].replace('Z', '+00:00')) <= now}
    games, _, resolved = nba_results(started, ROOT, fetch, offline) if started else ({}, {}, {})
    picks = []
    for gid, (at, rows) in sorted(quotes.items(), key=lambda kv: (kv[1][1][0]['commence_time'], kv[0])):
        game = games.get(resolved.get(gid, {}).get('nba_game_id'))
        final = bool(game and game.get('final'))
        for row, (prob, ev) in candidates(rows):
            result = outcome(row, game) if final else None
            units = {'win': decimal(row['price']) - 1, 'loss': -1.0, 'push': 0.0}.get(result, 0.0)
            picks.append(dict(event_id=gid, game=row['game'], commence_time=row['commence_time'], snapshot_at=str(at),
                              market=row['market'], side=row['side'], line=row.get('line'), book=row.get('book'),
                              price=row['price'], quoted_at=row.get('quoted_at'), model_version=row.get('model_version'),
                              model_probability=row['model_probability'], model_push_probability=row['model_push_probability'],
                              market_probability=row['other_book_probability'], other_books=row['other_books'],
                              blended_probability=round(prob, 4), expected_value=round(ev, 4),
                              result=result, units=round(units, 4)))
    by_market = {m: summarize([p for p in picks if p['market'] == m]) for m in GAME_MARKETS}
    return dict(generated_at=now.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
                status='shadow_only', report='FV-2026-05',
                rule=dict(model_weight=WEIGHT, min_expected_value=MIN_EV, min_books_at_line=MIN_BOOKS,
                          one_per='game and market (highest blended EV)', snapshot='last pregame snapshot in artifacts/nba/runs',
                          staking='one flat unit; pushes refunded'),
                summary=dict(all=summarize(picks), **by_market), picks=picks)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--offline', action='store_true', help='Grade with saved ESPN results only')
    args = parser.parse_args()
    report = build(offline=args.offline)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, sort_keys=False) + '\n')
    s = report['summary']['all']
    print(f"NBA shadow game lines: {s['picks']} candidates, {s['graded']} graded, "
          f"{s['wins']}-{s['losses']}-{s['pushes']}, {s['units']:+.2f} units")


if __name__ == '__main__':
    main()
