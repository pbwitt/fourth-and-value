"""Grade frozen decision ledgers without modifying them.

Reads artifacts/research/decisions/*.json (decision_ledger.py), verifies each ledger's
entries hash, settles every entry from final results, and writes a separate grade file
per ledger under artifacts/research/grades/. All three decision versions are scored on
the same universe:

  * probability accuracy (Brier, log loss) on binary settled outcomes, conditional on
    no push, for the model and, where available, the paired market at the same line;
  * calibration bins with Wilson intervals;
  * hypothetical flat one-unit returns at the RECORDED price (never a later price),
    with pushes and voids returning the stake;
  * coverage (selected / eligible universe) and game-clustered bootstrap intervals,
    because several entries can share one game.

Price movement is kept apart from results and from any closing comparison:
  * market_probability_move: later paired fair probability at the same line minus the
    decision-time market probability;
  * entry_price_value_vs_later: recorded decimal price x later fair probability - 1;
  * closing_comparison: only when the later same-line observation is within
    CLOSE_WINDOW_MINUTES of the start (the definition in nhl/v2/grading.closing_value);
    otherwise 'no_verified_close'. The latest snapshot held is not called a close.

Hypothetical decisions are not wagers. Bet Tracker records actual stakes separately.

  python scripts/research_grading.py --outcomes path/to/outcomes.json
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import decision_ledger
from nhl.v2.data import ROOT, iso, stamp, write_json

GRADES = ROOT/'artifacts/research/grades'
CLOSE_WINDOW_MINUTES = 30
SCHEMA = 'decision-grades-1'


def settle(entry, outcome):
    """won / lost / push / void / pending for one ledger entry.

    `outcome` is {'status': 'final'|'did_not_participate'|'postponed'|'unknown',
    'value': stat (props), 'home_score', 'away_score' (game markets)}.
    Missing participation is never a loss.
    """
    if not outcome or outcome.get('status') in (None, 'unknown'):
        return 'pending'
    if outcome['status'] in ('postponed', 'did_not_participate'):
        return 'void'
    if outcome['status'] != 'final':
        return 'pending'
    market, side, line = entry['market'], str(entry['side']), entry.get('line')
    if market in ('h2h', 'spreads', 'totals'):
        home, away = outcome['home_score'], outcome['away_score']
        if market == 'totals':
            margin = (home+away-line)*(1 if side.lower() == 'over' else -1)
        else:
            home_side = side == entry.get('home_team')
            margin = (home-away) if home_side else (away-home)
            if market == 'spreads':
                margin += line
    else:
        value = outcome.get('value')
        if not isinstance(value, (int, float)):
            return 'pending'
        margin = (value-line)*(1 if side.lower() == 'over' else -1)
    return 'won' if margin > 0 else 'lost' if margin < 0 else 'push'


def profit(entry, result):
    if result == 'won':
        return decision_ledger.decimal(entry['price'])-1
    if result == 'lost':
        return -1.0
    return 0.0


def bootstrap(groups, statistic, reps=1000, seed=20261005):
    """Resample games (clusters) and recompute statistic(list of rows)."""
    import random
    keys = sorted(groups)
    if not keys:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        sample = [row for k in (keys[rng.randrange(len(keys))] for _ in keys) for row in groups[k]]
        v = statistic(sample)
        if v is not None:
            values.append(v)
    if not values:
        return None
    values.sort()
    return [round(values[int(.025*(len(values)-1))], 6), round(values[int(.975*(len(values)-1))], 6)]


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [round(centre-half, 4), round(centre+half, 4)]


def probability_metrics(rows, field):
    binary = [r for r in rows if r['result'] in ('won', 'lost') and isinstance(r.get(field), (int, float))]
    if not binary:
        return dict(n=0)
    y = [1 if r['result'] == 'won' else 0 for r in binary]
    p = [min(max(r[field], 1e-6), 1-1e-6) for r in binary]
    brier = sum((a-b)**2 for a, b in zip(p, y))/len(y)
    loss = -sum(b*math.log(a)+(1-b)*math.log(1-a) for a, b in zip(p, y))/len(y)
    bins = []
    for low in [i/10 for i in range(10)]:
        idx = [i for i, v in enumerate(p) if low <= v < low+.1 or (low == .9 and v == 1)]
        if idx:
            k = sum(y[i] for i in idx)
            bins.append(dict(predicted=round(sum(p[i] for i in idx)/len(idx), 4), observed=round(k/len(idx), 4),
                             n=len(idx), observed_ci95=wilson(k, len(idx))))
    groups = defaultdict(list)
    for r in binary:
        groups[r['game_id']].append(r)
    ci = bootstrap(groups, lambda s: sum((min(max(x[field], 1e-6), 1-1e-6)-(x['result'] == 'won'))**2 for x in s)/len(s) if s else None)
    return dict(n=len(binary), games=len(groups), brier=round(brier, 6), brier_ci95=ci, log_loss=round(loss, 6), calibration_bins=bins)


def return_metrics(rows):
    staked = [r for r in rows if r['result'] in ('won', 'lost', 'push', 'void')]
    settled = [r for r in staked if r['result'] in ('won', 'lost', 'push')]
    groups = defaultdict(list)
    for r in settled:
        groups[r['game_id']].append(r)
    units = sum(r['profit'] for r in settled)
    return dict(selected=len(rows), settled=len(settled), won=sum(r['result'] == 'won' for r in rows),
                lost=sum(r['result'] == 'lost' for r in rows), pushes=sum(r['result'] == 'push' for r in rows),
                voids=sum(r['result'] == 'void' for r in rows), pending=sum(r['result'] == 'pending' for r in rows),
                units=round(units, 4), roi=round(units/len(settled), 4) if settled else None,
                roi_ci95=bootstrap(groups, lambda s: sum(x['profit'] for x in s)/len(s) if s else None),
                games=len(groups), staking='hypothetical flat 1 unit at the recorded price; pushes and voids return the stake')


def movement(entry, later):
    """Separate price-movement quantities; a closing comparison only under the definition."""
    if not later:
        return dict(status='no_later_observation')
    out = dict(observed_at=later.get('last_pregame_at') or later.get('observed_at'))
    fair = later.get('other_fair')
    if later.get('book_line') is not None and entry.get('line') is not None and later['book_line'] != entry['line']:
        out.update(status='line_changed', line_move=later['book_line']-entry['line'])
    if isinstance(fair, (int, float)):
        market = entry['probabilities'].get('market_conditional')
        if isinstance(market, (int, float)):
            out['market_probability_move'] = round(fair-market, 6)
        out['entry_price_value_vs_later'] = round(decision_ledger.decimal(entry['price'])*fair-1, 6)
    observed = stamp(out['observed_at']) if out.get('observed_at') else None
    start = stamp(entry['commence_time'])
    verified = (observed and observed < start and (start-observed).total_seconds() <= CLOSE_WINDOW_MINUTES*60
                and out.get('status') != 'line_changed' and isinstance(fair, (int, float)))
    out['closing_comparison'] = dict(status='verified_close_same_line', closing_value=out.get('entry_price_value_vs_later')) \
        if verified else dict(status='no_verified_close', reason='latest snapshot is not within the closing window or line changed')
    out.setdefault('status', 'same_line')
    return out


def grade(ledger, outcomes, later=None, now=None):
    """Grades for one ledger. `outcomes` maps entry_id -> outcome; `later` maps entry_id -> observation."""
    decision_ledger.verify(ledger)
    later = later or {}
    rows = []
    for e in ledger['entries']:
        result = settle(e, outcomes.get(e['entry_id']))
        rows.append(dict(entry_id=e['entry_id'], game_id=e['game_id'], sport=e['sport'], result=result,
                         profit=profit(e, result), decisions={v: e['decisions'][v]['decision'] for v in decision_ledger.VERSIONS},
                         model=e['probabilities']['model_win_conditional'],
                         adjusted=e['decisions']['adjusted'].get('model_win_conditional'),
                         market=e['probabilities']['market_conditional'],
                         movement=movement(e, later.get(e['entry_id']))))
    universe = [r for r in rows if r['decisions']['baseline'] != 'not_eligible']
    versions = {}
    for v in decision_ledger.VERSIONS:
        chosen = [r for r in rows if r['decisions'][v] == 'select']
        field = 'adjusted' if v == 'adjusted' else 'model'
        versions[v] = dict(returns=return_metrics(chosen),
                           probability=probability_metrics(chosen, field),
                           market_probability_same_entries=probability_metrics([r for r in chosen if r['market'] is not None], 'market'),
                           coverage=dict(selected=len(chosen), eligible_universe=len(universe),
                                         share=round(len(chosen)/len(universe), 4) if universe else None))
    filtered_out = [r for r in rows if r['decisions']['baseline'] == 'select' and r['decisions']['research_filtered'] != 'select']
    added = [r for r in rows if r['decisions']['research_filtered'] == 'select' and r['decisions']['baseline'] != 'select']
    return dict(schema=SCHEMA, ledger_id=ledger['ledger_id'], edition_id=ledger['edition_id'],
                ledger_entries_sha256=ledger['entries_sha256'], graded_at=iso(now or datetime.now(timezone.utc)),
                universe=dict(entries=len(rows), eligible=len(universe),
                              model=probability_metrics(universe, 'model'),
                              market_same_entries=probability_metrics([r for r in universe if r['market'] is not None], 'market')),
                versions=versions,
                research_effect=dict(filtered_out=return_metrics(filtered_out), added=return_metrics(added),
                                     note='Outcomes of baseline selections that research removed, and of research selections the baseline did not make. Small samples; not a causal estimate.'),
                entries=rows, evaluation_status='prospective_shadow_only',
                limitations=['Hypothetical flat stakes at recorded prices; no fills or limits verified.',
                             'Entries in one game are correlated; intervals resample games.',
                             'Pending and unresolved participation are excluded from scores, never counted as losses.',
                             'A single edition is far too small for conclusions; aggregate many ledgers.'])


def aggregate(grades):
    """Pool entries across ledgers for each version (same metrics, clustered by game)."""
    rows = [r for g in grades for r in g['entries']]
    out = {}
    for v in decision_ledger.VERSIONS:
        chosen = [r for r in rows if r['decisions'][v] == 'select']
        out[v] = dict(returns=return_metrics(chosen), probability=probability_metrics(chosen, 'adjusted' if v == 'adjusted' else 'model'))
    return dict(ledgers=len(grades), entries=len(rows), versions=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outcomes', type=Path, required=True,
                        help='JSON {entry_id: outcome} from a sport adapter (see RESEARCH_SYSTEM.md)')
    parser.add_argument('--later', type=Path, help='Optional JSON {entry_id: later observation}')
    parser.add_argument('--ledgers', type=Path, default=decision_ledger.LEDGERS)
    args = parser.parse_args()
    outcomes = json.loads(args.outcomes.read_text())
    later = json.loads(args.later.read_text()) if args.later else {}
    grades = []
    for path in sorted(args.ledgers.glob('*.json')):
        ledger = json.loads(path.read_text())
        g = grade(ledger, outcomes, later)
        write_json(GRADES/f"{ledger['ledger_id']}.json", g)
        grades.append(g)
    summary = aggregate(grades)
    write_json(GRADES/'summary.json', summary)
    print(json.dumps({v: s['returns'] for v, s in summary['versions'].items()}, indent=2))


if __name__ == '__main__':
    main()
