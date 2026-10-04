"""Cross-market coherence for NHL player props: research tracking only.

Every sportsbook price implies an expected count for a player. The four player markets are
linked: goals are shots times a shooting rate, points are goals plus assists, and a team's
goals are shared among its players. The independent model's levels have not beaten the
market, but its ratios between counts (goals per shot, assists per point, share of team
scoring) are a narrower claim. Each coherence estimate moves one market's implied mean into a
different market with the model's ratio, then prices the offer with the model's own count
distribution. An offer's own market never feeds its own estimate.

Logical containment is separate and model-free: a wider bet that wins whenever a narrower
one wins (1+ points versus 1+ goals) cannot be worth less. A cross-book bound flags a wider
bet priced below the market's own fair probability for a narrower bet it contains.

Signals are archived per snapshot under a fixed rule version and graded later. Nothing here
feeds Top Picks, Market Watch or recommendations.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path
from statistics import median
import sys

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import nbinom, poisson

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); __package__ = 'nhl.v2'
from .arbitrage import contains, subject
from .data import ROOT, iso, stamp, write_json
from .pricing import decimal, key

RULE_VERSION = 'nhl-coherence-1'
# Model order for means and dispersion.
PLAYER_MARKETS = ['player_shots_on_goal', 'player_goals', 'player_assists', 'player_points']
SOURCE_LABELS = {'player_shots_on_goal': 'shots', 'player_goals': 'goals', 'player_assists': 'assists',
                 'player_points': 'points', 'team': 'team'}
# Sources whose link to the target is mechanical (goals are shots times a shooting rate; points
# are goals plus assists; a team's goals are shared among its players). Only these form the
# combined estimate and the flag. Every other pair is still archived and graded separately.
STRUCTURAL = {'player_shots_on_goal': ['goals', 'team'], 'player_goals': ['shots', 'points', 'team'],
              'player_assists': ['points', 'team'], 'player_points': ['goals', 'assists', 'team']}
ALIGN_WINDOW = timedelta(minutes=30)
# Fixed before any coherence signal was graded: at least two linked markets must each price
# the offer favorably, and together by at least the 3% EV hurdle used elsewhere.
FLAG_EV = .03
FLAG_LINKS = 2
ARCHIVE = ROOT / 'artifacts/nhl/coherence'
OFFER_FIELDS = ['offer_id', 'forecast_id', 'event_id', 'nhl_game_id', 'commence_time', 'home_team', 'away_team', 'game',
                'book', 'book_label', 'market', 'market_label', 'player', 'player_id', 'side', 'line', 'price', 'quoted_at',
                'settlement_profile', 'settlement_verified', 'fair_probability', 'consensus_probability',
                'market_probability', 'best_price', 'conditional_probability', 'estimated_ev']


def distribution(models, manifest):
    """Count-distribution parameters of the selected player and team models, or None if unsupported."""
    fields, alphas = [], []
    for j in range(4):
        model = models['shots' if j == 0 else 'scoring']
        if model.kind == 'opportunity_hurdle':
            return None   # No negative-binomial or Poisson marginal to invert.
        fields.append('base_means' if model.kind == 'rate_poisson' else 'opportunity_means')
        alphas.append((model.alpha_shots if j == 0 else model.alpha_scoring) if model.kind == 'opportunity_nb' else 0.)
    return dict(version=manifest['version'], means_fields=fields, alphas=alphas, ot_home=float(models['team'].ot_home),
                artifact_sha256=manifest['artifact_sha256'])


def params_for(state):
    """Distribution parameters of the artifact that made the snapshot, or None.

    Snapshots record them from v2.2 on. Older snapshots resolve through the registry of retired
    artifacts or, when they match, the installed one; a mismatch is never filled in.
    """
    if state.get('model_distribution'):
        return state['model_distribution']
    sha = (state.get('model_manifest') or {}).get('artifact_sha256')
    if not sha:
        return None
    retired = ROOT/'models/nhl/v2/retired-distributions.json'
    if retired.exists() and sha in (known := json.loads(retired.read_text())):
        return known[sha]
    try:
        from .inference import bundle
        models, manifest = bundle()
    except Exception:
        return None
    return distribution(models, manifest) if manifest['artifact_sha256'] == sha else None


def _counts(mean, alpha):
    if alpha > 1e-6:
        r = 1/alpha
        return nbinom(r, r/(r+mean))
    return poisson(mean)


def outcome(mean, alpha, line, side):
    """Win/push/loss for an Over or Under at `line` when the count has this mean."""
    dist = _counts(mean, alpha)
    over = float(dist.sf(math.floor(line)))
    push = float(dist.pmf(line)) if float(line).is_integer() else 0.
    win = over if side == 'Over' else 1-over-push
    return dict(win=win, push=push, loss=max(0., 1-win-push))


def implied_mean(probability, line, alpha):
    """Mean whose Over probability, conditional on no push, equals a fair market probability."""
    def gap(mean):
        p = outcome(mean, alpha, line, 'Over')
        return p['win']/(1-p['push'])-probability
    lo, hi = 1e-4, 18.
    if not 0 < probability < 1 or gap(lo) > 0 or gap(hi) < 0:
        return None
    return brentq(gap, lo, hi, xtol=1e-7)


def _main_line(groups):
    """The line most books pair; ties go to the most even price, then the lower line."""
    return min(groups.items(), key=lambda kv: (-len({q['book'] for q in kv[1]}),
               abs(median(q['fair_probability'] for q in kv[1])-.5), kv[0]))


def _span(quotes):
    times = sorted(stamp(t) for q in quotes for t in (q['paired_at_min'], q['paired_at_max']))
    return iso(times[0]), iso(times[-1])


def _aligned(source, quoted_at):
    t = stamp(quoted_at)
    return all(abs(stamp(s)-t) <= ALIGN_WINDOW for s in (source['quoted_from'], source['quoted_to']))


def market_means(rows, alphas):
    """Implied mean per event, player, market and settlement profile from paired fair Over prices."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if (r['market'] in PLAYER_MARKETS and r['side'] == 'Over' and r.get('fair_probability') is not None
                and r.get('settlement_verified') and r.get('line') is not None):
            groups[(r['event_id'], subject(r), r['market'], r['settlement_profile'])][r['line']].append(r)
    out = {}
    for k, lines in groups.items():
        line, quotes = _main_line(lines)
        p = median(q['fair_probability'] for q in quotes)
        mean = implied_mean(p, line, alphas[PLAYER_MARKETS.index(k[2])])
        if mean is None:
            continue
        start, end = _span(quotes)
        out[k] = dict(mean=mean, line=line, probability=p, books=sorted({q['book'] for q in quotes}),
                      quoted_from=start, quoted_to=end)
    return out


def game_probabilities(home, away, line, ot_home):
    """Home win and Over (conditional on no push) from independent Poisson regulation scores.

    Matches the team model's settlement transform: every regulation tie gets exactly one
    overtime/shootout goal, to the home side with probability `ot_home`.
    """
    n = np.arange(30)
    reg = np.outer(poisson.pmf(n, home), poisson.pmf(n, away))
    i, j = np.indices(reg.shape)
    win = float(reg[i > j].sum()+reg[i == j].sum()*ot_home)
    total = i+j+(i == j)
    push = float(reg[total == line].sum())
    return win, float(reg[total > line].sum())/(1-push)


def fit_regulation(p_home, p_over, line, ot_home):
    """Regulation goal means that reproduce a fair moneyline and total, or None."""
    def residual(x):
        win, over = game_probabilities(*np.exp(x), line, ot_home)
        return [win-p_home, over-p_over]
    fit = least_squares(residual, np.log([3., 3.]), bounds=(np.log(.3), np.log(8.)))
    if max(abs(v) for v in fit.fun) > 1e-4:
        return None
    return [float(v) for v in np.exp(fit.x)]


def team_means(rows, ot_home):
    """Market-implied regulation goals per event from the consensus moneyline and main total."""
    by_event = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['market'] in ('h2h', 'totals') and r.get('fair_probability') is not None and r.get('settlement_verified'):
            by_event[r['event_id']][(r['market'], r['line'], r['side'])].append(r)
    out = {}
    for event, groups in by_event.items():
        sample = next(iter(groups.values()))[0]
        home = groups.get(('h2h', None, sample['home_team']))
        totals = {k[1]: v for k, v in groups.items() if k[0] == 'totals' and k[2] == 'Over'}
        if not home or not totals:
            continue
        line, overs = _main_line(totals)
        p_home, p_over = median(q['fair_probability'] for q in home), median(q['fair_probability'] for q in overs)
        fit = fit_regulation(p_home, p_over, line, ot_home)
        if fit is None:
            continue
        start, end = _span(home+overs)
        out[event] = dict(home=fit[0], away=fit[1], line=line, home_probability=p_home, over_probability=p_over,
                          books=sorted({q['book'] for q in home+overs}), quoted_from=start, quoted_to=end)
    return out


def _model_team_means(rows):
    out = {}
    for r in rows:
        if r.get('projected_home_reg_goals') is not None and r.get('projected_away_reg_goals') is not None:
            out.setdefault(r['event_id'], (r['projected_home_reg_goals'], r['projected_away_reg_goals']))
    return out


def _priced(estimate, alpha, row, dec):
    p = outcome(estimate['mean'], alpha, row['line'], row['side'])
    estimate.update(win=p['win'], push=p['push'], loss=p['loss'], conditional=p['win']/(1-p['push']) if p['push'] < 1 else None,
                    ev=p['win']*(dec-1)-p['loss'])
    return estimate


def coherence(state, params):
    """One entry per verified player-prop offer with at least one cross-market estimate."""
    rows = state.get('rows', [])
    alphas = params['alphas']
    means = market_means(rows, alphas)
    teams = team_means(rows, params['ot_home'])
    model_teams = _model_team_means(rows)
    sides = {e['nhl_game_id']: (e.get('home_id'), e.get('away_id')) for e in state.get('events', [])}
    entries = []
    for r in rows:
        if r['market'] not in PLAYER_MARKETS or not r.get('settlement_verified') or r.get('line') is None:
            continue
        t = PLAYER_MARKETS.index(r['market'])
        inputs = r.get('model_inputs') or {}
        if any(not inputs.get(field) for field in params['means_fields']):
            continue
        # Each market's mean from the field its selected model reads.
        model_means = [inputs[params['means_fields'][j]][j] for j in range(4)]
        if model_means[t] <= 0:
            continue
        try:
            dec = decimal(r['price'])
        except (TypeError, ValueError):
            continue
        sources = {}
        for s, market in enumerate(PLAYER_MARKETS):
            src = means.get((r['event_id'], subject(r), market, r['settlement_profile']))
            if s == t or not src or model_means[s] <= 0 or not _aligned(src, r['quoted_at']):
                continue
            ratio = model_means[t]/model_means[s]
            sources[SOURCE_LABELS[market]] = dict(mean=src['mean']*ratio, model_ratio=ratio, source_mean=src['mean'],
                                                  source_line=src['line'], source_probability=src['probability'],
                                                  books=src['books'])
        team, side_ids, model_team = teams.get(r['event_id']), sides.get(r.get('nhl_game_id')), model_teams.get(r['event_id'])
        if team and side_ids and model_team and r.get('player_team_id') in side_ids and _aligned(team, r['quoted_at']):
            i = side_ids.index(r['player_team_id'])
            market_goals, model_goals = (team['home'], team['away'])[i], model_team[i]
            if model_goals > 0:
                sources['team'] = dict(mean=model_means[t]*market_goals/model_goals, model_ratio=model_means[t]/model_goals,
                                       source_mean=market_goals, source_line=team['line'],
                                       source_probability=team['over_probability'], books=team['books'])
        if not sources:
            continue
        for estimate in sources.values():
            _priced(estimate, alphas[t], r, dec)
        # Geometric mean of the available structural means; fixed in rule version 1.
        linked = [name for name in STRUCTURAL[r['market']] if name in sources]
        combined = _priced(dict(mean=math.exp(sum(math.log(sources[n]['mean']) for n in linked)/len(linked)),
                                sources=linked), alphas[t], r, dec) if linked else None
        own = means.get((r['event_id'], subject(r), r['market'], r['settlement_profile']))
        entries.append(dict({f: r.get(f) for f in OFFER_FIELDS}, player_team_id=r.get('player_team_id'),
                            model_mean=model_means[t], own_market_mean=own['mean'] if own else None,
                            own_market_line=own['line'] if own else None, sources=sources, combined=combined,
                            cross_market_gap=combined['mean']/own['mean']-1 if own and combined else None,
                            model_agrees=r.get('estimated_ev') is not None and r['estimated_ev'] > 0,
                            agreeing_links=[n for n in linked if sources[n]['ev'] > 0],
                            flagged=(combined is not None and len(linked) >= FLAG_LINKS and combined['ev'] >= FLAG_EV
                                     and all(sources[n]['ev'] > 0 for n in linked))))
    return entries


def bounds(rows):
    """Wider offers priced below the market's fair probability for a narrower bet they contain.

    Half lines only, so neither bet can push. The narrower bet's fair probability is the
    median across books that pair both of its sides. The wider bet wins at least that often.
    """
    groups = defaultdict(list)
    for r in rows:
        if r.get('settlement_verified') and (r.get('line') is not None or r['market'] == 'h2h'):
            groups[(r['event_id'], subject(r), r.get('settlement_profile'))].append(r)
    found = []
    for group in groups.values():
        outcomes = defaultdict(list)
        for r in group:
            if r.get('fair_probability') is not None:
                outcomes[(r['market'], r['line'], r['side'])].append(r)
        for wide in group:
            try:
                dec = decimal(wide['price'])
            except (TypeError, ValueError):
                continue
            best = None
            for (market, line, side), quotes in outcomes.items():
                narrow = quotes[0]
                if not contains(wide, narrow):
                    continue
                aligned = [q for q in quotes if abs(stamp(q['quoted_at'])-stamp(wide['quoted_at'])) <= ALIGN_WINDOW]
                if not aligned:
                    continue
                p = median(q['fair_probability'] for q in aligned)
                if best is None or p > best[0]:
                    best = (p, narrow, aligned)
            if best and best[0]*dec-1 > 0:
                p, narrow, aligned = best
                found.append(dict({f: wide.get(f) for f in OFFER_FIELDS}, bound_probability=p, bound_ev=p*dec-1,
                                  narrow=dict(market=narrow['market'], market_label=narrow.get('market_label', narrow['market']),
                                              line=narrow['line'], side=narrow['side'], books=sorted({q['book'] for q in aligned}))))
    found.sort(key=lambda f: -f['bound_ev'])
    return found


def report(state, params):
    entries = coherence(state, params) if params else []
    best = [e for e in entries if e['best_price']]
    flagged = sorted((e for e in best if e['flagged']), key=lambda e: -e['combined']['ev'])
    gaps = sorted((e for e in best if e['cross_market_gap'] is not None), key=lambda e: -abs(e['cross_market_gap']))
    by_source = defaultdict(int)
    for e in entries:
        for name in e['sources']:
            by_source[name] += 1
    return dict(rule_version=RULE_VERSION, snapshot_id=state.get('snapshot_id'), checked_at=state.get('checked_at'),
                decision_session=state.get('decision_session'), feed_status=state.get('status'),
                params=params, flag_ev=FLAG_EV, flag_links=FLAG_LINKS, offers_checked=len(entries), best_price_offers=len(best),
                offers_flagged=len(flagged), structural_links=STRUCTURAL,
                estimates_by_source=dict(sorted(by_source.items())), flagged=flagged[:40], largest_gaps=gaps[:20],
                bounds=bounds(state.get('rows', []))[:40], entries=entries,
                status='research_only' if params else 'model_distribution_unavailable',
                note='Cross-market research signals. Not recommendations; separate from Top Picks and Market Watch.')


def public(data):
    """The page feed: everything except the full per-offer archive."""
    return {k: v for k, v in data.items() if k != 'entries'}


def archive(data, root=ARCHIVE, computed_at=None, backfilled=False):
    """Freeze one snapshot's signals; an existing file for the snapshot is never rewritten."""
    if not data.get('snapshot_id') or data.get('feed_status') != 'ready' or not data.get('params'):
        return None
    path = Path(root)/f"{data['snapshot_id']}.json.gz"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(data, computed_at=computed_at or iso(datetime.now(timezone.utc)), backfilled=backfilled,
                  flagged=None, largest_gaps=None)
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as f:
        f.write(json.dumps(record, separators=(',', ':'), allow_nan=False).encode())
    return path


# ---------------------------------------------------------------- grading

def _binary(p, won):
    p = min(max(p, 1e-6), 1-1e-6)
    return -(math.log(p) if won else math.log(1-p)), (p-won)**2


def _paired_difference(rows, field, reference, seed=48):
    """Mean log-loss difference (field minus reference) with a game-cluster bootstrap interval."""
    by_game = defaultdict(list)
    for r in rows:
        by_game[r['nhl_game_id']].append(_binary(r[field], r['won'])[0]-_binary(r[reference], r['won'])[0])
    sums = np.array([sum(v) for v in by_game.values()]); counts = np.array([len(v) for v in by_game.values()])
    idx = np.random.default_rng(seed).integers(0, len(sums), (1000, len(sums)))
    boot = sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
    return dict(mean=float(sums.sum()/counts.sum()), interval=np.quantile(boot, [.025, .975]).tolist(), games=len(sums))


def _scores(rows, fields):
    out = dict(count=len(rows), games=len({r['nhl_game_id'] for r in rows}))
    for f in fields:
        losses = [_binary(r[f], r['won']) for r in rows]
        out[f] = dict(log_loss=sum(l for l, _ in losses)/len(rows), brier=sum(b for _, b in losses)/len(rows))
    if out['games'] >= 2:
        out['coherence_minus_market_log_loss'] = _paired_difference(rows, 'coherence', 'market')
        out['model_minus_market_log_loss'] = _paired_difference(rows, 'model', 'market')
    return out


def _outcome_key(e):
    return (e['event_id'], e.get('player_id') or e['player'], e['market'], e['line'], e['side'])


def evaluate(records, games, players):
    """Probability quality for every source and flat-unit results for flagged offers.

    One observation per outcome: the first snapshot in which it was checked. Flagged offers
    use the first snapshot that flagged them, at that snapshot's offered price. Each model
    version is a separate cohort, and so are prospective and backfilled snapshots; backfills
    were computed later from archived quotes with this same fixed rule, never chosen by results.
    """
    from .grading import betting_metrics, settle
    by_game = {g['game_id']: g for g in games}
    by_player = {(r['game_id'], r['player_id']): r for r in players}
    groups = defaultdict(list)
    for rec in records:
        params = rec.get('params') or {}
        model = params.get('version') or 'artifact ' + str(params.get('artifact_sha256', 'unknown'))[:12]
        groups[('backfilled ' if rec.get('backfilled') else 'prospective ') + model].append(rec)
    cohorts = {}
    for label, group in sorted(groups.items()):
        first, flags, latest = {}, {}, {}
        for rec in sorted(group, key=lambda r: r['checked_at']):
            flagged_here = {}
            for e in rec['entries']:
                k = _outcome_key(e)
                if e.get('best_price'):
                    first.setdefault(k, e)
                # The best flagged price in the first snapshot that flags the outcome.
                if e['flagged'] and k not in flags and (k not in flagged_here or decimal(e['price']) > decimal(flagged_here[k]['price'])):
                    flagged_here[k] = e
                latest[(*k, e['book'])] = (rec['checked_at'], e)
            for k, e in flagged_here.items():
                flags[k] = (rec['checked_at'], e)
        scored, graded = defaultdict(list), []
        for k, e in first.items():
            result = settle(e, by_game.get(e['nhl_game_id']), by_player.get((e['nhl_game_id'], e.get('player_id'))))
            if result not in ('won', 'lost') or e.get('consensus_probability') is None:
                continue
            base = dict(nhl_game_id=e['nhl_game_id'], won=result == 'won', market=e['consensus_probability'])
            for name, est in [('combined', e['combined']), *e['sources'].items()]:
                if est is None or est.get('conditional') is None:
                    continue
                row = dict(base, coherence=est['conditional'])
                if e.get('conditional_probability') is not None:
                    row['model'] = e['conditional_probability']
                scored[(name, e['market'])].append(row)
        for k, (flagged_at, e) in flags.items():
            result = settle(e, by_game.get(e['nhl_game_id']), by_player.get((e['nhl_game_id'], e.get('player_id'))))
            # Movement of the same book's same-line consensus by the last later pregame snapshot.
            later_at, later = latest.get((*k, e['book']), (None, None))
            moved = (later['consensus_probability']-e['consensus_probability']
                     if later_at and later_at > flagged_at and later['consensus_probability'] is not None
                     and e['consensus_probability'] is not None else None)
            graded.append(dict(e, result=result, later_consensus_move=moved))
        # Same outcomes for every compared probability; rows without a model forecast drop out.
        quality = {f'{name}:{market}': _scores([r for r in rows if 'model' in r], ['coherence', 'market', 'model'])
                   for (name, market), rows in sorted(scored.items()) if any('model' in r for r in rows)}
        moves = [g['later_consensus_move'] for g in graded if g['later_consensus_move'] is not None]
        cohorts[label] = dict(
            probability_quality=quality,
            flagged=dict(betting_metrics(graded), unresolved=sum(g['result'] not in ('won', 'lost', 'push', 'void') for g in graded),
                         model_agrees=betting_metrics([g for g in graded if g['model_agrees']]),
                         later_snapshot_move=dict(count=len(moves), mean=float(np.mean(moves)) if moves else None,
                                                  toward_signal=sum(m > 0 for m in moves))),
            flagged_rows=[{k: g.get(k) for k in ('offer_id', 'game', 'player', 'market', 'side', 'line', 'book', 'price', 'quoted_at',
                                                  'result', 'model_agrees', 'later_consensus_move')} | dict(ev=g['combined']['ev'])
                          for g in graded])
    return dict(rule_version=RULE_VERSION, flag_ev=FLAG_EV, cohorts=cohorts, snapshots=len(records),
                limitations=['Quoted-price shadow results; no wager was placed or verified as executable.',
                             'Snapshots are hours before puck drop; later-snapshot movement is not closing-line value.',
                             'Unconfirmed participation stays unresolved; missing player data is never a loss.',
                             'Sources overlap (points contain goals), so per-source results are not independent tests.',
                             'Small early samples: intervals, not point estimates, describe what is known.'])


def load_archive(root=ARCHIVE):
    out = []
    for path in sorted(Path(root).glob('*.json.gz')):
        with gzip.open(path, 'rt') as f:
            out.append(json.load(f))
    return out


def _history(cached):
    from .data import load
    with gzip.open(ROOT/'models/nhl/v2/history.json.gz', 'rt') as f:
        history = json.load(f)
    if cached:
        season = json.loads((ROOT/'docs/nhl/data/latest.json').read_text())['season']
        games, players, _ = load(ROOT/'data/nhl/v2/history', [season])
        history['games'] = [g for g in history['games'] if g['season'] != season]+games
        history['players'] = [r for r in history['players'] if r['season'] != season]+players
    return history['games'], history['players']


def backfill(runs=ROOT/'artifacts/nhl/runs', root=ARCHIVE):
    """Compute rule-version signals for archived snapshots made by the current model artifact."""
    written = []
    for path in sorted(Path(runs).glob('*.json.gz')):
        with gzip.open(path, 'rt') as f:
            state = json.load(f)['snapshot']
        params = params_for(state)
        if not params:
            continue
        out = archive(report(state, params), root, backfilled=True)
        if out:
            written.append(str(out))
    return written


def render(data):
    """Static HTML section for the arbitrage page."""
    from html import escape

    def odds(price):
        return f'+{price:.0f}' if price > 0 else f'{price:.0f}'

    def offer(e):
        return escape(f"{e['player']} · {e.get('market_label') or e['market']} {e['side']} {e['line']:g}")

    intro = ('<section class="section" id="cross-market"><h2>Cross-market checks</h2>'
             '<p class="muted">Each player price implies an expected count. We move one market’s count into another '
             'with the model’s ratios (goals per shot, assists per point, share of team scoring), then price the offer. '
             'An offer’s own market never feeds its own check. Research signals only: not recommendations, and separate '
             'from Top Picks and Market Watch.</p>')
    if data.get('status') == 'model_distribution_unavailable':
        return intro+'<div class="empty">Model distribution unavailable for this snapshot, so no cross-market estimate was made.</div></section>'
    if data.get('status') != 'research_only':
        return intro+'<div class="empty">Cross-market checks are unavailable for this snapshot.</div></section>'
    counts = ', '.join(f'{escape(k)} {v}' for k, v in data['estimates_by_source'].items()) or 'none'
    parts = [intro, f'<p class="muted">{data["best_price_offers"]} best-price offers checked; {data["offers_flagged"]} priced at least '
                    f'{100*data["flag_ev"]:.0f}% better than the linked markets imply, with at least {data["flag_links"]} links agreeing. '
                    f'The flag uses mechanical links only: shots and goals, goals or assists and points, and team goals. '
                    f'Other pairs are tracked separately. Estimates by source: {counts}. '
                    f'Rule {escape(data["rule_version"])}.</p>']
    def line(value):
        return '' if value is None else f' {value:g}'

    def expected(value):
        return '—' if value is None else f'{value:.2f}'

    if data['flagged']:
        body = ''.join(
            f'<tr><td>{escape(e["game"])}</td><td>{offer(e)}</td><td>{odds(e["price"])} at {escape(e.get("book_label") or e["book"])}</td>'
            f'<td>{expected(e["own_market_mean"])}</td><td>{e["combined"]["mean"]:.2f}'
            f'<div class="meta">{escape(", ".join(e["combined"]["sources"]))}</div></td>'
            f'<td>{100*e["combined"]["conditional"]:.1f}%</td><td class="positive">{100*e["combined"]["ev"]:+.1f}%</td>'
            f'<td>{"Yes" if e["model_agrees"] else "No"}</td></tr>' for e in data['flagged'][:25])
        parts.append('<div class="table-wrap"><table><thead><tr><th>Game</th><th>Offer</th><th>Best price</th>'
                     '<th>Own market expects</th><th>Other markets expect</th><th>Cross-market win</th><th>Cross-market EV</th>'
                     f'<th>Model agrees</th></tr></thead><tbody>{body}</tbody></table></div>')
    else:
        parts.append('<div class="empty">No offer clears the cross-market threshold in this snapshot.</div>')
    if data['bounds']:
        body = ''.join(
            f'<tr><td>{escape(e["game"])}</td><td>{escape(str(e["player"] or e["side"]))} · {escape(e.get("market_label") or e["market"])} '
            f'{escape(str(e["side"]))}{line(e["line"])}</td><td>{odds(e["price"])} at {escape(e.get("book_label") or e["book"])}</td>'
            f'<td>{escape(e["narrow"]["market_label"])} {escape(str(e["narrow"]["side"]))}{line(e["narrow"]["line"])}: '
            f'{100*e["bound_probability"]:.1f}%</td><td class="positive">{100*e["bound_ev"]:+.1f}%</td></tr>' for e in data['bounds'][:25])
        parts.append('<h3>Priced below a bet it contains</h3><p class="muted">The wider bet wins every time the narrower one does, '
                     'so the market’s own fair price for the narrower bet is a floor.</p><div class="table-wrap"><table><thead><tr>'
                     '<th>Game</th><th>Wider bet</th><th>Price</th><th>Narrower bet, fair</th><th>EV at the floor</th></tr></thead>'
                     f'<tbody>{body}</tbody></table></div>')
    parts.append('<p class="muted">Assumptions: the model’s count distributions and ratios, a median of paired no-vig prices '
                 'at each market’s main line, and quotes within 30 minutes of each other. Every check is archived per snapshot '
                 'and graded against results before it can influence any pick.</p></section>')
    return ''.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('backfill')
    grade = subs.add_parser('grade')
    grade.add_argument('--cached-history', action='store_true', help='Add current-season cached game records')
    grade.add_argument('--output', type=Path, default=ARCHIVE/'evaluation.json')
    args = parser.parse_args()
    if args.command == 'backfill':
        print('\n'.join(backfill()) or 'No eligible snapshots')
    else:
        games, players = _history(args.cached_history)
        write_json(args.output, evaluate(load_archive(), games, players))


if __name__ == '__main__':
    main()
