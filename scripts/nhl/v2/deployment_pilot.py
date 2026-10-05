"""NHL shots-on-goal deployment pilot: research facts to ice-time scenarios (shadow only).

The selected v2.3 shots model is opportunity times production:
    mean = projected ice time x shots per minute x opponent factor
so a change in ice time scales the mean proportionally while the per-minute rate and
opponent factor stay as estimated. This module recomputes the frozen model's own count
distribution (negative binomial with the archived dispersion) for a different ice time.
It never alters the published forecast, a pick, or a ranking.

Mappings (research-fact-1 facts about this game, verified, not opinion):
  * a deployment fact whose excerpt itself states the minutes (effect=quantified) gives a
    point scenario at that ice time: status supported_adjustment;
  * a line/power-play/role fact without a number gives a scenario RANGE between the model's
    projected ice time and the 10th/90th percentile of the player's own last ten
    appearances, in the direction the fact implies: status scenario_only, no decision change;
  * unresolved or conflicting facts give the full 10th-90th range: status unresolved_fact.
No minutes-per-promotion effect size is assumed: none has been estimated or validated.

Double-counting safeguards: facts the reviewer marks as represented in model features are
excluded, and so is any fact whose source predates the model's feature cutoff (the games
under that deployment are already in the ice-time history). Starting-goalie and
participation facts are captured as context with no numerical effect: their effect on
shots has not been tested here.
"""
from math import exp, isfinite, lgamma, log
import re

from .data import digest, iso, stamp

PILOT_VERSION = 'nhl-shots-deployment-pilot-1'
MARKET = 'player_shots_on_goal'
MAPPABLE = {'ice_time', 'role', 'lineup_slot', 'power_play'}
CONTEXT_ONLY = {'goalie', 'participation'}
ESTABLISHING = ('official', 'original_report', 'secondary_report')
SUPPORT = 48
MINUTES = re.compile(r'(\d{1,2}(?:[.:]\d{1,2})?)\s*(?:minutes|mins?\b)', re.I)


def count_pmf(mean, alpha):
    """Same family as models.count_pmf: Poisson when alpha ~ 0, else NB(1/alpha, 1/(1+alpha*mean))."""
    mean = min(max(float(mean), 1e-6), 18.0)
    if alpha <= 1e-6:
        values = [exp(k*log(mean)-mean-lgamma(k+1)) for k in range(SUPPORT)]
    else:
        r, p = 1/alpha, 1/(1+alpha*mean)
        values = [exp(lgamma(k+r)-lgamma(r)-lgamma(k+1)+r*log(p)+k*log(1-p)) for k in range(SUPPORT)]
    if 1-sum(values) > 1e-6:
        raise ValueError('Distribution support insufficient')
    total = sum(values)
    return [v/total for v in values]


def outcome(pmf, line, side):
    win = sum(p for k, p in enumerate(pmf) if (k > line if side == 'Over' else k < line))
    push = sum(p for k, p in enumerate(pmf) if k == line)
    return win, push


def conditional(win, push):
    return win/(1-push) if push < 1 else None


def recent_toi(row):
    """Observed ice time (minutes) in the player's last ten appearances, from the forecast's own context."""
    trend = ((row.get('player_context') or {}).get('trend') or {}).get('rows') or []
    return sorted(float(t[4]) for t in trend if len(t) > 4 and isinstance(t[4], (int, float)) and t[4] > 0)


def quantile(values, q):
    if not values:
        return None
    position = (len(values)-1)*q
    low, high = int(position), min(int(position)+1, len(values)-1)
    return values[low]+(values[high]-values[low])*(position-low)


def reported_minutes(excerpt):
    found = [float(m.replace(':', '.')) if ':' not in m else int(m.split(':')[0])+int(m.split(':')[1])/60
             for m in MINUTES.findall(excerpt)]
    plausible = [m for m in found if 5 <= m <= 30]
    return plausible[0] if len(plausible) == 1 and len(found) == 1 else None


def classify(fact, cutoff):
    """Usage of one fact: (usage, reason)."""
    if fact.get('applies_to') != 'this_game':
        return 'context', 'not_about_this_game'
    if fact.get('usage') == 'excluded_already_in_model' or fact.get('represented_in') in ('model_features', 'hockey_features', 'both'):
        return 'excluded', 'already_in_model_features'
    effective = (fact.get('source') or {}).get('updated_at') or (fact.get('source') or {}).get('published_at')
    if cutoff and effective and stamp(effective) <= stamp(cutoff):
        return 'excluded', 'predates_model_feature_cutoff'
    if fact.get('assumption') in CONTEXT_ONLY:
        return 'context', 'numerical_effect_not_validated'
    if fact.get('assumption') not in MAPPABLE:
        return 'context', 'not_a_deployment_fact'
    if fact.get('verification') not in ESTABLISHING:
        return 'unresolved', 'not_verified'
    if fact.get('effect') == 'unresolved' or fact.get('conflicts_with') or fact.get('verification') == 'conflicting':
        return 'unresolved', 'unresolved_or_conflicting'
    if fact.get('effect') == 'quantified' and reported_minutes(fact.get('excerpt', '')) is not None:
        return 'quantified', 'reported_ice_time'
    if fact.get('effect') in ('scenario', 'quantified'):
        return 'scenario', 'direction_without_supported_quantity'
    return 'context', 'no_effect'


def assess(row, facts, distribution, now, *, ev_threshold=.02):
    """Shadow record for one NHL shots offer. Raises nothing for missing inputs; reports them."""
    from decision_ledger import outcome_key
    base = dict(schema='nhl-deployment-pilot-1', pilot_version=PILOT_VERSION, computed_at=iso(now),
                evaluation_status='shadow_only', outcome_key=outcome_key(row), offer_id=row.get('offer_id'),
                forecast_id=row.get('forecast_id'), player=row.get('player'), line=row.get('line'),
                side=row.get('side'), price=row.get('price'), ev_threshold=ev_threshold,
                adjusted=None, scenario_range=None, facts_considered=[], unvalidated_context=[])
    def done(status, **extra):
        base.update(status=status, **extra)
        base['record_id'] = digest([base['outcome_key'], base['forecast_id'], base['facts_considered'], status])[:24]
        return base
    if row.get('market') != MARKET:
        return done('not_in_pilot')
    inputs = row.get('model_inputs') or {}
    try:
        field = distribution['means_fields'][0]
        alpha = float(distribution['alphas'][0])
        mean, toi = float(inputs[field][0]), float(inputs['projected_toi'])
    except (KeyError, TypeError, ValueError, IndexError):
        return done('model_inputs_unavailable')
    if field == 'base_means' or toi <= 0:
        return done('model_kind_not_opportunity_based')
    win, push = outcome(count_pmf(mean, alpha), row['line'], row['side'])
    reproduced = isinstance(row.get('final_probability'), (int, float)) and abs(win-row['final_probability']) < 1e-6
    base['baseline'] = dict(mean=mean, projected_toi=toi, win=win, push=push, win_conditional=conditional(win, push),
                            reproduces_published_forecast=reproduced, feature_cutoff=inputs.get('feature_cutoff'))
    if not reproduced:
        # Fail closed: a scenario is only meaningful against the exact published baseline.
        return done('baseline_mismatch')
    rate = mean/toi
    history = recent_toi(row)
    lo, hi = quantile(history, .1), quantile(history, .9)
    usages = []
    for fact in facts:
        usage, reason = classify(fact, inputs.get('feature_cutoff'))
        usages.append((fact, usage))
        base['facts_considered'].append(dict(fact_id=fact.get('fact_id'), assumption=fact.get('assumption'),
                                             direction=fact.get('direction'), usage=usage, reason=reason))
        if usage == 'context' and fact.get('assumption') in CONTEXT_ONLY:
            base['unvalidated_context'].append(dict(fact_id=fact.get('fact_id'), assumption=fact['assumption'],
                                                    direction=fact.get('direction'), excerpt=fact.get('excerpt')))
    def at(minutes):
        w, p = outcome(count_pmf(rate*minutes, alpha), row['line'], row['side'])
        return dict(toi=minutes, mean=rate*minutes, win=w, push=p, win_conditional=conditional(w, p))
    quantified = sorted({round(reported_minutes(f['excerpt']), 2) for f, u in usages if u == 'quantified'})
    if len(quantified) == 1:
        adjusted = at(quantified[0])
        return done('supported_adjustment', adjusted=adjusted, adjusted_win_conditional=adjusted['win_conditional'],
                    push=adjusted['push'], adjustment_basis='reported_ice_time_in_source_excerpt')
    if len(quantified) > 1:
        return done('conflicting_quantities')
    kinds = [u for _, u in usages]
    if not ({'scenario', 'unresolved'} & set(kinds)):
        return done('no_mappable_fact' if facts else 'no_facts')
    if lo is None:
        return done('no_ice_time_history')
    directions = {f.get('direction') for f, u in usages if u == 'scenario'}
    # A supporting fact implies more opportunity for an Over and less for an Under.
    more = ('supports' in directions) == (row['side'] == 'Over')
    if 'unresolved' in kinds or len(directions) != 1 or directions == {'context'}:
        bounds, status = (lo, hi), 'unresolved_fact'
    else:
        bounds, status = ((toi, max(toi, hi)) if more else (min(toi, lo), toi)), 'scenario_only'
    ends = [at(b) for b in bounds]
    return done(status, scenario_range=dict(toi=[round(b, 2) for b in bounds],
                win_conditional=sorted(e['win_conditional'] for e in ends),
                basis='player_last_ten_appearances_p10_p90; no validated minutes effect'))


def run(rows, facts_by_offer, distribution, now):
    """Records for every NHL shots offer in the universe; one per offer."""
    records = []
    for row in rows:
        if row.get('sport', 'NHL') != 'NHL' or row.get('market') != MARKET:
            continue
        records.append(assess(row, facts_by_offer.get(row.get('offer_id'), []), distribution or {}, now))
    return records
