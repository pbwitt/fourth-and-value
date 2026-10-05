"""MLB decision gates: aggregate checks cannot hide side- or range-specific errors.

Pure Python so the freshness validator and site builder can use it without numpy.
Thresholds were fixed from the structure of the decision before looking at holdout
returns (none exist; no historical prices are archived):
  * BIN_MIN_N = 30: smallest bin where a Wilson interval is a usable approximation;
  * GAP = 0.03: a three-point probability overstatement is comparable in size to the
    3% EV hurdle near even odds, so it can turn a "pick" into a loss-making price;
  * MEAN_BIAS = 0.05: a 5% systematic error in the projected count (for example
    outs) shifts every threshold probability in one direction;
  * skill requires the game-clustered 95% interval of (model - reference) Brier to lie
    below zero when the report carries it.
Gate results describe accuracy against an empirical baseline only. They are not
evidence of performance against bookmaker prices, and a calibration gap is never
subtracted from expected value: it says where a probability is unreliable, not by how
much a price is mispriced.
"""
from math import sqrt

BIN_MIN_N = 30
GAP = 0.03
MEAN_BIAS = 0.05
MIN_FORECASTS_FOR_BIAS = 200
COUNT_MARKETS = {'pitcher_strikeouts', 'pitcher_outs', 'batter_hits', 'batter_total_bases', 'batter_home_runs', 'batter_rbis'}


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [centre-half, centre+half]


def reference_side(row):
    """Validation bins score P(Over) for props/totals and P(home) for moneylines/run lines."""
    if row['market'] in ('h2h', 'spreads'):
        return row.get('side') == row.get('home_team')
    return row.get('side') == 'Over'


def bin_for(bins, p):
    index = min(int(p*10), 9)
    return next((b for b in bins if min(int(b['predicted']*10), 9) == index), None)


def range_reason(audit, row):
    """Block a side whose probability range is overstated or untested in validation."""
    win, push = row.get('model_probability'), row.get('model_push_probability') or 0
    if not isinstance(win, (int, float)):
        return None
    same = reference_side(row)
    p_ref = win if same else max(0.0, 1-win-push)
    b = bin_for(audit.get('calibration_bins') or [], p_ref)
    if not b or b['n'] < BIN_MIN_N:
        return 'Research forecast: too few validation outcomes in this probability range'
    low, high = wilson(round(b['observed']*b['n']), b['n'])
    overstated = (b['predicted']-b['observed'] > GAP and b['predicted'] > high) if same else \
        (b['observed']-b['predicted'] > GAP and b['predicted'] < low)
    if overstated:
        return 'Research forecast: validation shows this side is overstated in this probability range'
    return None


def bias(audit):
    """Relative mean error of the projected count, with its interval when available."""
    mean_p, mean_a = audit.get('mean_prediction'), audit.get('mean_actual')
    if not isinstance(mean_p, (int, float)) or not isinstance(mean_a, (int, float)) or mean_a <= 0:
        return None
    interval = (audit.get('mean_bias') or {}).get('ci95')
    return dict(relative=(mean_p-mean_a)/mean_a, absolute=mean_p-mean_a, ci95=interval)


def bias_reason(audit, row):
    """A projected count that is systematically high blocks Overs; systematically low blocks Unders."""
    if row['market'] not in COUNT_MARKETS:
        return None
    b = bias(audit)
    if not b or abs(b['relative']) <= MEAN_BIAS:
        return None
    if b['ci95'] is not None:
        if b['ci95'][0] <= 0 <= b['ci95'][1]:
            return None
    elif (audit.get('forecasts') or 0) < MIN_FORECASTS_FOR_BIAS:
        return None
    favored = 'Over' if b['relative'] > 0 else 'Under'
    if row.get('side') == favored:
        return f"Research forecast: validation projects this count {abs(b['relative'])*100:.0f}% too {'high' if favored == 'Over' else 'low'}"
    return None


def skill_reason(audit):
    interval = (audit.get('brier_difference') or {}).get('ci95')
    if interval is not None and not interval[1] < 0:
        return 'Research forecast: accuracy gain over the empirical baseline is not established'
    return None


def reasons(audit, row):
    """All gate failures for one candidate, in decision order. Empty means eligible."""
    found = [skill_reason(audit), bias_reason(audit, row), range_reason(audit, row)]
    return [r for r in found if r]


def summary(audit, market):
    """Per-market diagnostics for the report and the Model Results page."""
    bins = audit.get('calibration_bins') or []
    flagged = []
    for b in bins:
        if b['n'] < BIN_MIN_N:
            flagged.append(dict(range=[int(b['predicted']*10)/10, int(b['predicted']*10)/10+.1], issue='too_few_outcomes', n=b['n']))
            continue
        low, high = wilson(round(b['observed']*b['n']), b['n'])
        if b['predicted']-b['observed'] > GAP and b['predicted'] > high:
            flagged.append(dict(range=[int(b['predicted']*10)/10, int(b['predicted']*10)/10+.1], issue='reference_side_overstated', n=b['n'],
                                predicted=b['predicted'], observed=b['observed'], observed_ci95=[round(low, 4), round(high, 4)]))
        elif b['observed']-b['predicted'] > GAP and b['predicted'] < low:
            flagged.append(dict(range=[int(b['predicted']*10)/10, int(b['predicted']*10)/10+.1], issue='opposite_side_overstated', n=b['n'],
                                predicted=b['predicted'], observed=b['observed'], observed_ci95=[round(low, 4), round(high, 4)]))
    blocked = []
    b = bias(audit)
    if market in COUNT_MARKETS and b and abs(b['relative']) > MEAN_BIAS and (
            (b['ci95'] is not None and not b['ci95'][0] <= 0 <= b['ci95'][1]) or
            (b['ci95'] is None and (audit.get('forecasts') or 0) >= MIN_FORECASTS_FOR_BIAS)):
        blocked.append('Over' if b['relative'] > 0 else 'Under')
    skill = skill_reason(audit)
    return dict(aggregate_passed=bool(audit.get('passed')), skill_established=None if (audit.get('brier_difference') or {}).get('ci95') is None else skill is None,
                mean_bias=b, sides_blocked_by_bias=blocked, flagged_ranges=flagged,
                basis='empirical baseline only; no bookmaker-price comparison', thresholds=dict(bin_min_n=BIN_MIN_N, gap=GAP, mean_bias=MEAN_BIAS))
