"""Explain the existing NFL forecast without changing its estimates or ranking."""
import hashlib
import json
import math
from statistics import NormalDist


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    return value


FAMILY_SAMPLE = {'rush': ['season', 'week', 'opponent_team', 'carries', 'rushing_yards'],
                 'receive': ['season', 'week', 'opponent_team', 'targets', 'receptions', 'receiving_yards']}


def family_trace(logs, latents, family, players, career_available):
    """Rushing and receiving: the recent sample and the latent components behind each mean."""
    from career_baseline import K_RECENT_MU
    import pandas as pd
    result = {}
    for player in players:
        recent = logs[logs.player.eq(player)].sort_index().tail(4) if logs is not None and not logs.empty else pd.DataFrame()
        columns = [c for c in FAMILY_SAMPLE[family] if c in recent]
        part = lambda key: float(latents[key][player]) if key in latents and player in latents[key].index else None
        parts = (dict(carries=part('volume_mu'), yards_per_carry=part('ypc_mu')) if family == 'rush' else
                 dict(targets=part('volume_mu'), catch_rate=part('cr_mu'), yards_per_reception=part('ypr_mu')))
        result[player] = clean(dict(version='nfl-projection-trace-1', family=family,
            current_sample=recent[columns].to_dict('records'),
            recent_mean_weight=len(recent)/(len(recent)+K_RECENT_MU) if career_available else None, **parts))
    return result


def opponent_trace(market, player, opponent_map, ratings):
    """The opposing defense the projection used: team, rating and rank (1 = toughest).

    Yards allowed per game, the league average and the data season are shown beside the
    rating when the ratings frame carries them.
    """
    team = (opponent_map or {}).get(player)
    if not team or ratings is None or getattr(ratings, 'empty', True) or team not in ratings.index:
        return None
    kind = 'rush' if market in ('rush_yds', 'rush_attempts') else 'pass'
    values = ratings[f'{kind}_def_rating'].dropna()
    if team not in values.index:
        return None
    out = dict(team=team, kind=kind, rating=float(values[team]),
               rank=int((values > values[team]).sum()) + 1, of=int(len(values)))
    allowed = f'{kind}_yds_per_game'
    if allowed in ratings and ratings[allowed].notna().get(team, False):
        out.update(allowed=float(ratings.at[team, allowed]), league=float(ratings[allowed].mean()))
    for key in ('games', 'season'):
        if key in ratings and ratings[key].notna().get(team, False):
            out[key] = int(ratings.at[team, key])
    return clean(out)


def passing_trace(logs, career, latents, season, players):
    """Capture the actual input sample and latent means used by this run."""
    from career_baseline import compute_position_pool, player_career_baseline, K_RECENT_MU
    import pandas as pd
    result = {}
    prior = career[career.season < season] if career is not None and not career.empty else pd.DataFrame()
    specs = [('attempts', 'attempts', None), ('yards_per_completion', 'passing_yards', 'completions')]
    pools = {label: compute_position_pool(career, 'QB', num, den) for label, num, den in specs} if not prior.empty else {}
    for player in players:
        recent = logs[logs.player.eq(player)].sort_index().tail(4) if logs is not None and not logs.empty else pd.DataFrame()
        base = {}
        if not prior.empty:
            own = prior[prior.player.eq(player)]
            for label, num, den in specs:
                pool = pools[label]
                mu, sigma, n = player_career_baseline(own, season, num, den, *pool)
                base[label] = dict(mean=mu, games=n, position_pool_mean=pool[0])
        columns = [c for c in ['season', 'week', 'opponent_team', 'attempts', 'completions', 'passing_yards'] if c in recent]
        result[player] = clean(dict(
            version='nfl-projection-trace-1', current_sample=recent[columns].to_dict('records'),
            recent_mean_weight=len(recent)/(len(recent)+K_RECENT_MU) if base else None,
            yards_per_completion_recent_weight=(int((recent.completions > 0).sum()) /
                (int((recent.completions > 0).sum())+K_RECENT_MU)) if base and 'completions' in recent else None,
            career_baselines=base,
            attempts=float(latents['volume_mu'][player]),
            completion_rate=float(latents['comp_pct_mu'][player]),
            yards_per_completion=float(latents['ypc_mu'][player]),
            limitations=['Appearance totals do not distinguish partial games from normal starter workloads.',
                        'Completion rate uses recent games without career shrinkage.',
                        'Variance omits completion-rate uncertainty and dependence among components.']))
    return result


def calibration_trace(market, raw, calibration):
    if not calibration or not math.isfinite(raw):
        return None
    if market not in calibration.get('_eligible_markets', []):
        return dict(status='not_calibrated')
    curve = calibration.get('markets', {}).get(market, calibration.get('_pooled_fallback'))
    if not curve:
        return None
    return clean(dict(version='nfl-calibration-trace-1',
        artifact_sha256=hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest(),
        fitted_season=calibration.get('_meta', {}).get('fitted_on_season') or calibration.get('_meta', {}).get('fitted_on_seasons'),
        fitted_weeks=calibration.get('_meta', {}).get('fitted_on_weeks'),
        total_graded_rows=calibration.get('_meta', {}).get('n_graded_picks') or calibration.get('_meta', {}).get('n_rows'),
        market_sample_size=(calibration.get('_market_counts', {}).get(market) or {}).get('rows'),
        market_games=(calibration.get('_market_counts', {}).get(market) or {}).get('games'), tail_sample_size=None,
        calibration_run=(calibration.get('_provenance') or {}).get('run_id'),
        model_version=(calibration.get('_provenance') or {}).get('model_version'),
        curve='market_specific' if market in calibration.get('markets', {}) else 'pooled',
        raw_probability=raw, fitted_raw_range=[min(curve['x']), max(curve['x'])],
        outside_fitted_range=raw < min(curve['x']) or raw > max(curve['x']),
        endpoint_probabilities=[curve['y'][0], curve['y'][-1]],
        limitation='Endpoint clipping is not held-out evidence; total rows are not independent games or tail sample size.'))


def unpack(value):
    try:
        return json.loads(value) if isinstance(value, str) else None
    except ValueError:
        return None


def context_key(row):
    return json.dumps([str(row.get('game_id')), row.get('player'), row.get('market_std')], separators=(',', ':'))


def build_context(frame, generated_at):
    """Exact snapshot context, including offered-book ladder and other-book lines.

    Raw Normal sensitivity is a stress calculation, never an adjusted forecast.
    Book rows retain line, side, price and timestamp; no cross-line de-vig.
    """
    import pandas as pd
    from market_math import add_market_comparisons
    d = add_market_comparisons(frame)
    # Diagnostic pairing adds an explicit five-minute timestamp window; stale or
    # missing opposite quotes cannot count as a paired comparison.
    timestamps = pd.to_datetime(d['last_update'], utc=True, errors='coerce')
    pair = d.assign(_quote_time=timestamps).groupby(['game_id','player','market_std','point','bookmaker'], dropna=False)['_quote_time']
    valid_pair = (pair.transform('max')-pair.transform('min')).dt.total_seconds().le(300) & pair.transform('count').eq(pair.transform('size'))
    d.loc[~valid_pair, 'prob_devig'] = float('nan')
    d = d[pd.to_numeric(d['point'], errors='coerce').notna() & d['model_prob'].notna()]
    groups = {}
    for _, rows in d.groupby(['game_id', 'player', 'market_std'], dropna=False):
        first = rows.iloc[0]
        fields = ['bookmaker', 'name', 'point', 'price', 'last_update', 'prob_devig']
        quotes = json.loads(rows[fields].drop_duplicates().to_json(orient='records'))
        groups[context_key(first)] = dict(
            projection=unpack(first.get('projection_diagnostics')),
            calibration=unpack(first.get('calibration_diagnostics')),
            quotes=quotes, mu=clean(float(first.mu)) if pd.notna(first.get('mu')) else None,
            sigma=clean(float(first.sigma)) if pd.notna(first.get('sigma')) else None,
            offers={})
        group = groups[context_key(first)]
        for _, r in rows.iterrows():
            if pd.isna(r.price) or abs(r.price) < 100 or pd.isna(r.last_update):
                continue
            offer = json.dumps([r.bookmaker, r['name'], float(r.point), int(r.price), r.last_update], separators=(',', ':'))
            group['offers'][offer] = dict(raw_probability=clean(float(r.model_prob_raw)) if pd.notna(r.get('model_prob_raw')) else None)
    return dict(schema_version=1, generated_at=generated_at, groups=groups)


def review_diagnostics(row, context, forecast_at):
    """Small, bound-to-forecast explanation packet; explicit missing provenance."""
    if not context or context.get('schema_version') != 1 or context.get('generated_at') != forecast_at:
        return None
    group = context.get('groups', {}).get(context_key(row))
    if not group:
        return None
    key = json.dumps([row['book'], row['side'], float(row['line']), int(row['price']), row['quoted_at']], separators=(',', ':'))
    if key not in group['offers']:
        return None
    quotes = group['quotes']
    # Other quotes must also be contemporaneous with the actual reviewed offer.
    from datetime import datetime
    at = datetime.fromisoformat(row['quoted_at'].replace('Z', '+00:00'))
    def contemporary(quote):
        try:
            qt = datetime.fromisoformat(quote['last_update'].replace('Z', '+00:00'))
            return abs((at-qt).total_seconds()) <= 300
        except (TypeError, ValueError, AttributeError):
            return False
    quotes = [q for q in quotes if contemporary(q) and q['price'] is not None]
    own = [q for q in quotes if q['bookmaker'] == row['book'] and q['name'] == row['side']]
    # Show central and neighboring prices, including the actual selection.
    own.sort(key=lambda q: q['point'])
    own = sorted(own, key=lambda q: (abs(q['point']-row['line']), q['point']))[:4]
    centers = []
    for book in sorted({q['bookmaker'] for q in quotes}):
        choices = [q for q in quotes if q['bookmaker'] == book and q['name'] == row['side'] and q['prob_devig'] is not None]
        if choices:
            centers.append(min(choices, key=lambda q: abs(q['prob_devig']-.5)))
    own_center = next((q for q in centers if q['bookmaker'] == row['book']), None)
    other_exact = [q for q in quotes if q['bookmaker'] != row['book'] and q['name'] == row['side'] and q['point'] == row['line'] and q['prob_devig'] is not None]
    q = 100/(row['price']+100) if row['price'] > 0 else -row['price']/(100-row['price'])
    sigma, mu = group['sigma'], group['mu']
    median = row.get('consensus_line')
    stress = None
    if row['market_std'] in ('pass_yds', 'pass_attempts', 'pass_completions') and sigma and sigma > 0 and median is not None:
        # Half lines only: integer settlement requires a push-aware inversion.
        if not float(row['line']).is_integer():
            z = NormalDist().inv_cdf(q)
            limit = row['line'] + (1 if row['side'].lower() == 'over' else -1)*sigma*z
            p_under = NormalDist(median, sigma).cdf(row['line'])
            p = 1-p_under if row['side'].lower() == 'over' else p_under
            stress = clean(dict(basis='Uncalibrated Normal stress test; same sigma, market median used as hypothetical mean, not a forecast.',
                mean_at_break_even=limit, market_centered_mean=median,
                probability=p, ev_per_unit=p/q-1))
    calibration = dict(group['calibration']) if group.get('calibration') else None
    if calibration and 'fitted_raw_range' in calibration:
        raw = group['offers'][key]['raw_probability']
        calibration.update(raw_probability=raw, outside_fitted_range=None if raw is None else
                           raw < calibration['fitted_raw_range'][0] or raw > calibration['fitted_raw_range'][1])
    return clean(dict(projection=group['projection'], calibration=calibration,
        sigma=sigma, model_minus_median_line=mu-median if mu is not None and median is not None else None,
        offered_line_minus_median=row['line']-median if median is not None else None,
        offered_book_central_quote=own_center, offered_book_nearby_quotes=sorted(own, key=lambda q:q['point']),
        other_book_central_quotes=[c for c in centers if c['bookmaker'] != row['book']][:5],
        other_books_at_exact_line=len({q['bookmaker'] for q in other_exact}),
        median_is_not_expected_mean=True, raw_distribution_stress=stress))
