"""NBA v1: lagged team and player states, a joint score model and a minutes x rate player model.

Everything a forecast uses is computed from games strictly before its Eastern game date;
games on the same date are featurized before any of them updates a state.

Game: possessions (pace) x points per possession from exponentially weighted offensive and
defensive efficiency, carried across seasons with shrinkage. Two ridge regressions map those
expectations, rest and back-to-backs to regulation points; their residuals form a bivariate
normal on integer regulation scores. Tied regulation mass plays empirical overtime periods,
so moneyline, spread and total all integrate one score matrix (push mass on whole lines).

Player: projected minutes (ridge on recency, role, rest, team change and blowout risk from
the game model's margin) x per-minute rates shrunk to a position prior, adjusted for pace and
the opponent's allowed rates. Counts are gamma-Poisson given minutes. Combos are simulated
jointly: one minutes draw and correlated gamma intensities (Gaussian copula) per draw.
Forecasts are conditional on the player appearing, as books void props for players who don't.
"""
from collections import defaultdict, deque
from datetime import date
import hashlib
import math

import numpy as np
from scipy import stats as sps

VERSION = 'nba-v1.0'
STATS = ('PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'TOV')
MARKET_STATS = {
    'player_points': ('PTS',), 'player_rebounds': ('REB',), 'player_assists': ('AST',),
    'player_threes': ('FG3M',), 'player_blocks': ('BLK',), 'player_steals': ('STL',),
    'player_turnovers': ('TOV',), 'player_points_rebounds_assists': ('PTS', 'REB', 'AST'),
    'player_points_rebounds': ('PTS', 'REB'), 'player_points_assists': ('PTS', 'AST'),
    'player_rebounds_assists': ('REB', 'AST'),
}
POSITION_GROUP = {'PG': 'G', 'SG': 'G', 'G': 'G', 'SF': 'F', 'PF': 'F', 'F': 'F', 'C': 'C'}
GAME_FEATURES = ('naive_home', 'naive_away', 'home_b2b', 'away_b2b', 'home_rest', 'away_rest', 'neutral')
MINUTE_FEATURES = ('m_short', 'm_long', 'm_last', 'start_rate', 'log_n', 'team_changed',
                   'long_layoff', 'b2b', 'blowout')
MIN_APPEARANCES = 3
SCORE_MIN, SCORE_MAX = 40, 200


STATE_TYPES = ('regular', 'cup_final')


def state_history(history):
    """Games that feed model states. Playoff and play-in games score 12-17 points lower in
    total than the regular season (2023-26) and would otherwise be the last games each
    team carries into a new season, biasing opening-night totals low."""
    return {k: [r for r in v if r.get('SEASON_TYPE') in STATE_TYPES] if isinstance(v, list) and v and isinstance(v[0], dict) else v
            for k, v in history.items()}


def possessions(row):
    if any(row.get(k) is None for k in ('FGA', 'OREB', 'TOV', 'FTA')):
        return None
    return row['FGA'] - row['OREB'] + row['TOV'] + 0.44 * row['FTA']


def day_of(value):
    return date.fromisoformat(str(value)[:10])


def season_key(day):
    return day.year if day.month >= 7 else day.year - 1


class EW:
    """Exponentially weighted sums with one decay step per observation."""
    __slots__ = ('decay', 'w', 'sums')

    def __init__(self, half_life, keys):
        self.decay = 0.5 ** (1 / half_life)
        self.w = 0.0
        self.sums = dict.fromkeys(keys, 0.0)

    def add(self, values, weight=1.0):
        self.w = self.decay * self.w + weight
        for key, value in values.items():
            self.sums[key] = self.decay * self.sums[key] + value

    def scale(self, factor):
        self.w *= factor
        for key in self.sums:
            self.sums[key] *= factor


TEAM_KEYS = ('poss', 'pts_for', 'pts_against', 'poss_raw') + tuple('allow_' + s for s in STATS)


class Teams:
    """Lagged team pace/efficiency and opponent allowances, with league references."""

    def __init__(self, half_life=20, carry=0.5, prior_games=5):
        self.half_life, self.carry, self.prior_games = half_life, carry, prior_games
        self.state, self.last_day, self.season = {}, {}, {}
        self.league = EW(600, TEAM_KEYS)

    def _team(self, tid):
        if tid not in self.state:
            self.state[tid] = EW(self.half_life, TEAM_KEYS)
        return self.state[tid]

    def rating(self, tid):
        lg, k = self.league, self.prior_games
        if lg.w <= 0:
            return None
        L = {key: lg.sums[key] / lg.w for key in TEAM_KEYS}
        t = self.state.get(tid) or EW(self.half_life, TEAM_KEYS)
        mix = {key: (t.sums[key] + k * L[key]) / (t.w + k) for key in TEAM_KEYS}
        return dict(pace=mix['poss'], off=mix['pts_for'] / mix['poss_raw'], deff=mix['pts_against'] / mix['poss_raw'],
                    allow={s: mix['allow_' + s] / mix['poss_raw'] for s in STATS})

    def league_ref(self):
        lg = self.league
        L = {key: lg.sums[key] / lg.w for key in TEAM_KEYS}
        return dict(pace=L['poss'], ppp=L['pts_for'] / L['poss_raw'],
                    allow={s: L['allow_' + s] / L['poss_raw'] for s in STATS})

    def rest(self, tid, day):
        last = self.last_day.get(tid)
        return 4 if last is None else max(0, min(4, (day - last).days))

    def features(self, home, away, day, neutral=False):
        h, a = self.rating(home), self.rating(away)
        if not h or not a:
            return None
        L = self.league_ref()
        pace = h['pace'] * a['pace'] / L['pace']
        ppp_h, ppp_a = h['off'] * a['deff'] / L['ppp'], a['off'] * h['deff'] / L['ppp']
        rh, ra = self.rest(home, day), self.rest(away, day)
        return dict(naive_home=pace * ppp_h, naive_away=pace * ppp_a, home_b2b=float(rh == 1),
                    away_b2b=float(ra == 1), home_rest=float(rh), away_rest=float(ra),
                    neutral=float(bool(neutral)), pace=pace, home_pace=h['pace'], away_pace=a['pace'],
                    league_pace=L['pace'], allow_home=h['allow'], allow_away=a['allow'], league_allow=L['allow'])

    def update(self, day, home_row, away_row, periods):
        """One completed game; possessions are scaled to 48 minutes for pace."""
        poss = [possessions(home_row), possessions(away_row)]
        if None in poss:
            return
        game_poss = sum(poss) / 2
        minutes = 48 + 5 * max(0, (periods or 4) - 4)
        for own, other in ((home_row, away_row), (away_row, home_row)):
            tid = own['TEAM_ID']
            season = season_key(day)
            t = self._team(tid)
            if self.season.get(tid) not in (None, season):
                t.scale(self.carry)  # new season: shrink last season toward the league
            self.season[tid] = season
            values = dict(poss=game_poss * 48 / minutes, poss_raw=game_poss, pts_for=own['PTS'], pts_against=other['PTS'])
            values.update({'allow_' + s: float(other.get(s) or 0) for s in STATS})
            t.add(values)
            self.league.add(values)
            self.last_day[tid] = day


def regulation(row):
    periods = row.get('PERIOD_PTS')
    return sum(periods[:4]) if periods and len(periods) >= 4 else None


def game_table(history, teams_kwargs):
    """Chronological pregame features for every game, plus overtime periods observed."""
    teams = Teams(**teams_kwargs)
    rows_by_game = defaultdict(dict)
    for row in history['teams']:
        rows_by_game[row['GAME_ID']]['home' if row['HOME'] else 'away'] = row
    games = sorted(history['games'], key=lambda g: (g['GAME_DATE'], g['GAME_ID']))
    by_day = defaultdict(list)
    for g in games:
        if len(rows_by_game[g['GAME_ID']]) == 2:
            by_day[g['GAME_DATE']].append(g)
    out, overtime = [], []
    for day_text in sorted(by_day):
        day = day_of(day_text)
        todays = by_day[day_text]
        for g in todays:
            pair = rows_by_game[g['GAME_ID']]
            f = teams.features(g['HOME_TEAM_ID'], g['AWAY_TEAM_ID'], day, g.get('NEUTRAL'))
            if f:
                out.append(dict(game=g, features=f, home_reg=regulation(pair['home']), away_reg=regulation(pair['away']),
                                home_pts=pair['home']['PTS'], away_pts=pair['away']['PTS']))
        for g in todays:
            pair = rows_by_game[g['GAME_ID']]
            teams.update(day, pair['home'], pair['away'], g.get('PERIODS'))
            hp, ap = pair['home'].get('PERIOD_PTS'), pair['away'].get('PERIOD_PTS')
            if hp and ap and len(hp) == len(ap) > 4:
                overtime.append(dict(day=day_text, periods=[(h, a) for h, a in zip(hp[4:], ap[4:])]))
    return out, overtime, teams


def design(rows, keys):
    return np.array([[r['features'][k] for k in keys] for r in rows], dtype=float)


def fit_ridge(X, y, alpha=1.0):
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1
    Z = (X - mean) / scale
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    penalty = alpha * np.eye(Z1.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(Z1.T @ Z1 + penalty, Z1.T @ y)
    return dict(mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist())


def predict_ridge(model, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Z = (X - np.array(model['mean'])) / np.array(model['scale'])
    return model['beta'][0] + Z @ np.array(model['beta'][1:])


def fit_game(rows, overtime, features=GAME_FEATURES):
    rows = [r for r in rows if r['home_reg'] is not None and r['away_reg'] is not None]
    X = design(rows, features)
    home = fit_ridge(X, np.array([r['home_reg'] for r in rows], float))
    away = fit_ridge(X, np.array([r['away_reg'] for r in rows], float))
    resid = np.column_stack([np.array([r['home_reg'] for r in rows]) - predict_ridge(home, X),
                             np.array([r['away_reg'] for r in rows]) - predict_ridge(away, X)])
    cov = np.cov(resid.T)
    pairs = defaultdict(int)
    for game in overtime:
        for pair in game['periods']:
            pairs[pair] += 1
    total = sum(pairs.values())
    ot = [[h, a, n / total] for (h, a), n in sorted(pairs.items())]
    out = dict(home=home, away=away, cov=cov.tolist(), overtime=ot, train_games=len(rows))
    if tuple(features) != GAME_FEATURES:
        out['features'] = list(features)  # ablations only; production uses GAME_FEATURES
    return out


def game_means(params, features):
    x = [[features[k] for k in params.get('features', GAME_FEATURES)]]
    return float(predict_ridge(params['home'], x)[0]), float(predict_ridge(params['away'], x)[0])


def score_matrix(params, features):
    """P(final home score, final away score) on [SCORE_MIN, SCORE_MAX]^2, overtime included."""
    mh, ma = game_means(params, features)
    cov = np.array(params['cov'])
    grid = np.arange(SCORE_MIN, SCORE_MAX + 1)
    H, A = np.meshgrid(grid, grid, indexing='ij')
    pts = np.stack([H - mh, A - ma], axis=-1)
    inv = np.linalg.inv(cov)
    density = np.exp(-0.5 * np.einsum('...i,ij,...j->...', pts, inv, pts))
    reg = density / density.sum()
    size = len(grid)
    final = reg.copy()
    tie = np.zeros_like(reg)
    np.fill_diagonal(tie, np.diag(reg))
    np.fill_diagonal(final, 0)
    ot = params['overtime']
    for _ in range(4):
        if tie.sum() < 1e-9:
            break
        nxt = np.zeros_like(reg)
        for dh, da, p in ot:
            shifted = np.zeros_like(reg)
            shifted[dh:, da:] = tie[:size - dh, :size - da]
            nxt += p * shifted
        new_tie = np.zeros_like(reg)
        np.fill_diagonal(new_tie, np.diag(nxt))
        np.fill_diagonal(nxt, 0)
        final += nxt
        tie = new_tie
    if tie.sum() > 0:  # Residual quadruple-overtime mass: split evenly by one point.
        d = np.diag(tie)
        final[np.arange(1, size), np.arange(size - 1)] += d[:-1] / 2
        final[np.arange(size - 1), np.arange(1, size)] += d[:-1] / 2
    return final / final.sum(), mh, ma


def game_market(matrix, market, side_is_home, line):
    """(win, push) for one game-market side from the score matrix."""
    grid = np.arange(SCORE_MIN, SCORE_MAX + 1)
    H, A = np.meshgrid(grid, grid, indexing='ij')
    if market == 'h2h':
        margin = H - A if side_is_home else A - H
        return float(matrix[margin > 0].sum()), 0.0
    if market == 'spreads':
        margin = (H - A if side_is_home else A - H) + line
    else:  # totals; side_is_home means Over
        margin = (H + A - line) if side_is_home else (line - H - A)
    return float(matrix[margin > 0].sum()), float(matrix[margin == 0].sum())


def game_median(matrix, market, side_is_home):
    """Median total, or median margin for the side, from the score matrix."""
    grid = np.arange(SCORE_MIN, SCORE_MAX + 1)
    H, A = np.meshgrid(grid, grid, indexing='ij')
    values = (H + A if market == 'totals' else H - A if side_is_home else A - H).ravel()
    order = np.argsort(values, kind='stable')
    return float(values[order][np.searchsorted(np.cumsum(matrix.ravel()[order]), .5)])


# ---------------------------------------------------------------- players

class Player:
    __slots__ = ('n', 'short', 'long', 'last_min', 'starts', 'rates', 'team', 'last_day', 'position', 'recent')

    def __init__(self, rate_half_lives):
        self.n = 0
        self.short, self.long = EW(5, ('m',)), EW(20, ('m',))
        self.starts = EW(5, ('s',))
        self.rates = {h: EW(h, ('MIN',) + STATS) for h in rate_half_lives}
        self.last_min, self.team, self.last_day, self.position = None, None, None, None
        self.recent = deque(maxlen=30)

    def add(self, row, day):
        m = float(row['MIN'])
        self.n += 1
        self.short.add(dict(m=m))
        self.long.add(dict(m=m))
        self.starts.add(dict(s=float(row.get('STARTER') is True)))
        for ew in self.rates.values():
            ew.add({'MIN': m, **{s: float(row.get(s) or 0) for s in STATS}})
        self.last_min, self.team, self.last_day = m, row['TEAM_ID'], day
        self.recent.append({st: float(row.get(st) or 0) for st in STATS})
        self.position = POSITION_GROUP.get(row.get('POSITION'), self.position)


def recent_means(p):
    games = list(p.recent)
    return {w: {st: sum(g[st] for g in games[-w:]) / len(games[-w:]) for st in STATS} for w in (10, 30)}


def minute_features(p, team_id, day, b2b, abs_margin):
    return dict(m_short=p.short.sums['m'] / p.short.w, m_long=p.long.sums['m'] / p.long.w, m_last=p.last_min,
                start_rate=p.starts.sums['s'] / p.starts.w, log_n=math.log1p(p.n),
                team_changed=float(team_id != p.team), long_layoff=float((day - p.last_day).days > 10),
                b2b=float(b2b), blowout=abs_margin * (p.short.sums['m'] / p.short.w) / 48)


def player_table(history, game_rows, rate_half_lives=(20, 50)):
    """Pregame player snapshots for each appearance (regular-season targets flagged)."""
    by_game = {r['game']['GAME_ID']: r for r in game_rows}
    apps = defaultdict(list)
    for row in history['players']:
        if not row.get('DNP') and row.get('MIN'):
            apps[row['GAME_DATE']].append(row)
    players, out = {}, []
    for day_text in sorted(apps):
        day = day_of(day_text)
        for row in apps[day_text]:
            p = players.get(row['PLAYER_ID'])
            g = by_game.get(row['GAME_ID'])
            if p is None or p.n < MIN_APPEARANCES or g is None or not all(row.get(s) is not None for s in STATS):
                continue
            home = row['HOME']
            f = g['features']
            rest = f['home_rest'] if home else f['away_rest']
            mh, ma = g.get('mean_home'), g.get('mean_away')
            margin = abs(mh - ma) if mh is not None else 0.0
            out.append(dict(
                row=row, minutes=minute_features(p, row['TEAM_ID'], day, rest == 1, margin),
                rates={h: {k: ew.sums[k] for k in ('MIN',) + STATS} for h, ew in p.rates.items()},
                recent=recent_means(p),
                position=p.position or POSITION_GROUP.get(row.get('POSITION')) or 'F',
                pace=f['pace'], own_pace=f['home_pace'] if home else f['away_pace'], league_pace=f['league_pace'],
                allow=f['allow_away'] if home else f['allow_home'], league_allow=f['league_allow'],
                season_type=row['SEASON_TYPE']))
        for row in apps[day_text]:
            players.setdefault(row['PLAYER_ID'], Player(rate_half_lives)).add(row, day)
    return out, players


def position_priors(snapshots):
    sums = defaultdict(lambda: defaultdict(float))
    for s in snapshots:
        r = s['row']
        for k in ('MIN',) + STATS:
            sums[s['position']][k] += float(r[k] if k != 'MIN' else r['MIN'])
    return {pos: {st: v[st] / v['MIN'] for st in STATS} for pos, v in sums.items()}


def rate_means(s, cfg, priors, scale=None):
    """Per-minute expected rate for each stat in this game (before minutes)."""
    ew = s['rates'][cfg['rate_half_life']]
    k = cfg['prior_minutes']
    prior = priors.get(s['position']) or priors['F']
    out = {}
    pace_factor = s['pace'] / s['own_pace']
    for st in STATS:
        rate = (ew[st] + k * prior[st]) / (ew['MIN'] + k)
        opp = (s['allow'][st] / s['league_allow'][st]) if s['league_allow'][st] > 0 else 1.0
        out[st] = rate * pace_factor * opp ** cfg['opponent_weight'] * (scale or {}).get(st, 1.0)
    return out


def projected_minutes(params, minutes_features):
    x = [[minutes_features[k] for k in MINUTE_FEATURES]]
    return float(np.clip(predict_ridge(params['minutes'], x)[0], 1, 46))


def stack_inputs(s, params, minutes_mu):
    r = rate_means(s, params['config'], params['priors'], params['scale'])
    return {st: [1.0, r[st] * minutes_mu, s['recent'][10][st], s['recent'][30][st]] for st in STATS}


def final_rates(s, params, minutes_mu):
    """Per-minute rates after the training-fitted blend with recent box-score averages."""
    inputs = stack_inputs(s, params, minutes_mu)
    return {st: max(float(np.dot(params['stack'][st], inputs[st])), 0.02) / minutes_mu for st in STATS}


def fit_dispersion(snapshots, rates, nodes):
    """Gamma shape per stat by maximum likelihood of the full predictive mixture (minutes
    uncertainty included), and Gaussian-copula correlations from residuals at actual minutes.

    ``rates`` is one per-minute rate dict per snapshot; ``nodes`` holds minutes quantiles.
    """
    shape = {}
    from scipy.optimize import minimize_scalar
    for st in STATS:
        y = np.array([s['row'][st] for s in snapshots])[:, None]
        means = np.maximum(np.array([r[st] for r in rates])[:, None] * nodes, 1e-9)

        def loss(log_k):
            k = math.exp(log_k)
            return -np.mean(np.log(sps.nbinom.pmf(y, k, k / (k + means)).mean(axis=1) + 1e-300))
        shape[st] = float(math.exp(minimize_scalar(loss, bounds=(math.log(0.5), math.log(2000)), method='bounded',
                                                   options=dict(xatol=0.02)).x))
    resid, mus = defaultdict(list), defaultdict(list)
    for s, r in zip(snapshots, rates):
        m = float(s['row']['MIN'])
        for st in STATS:
            resid[st].append(s['row'][st] - r[st] * m)
            mus[st].append(r[st] * m)
    corr = np.eye(len(STATS))
    for i, a in enumerate(STATS):
        for j, b in enumerate(STATS):
            if j <= i:
                continue
            ea, eb = np.array(resid[a]), np.array(resid[b])
            ma, mb = np.array(mus[a]), np.array(mus[b])
            cov_g = (ea * eb).sum() / (ma * mb).sum()
            corr[i, j] = corr[j, i] = float(np.clip(cov_g * math.sqrt(shape[a] * shape[b]), -0.9, 0.9))
    vals, vecs = np.linalg.eigh(corr)
    corr = vecs @ np.diag(np.clip(vals, 1e-3, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return shape, corr.tolist()


def minutes_sd(snapshots, model):
    """Residual SD of minutes by 4-minute bins of the projection (interpolated)."""
    X = np.array([[s['minutes'][k] for k in MINUTE_FEATURES] for s in snapshots])
    pred = predict_ridge(model, X)
    actual = np.array([float(s['row']['MIN']) for s in snapshots])
    edges = np.arange(0, 49, 4)
    centers, sds = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pred >= lo) & (pred < hi)
        if mask.sum() >= 50:
            centers.append((lo + hi) / 2)
            sds.append(float(np.sqrt(np.mean((actual[mask] - pred[mask]) ** 2))))
    return dict(centers=centers, sds=sds)


def minute_distribution(params, minutes_features, nodes=21):
    """Quantile nodes of the projected minutes (normal truncated to (0, 48])."""
    x = [[minutes_features[k] for k in MINUTE_FEATURES]]
    mu = float(np.clip(predict_ridge(params['minutes'], x)[0], 1, 46))
    sd = float(np.interp(mu, params['minutes_sd']['centers'], params['minutes_sd']['sds']))
    a, b = (0.5 - mu) / sd, (48 - mu) / sd
    q = (np.arange(nodes) + 0.5) / nodes
    return mu, sd, sps.truncnorm.ppf(q, a, b, loc=mu, scale=sd)


def stat_pmf(rate, shape, minute_nodes, support):
    """P(Y = y) mixing gamma-Poisson over the minutes nodes."""
    pmf = np.zeros(len(support))
    for m in minute_nodes:
        mu = max(rate * m, 1e-9)
        pmf += sps.nbinom.pmf(support, shape, shape / (shape + mu))
    return pmf / len(minute_nodes)


def simulate(params, rates, minutes_mu, minutes_sd_value, n=20000, seed=0):
    """Joint draws of every stat: shared minutes, Gaussian-copula gamma intensities."""
    rng = np.random.default_rng(seed)
    a, b = (0.5 - minutes_mu) / minutes_sd_value, (48 - minutes_mu) / minutes_sd_value
    minutes = sps.truncnorm.rvs(a, b, loc=minutes_mu, scale=minutes_sd_value, size=n, random_state=rng)
    z = rng.multivariate_normal(np.zeros(len(STATS)), np.array(params['correlation']), size=n)
    u = np.clip(sps.norm.cdf(z), 1e-12, 1 - 1e-12)
    draws = {}
    for i, st in enumerate(STATS):
        k = params['shape'][st]
        intensity = sps.gamma.ppf(u[:, i], k, scale=1 / k)
        draws[st] = rng.poisson(rates[st] * minutes * intensity)
    return draws, minutes


def seed_for(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def calibrate(p, slope):
    """Logit-slope recalibration of a binary probability (slope < 1 pulls toward 50%)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-slope * np.log(p / (1 - p))))


def player_market(draws, market, side, line, slope=1.0):
    value = sum(draws[s] for s in MARKET_STATS[market])
    over, push = float((value > line).mean()), float((value == line).mean())
    if push < 1 and slope != 1.0:
        # Calibrated on the over/under decision; push mass is left as simulated.
        over = float(calibrate(over / (1 - push), slope)) * (1 - push)
    under = 1 - over - push
    return (over if side == 'Over' else under), push, float(value.mean())


def player_median(draws, market):
    """Smallest simulated value with at least half the draws at or below it (lines sit near it)."""
    return float(np.quantile(sum(draws[s] for s in MARKET_STATS[market]), .5, method='inverted_cdf'))
