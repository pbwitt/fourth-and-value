"""One-week pregame test of NHL milestone props, priced by the frozen player model.

Books also sell an anytime goal scorer and alternate shots and points lines ("3+
shots"). During WINDOW the daily refresh asks for them alongside the usual props,
only for games starting within 24 hours: about 3 extra Odds API credits per game
per refresh, charged only for markets the books return. Each is the same contract
as a base prop (anytime scorer = goals over 0.5; 3+ shots = shots over 2.5), so
the same model, comparison and settlement rules apply unchanged.

Research only. These quotes never enter the public snapshot, Top Picks, Market
Watch or the candidate board. They are kept in artifacts/nhl/ladder-test and
graded by ladder_test.py.
"""
from datetime import date, datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from nba.pipeline import implied_probability, iso, timestamp
from .data import ROOT
from .pricing import settlement_basis

MARKETS = {'player_goal_scorer_anytime': 'player_goals',
           'player_shots_on_goal_alternate': 'player_shots_on_goal',
           'player_points_alternate': 'player_points'}
LABELS = {'player_goals': 'Goals', 'player_shots_on_goal': 'Shots on goal', 'player_points': 'Points'}
WINDOW = (date(2026, 10, 4), date(2026, 10, 11))   # Eastern dates; the end is exclusive
HORIZON = timedelta(hours=24)
OUT = ROOT / 'artifacts/nhl/ladder-test'
EASTERN = ZoneInfo('America/New_York')
KEEP = ['event_id', 'nhl_game_id', 'commence_time', 'game', 'home_team', 'away_team', 'book', 'offered_market',
        'offered_side', 'market', 'player', 'player_id', 'side', 'line', 'price', 'book_probability', 'quoted_at',
        'settlement_verified', 'offer_id', 'fair_probability', 'other_book_probability', 'other_books',
        'model_probability', 'conditional_probability', 'push_probability', 'estimated_ev', 'projected_mean',
        'model_status']


def active(now):
    return WINDOW[0] <= now.astimezone(EASTERN).date() < WINDOW[1]


def wanted(event, now):
    start = timestamp(event.get('commence_time'))
    return bool(start) and active(now) and timedelta(0) < start - now <= HORIZON


def quotes(event, now, rules):
    """Milestone outcomes as rows on their base market; the offered market and side are kept."""
    rows = []
    start = timestamp(event.get('commence_time'))
    if not start or start <= now:
        return rows
    home, away = event.get('home_team'), event.get('away_team')
    for book in event.get('bookmakers', []):
        policy = rules.get(book.get('key'), {})
        profile = policy.get('player')
        for market in book.get('markets', []):
            offered = market.get('key')
            base = MARKETS.get(offered)
            if not base:
                continue
            updated = timestamp(market.get('last_update') or book.get('last_update'))
            if not updated or not timedelta(minutes=-5) <= now - updated <= timedelta(hours=24):
                continue
            for outcome in market.get('outcomes', []):
                name = str(outcome.get('name', ''))
                player = str(outcome.get('description') or '').strip()
                probability = implied_probability(outcome.get('price'))
                if not player or not math.isfinite(probability):
                    continue
                if offered == 'player_goal_scorer_anytime':
                    if name not in ('Yes', 'No'):
                        continue
                    side, line = ('Over' if name == 'Yes' else 'Under'), 0.5
                else:
                    if name not in ('Over', 'Under'):
                        continue
                    try:
                        line = float(outcome.get('point'))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(line):
                        continue
                    side = name
                rows.append(dict(event_id=event['id'], commence_time=event['commence_time'], home_team=home,
                    away_team=away, game=f'{away} @ {home}', book=book['key'], book_label=book.get('title') or book['key'],
                    market=base, market_label=LABELS[base], offered_market=offered, offered_side=name, player=player,
                    side=side, line=line, price=float(outcome['price']), book_probability=float(probability),
                    quoted_at=iso(updated), nhl_game_id=event.get('nhl_game_id'), ingested_at=iso(now),
                    settlement_profile=profile or f"unverified:{book['key']}:{base}", settlement_verified=bool(profile),
                    **settlement_basis(policy, 'player', profile)))
    return rows


def _plain(value):
    # NumPy scalars become plain values; a missing estimate stays missing, never NaN.
    if hasattr(value, 'item') and not isinstance(value, (str, bool)):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def record(rows, state, now, out=OUT):
    """Price milestone quotes with the frozen model and keep a compact copy for grading."""
    if not rows:
        return None
    from .inference import annotate, bundle, live_history
    from .pricing import compare
    models, manifest = bundle()
    games, players, checked = live_history(now)
    decided = datetime.now(timezone.utc)
    priced = annotate(compare(rows, now), games, players, state['events'], models, manifest, decided, checked)
    snapshot = dict(snapshot_id=state.get('snapshot_id'), decided_at=iso(decided),
                    window=[d.isoformat() for d in WINDOW], model_version=manifest.get('version'),
                    model_sha256=manifest.get('artifact_sha256'),
                    rows=[{k: _plain(r.get(k)) for k in KEEP} for r in priced])
    path = Path(out) / f"{decided.strftime('%Y%m%dT%H%M%SZ')}-{str(state.get('snapshot_id') or 'snapshot')[:12]}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as f:
        f.write(json.dumps(snapshot, separators=(',', ':'), allow_nan=False).encode())
    return path
