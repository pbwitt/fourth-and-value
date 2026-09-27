"""Fixed prospective research screen, independent of Market Watch membership."""
from collections import Counter
from copy import deepcopy
from datetime import timedelta
import math
from zoneinfo import ZoneInfo

from .data import digest, iso, stamp
from .pricing import decimal

EASTERN = ZoneInfo('America/New_York')
MARKETS = {'h2h', 'spreads', 'totals', 'player_shots_on_goal',
           'player_goals', 'player_assists', 'player_points'}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def exclusion(row, now, config):
    """Fail closed on missing forecasts. Consensus is context, never an entry gate."""
    try:
        if row.get('market') not in MARKETS:
            return 'unsupported_market'
        if not all(row.get(k) for k in ('offer_id', 'forecast_id', 'nhl_game_id', 'model_version')):
            return 'missing_identity_or_forecast'
        if row['market'].startswith('player_') and not row.get('player_id'):
            return 'missing_player_identity'
        start = stamp(row['commence_time'])
        if start <= now or start.astimezone(EASTERN).date() != now.astimezone(EASTERN).date():
            return 'outside_today_slate'
        if not timedelta(0) <= now-stamp(row['quoted_at']) <= timedelta(minutes=config['quote_max_minutes']):
            return 'expired_quote'
        if not timedelta(0) <= now-stamp(row['model_data_checked_at']) < timedelta(hours=config['model_max_hours']):
            return 'expired_model_inputs'
        if not timedelta(0) <= now-stamp(row['decision_at']) <= timedelta(minutes=config['quote_max_minutes']):
            return 'expired_forecast'
        if not row.get('settlement_verified') or str(row.get('settlement_profile', 'unverified')).startswith('unverified'):
            return 'unverified_settlement'
        fields = ('independent_probability', 'final_probability', 'push_probability', 'loss_probability',
                  'estimated_ev', 'minimum_acceptable_decimal', 'rank_score')
        if not all(number(row.get(k)) for k in fields):
            return 'missing_probability_or_price'
        win, push, loss = (row[k] for k in ('final_probability', 'push_probability', 'loss_probability'))
        if min(win, push, loss) < 0 or max(win, push, loss) > 1 or abs(win+push+loss-1) > 1e-7:
            return 'incoherent_probability'
        if not 0 < row['independent_probability'] < 1 or not row.get('sensitivity'):
            return 'missing_independent_sensitivity'
        dec = decimal(row['price'])
        ev = win*(dec-1)-loss
        if abs(ev-row['estimated_ev']) > 1e-7:
            return 'incoherent_ev'
        if ev < config['minimum_ev'] or row['rank_score'] <= 0:
            return 'insufficient_model_value'
        if row['minimum_acceptable_decimal'] <= 1 or dec+1e-10 < row['minimum_acceptable_decimal']:
            return 'fails_sensitivity_price'
        if row.get('validation_status') in (None, 'unavailable'):
            return 'missing_validation_status'
    except (KeyError, ValueError, TypeError, OverflowError):
        return 'invalid_record'
    return None


def shortlist(state, now, config):
    if not isinstance(config['max_candidates'], int) or not 1 <= config['max_candidates'] <= 4:
        raise ValueError('The research policy supports one to four candidates')
    board = dict(schema_version=1, policy_version=config['policy_version'], generated_at=iso(now),
                 decision_date=now.astimezone(EASTERN).date().isoformat(),
                 session='morning' if now.astimezone(EASTERN).hour < 14 else 'afternoon',
                 source_snapshot_id=state.get('snapshot_id') or digest(state)[:24],
                 source_last_success_at=state.get('last_success_at'),
                 policy={k: config[k] for k in ('max_candidates', 'minimum_ev', 'quote_max_minutes', 'model_max_hours')},
                 validation_status='experimental_prospective_shadow', recommendations=[],
                 candidates=[], eligible_count=0, exclusions={}, review_status='not_requested',
                 status='ready')
    try:
        healthy = state.get('status') in ('ready', 'waiting_for_markets') and not state.get('model_error') and (
            timedelta(0) <= now-stamp(state['last_success_at']) <= timedelta(minutes=config['quote_max_minutes']))
    except (KeyError, TypeError, ValueError):
        healthy = False
    if not healthy:
        board.update(status='unavailable', review_status='feed_unavailable')
    else:
        counts, eligible = Counter(), []
        for row in state.get('rows', []):
            reason = exclusion(row, now, config)
            if reason:
                counts[reason] += 1
            else:
                eligible.append(row)
        # Rank is the worst scenario's fixed-fraction log growth, not payout or nominal EV.
        # One/game also prevents opposing, duplicate-book and related player exposures.
        seen = set()
        for row in sorted(eligible, key=lambda r: (-r['rank_score'], -decimal(r['price']), r['offer_id'])):
            if row['nhl_game_id'] in seen:
                counts['same_game_exposure'] += 1
                continue
            if len(board['candidates']) >= config['max_candidates']:
                counts['shortlist_limit'] += 1
                continue
            seen.add(row['nhl_game_id'])
            candidate = deepcopy(row)
            candidate.update(candidate_id=digest([config['policy_version'], row['offer_id'], row['forecast_id']])[:24],
                             candidate_rank=len(board['candidates'])+1, recommendation=False,
                             candidate_status='experimental_candidate', human_decision='unreviewed',
                             qualitative_review=None)
            board['candidates'].append(candidate)
        board.update(eligible_count=len(eligible), exclusions=dict(counts))
        if not eligible:
            board['status'] = 'no_candidates'
            board['review_status'] = 'no_candidates'
    board['board_id'] = digest(board)[:24]
    return board
