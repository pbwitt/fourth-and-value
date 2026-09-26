#!/usr/bin/env python3
"""Validate public NHL content timestamps, never filesystem modification times."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nhl.refresh import ROOT, season_for, timestamp


def validate(state, now):
    errors = []
    checked = timestamp(state.get('last_success_at'))
    if state.get('status') not in ['ready', 'waiting_for_markets']:
        errors.append('NHL market feed is not healthy')
    if not checked or not timedelta(minutes=-5) <= now - checked <= timedelta(hours=24):
        errors.append('Last successful snapshot is missing or expired')
    if state.get('season') != season_for(now):
        errors.append('Snapshot belongs to another NHL season')
    for game in state.get('events', []):
        if game.get('game_type') != 2 or game.get('season') != season_for(now):
            errors.append('Schedule contains a non-regular-season or wrong-season event')
            break
    for row in state.get('rows', []):
        quote, start = timestamp(row.get('quoted_at')), timestamp(row.get('commence_time'))
        if not quote or not timedelta(minutes=-5) <= now - quote <= timedelta(hours=24) or not start or start <= now:
            errors.append('Snapshot contains an expired quote or a started game')
            break
        probability = row.get('independent_probability')
        if probability is not None:
            model_time = timestamp(row.get('model_data_checked_at'))
            if not model_time or not timedelta(0) <= now-model_time < timedelta(hours=36):
                errors.append('Independent model inputs are expired')
                break
            push, loss = row.get('push_probability'), row.get('loss_probability')
            if (not isinstance(push,(int,float)) or not isinstance(loss,(int,float)) or
                min(probability,push,loss)<0 or abs(probability+push+loss-1)>1e-7):
                errors.append('Independent settlement probabilities are invalid')
                break
            if not row.get('model_version') or not row.get('forecast_id'):
                errors.append('Independent forecast provenance is missing')
                break
    return errors


if __name__ == '__main__':
    path = ROOT / 'docs/nhl/data/latest.json'
    try:
        errors = validate(json.loads(path.read_text()), datetime.now(timezone.utc))
    except (OSError, ValueError):
        errors = ['NHL public snapshot is missing or invalid']
    for error in errors:
        print(error, file=sys.stderr)
    print('NHL freshness: ' + ('FAIL' if errors else 'PASS'))
    raise SystemExit(bool(errors))
