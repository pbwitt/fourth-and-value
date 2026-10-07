"""Compact pregame archive of each MLB refresh, for weekly recaps.

docs/mlb/data/latest.json is overwritten on every refresh, so it cannot show what
the site knew before first pitch a week later. Each refresh therefore also saves a
small gzip snapshot of its pregame rows to artifacts/mlb/runs/<UTC stamp>.json.gz.
A recap grades each game from the last snapshot saved before that game started.
"""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'artifacts/mlb/runs'
# Per-book quote fields plus the forecast attached to that quote. Player context and
# display text stay out: they are large and the recap does not grade them.
ROW_FIELDS = ('event_id', 'mlb_game_id', 'game', 'home_team', 'away_team', 'commence_time', 'game_type',
              'phase', 'series_game', 'market', 'player', 'model_player_id', 'side', 'line', 'price', 'book',
              'quoted_at', 'model_probability', 'model_push_probability', 'model_mean', 'is_model_pick',
              'model_status', 'model_version', 'model_policy')
EVENT_FIELDS = ('mlb_game_id', 'commence_time', 'home_team', 'away_team', 'game_type', 'phase', 'series_game')


def compact(state):
    """The archive payload for one refresh: its pregame quotes and the scheduled games."""
    def pick(record, fields):
        out = {k: record[k] for k in fields if record.get(k) is not None}
        for side in ('home_pitcher', 'away_pitcher'):
            if fields is EVENT_FIELDS and isinstance(record.get(side), dict):
                out[side] = {k: record[side].get(k) for k in ('id', 'fullName')}
        return out
    return dict(schema=1, sport='MLB', checked_at=state.get('checked_at'), model_version=state.get('model_version'),
                events=[pick(e, EVENT_FIELDS) for e in state.get('events') or []],
                rows=[pick(r, ROW_FIELDS) for r in state.get('rows') or []])


def save_run(state, now, root=RUNS):
    """Write this refresh's snapshot; never overwrite an existing one."""
    path = Path(root) / (now.strftime('%Y%m%dT%H%M%SZ') + '.json.gz')
    if path.exists() or not state.get('rows'):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        json.dump(compact(state), f, separators=(',', ':'), sort_keys=True)
    return path
