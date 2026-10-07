"""Durable immutable NBA refresh snapshots, ready for later weekly grading."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save_nba_run(state, now, root=ROOT):
    if state.get('status') != 'ready' or not state.get('rows'):
        return None
    path = Path(root)/'artifacts/nba/runs'/f'{now:%Y%m%dT%H%M%SZ}.json.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return None
    with gzip.open(path, 'xt', encoding='utf-8') as f:
        json.dump(state, f, separators=(',', ':'), sort_keys=True, allow_nan=False)
    return path
