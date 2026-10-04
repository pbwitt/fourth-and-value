"""Published track record for the live NHL model: held-out reliability, for display only.

Reads the committed evaluation report for the running model version and keeps the
"when the model said X%, it happened Y%" bins per market. Nothing here feeds a forecast.
"""
import hashlib
import json
from pathlib import Path

from . import VERSION, EVIDENCE
from .data import ROOT

# Each version's own evidence folder; a version without published evidence gets no track record.
REPORTS = {version: folder + '/evaluation.json' for version, folder in EVIDENCE.items()}
PLAYER = {'player_shots_on_goal': 'shots', 'player_goals': 'goals', 'player_assists': 'assists', 'player_points': 'points'}
GAME = {'h2h': ('moneyline', 'Home win'), 'totals': ('total_5.5', 'Over 5.5 goals'), 'spreads': ('puck_-1.5', 'Home −1.5')}
OUT = ROOT / 'docs/nhl/data/track-record.json'


def merged(bins, minimum=100):
    """Neighbouring bins join until each holds at least `minimum` forecasts."""
    groups, current = [], []
    for b in bins:
        current.append(b)
        if sum(x['count'] for x in current) >= minimum:
            groups.append(current); current = []
    if current:
        if groups:
            groups[-1] += current
        else:
            groups.append(current)
    out = []
    for g in groups:
        n = sum(x['count'] for x in g)
        out.append(dict(predicted=round(sum(x['forecast'] * x['count'] for x in g) / n, 4),
                        observed=round(sum(x['observed'] * x['count'] for x in g) / n, 4), n=n))
    return out


def market(entry, side):
    return dict(side=side, forecasts=entry['n'], brier=round(entry['brier'], 4), ece=round(entry['ece'], 4),
                calibration_bins=merged(entry['calibration']))


def build(version=VERSION, root=ROOT):
    path = root / REPORTS[version]
    raw = path.read_bytes()
    report = json.loads(raw)
    final, selection = report['final'], report['selection']
    markets = {}
    for key, stat in PLAYER.items():
        entry = final['player'][selection['shots' if stat == 'shots' else 'scoring']][stat]
        markets[key] = dict(market(entry, f"Over {entry['reference_line']:g}"), line=entry['reference_line'])
    team = final['team'][selection['team']]['markets']
    for key, (name, side) in GAME.items():
        markets[key] = market(team[name], side)
    return dict(schema_version=1, model_version=version, source=REPORTS[version],
                source_sha256=hashlib.sha256(raw).hexdigest(), test_start=final['test']['start'], test_end=final['test']['end'],
                trained_through=final['training']['end'] if isinstance(final.get('training'), dict) else None,
                note='One held-out season the model never saw while training. Player markets include every skater who played, '
                     'not only players with posted props, and count only games he played. Each market is checked at one line.',
                markets=markets)


def write(out=OUT, version=VERSION):
    data = build(version)
    text = json.dumps(data, indent=1, sort_keys=True) + '\n'
    if not out.exists() or out.read_text() != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return data


if __name__ == '__main__':
    print(json.dumps(write(), indent=1)[:2000])
