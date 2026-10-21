"""Alert when a refreshed sport recorded no Pinnacle reference today.

Pinnacle rides along with every core odds refresh (scripts/sharp_reference.py). This check
compares each sport's published feed with today's reference file (Eastern date) and fails
when the reference is missing, so a silent stall like Oct 18-21, 2026 (the optional
snapshot job blocked by the credit reserve for four days) cannot happen unnoticed. It runs
from .github/workflows/sharp-reference-check.yml, which opens a GitHub issue on failure.

A sport only counts when its feed refreshed successfully today and has upcoming games.
It fails when such a sport has no reference file today, when every such sport recorded
zero Pinnacle quotes, or when a sport with at least MIN_GAMES upcoming games recorded
zero quotes. Pinnacle can skip a single game, so smaller slates only warn.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from display_time import EASTERN, eastern_label  # noqa: E402
from sharp_reference import load, path_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIN_GAMES = 3
FEEDS = {
    'nfl': ('docs/nfl/data/quotes.json', 'generated_at'),
    'nhl': ('docs/nhl/data/latest.json', 'last_success_at'),
    'mlb': ('docs/mlb/data/latest.json', 'last_success_at'),
    'nba': ('docs/nba/data/latest.json', 'last_success_at'),
}


def stamp(value):
    try:
        t = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    return t if t.tzinfo else None


def feed_state(root, sport, now):
    """(refreshed today, upcoming games at refresh time, refresh time) for a sport's feed."""
    path, field = FEEDS[sport]
    try:
        feed = json.loads((root / path).read_text())
    except (OSError, ValueError):
        return False, 0, None
    refreshed = stamp(feed.get(field))
    if not refreshed or feed.get('status') != 'ready':
        return False, 0, refreshed
    today = refreshed.astimezone(EASTERN).date() == now.astimezone(EASTERN).date()
    games = {r.get('event_id') or r.get('game_id') for r in feed.get('rows') or []
             if (stamp(r.get('commence_time')) or refreshed) > refreshed}
    games.discard(None)
    return today, len(games), refreshed


def check(root, now):
    rows, errors, warnings = [], [], []
    for sport in FEEDS:
        today, games, refreshed = feed_state(root, sport, now)
        if not today or not games:
            rows.append(dict(sport=sport, expected=False, games=games, refreshed=refreshed))
            continue
        data = load(path_for(sport, now, root))
        runs = (data or {}).get('runs', [])
        quotes = len((data or {}).get('quotes', []))
        events = len({q[0] for q in (data or {}).get('quotes', [])})
        rows.append(dict(sport=sport, expected=True, games=games, refreshed=refreshed, runs=len(runs),
                         quotes=quotes, events=events))
        slate = f"{games} upcoming game{'s' if games != 1 else ''}"
        if data is None:
            errors.append(f'{sport.upper()} refreshed today with {slate} but recorded no Pinnacle reference file.')
        elif quotes == 0 and games >= MIN_GAMES:
            errors.append(f'{sport.upper()} has {slate} but zero Pinnacle quotes today.')
        elif quotes == 0:
            warnings.append(f'{sport.upper()} has {slate} and zero Pinnacle quotes today.')
    expected = [r for r in rows if r['expected']]
    if expected and all(r.get('quotes', 0) == 0 for r in expected) and not errors:
        errors.append('Every refreshed sport recorded zero Pinnacle quotes today.')
    return rows, errors, warnings


def report(rows, errors, warnings, now):
    lines = [f'## Pinnacle reference check, {eastern_label(now)}', '',
             '| Sport | Refreshed | Upcoming games | Runs today | Pinnacle quotes | Pinnacle games |',
             '|---|---|---|---|---|---|']
    for r in rows:
        when = eastern_label(r['refreshed']) if r.get('refreshed') else 'never'
        if not r['expected']:
            lines.append(f"| {r['sport'].upper()} | {when} | {r['games']} | not expected | | |")
        else:
            lines.append(f"| {r['sport'].upper()} | {when} | {r['games']} | {r.get('runs', 0)} | "
                         f"{r.get('quotes', 0)} | {r.get('events', 0)} |")
    lines.append('')
    lines += [f'- ERROR: {e}' for e in errors] + [f'- Warning: {w}' for w in warnings]
    if not errors:
        lines.append('Status: OK' if any(r['expected'] for r in rows) else 'Status: no sport refreshed today; nothing to check.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    rows, errors, warnings = check(args.root, now)
    text = report(rows, errors, warnings, now)
    print(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as handle:
            handle.write(text + '\n')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
