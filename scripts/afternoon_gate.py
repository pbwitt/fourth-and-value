"""Gate for the afternoon NHL/MLB market refresh: each sport refreshes at most once per afternoon.

Supabase starts the refresh at 4:30 p.m. Eastern (supabase/afternoon_scheduler.sql). GitHub's
own schedule is only a backup and can start hours late, so an automatic start skips a sport
whose feed already published after 4 p.m. Eastern today, and does nothing outside the
4 p.m.–midnight window. An operator's manual start always refreshes both sports.
"""
import argparse
import json
import os
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo('America/New_York')
FEEDS = {'nhl': 'docs/nhl/data/latest.json', 'mlb': 'docs/mlb/data/latest.json'}
AFTERNOON = time(16, 0)


def published_at(path):
    """The feed's last successful refresh, or None when it cannot be read."""
    try:
        value = json.loads(Path(path).read_text()).get('last_success_at')
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    return stamp if stamp.tzinfo else None


def decide(root=ROOT, now=None, source='operator'):
    """{sport: True to refresh}. Manual starts always refresh; automatic starts once per afternoon."""
    now = (now or datetime.now(timezone.utc)).astimezone(ET)
    if source == 'operator':
        return {sport: True for sport in FEEDS}
    start = datetime.combine(now.date(), AFTERNOON, ET)
    if now < start:
        # A backup start delayed past midnight would refresh after the slate; skip it.
        return {sport: False for sport in FEEDS}
    result = {}
    for sport, path in FEEDS.items():
        stamp = published_at(Path(root) / path)
        result[sport] = not (stamp and start <= stamp.astimezone(ET) <= now)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='operator', choices=['operator', 'supabase', 'github-schedule'])
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    plan = decide(ROOT, now, args.source)
    for sport, run in plan.items():
        stamp = published_at(ROOT / FEEDS[sport])
        last = stamp.astimezone(ET).strftime('%Y-%m-%d %H:%M ET') if stamp else 'unknown'
        print(f"{sport.upper()}: {'refresh' if run else 'skip'} (started by {args.source}; last published {last})")
    if os.getenv('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            for sport, run in plan.items():
                output.write(f"{sport}={'true' if run else 'false'}\n")
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(f"## Afternoon refresh gate ({args.source}, {now.astimezone(ET):%H:%M} ET)\n\n"
                          + ''.join(f"- {sport.upper()}: {'refresh' if run else 'already refreshed or outside window'}\n"
                                    for sport, run in plan.items()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
