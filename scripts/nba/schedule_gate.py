"""Gate for GitHub's backup NBA starts: each refresh slot runs at most once.

Supabase starts the NBA refresh at 6:45 a.m. and 5 p.m. Eastern (supabase/nba_scheduler.sql).
GitHub's own schedule is only a backup and can start hours late, so a scheduled start skips
when the feed already published after the latest slot. Supabase and operator starts
(workflow_dispatch) always refresh.
"""
import argparse
import json
import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo('America/New_York')
FEED = 'docs/nba/data/latest.json'
SLOTS = (time(6, 45), time(17, 0))


def published_at(path):
    """The feed's last successful refresh, or None when it cannot be read."""
    try:
        value = json.loads(Path(path).read_text()).get('last_success_at')
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    return stamp if stamp.tzinfo else None


def latest_slot(now):
    """The most recent slot start at or before `now`, in Eastern time."""
    now = now.astimezone(ET)
    for day in (now.date(), now.date() - timedelta(days=1)):
        for slot in reversed(SLOTS):
            start = datetime.combine(day, slot, ET)
            if start <= now:
                return start
    raise AssertionError('unreachable')


def decide(root=ROOT, now=None, event='workflow_dispatch'):
    """True to refresh. Only a GitHub schedule start can skip, when its slot already published."""
    if event != 'schedule':
        return True
    stamp = published_at(Path(root) / FEED)
    return not (stamp and stamp >= latest_slot(now or datetime.now(timezone.utc)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event', default='workflow_dispatch')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    refresh = decide(ROOT, now, args.event)
    stamp = published_at(ROOT / FEED)
    last = stamp.astimezone(ET).strftime('%Y-%m-%d %H:%M ET') if stamp else 'unknown'
    slot = latest_slot(now).strftime('%Y-%m-%d %H:%M ET')
    print(f"NBA: {'refresh' if refresh else 'skip'} (started by {args.event}; slot {slot}; last published {last})")
    if os.getenv('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write(f"refresh={'true' if refresh else 'false'}\n")
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(f"## NBA refresh gate ({args.event}, {now.astimezone(ET):%H:%M} ET)\n\n"
                          f"- {'Refresh' if refresh else 'Skipped: already published after the ' + slot + ' slot'}\n")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
