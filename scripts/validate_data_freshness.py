#!/usr/bin/env python3
"""Reject unknown/old quote timestamps; kickoff alone does not establish freshness."""
import argparse
import sys
import pandas as pd


def validate_props_freshness(props_csv, max_age_hours=48, now=None):
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize('UTC')
    try:
        df = pd.read_csv(props_csv)
    except (OSError, pd.errors.EmptyDataError) as exc:
        print(f'Invalid props input: {exc}', file=sys.stderr)
        return False
    if df.empty:
        print('No props available in this snapshot.', file=sys.stderr)
        return False
    if not {'commence_time', 'last_update'}.issubset(df.columns):
        print('Missing kickoff or sportsbook quote timestamps.', file=sys.stderr)
        return False
    kick = pd.to_datetime(df['commence_time'], utc=True, errors='coerce')
    quotes = pd.to_datetime(df['last_update'], utc=True, errors='coerce')
    if kick.isna().any():
        print('Invalid game kickoff timestamp.', file=sys.stderr)
        return False
    upcoming = kick > now
    if not upcoming.any():
        print('No upcoming games in the snapshot.', file=sys.stderr)
        return False
    age = (now - quotes[upcoming]).dt.total_seconds() / 3600
    valid = age.notna() & age.between(0, max_age_hours)
    if not valid.all():
        print(f'{int((~valid).sum())} upcoming quotes have missing, future or stale timestamps.', file=sys.stderr)
        return False
    if ((kick[upcoming] - now).dt.total_seconds() > 8 * 86400).any():
        print('Snapshot includes games beyond the next eight days.', file=sys.stderr)
        return False
    print(f'Validated {int(upcoming.sum())} upcoming quotes; oldest {age.max():.1f} hours.')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--props', default='data/props/latest_all_props.csv')
    parser.add_argument('--max-age-hours', type=float, default=48)
    args=parser.parse_args()
    sys.exit(0 if validate_props_freshness(args.props,args.max_age_hours) else 1)

if __name__=='__main__':
    main()
