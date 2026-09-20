#!/usr/bin/env python3
"""Report site traffic from the Cloudflare GraphQL Analytics API.

Only counts requests that pass through the Cloudflare proxy, so the DNS
records for the zone must be Proxied (orange cloud) rather than DNS only.
Numbers start from the moment proxying was switched on; there is no
retroactive data.

Setup: put a token in .env as CLOUDFLARE_API_TOKEN. Create it at
Cloudflare > My Profile > API Tokens > Create Token, using the
"Read analytics and logs" template, scoped to this zone. The zone id is
looked up from the domain, or set CLOUDFLARE_ZONE_ID to skip the lookup.

  python scripts/cloudflare_traffic.py             # last 7 days
  python scripts/cloudflare_traffic.py --days 30
"""
import argparse
import os
import sys
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

API = 'https://api.cloudflare.com/client/v4'
GRAPHQL = f'{API}/graphql'

QUERY = """
query ($zone: String!, $since: Date!, $until: Date!) {
  viewer {
    zones(filter: {zoneTag: $zone}) {
      httpRequests1dGroups(
        limit: 60,
        filter: {date_geq: $since, date_leq: $until},
        orderBy: [date_ASC]
      ) {
        dimensions { date }
        sum { requests pageViews }
        uniq { uniques }
      }
    }
  }
}
"""


def headers(token):
    return {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}


def zone_id(token, domain):
    cached = os.getenv('CLOUDFLARE_ZONE_ID')
    if cached:
        return cached
    r = requests.get(f'{API}/zones', headers=headers(token),
                     params={'name': domain}, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f'Zone lookup failed [{r.status_code}]: {r.text[:300]}')
    results = r.json().get('result', [])
    if not results:
        raise SystemExit(f'No Cloudflare zone found for {domain}. '
                         'Check the token has Zone:Read for this zone.')
    return results[0]['id']


def fetch(token, zone, days):
    until = date.today()
    since = until - timedelta(days=days - 1)
    r = requests.post(GRAPHQL, headers=headers(token), timeout=30, json={
        'query': QUERY,
        'variables': {'zone': zone, 'since': since.isoformat(), 'until': until.isoformat()},
    })
    if r.status_code != 200:
        raise SystemExit(f'Analytics query failed [{r.status_code}]: {r.text[:400]}')
    payload = r.json()
    if payload.get('errors'):
        raise SystemExit(f'Analytics query returned errors: {payload["errors"]}')
    zones = payload['data']['viewer']['zones']
    if not zones:
        raise SystemExit('Token authenticated but returned no zone data. '
                         'The token likely lacks Zone Analytics: Read.')
    return zones[0]['httpRequests1dGroups']


def main():
    ap = argparse.ArgumentParser(description='Report Cloudflare traffic')
    ap.add_argument('--domain', default='fourthandvalue.com')
    ap.add_argument('--days', type=int, default=7)
    args = ap.parse_args()

    token = os.getenv('CLOUDFLARE_API_TOKEN')
    if not token:
        raise SystemExit(
            'CLOUDFLARE_API_TOKEN not set in .env.\n'
            'Create one at Cloudflare > My Profile > API Tokens > Create Token,\n'
            'template "Read analytics and logs", scoped to this zone.')

    rows = fetch(token, zone_id(token, args.domain), args.days)
    if not rows:
        print('No data returned. If proxying was only just enabled, Cloudflare '
              'can take 15-30 minutes to report the first requests.')
        return

    print(f'{args.domain} - last {args.days} days\n')
    print(f'{"date":<12}{"requests":>10}{"page views":>12}{"visitors":>10}')
    print('-' * 44)
    total_req = total_pv = total_uniq = 0
    for row in rows:
        d = row['dimensions']['date']
        req = row['sum'].get('requests', 0)
        pv = row['sum'].get('pageViews', 0)
        uq = row['uniq'].get('uniques', 0)
        total_req += req; total_pv += pv; total_uniq += uq
        print(f'{d:<12}{req:>10,}{pv:>12,}{uq:>10,}')
    print('-' * 44)
    print(f'{"total":<12}{total_req:>10,}{total_pv:>12,}{total_uniq:>10,}')
    print('\nVisitors are summed per day, so a repeat visitor counts once per day.')
    print('Your own visits are included, and some traffic is bots rather than people.')


if __name__ == '__main__':
    main()
