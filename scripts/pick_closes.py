"""Register a verified edition's picks for closing-price capture (public.pick_closes).

The closing-lines Edge Function prices each one in the minutes before its game, exactly
like a tracked bet and from the same per-game snapshot. Rows carry Bet Tracker's ticket
fields (briefing-picks.js ticketData), so they map to the same markets. Registering is
idempotent per (edition, pick); a newer edition of the same date and kind replaces the
older one's picks that have not closed yet. Without the Supabase settings it does
nothing and never fails the edition.

  python scripts/pick_closes.py --kind morning --edition <edition_id>
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import requests

ROOT = Path(__file__).resolve().parents[1]
FILES = {'morning': 'morning-card.json', 'afternoon': 'afternoon-card.json'}
SETTINGS = ('SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY')
TICKETS = r'''const {ticketData}=require(process.argv[1]);
const card=JSON.parse(require('fs').readFileSync(0,'utf8')),out=[];
(card.rows||[]).forEach((r,i)=>{let t;try{t=ticketData(r,r.price,1);}catch{return;}
 const id=String(r.event_id||r.game_id||'');
 out.push({pick_index:i,league:t.league,market_type:t.market_type,side:t.side,line:t.line,player:t.player,book:t.book,
  odds:t.odds,game_date:t.game_date,team_home:t.team_home,team_away:t.team_away,commence_time:r.commence_time,
  decision_market_prob:t.market_prob??null,final_prob:t.final_prob??null,
  // An Odds API event id saves the closing job a lookup; anything else it matches itself.
  event_id:/^[0-9a-f]{32}$/.test(id)?id:null});});
process.stdout.write(JSON.stringify(out));'''


def rows(card, root=ROOT):
    """One pick_closes row per published pick, in card order (the email's pick numbers)."""
    result = subprocess.run(['node', '-e', TICKETS, str(root / 'docs/assets/briefing-picks.js')],
                            input=json.dumps(card), capture_output=True, text=True, check=True, timeout=30)
    keep = dict(edition_id=card['edition_id'], edition_kind=card['kind'], decision_date=card['decision_date'],
                policy_version=card.get('policy_version'), published_at=card['published_at'])
    return [dict(keep, **r) for r in json.loads(result.stdout)]


def register(card, env, http=requests):
    base = env['SUPABASE_URL'].rstrip('/') + '/rest/v1/pick_closes'
    key = env['SUPABASE_SERVICE_ROLE_KEY']
    headers = {'apikey': key, 'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    picks = rows(card)
    if picks:
        response = http.post(base, params={'on_conflict': 'edition_id,pick_index'}, json=picks, timeout=30,
                             headers={**headers, 'Prefer': 'resolution=merge-duplicates,return=minimal'})
        if not response.ok:
            raise RuntimeError(f'Pick registration HTTP {response.status_code}')
    # A recovery that published a newer edition replaces the older one's open picks.
    response = http.delete(base, timeout=30, headers=headers, params={
        'decision_date': 'eq.' + card['decision_date'], 'edition_kind': 'eq.' + card['kind'],
        'edition_id': 'neq.' + card['edition_id'], 'clv_status': 'is.null'})
    if not response.ok:
        raise RuntimeError(f'Superseded pick cleanup HTTP {response.status_code}')
    return len(picks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=sorted(FILES), required=True)
    parser.add_argument('--edition', required=True, help='the verified edition id')
    args = parser.parse_args()
    card = json.loads((ROOT / 'docs/briefing' / FILES[args.kind]).read_text())
    if card.get('edition_id') != args.edition or card.get('kind') != args.kind:
        raise SystemExit('The edition on disk is not the verified one; nothing registered.')
    if not all(os.environ.get(name) for name in SETTINGS):
        print('Closing-price registration not configured (needs ' + ', '.join(SETTINGS) + '); nothing registered.')
        return 0
    try:
        count = register(card, os.environ)
    except (RuntimeError, requests.RequestException, subprocess.CalledProcessError) as error:
        raise SystemExit(f'Closing-price registration failed: {error}. The edition is published; nothing else changed.')
    print(f'Registered {count} {args.kind} pick{"s" if count != 1 else ""} for closing prices.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
