"""Publish an immutable, dated research edition; market refreshes cannot erase it."""
from copy import deepcopy
import json
import subprocess
from zoneinfo import ZoneInfo
from nhl.v2.data import ROOT, iso, digest, write_json
from nhl.analyst import immutable

ET=ZoneInfo('America/New_York')

def existing_today(root,now):
    try:
        card=json.loads((root/'docs/briefing/morning-card.json').read_text())
        return card.get('decision_date')==now.astimezone(ET).date().isoformat()
    except (OSError,ValueError):
        return False


def publish_card(feeds,reviews,now,*,root=ROOT,kind='morning'):
    if kind not in ('morning','test'):raise ValueError('Unknown edition kind')
    merged=deepcopy(feeds);merged['Reviews']=reviews
    result=subprocess.run(['node',str(ROOT/'scripts/analyst_shortlist.cjs')],
        input=json.dumps({'feeds':merged,'asof':iso(now)},allow_nan=False),
        text=True,capture_output=True,check=True,timeout=30,cwd=ROOT)
    selected=json.loads(result.stdout)
    rows=selected['card']
    card=dict(schema_version=1,kind=kind,decision_date=now.astimezone(ET).date().isoformat(),
        published_at=iso(now),policy_version='morning-edition-1',rows=rows,
        coverage=selected['coverage'],budget=reviews.get('budget'),
        discovery_status=reviews.get('discovery_status'),
        status='published' if rows else 'no_reviewed_candidates',
        basis='Original reviewed forecasts and quotes at publication; no automatic intraday reassessment.')
    card['edition_id']=digest(card)[:24]
    relative='docs/briefing/cards/'+card['decision_date']+'-'+card['edition_id']+'.json'
    card['archive_url']='/'+relative.removeprefix('docs/')
    immutable(root/relative,card)
    write_json(root/'docs/briefing/morning-card.json',card)
    return card
