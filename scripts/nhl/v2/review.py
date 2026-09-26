"""Append-only analyst evidence, kept separate from the original model forecast."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2])); __package__='nhl.v2'
from .data import ROOT, stamp, digest

KINDS={'goalie','deployment','injury','return','trade','coaching','tactical','other'}


def validate_review(record,asof):
    required=['offer_id','forecast_id','analyst','source_url','source_published_at','recorded_at','reason','kind','status','represented_in']
    if any(not record.get(k) and k!='represented_in' for k in required): raise ValueError('Incomplete analyst evidence')
    if record['kind'] not in KINDS or record['status'] not in ['reviewed','watch','rejected']: raise ValueError('Invalid review category/status')
    if not record['source_url'].startswith(('https://','http://')): raise ValueError('Source URL required')
    published,recorded=stamp(record['source_published_at']),stamp(record['recorded_at'])
    if published>recorded or recorded>asof: raise ValueError('Future review evidence')
    if not isinstance(record['represented_in'],list): raise ValueError('Represented-in feature/market list required')
    override=record.get('override_probability')
    if override is not None:
        if not 0<override<1 or not record.get('adjustment_method') or not record.get('double_counting_check'):
            raise ValueError('Explicit probability adjustment needs method and double-counting explanation')
    return dict(record,review_id=digest(record)[:24],evaluation_status='prospective_shadow_only')


def apply_review(row,record):
    if row['offer_id']!=record['offer_id'] or row['forecast_id']!=record['forecast_id']:
        raise ValueError('Review belongs to another offer or forecast')
    # A manual override never rewrites calibrated/model/market probabilities.
    output=dict(row,analyst_review=record,analyst_status=record['status'])
    output['original_forecast']={k:row.get(k) for k in ['independent_probability','market_probability','final_probability','push_probability','estimated_ev']}
    if record.get('override_probability') is not None:
        if record['override_probability']+row.get('push_probability',0)>1: raise ValueError('Override exceeds action probability')
        output['analyst_probability']=record['override_probability']
        output['analyst_forecast_status']='unvalidated_override'
    return output


def append_review(path,record,asof):
    record=validate_review(record,asof); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive append by one local operator; original entries are never edited.
    if path.exists() and any(json.loads(s)['review_id']==record['review_id'] for s in path.read_text().splitlines()):
        return record
    with path.open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
    return record


if __name__=='__main__':
    from datetime import datetime,timezone
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('record',type=Path)
    p.add_argument('--ledger',type=Path,default=ROOT/'artifacts/nhl/reviews.jsonl')
    args=p.parse_args(); print(append_review(args.ledger,json.loads(args.record.read_text()),datetime.now(timezone.utc))['review_id'])
