"""Conservative pre-call reservations for the newsroom's rolling seven-day budget."""
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import subprocess
import editorial as ed

PATH=ed.DOCS/'editorial/budget.json'
# Per-token (input, output) rates. Input includes the higher cache-write rate
# (1.25x list); output is list price. Astra was checked September 22; GPT-6.1 Sol
# ($2/$10 per million list) is used only for the shadow comparison drafts.
RATES={'gpt-6-astra':(12.5/1000000,50/1000000),'gpt-6.1-sol':(2.5/1000000,10/1000000)}
INPUT_RATE,OUTPUT_RATE=RATES['gpt-6-astra']
LIMITS={'write':(18000,3200),'audit':(24000,1000)}
ASTRA='gpt-6-astra'

def bounds(payload,phase):
    if payload.get('tools') or payload.get('model') not in RATES or payload.get('service_tier')!='default':raise ValueError('Unbudgeted model, tool or service tier')
    encoded=sum(len(payload[k].encode('utf-8')) for k in ('instructions','input'))
    cap,output=LIMITS[phase]
    if encoded>cap or payload['max_output_tokens']>output:raise ValueError('Request exceeds budgeted size')
    inp,out=RATES[payload['model']]
    # Byte count upper-bounds the byte-based tokenizer; reserve extra framing space.
    return round(((encoded+1024)*inp+output*out)*1.10,6)

def maximum(models=None):
    """Largest write+audit charge; models maps a phase to a non-default model."""
    models=models or {}
    return round(sum(((cap+1024)*RATES[models.get(phase,ASTRA)][0]+out*RATES[models.get(phase,ASTRA)][1])*1.10
                     for phase,(cap,out) in LIMITS.items()),6)

def cost(usage,model=ASTRA):
    # No cached-input discount assumed. Exact invoice may be lower.
    inp,out=RATES[model]
    return round(usage['input_tokens']*inp+usage['output_tokens']*out,6)

def read():
    return json.loads(PATH.read_text()) if PATH.exists() else {'version':2,'entries':[]}

def used(data,now):
    return sum(e['charge_usd'] for e in data['entries'] if ed.stamp(e['at'])>=now-timedelta(days=7))

def reserve(key,now,cap,amount=None):
    data=read()
    prior=next((e for e in data['entries'] if e['key']==key),None)
    # An operator may verify a legacy pre-request failure. Preserve that record;
    # a paid or uncertain attempt can never be released through this exception.
    if prior and not (prior.get('verified_pre_request_failure') and prior.get('status')=='settled' and prior.get('charge_usd')==0):return False
    amount=maximum() if amount is None else amount
    if used(data,now)+amount>cap:return False
    if prior:
        history=prior.setdefault('prior_attempts',[])
        history.append({k:prior[k] for k in ('at','status','charge_usd','verified_pre_request_failure')})
        prior.pop('verified_pre_request_failure')
        prior.update(at=now.isoformat(),status='reserved',charge_usd=amount)
    else:data['entries'].append(dict(key=key,at=now.isoformat(),status='reserved',charge_usd=amount))
    ed.write_json(PATH,data);return True

def settle(key,usages,complete):
    data=read();entry=next(e for e in data['entries'] if e['key']==key)
    if complete:
        # A usage is a response usage (Astra) or a (usage, model) pair.
        entry.update(status='settled',charge_usd=round(sum(cost(*u) if isinstance(u,tuple) else cost(u) for u in usages),6))
    else:entry['status']='uncertain-reservation-retained'
    ed.write_json(PATH,data)

def checkpoint(statepath):
    # CI must durably reserve money and mark the slot started before making a call.
    # A crash or deployment failure cannot reset the budget on the next runner.
    if os.getenv('GITHUB_ACTIONS')!='true':return
    commands=[['git','config','user.name','Fourth & Value Editorial'],['git','config','user.email','actions@github.com'],
        ['git','add',str(PATH.relative_to(ed.ROOT)),str(statepath.relative_to(ed.ROOT)), 'docs/editorial/articles/', 'docs/editorial/evidence/', 'docs/editorial/published.json'],
        ['git','commit','-m','Editorial: reserve bounded writing budget'],
        ['git','pull','--rebase','--autostash','origin','main'],['git','push','origin','HEAD:main']]
    for command in commands:
        result=subprocess.run(command,cwd=ed.ROOT,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Budget checkpoint failed; no paid request permitted')
