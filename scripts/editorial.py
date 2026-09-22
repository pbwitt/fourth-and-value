"""Build factual daily editions and publish only owner-approved private drafts.

No writing-model calls. Never logs private queue content or API URLs with keys.
"""
import argparse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import html as html_lib
import os
from pathlib import Path
import re
from statistics import median
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit

import requests
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
CFG=json.loads((ROOT/'config/editorial.json').read_text())
ETZ=ZoneInfo(CFG['timezone'])
ENV=Environment(loader=FileSystemLoader(ROOT/'scripts/editorial_templates'),autoescape=select_autoescape(['html']))
PUBLIC=DOCS/'briefing'

def stamp(s):
    return datetime.fromisoformat(s.replace('Z','+00:00')).astimezone(timezone.utc)

def safe_url(url):
    u=urlsplit(url)
    return u.scheme=='https' and bool(u.netloc) and not u.username and not u.password

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def summarize_events(sport,events,now,previous):
    rows=[]
    old={g['id']:g for g in previous.get('games',[]) if g['sport']==sport}
    for event in events:
        try:start=stamp(event['commence_time'])
        except (KeyError,ValueError):continue
        horizon=168 if sport=='NFL' else 48
        if not now < start <= now+timedelta(hours=horizon):continue
        books={};quotes=[]
        for book in event.get('bookmakers',[]):
            for market in book.get('markets',[]):
                if market.get('key')!='totals':continue
                try:quoted=stamp(market.get('last_update') or book['last_update'])
                except (KeyError,ValueError):continue
                if not timedelta(0)<=now-quoted<=timedelta(hours=CFG['max_quote_age_hours']):continue
                outcomes=market.get('outcomes',[])
                over=next((x for x in outcomes if x.get('name')=='Over'),None)
                under=next((x for x in outcomes if x.get('name')=='Under'),None)
                if not over or not under or over.get('point')!=under.get('point'):continue
                if not all(isinstance(x.get('price'),(int,float)) and abs(x['price'])>=100 for x in [over,under]):continue
                point=over.get('point')
                if not isinstance(point,(int,float)) or not 0<point<1000:continue
                books[book['key']]=point
                quotes.append(dict(book=book['key'],label=book.get('title',book['key']),line=point,
                                   over_price=over['price'],under_price=under['price'],quoted_at=quoted.isoformat()))
        if len(books)<2:continue
        prior=old.get(event['id'],{});common=set(books)&set(prior.get('books',{}))
        change=None
        if len(common)>=2:
            change=median(books[b] for b in common)-median(prior['books'][b] for b in common)
        rows.append(dict(id=event['id'],sport=sport,game=f"{event['away_team']} @ {event['home_team']}",
            commence_time=start.isoformat(),start_label=start.astimezone(ETZ).strftime('%a %I:%M %p ET'),
            median=median(books.values()),minimum=min(books.values()),maximum=max(books.values()),
            books=books,quotes=quotes,previous_books={b:prior['books'][b] for b in sorted(common)},change_from=previous.get('generated_at'),change=change,change_label=f'{change:+g} ({len(common)} matched books)' if change is not None else 'First comparable snapshot'))
    return rows

def fetch_news(now):
    items=[];seen=set();status={}
    for sport,url in CFG['news_feeds'].items():
        try:
            r=requests.get(url,timeout=25,headers={'User-Agent':'FourthAndValue/1.0 editorial feed reader'});r.raise_for_status()
            tree=ET.fromstring(r.content)
            found=0
            for item in tree.findall('.//item'):
                title=' '.join((item.findtext('title') or '').split())
                link=item.findtext('link') or ''
                try:published=parsedate_to_datetime(item.findtext('pubDate') or '').astimezone(timezone.utc)
                except (ValueError,TypeError):continue
                if not safe_url(link) or not title or link in seen or not timedelta(0)<=now-published<=timedelta(hours=CFG['max_news_age_hours']):continue
                # Headlines only, no full article excerpts or inferred injury facts.
                title=' '.join(title.split()[:24])
                items.append(dict(sport=sport,title=title,url=link,published_at=published.isoformat(),published_label=published.astimezone(ETZ).strftime('%b %d, %I:%M %p ET')))
                seen.add(link);found+=1
                if found>=3:break
            status[sport]=f'{found} recent headlines'
        except (requests.RequestException,ET.ParseError):status[sport]='News feed unavailable'
    return sorted(items,key=lambda x:x['published_at'],reverse=True),status

def refresh(now):
    previous=json.loads((PUBLIC/'latest.json').read_text()) if (PUBLIC/'latest.json').exists() else {}
    games=[];coverage={}
    for sport,key in CFG['sports'].items():
        secret=os.getenv(f'{sport}_ODDS_API_KEY') or os.getenv('ODDS_API_KEY')
        if not secret:coverage[sport]='No odds credential; no prices published';continue
        try:
            r=requests.get(f'https://api.the-odds-api.com/v4/sports/{key}/odds',
                params={'apiKey':secret,'regions':'us','markets':'totals','oddsFormat':'american'},timeout=35)
            if r.status_code==404:coverage[sport]='No active market returned';continue
            r.raise_for_status()
            selected=summarize_events(sport,r.json(),now,previous)
            games.extend(selected);coverage[sport]=f'{len(selected)} upcoming games with fresh paired totals at two or more books'
        except (requests.RequestException,ValueError,TypeError):coverage[sport]='Price feed unavailable; prior quotes not carried forward'
    news,news_status=fetch_news(now)
    data=dict(generated_at=now.isoformat(),previous_generated_at=previous.get('generated_at'),games=sorted(games,key=lambda x:x['commence_time']),news=news,coverage=coverage,news_coverage=news_status)
    write_json(PUBLIC/'latest.json',data)
    write_json(PUBLIC/'history'/f'{now.astimezone(ETZ).date()}.json',data)
    write_json(PUBLIC/'snapshots'/f'{now.strftime("%Y%m%dT%H%M%SZ")}.json',data)
    return data

def context(data,now):
    # Even a non-refresh render must not revive stale or already-started quotes.
    games=[]
    fresh=bool(data.get('generated_at')) and timedelta(0)<=now-stamp(data['generated_at'])<=timedelta(hours=6)
    if fresh:
        for g in data.get('games',[]):
            if stamp(g['commence_time'])>now and all(timedelta(0)<=now-stamp(q['quoted_at'])<=timedelta(hours=6) for q in g.get('quotes',[])):
                games.append(g)
    news=[n for n in data.get('news',[]) if timedelta(0)<=now-stamp(n['published_at'])<=timedelta(hours=36)]
    candidates=sorted(games,key=lambda x:(-(x['maximum']-x['minimum']),x['commence_time']))[:3]
    cards=[]
    for g in candidates:
        cards.append(dict(sport=g['sport'],title=g['game'],text=f"Observed totals range from {g['minimum']:g} to {g['maximum']:g} across {len(g['books'])} books. The median is {g['median']:g}. Compare the price as well as the number before deciding whether the difference is useful.",
            note=f"{g['start_label']} · {g['change_label']}",url='/nfl/totals/' if g['sport']=='NFL' else f"/{g['sport'].lower()}/totals/"))
    date=stamp(data['generated_at']).astimezone(ETZ) if data.get('generated_at') else now.astimezone(ETZ)
    return dict(date_label=date.strftime('%A, %B %d'),snapshot_label='Prices checked '+date.strftime('%b %d at %I:%M %p ET')+(' · refresh pending' if not fresh else ''),
        summary=f"{len(games)} upcoming games have comparable fresh totals in this edition. We’re watching differences between books and changes since the previous snapshot." if games else 'No current price comparison is being promoted. Read the dated analysis below while the next snapshot is collected.',
        cards=cards,games=games,news=news[:10],coverage=data.get('coverage',{}))

def render_home(data,now):
    catalog=list(CFG['articles'])
    published=DOCS/'editorial/published.json'
    if published.exists():catalog+=json.loads(published.read_text())
    catalog.sort(key=lambda a:a['date'],reverse=True)
    eligible=[a for a in catalog if a.get('featured') and a['kind']!='Opinion']
    lead=eligible[0] if eligible else next(a for a in catalog if a['kind']!='Opinion')
    ctx=context(data,now)
    ctx.update(lead=lead,features=[a for a in catalog if a['kind']!='Opinion'][:3],opinions=[a for a in catalog if a['kind']=='Opinion'][:2])
    (DOCS/'index.html').write_text(ENV.get_template('home.html').render(**ctx)+'\n')
    # Opinion remains a distinct, permanent archive; approved analysis also
    # appears in the existing blog without rebuilding any authored article.
    opinion_cards=''.join('<article class="card opinion"><p class="eyebrow">Opinion · '+html_lib.escape(a['date'])+'</p><h2><a href="'+html_lib.escape(a['url'],quote=True)+'">'+html_lib.escape(a['title'])+'</a></h2><p>'+html_lib.escape(a['excerpt'])+'</p></article>' for a in catalog if a['kind']=='Opinion')
    (DOCS/'editorial/index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Opinion | Fourth &amp; Value</title><meta name="description" content="Independent opinion on sports, accountability and the institutions behind the games."><link rel="canonical" href="https://fourthandvalue.com/editorial/"><link rel="stylesheet" href="/assets/editorial.css"></head><body><div id="nav-root"></div><script src="/nav.js?v=42"></script><main class="newsroom article"><p class="eyebrow">Independent perspectives</p><h1>Opinion.</h1><p class="lead">The arguments beyond the numbers. Each piece is clearly labeled and approved by its author.</p>'+opinion_cards+'<footer><a href="/">Home</a> · <a href="/editorial/inbox.html">Editorial desk</a></footer></main></body></html>\n')
    blog=DOCS/'blog/index.html';text=blog.read_text()
    text=re.sub(r'<!-- editorial-managed:start -->.*?<!-- editorial-managed:end -->','',text,flags=re.S)
    entries='<li class="post" data-title="Daily market briefing" data-excerpt="Fresh prices and reporting"><h2><a href="/briefing/">The daily market briefing</a></h2><p class="excerpt">Observed prices, differences between books and recent reporting. Updated each morning.</p></li>'
    for a in catalog:
        if not a['url'].startswith('/editorial/articles/') or a['kind']=='Opinion':continue
        entries+='<li class="post" data-title="'+html_lib.escape(a['title'],quote=True)+'" data-excerpt="'+html_lib.escape(a['excerpt'],quote=True)+'"><h2><a href="'+a['url']+'">'+html_lib.escape(a['title'])+'</a></h2><div class="meta">'+a['date']+' · Analysis</div><p class="excerpt">'+html_lib.escape(a['excerpt'])+'</p></li>'
    text=text.replace('<ul id="posts" class="list">','<ul id="posts" class="list"><!-- editorial-managed:start -->'+entries+'<!-- editorial-managed:end -->')
    blog.write_text(text)

def render_briefing(data,now):
    ctx=context(data,now);day=now.astimezone(ETZ).date().isoformat()
    ctx.update(title=f"The market rundown: {now.astimezone(ETZ).strftime('%B %d, %Y')}",url=f'/briefing/{day}.html',evidence_url=f'/briefing/history/{day}.json')
    PUBLIC.mkdir(exist_ok=True,parents=True)
    html=ENV.get_template('briefing.html').render(**ctx)+'\n'
    (PUBLIC/f'{day}.html').write_text(html)
    (PUBLIC/'index.html').write_text(html.replace(f'https://fourthandvalue.com/briefing/{day}.html','https://fourthandvalue.com/briefing/'))

def api_headers():
    key=os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    return {'apikey':key,'Authorization':f'Bearer {key}','Content-Type':'application/json','Prefer':'return=representation'}

def publish_approved(now,receipt):
    base=os.getenv('SUPABASE_URL','').rstrip('/')
    if not base or not os.getenv('SUPABASE_SERVICE_ROLE_KEY'):
        print('Private queue not configured; public briefing can still run.');return
    response=requests.get(base+'/rest/v1/editorial_ideas',headers=api_headers(),params={'status':'in.(approved,publishing)','select':'*'},timeout=30)
    if response.status_code==404:
        print('Private queue schema not installed; no private content published.');return
    if not response.ok:raise RuntimeError(f'Private queue HTTP {response.status_code}')
    catalog_path=DOCS/'editorial/published.json'
    catalog=json.loads(catalog_path.read_text()) if catalog_path.exists() else []
    done=[]
    for row in response.json():
        if not row.get('approved_hash') or (row.get('publish_on') and row['publish_on']>now.astimezone(ETZ).date().isoformat()):continue
        # A revoked editor cannot publish through an old queued approval.
        user=requests.get(base+'/auth/v1/admin/users/'+row['user_id'],headers=api_headers(),timeout=20)
        if not user.ok or user.json().get('app_metadata',{}).get('fv_editor') is not True:continue
        if not re.fullmatch(r'[0-9a-f-]{36}',row['id']):continue
        if not row['title'].strip() or len(row['body'].strip())<100 or not row['byline'].strip():continue
        links=[line.strip() for line in row['sources'].splitlines() if line.strip()]
        if any(not safe_url(link) for link in links):continue
        if row['kind']=='analysis' and not links:continue
        if row['status']=='approved':
            claim=requests.patch(base+'/rest/v1/editorial_ideas',headers=api_headers(),params={'id':'eq.'+row['id'],'status':'eq.approved','updated_at':'eq.'+row['updated_at']},json={'status':'publishing'},timeout=30)
            if not claim.ok:raise RuntimeError(f'Publication claim HTTP {claim.status_code}')
            claimed=claim.json()
            if not claimed:continue
            row=claimed[0]
        url=f"/editorial/articles/{row['id']}.html"
        prior=next((a for a in catalog if a['url']==url),{})
        date=row.get('publish_on') or prior.get('date') or now.astimezone(ETZ).date().isoformat()
        item=dict(title=row['title'],url=url,date=date,kind=row['kind'].title(),sport=row['sport'],featured=row['featured'],excerpt=row['body'].split('\n')[0][:220])
        target=DOCS/url.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(ENV.get_template('article.html').render(**item,byline=row['byline'],paragraphs=re.split(r'\n\s*\n',row['body']),links=links)+'\n')
        catalog=[a for a in catalog if a['url']!=url]+[item]
        done.append(dict(id=row['id'],approved_hash=row['approved_hash'],updated_at=row['updated_at'],url=url))
    if done:write_json(catalog_path,catalog)
    write_json(receipt,done)
    print(f'{len(done)} approved articles prepared; private ideas/drafts remain in Supabase.')

def acknowledge(receipt):
    if not receipt.exists():return
    base=os.environ['SUPABASE_URL'].rstrip('/')
    for row in json.loads(receipt.read_text()):
        r=requests.patch(base+'/rest/v1/editorial_ideas',headers=api_headers(),params={'id':'eq.'+row['id'],'status':'eq.publishing','approved_hash':'eq.'+row['approved_hash'],'updated_at':'eq.'+row['updated_at']},json={'status':'published','published_url':row['url']},timeout=30)
        if not r.ok:raise RuntimeError(f'Publication acknowledgment HTTP {r.status_code}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--refresh',action='store_true');ap.add_argument('--publish-approved',action='store_true');ap.add_argument('--ack',action='store_true');ap.add_argument('--receipt',type=Path,default=Path('/tmp/fv-editorial-published.json'));args=ap.parse_args()
    if args.ack:acknowledge(args.receipt);return
    now=datetime.now(timezone.utc)
    if args.publish_approved:publish_approved(now,args.receipt)
    if args.refresh:data=refresh(now);render_briefing(data,now)
    else:data=json.loads((PUBLIC/'latest.json').read_text()) if (PUBLIC/'latest.json').exists() else {}
    render_home(data,now)
    # Only published URLs are discoverable; private desk itself stays noindex.
    sitemap=DOCS/'sitemap.xml';text=sitemap.read_text()
    paths=['/briefing/']+[f'/briefing/{now.astimezone(ETZ).date()}.html'] if args.refresh else []
    p=DOCS/'editorial/published.json'
    if p.exists():paths += [a['url'] for a in json.loads(p.read_text())]
    for path in paths:
        link='https://fourthandvalue.com'+path
        if f'<loc>{link}</loc>' not in text:text=text.replace('</urlset>',f'  <url><loc>{link}</loc></url>\n</urlset>')
    sitemap.write_text(text)
    print('Public homepage rendered. No writing API calls made.')

if __name__=='__main__':main()
