"""Build factual daily editions, render published features and publish approved private drafts.

Paid research runs separately in editorial_writer.py. Never logs private queue content or API URLs with keys.
"""
import argparse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import json
import html as html_lib
import math
import os
from pathlib import Path
import re
from statistics import median
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit

import sys

import requests
from jinja2 import Environment, FileSystemLoader, select_autoescape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odds_budget import CreditBudget, CreditFloorReached, estimate_cost, requests_preflight

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
    games=[];coverage={};budget=CreditBudget(label='Editorial prices')
    for sport,key in CFG['sports'].items():
        secret=os.getenv(f'{sport}_ODDS_API_KEY') or os.getenv('ODDS_API_KEY')
        if not secret:coverage[sport]='No odds credential; no prices published';continue
        params={'regions':'us','markets':'totals','oddsFormat':'american'}
        try:
            # Shared 2,000-credit floor (scripts/odds_budget.py); a refusal withholds prices.
            budget.ensure(estimate_cost(f'sports/{key}/odds',params),preflight=requests_preflight(secret))
        except CreditFloorReached as error:
            print(error);coverage[sport]='Odds credit floor reached; no prices published';continue
        try:
            r=requests.get(f'https://api.the-odds-api.com/v4/sports/{key}/odds',
                params={'apiKey':secret,**params},timeout=35)
            budget.observe(r.headers)
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

def market_cards(games):
    """Highlight distinct observations, not three copies of the same range sentence."""
    if not games:return []
    selected=[];used=set()
    movers=sorted((g for g in games if g.get('change')),key=lambda g:-abs(g['change'])/g['median'])
    spread=lambda g:max(q['line'] for q in supported_quotes(g))-min(q['line'] for q in supported_quotes(g)) if g.get('quotes') else g['maximum']-g['minimum']
    gaps=sorted((g for g in games if spread(g)>0),key=lambda g:-spread(g)/g['median'])
    for kind,pool in [('Movement',movers),('Book disagreement',gaps),('Next up',sorted(games,key=lambda g:g['commence_time']))]:
        g=next((g for g in pool if g['id'] not in used),None)
        if not g:continue
        used.add(g['id'])
        if kind=='Movement':
            since=stamp(g['change_from']).astimezone(ETZ).strftime('%b %d at %I:%M %p ET') if g.get('change_from') else 'the previous snapshot'
            text=f"The matched-book median moved {'up' if g['change']>0 else 'down'} {abs(g['change']):g} since {since}. The current all-book median is {g['median']:g}; this measures movement, not its cause."
        elif kind=='Book disagreement':
            text=f"A {spread(g):g}-point gap separates the lowest and highest totals shared by at least two of {len(g['books'])} books." if supported_quotes(g) is not g.get('quotes') else f"A {spread(g):g}-point gap separates the lowest and highest totals across {len(g['books'])} books."
            text+=" The number available depends on where you bet; compare the attached prices too."
        else:
            text=f"Starts {g['start_label']}. "
            text+=(f"All {len(g['books'])} books show a total of {g['median']:g}; the prices can still differ." if g['minimum']==g['maximum'] else f"Books show totals from {g['minimum']:g} to {g['maximum']:g}, with a median of {g['median']:g}.")
            if g.get('change')==0:text+=' No net change in the matched-book median since the previous check.'
        quotes=supported_quotes(g);prices=[]
        if quotes:
            low=min(quotes,key=lambda q:(q['line'],-q['over_price']))
            high=max(quotes,key=lambda q:(q['line'],q['under_price']))
            prices=[f"Lowest over total: {low['label']} · Over {low['line']:g} ({low['over_price']:+g})",f"Highest under total: {high['label']} · Under {high['line']:g} ({high['under_price']:+g})"]
        selected.append(dict(sport=g['sport'],title=g['game'],kind=kind,text=text,prices=prices,
            note=f"{g['start_label']} · {len(g['books'])} books observed",url='/nfl/totals/' if g['sport']=='NFL' else f"/{g['sport'].lower()}/totals/"))
    return selected

def american(price):
    return f'{price:+d}' if isinstance(price,int) else f'{price:+g}'

def supported_quotes(g):
    """With four or more books, a total posted by one book alone does not set the range.

    Books move their main number at different juice thresholds, so a lone book one hook
    above everyone else is usually the same view priced on the other side, not a better line."""
    quotes=g.get('quotes',[])
    if len(quotes)<4:return quotes
    counts={}
    for q in quotes:counts[q['line']]=counts.get(q['line'],0)+1
    shared=[q for q in quotes if counts[q['line']]>=2]
    return shared or quotes

def quote_view(q):
    return dict(label=q['label'],line=q['line'],over=american(q['over_price']),under=american(q['under_price']))

def edge_quote(g,side):
    # Lowest total with the best Over price, or highest total with the best Under price,
    # among lines at least two books share; lone-book lines beyond it are listed separately.
    quotes=supported_quotes(g)
    if not quotes:return None
    line=min(q['line'] for q in quotes) if side=='over' else max(q['line'] for q in quotes)
    q=max((q for q in quotes if q['line']==line),key=lambda q:(q[f'{side}_price'],q['label']))
    lone=sorted((x for x in g.get('quotes',[]) if (x['line']<line if side=='over' else x['line']>line)),key=lambda x:x['line'])
    return dict(quote_view(q),lone=[quote_view(x) for x in lone])

def pulled_label(games):
    times=sorted(stamp(q['quoted_at']).astimezone(ETZ) for g in games for q in g.get('quotes',[]))
    if not times:return ''
    clock=lambda t:t.strftime('%I:%M %p').lstrip('0')
    first,last=clock(times[0]),clock(times[-1])
    day=f"{times[0]:%b} {times[0].day}" if times[0].date()==times[-1].date() else None
    if day:return f'Book prices pulled {day}, '+(first if first==last else f'{first}–{last}')+' ET'
    return f"Book prices pulled {times[0]:%b} {times[0].day} {first} – {times[-1]:%b} {times[-1].day} {last} ET"

def nhl_model_totals(now,path=None):
    """Expected full-game goals per Odds API event from the current NHL model board.

    Mirrors TeamModel.joint: independent Poisson regulation scores, plus exactly one
    OT/shootout settlement goal whenever regulation ends tied. Expired or failed
    forecasts are omitted rather than shown beside current prices."""
    path=path or DOCS/'nhl/data/latest.json'
    try:board=json.loads(path.read_text());made=stamp(board['model_prediction_at'])
    except (OSError,ValueError,KeyError,TypeError,AttributeError):return {}
    if board.get('status')!='ready' or board.get('model_error') or not timedelta(0)<=now-made<=timedelta(hours=36):return {}
    totals={}
    for row in board.get('rows',[]):
        home,away=row.get('projected_home_reg_goals'),row.get('projected_away_reg_goals')
        if row.get('player') or row.get('event_id') in totals or not all(isinstance(x,(int,float)) and 0<x<18 for x in [home,away]):continue
        tie=sum(math.exp(-home-away)*(home*away)**n/math.factorial(n)**2 for n in range(48))
        totals[row['event_id']]=dict(total=home+away+tie,home=home,away=away,version=row.get('model_version') or board.get('model_version'))
    return totals

def line_movement(now,root=None):
    """Per-sport summary of how prices moved after published picks (scripts/line_movement.py)."""
    rows=[]
    for sport in ['MLB','NHL']:
        try:ledger=json.loads(((root or DOCS)/sport.lower()/'data/line-movement.json').read_text())
        except (OSError,ValueError):continue
        updated=stamp(ledger['updated_at']) if ledger.get('updated_at') else None
        s=ledger.get('summary') or {}
        if not updated or not timedelta(0)<=now-updated<=timedelta(days=3) or not s.get('observed'):continue
        avg=s.get('average_probability_move')
        rows.append(dict(sport=sport,observed=s['observed'],picks=s['picks'],same_line=s.get('same_line',0),beat=s.get('same_line_beat',0),
            average=f'{100*avg:+.1f} pp' if isinstance(avg,(int,float)) else '—',line_moves=s.get('line_moves',0),line_favorable=s.get('line_moves_favorable',0)))
    return rows

def context(data,now,model_totals=None):
    # Even a non-refresh render must not revive stale or already-started quotes.
    games=[]
    model_totals=nhl_model_totals(now) if model_totals is None else model_totals
    fresh=bool(data.get('generated_at')) and timedelta(0)<=now-stamp(data['generated_at'])<=timedelta(hours=6)
    if fresh:
        for g in data.get('games',[]):
            if stamp(g['commence_time'])>now and all(timedelta(0)<=now-stamp(q['quoted_at'])<=timedelta(hours=6) for q in g.get('quotes',[])):
                games.append(dict(g,low=edge_quote(g,'over'),high=edge_quote(g,'under'),model=model_totals.get(g['id']) if g['sport']=='NHL' else None))
    news=[n for n in data.get('news',[]) if timedelta(0)<=now-stamp(n['published_at'])<=timedelta(hours=36)]
    cards=market_cards(games)
    divided=sum(g['maximum']>g['minimum'] for g in games)
    date=stamp(data['generated_at']).astimezone(ETZ) if data.get('generated_at') else now.astimezone(ETZ)
    return dict(date_label=date.strftime('%A, %B %d'),snapshot_label='Prices checked '+date.strftime('%b %d at %I:%M %p ET')+(' · refresh pending' if not fresh else ''),
        summary=f"{len(games)} upcoming games checked · {divided} with different totals across books." if games else 'No upcoming games currently have fresh, comparable totals. The next scheduled price check will update this board.',
        cards=cards,games=games,pulled_label=pulled_label(games),movement=line_movement(now),sports=sorted({g['sport'] for g in games}),news=news[:10],coverage=data.get('coverage',{}))

def featured_now(article,now):
    if article['kind']=='Opinion':return False
    expiry=stamp(article['featured_until']) if article.get('featured_until') else stamp(article['date']+'T00:00:00+00:00')+timedelta(days=3)
    return now<expiry

def home_slides(current,fallback):
    # The homepage slider rotates the newest featured pieces: morning analysis
    # and one-off blog features share it. The newest featured blog piece keeps
    # a slide for its featured window even when new articles fill the others.
    eligible=[a for a in current if a.get('featured')]
    slides=(eligible or current)[:3] or [fallback]
    blog=next((a for a in eligible if a['url'].startswith('/blog/')),None)
    if blog and blog not in slides:slides=slides[:2]+[blog]
    return slides

def render_home(data,now):
    catalog=list(CFG['articles'])
    published=DOCS/'editorial/published.json'
    if published.exists():catalog+=json.loads(published.read_text())
    catalog.sort(key=lambda a:(a['date'],a.get('published_at','')),reverse=True)
    current=[a for a in catalog if featured_now(a,now)]
    fallback=dict(title='The daily market briefing',excerpt='Compare current prices across the leagues and follow what changes next.',sport='Sports',kind='Market watch',url='/briefing/',date=now.astimezone(ETZ).date().isoformat())
    slides=home_slides(current,fallback)
    lead=slides[0];shown={a['url'] for a in slides}
    ctx=context(data,now)
    ctx.update(lead=lead,slides=slides,features=[a for a in current if a['url'] not in shown][:6],opinions=[a for a in catalog if a['kind']=='Opinion'][:2])
    (DOCS/'index.html').write_text(ENV.get_template('home.html').render(**ctx)+'\n')
    # Opinion remains a distinct, permanent archive; approved analysis also
    # appears in the existing blog without rebuilding any authored article.
    opinion_cards=''.join('<article class="card opinion"><p class="eyebrow">Opinion · '+html_lib.escape(a['date'])+'</p><h2><a href="'+html_lib.escape(a['url'],quote=True)+'">'+html_lib.escape(a['title'])+'</a></h2><p>'+html_lib.escape(a['excerpt'])+'</p></article>' for a in catalog if a['kind']=='Opinion')
    (DOCS/'editorial/index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Opinion | Fourth &amp; Value</title><meta name="description" content="Independent opinion on sports, accountability and the institutions behind the games."><link rel="canonical" href="https://fourthandvalue.com/editorial/"><meta property="og:type" content="website"><meta property="og:title" content="Opinion | Fourth &amp; Value"><meta property="og:description" content="Independent opinion on sports, accountability and the institutions behind the games."><meta property="og:url" content="https://fourthandvalue.com/editorial/"><meta property="og:image" content="https://fourthandvalue.com/assets/social-card.png"><meta name="twitter:card" content="summary_large_image"><meta name="twitter:image" content="https://fourthandvalue.com/assets/social-card.png"><link rel="stylesheet" href="/assets/editorial.css"></head><body><div id="nav-root"></div><script src="/nav.js?v=47"></script><main class="newsroom article"><p class="eyebrow">Independent perspectives</p><h1>Opinion.</h1><p class="lead">The arguments beyond the numbers. Each piece is clearly labeled and approved by its author.</p>'+opinion_cards+'<footer><a href="/">Home</a> · <a href="/editorial/inbox.html">Editorial desk</a></footer></main></body></html>\n')
    blog=DOCS/'blog/index.html';text=blog.read_text()
    text=re.sub(r'<!-- editorial-managed:start -->.*?<!-- editorial-managed:end -->','',text,flags=re.S)
    entries='<li class="post" data-title="Daily market briefing" data-excerpt="Fresh prices and reporting"><h2><a href="/briefing/">The daily market briefing</a></h2><p class="excerpt">Observed prices, differences between books and recent reporting. Updated throughout the day.</p></li>'
    for a in catalog:
        if not a['url'].startswith('/editorial/articles/') or a['kind']=='Opinion':continue
        entries+='<li class="post" data-title="'+html_lib.escape(a['title'],quote=True)+'" data-excerpt="'+html_lib.escape(a['excerpt'],quote=True)+'"><h2><a href="'+a['url']+'">'+html_lib.escape(a['title'])+'</a></h2><div class="meta">'+a['date']+' · Analysis</div><p class="excerpt">'+html_lib.escape(a['excerpt'])+'</p></li>'
    text=text.replace('<ul id="posts" class="list">','<ul id="posts" class="list"><!-- editorial-managed:start -->'+entries+'<!-- editorial-managed:end -->')
    blog.write_text(text)

def render_briefing(data,now,archive=True):
    ctx=context(data,now);day=stamp(data['generated_at']).astimezone(ETZ).date().isoformat() if data.get('generated_at') else now.astimezone(ETZ).date().isoformat()
    path=DOCS/'editorial/published.json'
    articles=json.loads(path.read_text()) if path.exists() else []
    ctx['analysis']=sorted((a for a in articles if featured_now(a,now)),key=lambda a:(a['date'],a.get('published_at','')),reverse=True)[:3]
    ctx.update(title=f"The market rundown: {now.astimezone(ETZ).strftime('%B %d, %Y')}",url=f'/briefing/{day}.html',evidence_url=f'/briefing/history/{day}.json')
    PUBLIC.mkdir(exist_ok=True,parents=True)
    html=ENV.get_template('briefing.html').render(**ctx,live_picks=False)+'\n'
    if archive:(PUBLIC/f'{day}.html').write_text(html)
    ctx['url']='/briefing/'
    (PUBLIC/'index.html').write_text(ENV.get_template('briefing.html').render(**ctx,live_picks=True)+'\n')

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
        # Defense in depth: even if a queue response is malformed, a contributor
        # draft must carry a real editor's explicit publication authorization.
        if row.get('status') not in ('approved','publishing'):continue
        if row.get('requires_review') and not row.get('approved_by'):continue
        if not row.get('approved_hash') or (row.get('publish_on') and row['publish_on']>now.astimezone(ETZ).date().isoformat()):continue
        # A revoked editor cannot publish through an old queued approval.
        user=requests.get(base+'/auth/v1/admin/users/'+(row.get('approved_by') or row['user_id']),headers=api_headers(),timeout=20)
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
        item['published_at']=prior.get('published_at') or now.isoformat()
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
    if args.refresh:data=refresh(now)
    else:data=json.loads((PUBLIC/'latest.json').read_text()) if (PUBLIC/'latest.json').exists() else {}
    render_briefing(data,now,archive=args.refresh)
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
    print('Public homepage and archives rendered.')

if __name__=='__main__':main()
