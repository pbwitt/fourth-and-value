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
from urllib.parse import urlencode, urlsplit

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

NEWS_HEADERS={'User-Agent':'FourthAndValue/1.0 editorial feed reader'}

def espn_api_news(url):
    """(title, link, published) from ESPN's public news API. ESPN+ stories are skipped."""
    r=requests.get(url,timeout=25,headers=NEWS_HEADERS);r.raise_for_status()
    data=r.json()
    articles=data.get('articles') if isinstance(data,dict) else None
    if not isinstance(articles,list):raise ValueError('response has no articles list')
    out=[]
    for a in articles:
        if not isinstance(a,dict) or a.get('premium'):continue
        link=((a.get('links') or {}).get('web') or {}).get('href') or ''
        try:published=datetime.fromisoformat(str(a.get('published') or '').replace('Z','+00:00')).astimezone(timezone.utc)
        except ValueError:continue
        out.append((str(a.get('headline') or ''),link,published))
    return out

def rss_news(url):
    r=requests.get(url,timeout=25,headers=NEWS_HEADERS);r.raise_for_status()
    out=[]
    for item in ET.fromstring(r.content).findall('.//item'):
        try:published=parsedate_to_datetime(item.findtext('pubDate') or '').astimezone(timezone.utc)
        except (ValueError,TypeError):continue
        out.append((item.findtext('title') or '',item.findtext('link') or '',published))
    return out

def fetch_news(now):
    """Up to three recent headlines per league: the ESPN news API first, then the RSS feed.

    A league whose sources all fail prints why, so the scheduled run's log shows the cause."""
    items=[];seen=set();status={}
    sources=[('ESPN news API',CFG.get('news_api',{}),espn_api_news),('ESPN RSS',CFG.get('news_feeds',{}),rss_news)]
    for sport in dict.fromkeys([*CFG.get('news_api',{}),*CFG.get('news_feeds',{})]):
        entries,errors=None,[]
        for label,urls,read in sources:
            if not urls.get(sport):continue
            try:entries=read(urls[sport]);break
            except (requests.RequestException,ValueError,ET.ParseError) as error:
                errors.append(f'{label}: {type(error).__name__}: {error}'[:300])
        if entries is None:
            print(f'News {sport}: unavailable ({"; ".join(errors) or "no source configured"})',flush=True)
            status[sport]='News feed unavailable';continue
        found=0
        for title,link,published in sorted(entries,key=lambda e:e[2],reverse=True):
            title=' '.join(title.split())
            if not safe_url(link) or not title or link in seen or not timedelta(0)<=now-published<=timedelta(hours=CFG['max_news_age_hours']):continue
            # Headlines only, no full article excerpts or inferred injury facts.
            title=' '.join(title.split()[:24])
            items.append(dict(sport=sport,title=title,url=link,source='ESPN',published_at=published.isoformat(),published_label=published.astimezone(ETZ).strftime('%b %d, %I:%M %p ET')))
            seen.add(link);found+=1
            if found>=3:break
        status[sport]=f'{found} recent headlines'
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

def market_rows(games,limit=5):
    """Home-page table rows: line moves first, then book disagreement, then the next starts.

    Each row names what changed in a few words plus the best over and under with their
    books, so the table carries numbers rather than the cards' explanatory sentences."""
    def spread(g):
        quotes=supported_quotes(g)
        return max(q['line'] for q in quotes)-min(q['line'] for q in quotes) if quotes else g['maximum']-g['minimum']
    movers=sorted((g for g in games if g.get('change')),key=lambda g:-abs(g['change'])/g['median'])
    gaps=sorted((g for g in games if spread(g)>0),key=lambda g:-spread(g)/g['median'])
    rows=[];used=set()
    for kind,pool in [('move',movers),('split',gaps),('next',sorted(games,key=lambda g:g['commence_time']))]:
        for g in pool:
            # Unmoved games only fill a short table; they are context, not news.
            if len(rows)>=(3 if kind=='next' else limit):break
            if g['id'] in used:continue
            used.add(g['id']);books=len(g['books'])
            if kind=='move':
                common=g.get('previous_books') or {}
                before,after=median(common.values()),median(g['books'][b] for b in common if b in g['books'])
                since=stamp(g['change_from']).astimezone(ETZ).strftime('%I:%M %p ET').lstrip('0') if g.get('change_from') else 'the previous check'
                change,detail=f'Total {before:g} → {after:g}',f'since {since}'
            elif kind=='split':
                lines=[q['line'] for q in supported_quotes(g)] or [g['minimum'],g['maximum']]
                change,detail=f'Books split {min(lines):g} to {max(lines):g}',f'a {spread(g):g}-point gap'
            elif g['minimum']==g['maximum']:
                change,detail=f'Total {g["median"]:g} at all {books} books','only the prices differ'
            else:
                change,detail=f'Totals {g["minimum"]:g} to {g["maximum"]:g}',f'median {g["median"]:g}'
            over=under=None
            if quotes:=supported_quotes(g):
                low=min(quotes,key=lambda q:(q['line'],-q['over_price']))
                high=max(quotes,key=lambda q:(q['line'],q['under_price']))
                over=dict(text=f"O {low['line']:g} ({low['over_price']:+g})",book=low['label'])
                under=dict(text=f"U {high['line']:g} ({high['under_price']:+g})",book=high['label'])
            rows.append(dict(sport=g['sport'],game=g['game'],start=g['commence_time'],start_label=g['start_label'],books=books,
                change=change,detail=detail,over=over,under=under,url='/nfl/totals/' if g['sport']=='NFL' else f"/{g['sport'].lower()}/totals/"))
    return rows

NEXT_UP_FEEDS={'MLB':'mlb/data/latest.json','NHL':'nhl/data/latest.json','NBA':'nba/data/latest.json','NFL':'nfl/data/quotes.json'}
TWO_WORD_NAMES=('Red Sox','White Sox','Blue Jays','Maple Leafs','Golden Knights','Red Wings','Blue Jackets','Trail Blazers')

def nickname(team):
    team=str(team or '')
    return next((n for n in TWO_WORD_NAMES if team.endswith(' '+n)),team.split(' ')[-1] if team else '')

def next_up(now,limit=4,root=None):
    """The next games across the leagues with the best current moneyline and total.

    Reads each league's committed feed at render time. A price counts only if it was
    quoted in the last six hours, the same rule as the market table; games that have
    started, or start more than 36 hours out, are left out. MLB adds the probable
    starters and their strikeout lines beside our forecast; NHL adds projected goalies."""
    games={}
    for sport,rel in NEXT_UP_FEEDS.items():
        try:feed=json.loads(((root or DOCS)/rel).read_text())
        except (OSError,ValueError):continue
        goalies=feed.get('goalie_projections') or {}
        for r in feed.get('rows') or []:
            try:start=stamp(r['commence_time'])
            except (KeyError,TypeError,ValueError):continue
            if not timedelta(0)<start-now<=timedelta(hours=36):continue
            g=games.setdefault((sport,r.get('game'),r['commence_time']),dict(sport=sport,start=r['commence_time'],
                start_label=start.astimezone(ETZ).strftime('%a %I:%M %p ET').replace(' 0',' '),away=r.get('away_team',''),home=r.get('home_team',''),
                label='',who='',ml={},totals={},props={}))
            if sport=='MLB' and r.get('game_type') not in (None,'R') and r.get('phase') and not g['label']:
                g['label']=r['phase']+(f" · Game {r['series_game']}" if r.get('series_game') else '')
            if sport=='MLB' and not g['who'] and (r.get('away_pitcher') or r.get('home_pitcher')):
                g['who']=f"{(r.get('away_pitcher') or {}).get('fullName') or 'TBD'} vs {(r.get('home_pitcher') or {}).get('fullName') or 'TBD'}"
            if sport=='NHL' and not g['who'] and str(r.get('nhl_game_id')) in goalies:
                gp=goalies[str(r['nhl_game_id'])];names=[]
                for side in ('away','home'):
                    starters=sorted((gp.get(side) or {}).get('goalies') or [],key=lambda x:-(x.get('start_probability') or 0))
                    names.append(starters[0]['player'] if starters else 'TBD')
                confirmed=all((gp.get(s) or {}).get('confirmed') for s in ('away','home'))
                g['who']=' vs '.join(names)+('' if confirmed else ' (projected)')
            quoted=stamp(r['quoted_at']) if r.get('quoted_at') else None
            if not quoted or not timedelta(0)<=now-quoted<=timedelta(hours=6) or not isinstance(r.get('price'),(int,float)):continue
            label=r.get('book_label') or r.get('book')
            if r.get('market')=='h2h' and not r.get('player'):
                best=g['ml'].get(r['side'])
                if not best or r['price']>best[0]:g['ml'][r['side']]=(r['price'],label)
            elif r.get('market')=='totals' and not r.get('player') and isinstance(r.get('line'),(int,float)):
                side=g['totals'].setdefault(r['line'],{}).setdefault(r['side'],[])
                side.append((r['price'],label))
            elif sport=='MLB' and r.get('market')=='pitcher_strikeouts' and r.get('player') and isinstance(r.get('line'),(int,float)):
                p=g['props'].setdefault(r['player'],dict(lines={},mean=r.get('model_mean')))
                p['lines'][r['line']]=p['lines'].get(r['line'],0)+1
    out=[]
    for g in sorted(games.values(),key=lambda g:g['start'])[:limit]:
        ml=[f"{nickname(team)} {american(price)} {book}" for team in (g['away'],g['home']) if team in g['ml'] for price,book in [g['ml'][team]]]
        total=None
        if g['totals']:
            line=max(g['totals'],key=lambda l:(sum(len(v) for v in g['totals'][l].values()),-l))
            best={s:max(v) for s,v in g['totals'][line].items()}
            parts=[f"{s[0]} {american(best[s][0])} {best[s][1]}" for s in ('Over','Under') if s in best]
            total=dict(line=f'{line:g}',parts=parts)
        props=[]
        starters=[x.split(' vs ')[i] for x in [g['who']] for i in (0,1)] if g['sport']=='MLB' and ' vs ' in g['who'] else []
        for name in starters:
            p=g['props'].get(name)
            if not p:continue
            line=max(p['lines'],key=lambda l:(p['lines'][l],-l))
            forecast=f" · forecast {p['mean']:.1f}" if isinstance(p['mean'],(int,float)) else ''
            props.append(dict(text=f"{name.split(' ')[-1]} strikeouts {line:g}{forecast}",
                url='/mlb/props/?'+urlencode({'q':name,'market':'pitcher_strikeouts'})))
        out.append(dict(sport=g['sport'],label=g['label'],start=g['start'],start_label=g['start_label'],away=g['away'],home=g['home'],
            who=g['who'],ml=ml,total=total,props=props))
    return out

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
        cards=cards,rows=market_rows(games),games=games,pulled_label=pulled_label(games),movement=line_movement(now),sports=sorted({g['sport'] for g in games}),news=news[:10],coverage=data.get('coverage',{}))

def featured_now(article,now):
    if article['kind']=='Opinion':return False
    expiry=stamp(article['featured_until']) if article.get('featured_until') else stamp(article['date']+'T00:00:00+00:00')+timedelta(days=3)
    return now<expiry

def home_lead(current,fallback):
    """The newest featured analysis or feature leads the page. Opinion has its own slot."""
    return ([a for a in current if a.get('featured')] or current or [fallback])[0]

def featured_opinions(catalog,now):
    # Explicit opt-in and expiry only; never include these in market analysis.
    return [a for a in catalog if a.get('kind')=='Opinion' and a.get('featured')
            and a.get('featured_until') and now<stamp(a['featured_until'])]

def home_stories(current,catalog,lead,now,limit=8):
    """Stories after the lead, newest first. Current features and any opinion an editor
    has featured come first, then the newest dated analysis and opinion top the list up
    to five. Opinion sits with the analysis, labeled with its byline."""
    key=lambda a:(a['date'],a.get('published_at',''))
    pool=sorted(current+featured_opinions(catalog,now),key=key,reverse=True)
    seen={lead['url']};out=[]
    for a in [*pool,*sorted(catalog,key=key,reverse=True)]:
        if a['url'] in seen:continue
        if len(out)>=5 and a not in pool:break
        seen.add(a['url']);out.append(a)
        if len(out)>=limit:break
    return out

def render_home(data,now):
    catalog=list(CFG['articles'])
    published=DOCS/'editorial/published.json'
    if published.exists():catalog+=json.loads(published.read_text())
    catalog.sort(key=lambda a:(a['date'],a.get('published_at','')),reverse=True)
    current=[a for a in catalog if featured_now(a,now)]
    fallback=dict(title='The daily market briefing',excerpt='Compare current prices across the leagues and follow what changes next.',sport='Sports',kind='Market watch',url='/briefing/',date=now.astimezone(ETZ).date().isoformat())
    lead=home_lead(current,fallback)
    stories=home_stories(current,catalog,lead,now)
    ctx=context(data,now)
    # The lead stands alone; the rest fill full rows of three and stay one click away.
    features=stories[:6];features=features[:len(features)-len(features)%3] if len(features)>3 else features
    ctx.update(lead=lead,features=features,next_up=next_up(now))
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
