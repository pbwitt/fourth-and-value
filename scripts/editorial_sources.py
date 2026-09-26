"""Bounded public-source collection. No paid search or generated source URLs."""
from datetime import timedelta,timezone
from email.utils import parsedate_to_datetime
from html import unescape
import json
import re
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
import requests
import editorial as ed

HEADERS={'User-Agent':'FourthAndValue/1.0 (+https://fourthandvalue.com/)'}
HOSTS=('espn.com','cbssports.com','mlb.com','nba.com','nhl.com','nhle.com','nfl.com','sports.yahoo.com')

def trusted(url):
    host=(urlsplit(url).hostname or '').lower()
    return ed.safe_url(url) and any(host==h or host.endswith('.'+h) for h in HOSTS)

def fetch(url):
    if not trusted(url):raise ValueError('Unapproved publisher')
    # Check each redirect before following it; never fetch generated or private URLs.
    for _ in range(4):
        r=requests.get(url,headers=HEADERS,timeout=15,allow_redirects=False,stream=True)
        if r.is_redirect:
            from urllib.parse import urljoin
            url=urljoin(url,r.headers.get('Location',''));r.close()
            if not trusted(url):raise ValueError('Unapproved redirect')
            continue
        r.raise_for_status()
        chunks=[];size=0
        for chunk in r.iter_content(65536):
            size+=len(chunk)
            if size>2500000:r.close();raise ValueError('Publisher page too large')
            chunks.append(chunk)
        r.close();return b''.join(chunks).decode('utf-8',errors='replace')
    raise ValueError('Too many redirects')

def text_content(html):
    # Prefer the publisher's structured article body when present.
    def bodies(value):
        if isinstance(value,dict):
            if isinstance(value.get('articleBody'),str):yield value['articleBody']
            for v in value.values():yield from bodies(v)
        elif isinstance(value,list):
            for v in value:yield from bodies(v)
    found=[]
    for script in re.findall(r'<script[^>]+type=[\"\']application/ld\+json[\"\'][^>]*>(.*?)</script>',html,re.S|re.I):
        try:found+=list(bodies(json.loads(script)))
        except ValueError:pass
    if found:return ' '.join(unescape(max(found,key=len)).split())
    clean=re.sub(r'<(script|style|nav|footer|header)\b[^>]*>.*?</\1>','',html,flags=re.S|re.I)
    articles=re.findall(r'<article\b[^>]*>(.*?)</article>',clean,re.S|re.I)
    if articles:clean=max(articles,key=len)
    paragraphs=[]
    for p in re.findall(r'<p\b[^>]*>(.*?)</p>',clean,re.S|re.I):
        p=' '.join(unescape(re.sub('<[^>]+>',' ',p)).split())
        if len(p.split())>=12 and not re.search(r'privacy policy|all rights reserved|subscribe to|sign up for|terms of use',p,re.I):paragraphs.append(p)
    return '\n\n'.join(dict.fromkeys(paragraphs))

def candidates(sport,now,limit=3):
    feeds=[f'https://sports.yahoo.com/{sport.lower()}/rss.xml',f'https://www.cbssports.com/rss/headlines/{sport.lower()}/',ed.CFG['news_feeds'][sport]]
    if sport=='MLB':feeds.insert(0,'https://www.mlb.com/feeds/news/rss.xml')
    found=[]
    for feed in feeds:
        try:
            tree=ET.fromstring(fetch(feed));count=0
            for item in tree.findall('.//item'):
                url=(item.findtext('link') or '').strip()
                try:date=parsedate_to_datetime(item.findtext('pubDate') or '').astimezone(timezone.utc)
                except (ValueError,TypeError):continue
                if not trusted(url) or not timedelta(0)<=now-date<=timedelta(hours=72):continue
                title=' '.join((item.findtext('title') or '').split())
                if not title:continue
                found.append(dict(title=title[:200],url=url,published_at=date.date().isoformat(),published_timestamp=date.isoformat()))
                count+=1
                if count>=limit:break
        except (requests.RequestException,ValueError,ET.ParseError):continue
    return found

def nfl_reporting(now,terms=()):
    """Read dated league reporting when syndicated feeds cannot supply diversity."""
    try:index=fetch('https://www.nfl.com/news')
    except (requests.RequestException,ValueError):return []
    urls=list(dict.fromkeys(unescape(url) for url in re.findall(
        r'href=[\"\'](https://www\.nfl\.com/news/[^\"\'?#]+)[\"\']',index)))
    urls=[url for url in urls if '/news/series/' not in url
        and (not terms or any(term in url.replace('-',' ').lower() for term in terms))]
    for url in urls[:8]:
        try:
            html=fetch(url)
            for script in re.findall(r'<script[^>]+type=[\"\']application/ld\+json[\"\'][^>]*>(.*?)</script>',html,re.S|re.I):
                try:article=json.loads(script)
                except ValueError:continue
                if not isinstance(article,dict) or article.get('@type')!='NewsArticle':continue
                date=ed.stamp(article.get('datePublished',''))
                if not timedelta(0)<=now-date<=timedelta(hours=72):continue
                title=str(article.get('headline','')).strip()
                excerpt=text_content(html)
                if not title or len(excerpt.split())<100:continue
                return [dict(title=title[:200],url=url,published_at=date.date().isoformat(),
                    published_timestamp=date.isoformat(),excerpt=excerpt[:2300],retrieved_at=now.isoformat())]
        except (requests.RequestException,ValueError,TypeError):continue
    return []


def collect(sport,now,seen_urls=(),terms=()):
    # Three headlines per publisher can all be blocked or too short. Search a
    # bounded deeper pool before declaring an entire league unavailable.
    found=candidates(sport,now,limit=30 if terms else 12)
    if terms:found=[s for s in found if any(term in (s['title']+' '+s['url'].replace('-', ' ').replace('_', ' ')).lower() for term in terms) and not re.search(r'promo code|bonus bets|sign.up offer',s['title'],re.I)]
    # Prefer new reporting; a materially updated story may still use an older source.
    found.sort(key=lambda s:(s['url'] in seen_urls,-ed.stamp(s['published_timestamp']).timestamp()))
    selected=[];hosts=set();failures={}
    for source in found:
        host=urlsplit(source['url']).hostname
        if len(selected)>=2 and host in hosts:continue
        try:
            excerpt=text_content(fetch(source['url']))
            if len(excerpt.split())<100:
                failures[host]='No readable article body';continue
            selected.append(dict(source,id=f's{len(selected)+1}',excerpt=excerpt[:2300],retrieved_at=now.isoformat()))
            hosts.add(host)
            if len(selected)>=3 and len(hosts)>=2:break
        except (requests.RequestException,ValueError) as exc:
            failures[host]=type(exc).__name__+': '+str(exc)[:180]
    if len(hosts)<2 and sport=='NFL':
        for source in nfl_reporting(now,terms):
            host=urlsplit(source['url']).hostname
            if host not in hosts:
                selected.append(dict(source,id=f's{len(selected)+1}'));hosts.add(host)
    if len(hosts)<2:
        print('Source availability: '+json.dumps(dict(sport=sport,candidates=len(found),readable_hosts=sorted(hosts),failures=failures)),flush=True)
    return selected if len(hosts)>=2 else []
