"""Bounded public NHL reporting, with actual retrieval and publication timestamps."""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import hashlib
import json
import re
import unicodedata
from urllib.parse import urljoin, urlsplit
import xml.etree.ElementTree as ET

import requests

from .data import digest, iso, stamp

NFL_TEAM_SITES = dict(zip(
    ['Arizona Cardinals','Atlanta Falcons','Baltimore Ravens','Buffalo Bills','Carolina Panthers','Chicago Bears',
     'Cincinnati Bengals','Cleveland Browns','Dallas Cowboys','Denver Broncos','Detroit Lions','Green Bay Packers',
     'Houston Texans','Indianapolis Colts','Jacksonville Jaguars','Kansas City Chiefs','Las Vegas Raiders',
     'Los Angeles Chargers','Los Angeles Rams','Miami Dolphins','Minnesota Vikings','New England Patriots',
     'New Orleans Saints','New York Giants','New York Jets','Philadelphia Eagles','Pittsburgh Steelers',
     'San Francisco 49ers','Seattle Seahawks','Tampa Bay Buccaneers','Tennessee Titans','Washington Commanders'],
    ['azcardinals.com','atlantafalcons.com','baltimoreravens.com','buffalobills.com','panthers.com','chicagobears.com',
     'bengals.com','clevelandbrowns.com','dallascowboys.com','denverbroncos.com','detroitlions.com','packers.com',
     'houstontexans.com','colts.com','jaguars.com','chiefs.com','raiders.com','chargers.com','therams.com',
     'miamidolphins.com','vikings.com','patriots.com','neworleanssaints.com','giants.com','newyorkjets.com',
     'philadelphiaeagles.com','steelers.com','49ers.com','seahawks.com','buccaneers.com','tennesseetitans.com','commanders.com']))
HOSTS = ('nhl.com', 'mlb.com', 'nfl.com', 'espn.com', 'cbssports.com', 'actionnetwork.com', 'covers.com', 'vsin.com', *NFL_TEAM_SITES.values())
FEEDS = ('https://www.espn.com/espn/rss/nhl/news',
         'https://www.cbssports.com/rss/headlines/nhl/')


def trusted(url):
    try:
        p = urlsplit(url)
        return p.scheme == 'https' and not p.username and not p.password and p.port in (None, 443) and any(
            p.hostname == h or (p.hostname or '').endswith('.'+h) for h in HOSTS)
    except (ValueError, TypeError):
        return False


def fetch(url):
    for _ in range(4):
        if not trusted(url):
            raise ValueError('Publisher not allowed')
        with requests.get(url, headers={'User-Agent': 'FourthAndValue/1.0 (+https://fourthandvalue.com/)'},
                          timeout=(5, 8), stream=True, allow_redirects=False) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers.get('Location', ''))
                continue
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 2500000:
                    raise ValueError('Publisher response too large')
                chunks.append(chunk)
            return b''.join(chunks).decode('utf-8', errors='replace')
    raise ValueError('Too many redirects')


def failure_category(error):
    """Sanitized retrieval failure label: no URL, header or body text is retained."""
    if isinstance(error, requests.HTTPError) and getattr(error, 'response', None) is not None:
        return f'http_{error.response.status_code}'
    if isinstance(error, requests.Timeout):
        return 'timeout'
    if isinstance(error, requests.ConnectionError):
        return 'connection'
    return {'Publisher not allowed': 'not_allowed', 'Publisher response too large': 'too_large',
            'Too many redirects': 'redirects'}.get(str(error), 'parse_or_validation')


def structured(html):
    def walk(value):
        if isinstance(value, dict):
            if value.get('@type') in ('NewsArticle', 'Article', 'ReportageNewsArticle'):
                yield value
            for v in value.values():
                yield from walk(v)
        elif isinstance(value, list):
            for v in value:
                yield from walk(v)
    for content in re.findall(r'<script[^>]+type=[\"\']application/ld\+json[\"\'][^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            yield from walk(json.loads(content))
        except ValueError:
            continue


def plain(html):
    return ' '.join(unescape(re.sub('<[^>]+>', ' ', html)).split())


def article_text(html):
    # Injury reports often put all actual designations in HTML tables while
    # articleBody contains only section names and the legend.
    tables = injury_tables(html)
    if tables:
        return ' '.join(tables)
    bodies = [a['articleBody'] for a in structured(html) if isinstance(a.get('articleBody'), str)]
    if bodies:
        return plain(max(bodies, key=len))
    clean = re.sub(r'<(script|style|nav|header|footer)\b[^>]*>.*?</\1>', '', html, flags=re.S | re.I)
    articles = re.findall(r'<article\b[^>]*>(.*?)</article>', clean, re.S | re.I)
    clean = max(articles, key=len) if articles else clean
    paras = [plain(p) for p in re.findall(r'<p\b[^>]*>(.*?)</p>', clean, re.S | re.I)]
    return ' '.join(dict.fromkeys(p for p in paras if len(p.split()) >= 12 and not re.search(
        r'privacy policy|subscribe|sign up|terms of use|all rights reserved', p, re.I)))


def injury_tables(html):
    result = []
    for table in re.finditer(r'<table\b[^>]*>(.*?)</table>', html, re.S | re.I):
        rows = [[plain(c) for c in re.findall(r'<t[hd]\b[^>]*>(.*?)</t[hd]>', r, re.S | re.I)]
                for r in re.findall(r'<tr\b[^>]*>(.*?)</tr>', table[1], re.S | re.I)]
        if not rows:
            continue
        header = [c.upper() for c in rows[0]]
        if not {'PLAYER', 'INJURY', 'GAME STATUS'}.issubset(header):
            continue
        headings = re.findall(r'<h[1-6]\b[^>]*>(.*?)</h[1-6]>', html[:table.start()], re.S | re.I)
        team = plain(headings[-1]) if headings else 'Injury report'
        # Keep complete table rows, ordered by relevance to offensive props.
        if any(len(r) != len(header) for r in rows[1:]):
            raise ValueError('Incomplete official injury row')
        parsed = [dict(zip(header, r)) for r in rows[1:]]
        parsed.sort(key=lambda r: r.get('POSITION', '') not in ('QB','WR','RB','FB','TE','T','G','C','OT','OG','OL'))
        for r in parsed:
            practice = next((f"; practice {day}: {r[day]}" for day in ('SAT','FRI','THURS','THU','WED') if r.get(day)), '')
            result.append(f"{team}: {r.get('POSITION', '')} {r['PLAYER']}; injury: {r['INJURY']}; game status: {r['GAME STATUS'] or 'not listed'}{practice}.")
    return result


def matching_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    text = re.sub(r'[^a-z0-9]+', ' ', text.casefold()).strip()
    # C.J., C-J and CJ are equivalent full-name forms, without surname-only matches.
    return re.sub(r'\b([a-z]) ([a-z])\b', r'\1\2', text)


def terms(row):
    names = [row.get('player', ''), row.get('home_team', ''), row.get('away_team', '')]
    # Full names and unambiguous team nicknames; never a city alone or player surname alone.
    names += [' '.join(team.split()[-2:]) if team.endswith(('Maple Leafs', 'Red Wings', 'Blue Jackets', 'Golden Knights', 'Red Sox', 'White Sox', 'Blue Jays'))
              else team.split()[-1] for team in names[1:] if team]
    return [matching_text(n) for n in names if len(n) >= 4]


def matches(row, text):
    text = matching_text(text)
    return any(re.search(r'(?<!\w)'+re.escape(t)+r'(?!\w)', text) for t in terms(row))


def reporting_priority(row, source):
    text = (source['title']+' '+source['url']).casefold()
    # Facts affecting opportunity get priority over generic team coverage. Other
    # outlets' betting selections are not evidence of our own edge.
    if re.search(r'best.bets|betting.picks|picks.odds|odds.best|promo.code|expert.picks', text):
        return -1
    priority = 5 if re.search(r'injur|lineup|practice|active|starter|weather|bullpen|scratch|pitcher', text) else 0
    if row.get('player') and matching_text(row['player']) in matching_text(text):
        priority += 3
    # A matchup-specific report beats an unrelated old team injury article.
    teams = [row.get('home_team'), row.get('away_team')]
    if all(teams) and all(matching_text(t.split()[-1]) in matching_text(text) for t in teams):
        priority += 2
    if (urlsplit(source['url']).hostname or '').removeprefix('www.') in ('nfl.com','mlb.com','nhl.com', *NFL_TEAM_SITES.values()):
        priority += 1
    return priority


def usable(source, row, asof):
    try:
        if source.get('source_kind') == 'live_injury_table':
            return (source['url'] in ('https://www.cbssports.com/mlb/injuries/', 'https://www.cbssports.com/nhl/injuries/')
                    and source.get('published_at') is None
                    and asof-timedelta(minutes=90) <= stamp(source['retrieved_at']) <= asof
                    and row['candidate_id'] in source['candidate_ids'] and bool(source.get('injury_rows'))
                    and bool(source['excerpt']) and matches(row, source['title']+' '+source['excerpt']))
        published, retrieved = stamp(source['published_at']), stamp(source['retrieved_at'])
        updated = stamp(source['updated_at']) if source.get('updated_at') else published
        return (trusted(source['url']) and asof-timedelta(days=7) <= published <= updated <= retrieved <= asof
                and asof-timedelta(hours=72) <= updated
                and row['candidate_id'] in source['candidate_ids'] and bool(source['excerpt'])
                and matches(row, source['title']+' '+source['excerpt']))
    except (KeyError, TypeError, ValueError):
        return False


def collect(rows, clock=lambda: datetime.now(timezone.utc), sport='NHL'):
    """Direct MLB/NHL injury table, league/team indexes, at most 16 articles."""
    if not rows:
        return [], {'status': 'no_candidates', 'failures': []}
    pool, sources, failures = [], [], []
    now = clock()
    if sport not in ('NHL', 'MLB', 'NFL'):
        raise ValueError('Unsupported reporting sport')
    injury_status = {'status': 'official_articles'}
    if sport in ('MLB', 'NHL'):
        from .injuries import collect as collect_injuries
        sources, injury_status = collect_injuries(rows, clock, sport, fetch)
    feeds = FEEDS if sport == 'NHL' else (
        f'https://www.espn.com/espn/rss/{sport.lower()}/news',
        f'https://www.cbssports.com/rss/headlines/{sport.lower()}/')
    for feed in feeds:
        try:
            tree = ET.fromstring(fetch(feed))
            for item in tree.findall('.//item')[:40]:
                try:
                    date = parsedate_to_datetime(item.findtext('pubDate') or '')
                    if date.tzinfo is None:
                        continue
                    title, url = item.findtext('title') or '', item.findtext('link') or ''
                    if trusted(url) and timedelta(0) <= now-date <= timedelta(days=7):
                        pool.append(dict(title=plain(title)[:200], url=url.strip(), published_at=iso(date)))
                except (ValueError, TypeError):
                    continue
        except (requests.RequestException, ValueError, ET.ParseError) as error:
            failures.append({'host': urlsplit(feed).hostname, 'stage': 'feed_unavailable', 'category': failure_category(error)})
    try:
        domain = sport.lower()+'.com'
        index = fetch(f'https://www.{domain}/news/')
        for href in dict.fromkeys(unescape(u) for u in re.findall(r'href=[\"\']([^\"\'?#]+)[\"\']', index)):
            url = urljoin('https://www.'+domain, href)
            if trusted(url) and '/news/' in url and any(matches(r, url) for r in rows):
                pool.append(dict(url=url, title='', published_at=None))
    except (requests.RequestException, ValueError) as error:
        failures.append({'host': 'www.'+domain, 'stage': 'index_unavailable', 'category': failure_category(error)})
    if sport == 'NFL':
        domains = list(dict.fromkeys(NFL_TEAM_SITES[t] for r in rows for t in
            (r.get('home_team'), r.get('away_team')) if t in NFL_TEAM_SITES))[:8]
        for team_domain in domains:
            try:
                index = fetch('https://www.'+team_domain+'/news/')
                for href in dict.fromkeys(unescape(u) for u in re.findall(r'href=[\"\']([^\"\'?#]+)[\"\']', index)):
                    url = urljoin('https://www.'+team_domain, href)
                    if trusted(url) and '/news/' in url and any(matches(r, url) for r in rows):
                        pool.append(dict(url=url, title='', published_at=None))
            except (requests.RequestException, ValueError) as error:
                failures.append({'host': team_domain, 'stage': 'team_index_unavailable', 'category': failure_category(error)})
    seen, attempts, counts = set(), 0, {r['candidate_id']: sum(r['candidate_id'] in s['candidate_ids'] for s in sources) for r in rows}
    rejected = []
    # Round-robin by candidate avoids spending every fetch on the first matchup.
    queues = [sorted([s for s in pool if matches(r, s['title']+' '+s['url']) and reporting_priority(r, s) >= 0],
                     key=lambda s: (reporting_priority(r, s), s.get('published_at') or ''), reverse=True) for r in rows]
    for n in range(max((len(q) for q in queues), default=0)):
        for row, queue in zip(rows, queues):
            if n >= len(queue) or counts[row['candidate_id']] >= 2 or attempts >= 16:
                continue
            source = queue[n]
            if source['url'] in seen:
                continue
            seen.add(source['url']); attempts += 1
            try:
                html = fetch(source['url']); retrieved = clock()
                articles = list(structured(html))
                dated = next((a for a in articles if a.get('datePublished') and a.get('headline')), None)
                if not source['published_at']:
                    if not dated:
                        rejected.append(dict(url=source['url'], reason='publication_time_missing'))
                        continue
                    source = dict(source, title=plain(dated['headline'])[:200], published_at=iso(stamp(dated['datePublished'])))
                if dated and dated.get('dateModified'):
                    source = dict(source, updated_at=iso(stamp(dated['dateModified'])))
                text = article_text(html)
                published = stamp(source['published_at'])
                effective = stamp(source.get('updated_at') or source['published_at'])
                if len(text.split()) < 60 or not (retrieved-timedelta(days=7) <= published <= effective <= retrieved
                        and retrieved-timedelta(hours=72) <= effective):
                    rejected.append(dict(url=source['url'], reason='insufficient_text_or_stale_timestamp'))
                    continue
                table_rows = injury_tables(html)
                excerpt = text[:1400]
                if table_rows:
                    player = matching_text(row.get('player') or '')
                    ordered = sorted(table_rows, key=lambda t: not (player and player in matching_text(t)))
                    excerpt = 'Injury table excerpt; additional players may be listed in the full report.'
                    for line in ordered:
                        if len(excerpt)+len(line)+1 <= 1400:
                            excerpt += '\n'+line
                ids = [r['candidate_id'] for r in rows if counts[r['candidate_id']] < 2 and matches(r, source['title']+' '+excerpt)]
                if not ids:
                    continue
                source = dict(source, excerpt=excerpt, retrieved_at=iso(retrieved), candidate_ids=ids,
                              content_sha256=hashlib.sha256(html.encode()).hexdigest(),
                              publication_basis='publisher_rss_or_article_metadata')
                if table_rows:
                    source['source_kind'] = 'official_injury_report'
                    source['injury_table_rows'] = table_rows
                source['source_id'] = digest(source)[:20]
                sources.append(source)
                for cid in ids:
                    counts[cid] += 1
            except (requests.RequestException, ValueError, TypeError) as error:
                failures.append({'host': urlsplit(source['url']).hostname, 'stage': 'article_unavailable', 'category': failure_category(error)})
    return sources, dict(status='available' if sources else 'no_usable_reporting', attempts=attempts,
                         failures=failures, rejected=rejected, coverage=counts, injury_tables=injury_status)


def attach_context(board, diagnostics):
    for row in board['candidates']:
        status = diagnostics.get('injury_tables', {})
        row['injury_context'] = status.get('candidates', {}).get(row['candidate_id'], {'status': status.get('status', 'unavailable')})


def targeted(rows, urls, clock=lambda: datetime.now(timezone.utc)):
    """Retrieve bounded search leads; search snippets alone never support a bet."""
    sources, failures = [], []
    for url in list(dict.fromkeys(urls))[:24]:
        if not trusted(url):
            failures.append(dict(url=url, reason='publisher_not_allowed')); continue
        try:
            html=fetch(url); retrieved=clock()
            article=next((a for a in structured(html) if a.get('datePublished') and a.get('headline')),None)
            if not article: raise ValueError('Dated article unavailable')
            text=article_text(html)
            if len(text.split())<60: raise ValueError('Insufficient article text')
            # Keep the part that actually concerns the researched subject/game.
            parts=re.split(r'(?<=[.!?])\s+',text)
            relevant=[p for p in parts if any(matches(r,p) for r in rows)]
            excerpt=' '.join(relevant+parts)[:1400]
            source=dict(url=url,title=plain(article['headline'])[:200],excerpt=excerpt,
                published_at=iso(stamp(article['datePublished'])),
                updated_at=iso(stamp(article.get('dateModified') or article['datePublished'])),
                retrieved_at=iso(retrieved),publication_basis='publisher_article_metadata',
                content_sha256=hashlib.sha256(html.encode()).hexdigest(),
                source_kind='professional_opinion' if re.search(r'best bets|picks|predictions',article['headline'],re.I) else 'reporting',
                candidate_ids=[r['candidate_id'] for r in rows if matches(r,article['headline']+' '+excerpt)])
            source['source_id']=digest(source)[:20]
            if any(usable(source,r,retrieved) for r in rows): sources.append(source)
            else: raise ValueError('Stale or mismatched article')
        except (requests.RequestException,ValueError,TypeError) as error:
            failures.append(dict(url=url,reason='unverified_or_unavailable',category=failure_category(error)))
    return sources,failures
