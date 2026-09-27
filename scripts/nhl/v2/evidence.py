"""Bounded public NHL reporting, with actual retrieval and publication timestamps."""
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import hashlib
import json
import re
from urllib.parse import urljoin, urlsplit
import xml.etree.ElementTree as ET

import requests

from .data import digest, iso, stamp

HOSTS = ('nhl.com', 'mlb.com', 'nfl.com', 'espn.com', 'cbssports.com')
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
    bodies = [a['articleBody'] for a in structured(html) if isinstance(a.get('articleBody'), str)]
    if bodies:
        return plain(max(bodies, key=len))
    clean = re.sub(r'<(script|style|nav|header|footer)\b[^>]*>.*?</\1>', '', html, flags=re.S | re.I)
    articles = re.findall(r'<article\b[^>]*>(.*?)</article>', clean, re.S | re.I)
    clean = max(articles, key=len) if articles else clean
    paras = [plain(p) for p in re.findall(r'<p\b[^>]*>(.*?)</p>', clean, re.S | re.I)]
    return ' '.join(dict.fromkeys(p for p in paras if len(p.split()) >= 12 and not re.search(
        r'privacy policy|subscribe|sign up|terms of use|all rights reserved', p, re.I)))


def terms(row):
    names = [row.get('player', ''), row.get('home_team', ''), row.get('away_team', '')]
    # Full names and unambiguous team nicknames; never a city alone or player surname alone.
    names += [' '.join(team.split()[-2:]) if team.endswith(('Maple Leafs', 'Red Wings', 'Blue Jackets', 'Golden Knights', 'Red Sox', 'White Sox', 'Blue Jays'))
              else team.split()[-1] for team in names[1:] if team]
    return [n.casefold() for n in names if len(n) >= 4]


def matches(row, text):
    text = text.casefold().replace('-', ' ')
    return any(re.search(r'(?<!\w)'+re.escape(t)+r'(?!\w)', text) for t in terms(row))


def reporting_priority(row, source):
    text = (source['title']+' '+source['url']).casefold()
    # Facts affecting opportunity get priority over generic team coverage. Other
    # outlets' betting selections are not evidence of our own edge.
    if re.search(r'best.bets|betting.picks|picks.odds|odds.best|promo.code|expert.picks', text):
        return -1
    priority = 5 if re.search(r'injur|lineup|practice|active|starter|weather|bullpen|scratch|pitcher', text) else 0
    if row.get('player') and row['player'].casefold() in text.replace('-', ' '):
        priority += 3
    if urlsplit(source['url']).hostname in ('www.nfl.com','www.mlb.com','www.nhl.com'):
        priority += 1
    return priority


def usable(source, row, asof):
    try:
        published, retrieved = stamp(source['published_at']), stamp(source['retrieved_at'])
        return (trusted(source['url']) and asof-timedelta(hours=72) <= published <= retrieved <= asof
                and row['candidate_id'] in source['candidate_ids'] and bool(source['excerpt'])
                and matches(row, source['title']+' '+source['excerpt']))
    except (KeyError, TypeError, ValueError):
        return False


def collect(rows, clock=lambda: datetime.now(timezone.utc), sport='NHL'):
    """Max three index requests, eight article requests, two sources/candidate."""
    if not rows:
        return [], {'status': 'no_candidates', 'failures': []}
    pool, sources, failures = [], [], []
    now = clock()
    if sport not in ('NHL', 'MLB', 'NFL'):
        raise ValueError('Unsupported reporting sport')
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
                    if trusted(url) and timedelta(0) <= now-date <= timedelta(hours=72):
                        pool.append(dict(title=plain(title)[:200], url=url.strip(), published_at=iso(date)))
                except (ValueError, TypeError):
                    continue
        except (requests.RequestException, ValueError, ET.ParseError):
            failures.append({'host': urlsplit(feed).hostname, 'stage': 'feed_unavailable'})
    try:
        domain = sport.lower()+'.com'
        index = fetch(f'https://www.{domain}/news/')
        for href in dict.fromkeys(unescape(u) for u in re.findall(r'href=[\"\']([^\"\'?#]+)[\"\']', index)):
            url = urljoin('https://www.'+domain, href)
            if trusted(url) and '/news/' in url and any(matches(r, url) for r in rows):
                pool.append(dict(url=url, title='', published_at=None))
    except (requests.RequestException, ValueError):
        failures.append({'host': 'www.'+domain, 'stage': 'index_unavailable'})
    seen, attempts, counts = set(), 0, {r['candidate_id']: 0 for r in rows}
    # Round-robin by candidate avoids spending every fetch on the first matchup.
    queues = [sorted([s for s in pool if matches(r, s['title']+' '+s['url']) and reporting_priority(r, s) >= 0],
                     key=lambda s: (reporting_priority(r, s), s.get('published_at') or ''), reverse=True) for r in rows]
    for n in range(max((len(q) for q in queues), default=0)):
        for row, queue in zip(rows, queues):
            if n >= len(queue) or counts[row['candidate_id']] >= 2 or attempts >= 8:
                continue
            source = queue[n]
            if source['url'] in seen:
                continue
            seen.add(source['url']); attempts += 1
            try:
                html = fetch(source['url']); retrieved = clock()
                articles = list(structured(html))
                if not source['published_at']:
                    dated = next((a for a in articles if a.get('datePublished') and a.get('headline')), None)
                    if not dated:
                        continue
                    source = dict(source, title=plain(dated['headline'])[:200], published_at=iso(stamp(dated['datePublished'])))
                text = article_text(html)
                if len(text.split()) < 60 or not retrieved-timedelta(hours=72) <= stamp(source['published_at']) <= retrieved:
                    continue
                ids = [r['candidate_id'] for r in rows if counts[r['candidate_id']] < 2 and matches(r, source['title']+' '+text[:1400])]
                if not ids:
                    continue
                source = dict(source, excerpt=text[:1400], retrieved_at=iso(retrieved), candidate_ids=ids,
                              content_sha256=hashlib.sha256(html.encode()).hexdigest(),
                              publication_basis='publisher_rss_or_article_metadata')
                source['source_id'] = digest(source)[:20]
                sources.append(source)
                for cid in ids:
                    counts[cid] += 1
            except (requests.RequestException, ValueError, TypeError):
                failures.append({'host': urlsplit(source['url']).hostname, 'stage': 'article_unavailable'})
    return sources, dict(status='available' if sources else 'no_usable_reporting', attempts=attempts,
                         failures=failures, coverage=counts)
