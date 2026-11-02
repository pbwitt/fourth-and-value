#!/usr/bin/env python3
"""Owner-only "ready to post on X" email: today's morning card and new articles.

Free by design: nothing here calls the X API (pay-per-use since February 2026, with a
higher price for posts that contain a link). Each draft is an x.com web intent: one tap
opens X's composer pre-filled, the owner can edit, and the owner posts it.

Picks come from the shared selector (docs/assets/briefing-picks.js editionRows), so a
draft lists exactly the rows Today's Picks shows, with their offered lines, prices and
books unchanged. Times are Eastern. A small ledger (one per Eastern day, kept in the
Actions cache) stops a card or article from being emailed twice.

  python scripts/social_drafts.py --preview /tmp/drafts.html   # build only, send nothing
  python scripts/social_drafts.py --ledger L.json --verify-live --send
"""
import argparse
from datetime import datetime, time, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata
from urllib.parse import quote

from display_time import EASTERN

ROOT = Path(__file__).resolve().parents[1]
SITE = 'https://fourthandvalue.com'
CARD_FILE = ROOT / 'docs/briefing/morning-card.json'
ARTICLES_FILE = ROOT / 'docs/editorial/published.json'
SELECTOR = ROOT / 'docs/assets/briefing-picks.js'
PICKS_URL = SITE + '/briefing/'
INTENT = 'https://x.com/intent/tweet?text='
LIMIT = 280
URL_WEIGHT = 23  # X shortens every link to t.co, which counts as 23 characters.
# Wait this long for a finished morning card so the card and articles share one email.
HOLD_UNTIL = time(9, 45)

SPORT_EMOJI = {'NFL': '🏈', 'NBA': '🏀', 'MLB': '⚾', 'NHL': '🏒'}
# Labels like 'BetOnline.ag' would turn into links on X, so offshore books drop the domain.
BOOKS = {'draftkings': 'DK', 'fanduel': 'FD', 'betmgm': 'MGM', 'williamhill_us': 'Caesars', 'espnbet': 'ESPN BET',
         'betonlineag': 'BetOnline', 'lowvig': 'LowVig', 'mybookieag': 'MyBookie'}
# Abbreviations used on betting X. Unknown markets fall back to the published label.
MARKETS = {
    'passing attempts': 'pass att', 'passing completions': 'pass comp', 'passing touchdowns': 'pass TD',
    'passing yards': 'pass yds', 'receiving yards': 'rec yds', 'receptions': 'rec',
    'rushing attempts': 'rush att', 'rushing yards': 'rush yds',
    'points': 'pts', 'rebounds': 'reb', 'assists': 'ast', 'threes': '3PM', 'three pointers made': '3PM',
    'points + rebounds + assists': 'PRA', 'shots on goal': 'SOG', 'blocked shots': 'blocks',
    'batter hits': 'hits', 'batter rbis': 'RBI', 'batter total bases': 'TB', 'batter home runs': 'HR',
    'pitcher outs': 'outs', 'pitcher strikeouts': 'K',
}
GAME_MARKETS = {'moneyline': 'ML', 'h2h': 'ML', 'run line': 'run line', 'puck line': 'puck line',
                'spread': 'spread', 'spreads': 'spread', 'game total': 'total', 'totals': 'total'}
# Mirrors twitter-text v3: these code points weigh 1, everything else 2.
LIGHT_RANGES = ((0, 4351), (8192, 8205), (8208, 8223), (8242, 8247))
# Full links, plus bare domains such as 'example.ag' that X also turns into 23-character links.
URL_RE = re.compile(r'https?://\S+|\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b(?:/\S*)?')


def weighted_length(text):
    """X's weighted count, rounded up for emoji sequences and link-like text, so a draft never runs over."""
    # X counts every full link as 23; a bare domain it may leave as text, so take the longer.
    text = URL_RE.sub(lambda m: 'x' * (URL_WEIGHT if m.group().startswith('http')
                                         else max(URL_WEIGHT, len(m.group()))), text)
    text = unicodedata.normalize('NFC', text)
    return sum(1 if any(lo <= ord(c) <= hi for lo, hi in LIGHT_RANGES) else 2 for c in text)


def parse_time(value):
    t = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('A source time zone is required')
    return t


def clock(value):
    """'7:21 AM EST' in Eastern time, EDT in daylight time."""
    t = parse_time(value).astimezone(EASTERN)
    return f"{t.hour % 12 or 12}:{t:%M} {'AM' if t.hour < 12 else 'PM'} {t:%Z}"


def day_label(iso_date):
    d = datetime.fromisoformat(iso_date)
    return f'{d:%a}, {d:%b} {d.day}'


def number(value, signed=False):
    """The offered line exactly: 31.5 stays 31.5, a whole line drops the trailing .0."""
    v = float(value)
    text = str(int(v)) if v.is_integer() else repr(v)
    return ('+' + text) if signed and v > 0 else text


def american(price):
    try:
        v = float(price)
    except (TypeError, ValueError):
        return str(price)
    if not v.is_integer():
        return str(price)
    return f'+{int(v)}' if v > 0 else str(int(v))


def side_prefix(side):
    s = str(side or '').strip().lower()
    return 'o' if s == 'over' else 'u' if s == 'under' else None


def pick_line(row):
    """One card row as '🏈 Daniel Jones o31.5 pass att -104 DK'."""
    emoji = SPORT_EMOJI.get(row.get('sport'), '•')
    label = str(row.get('market_label') or row.get('market') or '').strip()
    book = BOOKS.get(row.get('book')) or row.get('book_label') or row.get('book') or ''
    side = str(row.get('side') or '').strip()
    line = row.get('line')
    prefix = side_prefix(side)
    if row.get('player'):
        market = MARKETS.get(label.lower(), label.lower())
        if prefix and line is not None:
            body = f"{row['player']} {prefix}{number(line)} {market}"
        else:
            body = ' '.join(x for x in (row['player'], side, number(line) if line is not None else '', market) if x)
    else:
        market = GAME_MARKETS.get(label.lower()) or GAME_MARKETS.get(str(row.get('market') or '').lower()) or label.lower()
        if market == 'total' and prefix and line is not None:
            body = f"{row.get('game') or ''} {prefix}{number(line)} total".strip()
        elif market == 'ML':
            body = f'{side} ML'
        else:
            body = ' '.join(x for x in (side, number(line, signed=True) if line is not None else '', market) if x)
    return f'{emoji} {body} {american(row.get("price"))} {book}'.strip()


def card_draft(card, rows):
    """Fit as many picks as X allows, in card order; the rest are counted, not dropped silently."""
    head = f"Today’s picks · {day_label(card['decision_date'])}"
    foot = f"Odds as posted {clock(card['published_at'])}; lines move.\n{PICKS_URL}"
    lines = [pick_line(r) for r in rows]
    shown, hidden = list(lines), 0
    while shown:
        more = [f'+{hidden} more on the card'] if hidden else []
        text = '\n'.join([head, ''] + shown + more + ['', foot])
        if weighted_length(text) <= LIMIT:
            return text, len(shown)
        shown.pop()
        hidden += 1
    return '\n'.join([head, '', f'{len(lines)} picks on today’s card', '', foot]), 0


def trim(text, budget):
    """Cut a headline at a word boundary with an ellipsis so it weighs at most budget."""
    if weighted_length(text) <= budget:
        return text
    words = text.split()
    while words and weighted_length(' '.join(words) + '…') > budget:
        words.pop()
    return (' '.join(words).rstrip(',;:—–-') + '…') if words else ''


def whole_sentences(text, budget):
    """Leading whole sentences that fit; a cut mid-sentence could reverse the meaning."""
    kept = ''
    for sentence in re.split(r'(?<=[.!?])\s+', text.strip()):
        candidate = f'{kept} {sentence}'.strip()
        if weighted_length(candidate) > budget:
            break
        kept = candidate
    return kept


def article_draft(article):
    emoji = SPORT_EMOJI.get(article.get('sport'))
    title = ' '.join(x for x in (emoji, str(article.get('title') or '').strip()) if x)
    url = SITE + article['url']
    title = trim(title, LIMIT - URL_WEIGHT - 2)
    room = LIMIT - weighted_length(title) - URL_WEIGHT - 4
    excerpt = whole_sentences(str(article.get('excerpt') or ''), room)
    parts = [title] + ([excerpt] if excerpt else []) + [url]
    return '\n\n'.join(parts)


def intent_url(text):
    return INTENT + quote(text, safe='')


def eastern_date(now):
    return now.astimezone(EASTERN).date().isoformat()


def card_state(card, now):
    """ready: today's finished card with picks; empty: finished, no picks; pending: anything else."""
    try:
        if (card.get('schema_version') != 1 or card.get('kind') != 'morning'
                or card.get('decision_date') != eastern_date(now)
                or eastern_date(parse_time(card['published_at'])) != card['decision_date']
                or parse_time(card['published_at']) > now or not card.get('edition_id')):
            return 'pending'
    except (KeyError, TypeError, ValueError, AttributeError):
        return 'pending'
    if card.get('status') == 'no_reviewed_candidates':
        return 'empty'
    return 'ready' if card.get('status') == 'published' else 'pending'


def displayed_rows(card, now):
    """The rows Today's Picks shows, from the shared browser/Node selector."""
    script = ("const p=require(process.argv[1]);let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{"
              "const {card,now}=JSON.parse(s);process.stdout.write(JSON.stringify(p.editionRows(card,Date.parse(now))))})")
    result = subprocess.run(['node', '-e', script, str(SELECTOR)], input=json.dumps({'card': card, 'now': now.isoformat()}),
                            text=True, capture_output=True, check=True, timeout=30, cwd=ROOT)
    return json.loads(result.stdout)


def todays_articles(articles, now):
    today = eastern_date(now)
    picked = []
    for a in articles:
        try:
            published = parse_time(a['published_at'])
        except (KeyError, TypeError, ValueError):
            continue
        if eastern_date(published) == today and published <= now and str(a.get('url', '')).startswith('/'):
            picked.append(a)
    return sorted(picked, key=lambda a: a['published_at'])


def live(path, check=None):
    """True once GitHub Pages serves the page; a cache-busting query skips stale CDN copies."""
    import requests
    try:
        r = requests.get(f'{SITE}{path}?fv={int(datetime.now(timezone.utc).timestamp())}', timeout=20)
        return r.status_code == 200 and (check is None or check(r))
    except requests.RequestException:
        return False


def collect(card, articles, now, sent, *, rows_for=displayed_rows, is_live=None, hold=True):
    """Return (drafts, handled ids, status message). handled ids join the ledger after sending."""
    drafts, handled = [], []
    state = card_state(card, now)
    if state == 'pending' and hold and now.astimezone(EASTERN).time() < HOLD_UNTIL:
        return [], [], 'Waiting for today’s finished morning card.'
    if state in ('ready', 'empty'):
        card_id = 'card:' + card['edition_id']
        if card_id not in sent:
            handled.append(card_id)
            if state == 'ready':
                rows = [r for r in rows_for(card, now) if parse_time(r['commence_time']) > now]
                if rows and (is_live is None or is_live('/briefing/morning-card.json',
                                                        lambda r: r.json().get('edition_id') == card['edition_id'])):
                    text, shown = card_draft(card, rows)
                    first = min(parse_time(r['commence_time']) for r in rows)
                    replaces = any(s.startswith('card:') for s in sent)
                    note = (f'{len(rows)} picks; first game {clock(first)}.'
                            + (f' The draft fits {shown}; the rest are counted.' if shown < len(rows) else '')
                            + (' This replaces the card in an earlier email today.' if replaces else ''))
                    drafts.append(dict(id=card_id, title='Today’s card', text=text, note=note))
                elif rows:
                    handled.remove(card_id)  # Not live yet: try again next run.
    for a in todays_articles(articles, now):
        article_id = 'article:' + a['url']
        if article_id in sent:
            continue
        if is_live is not None and not is_live(a['url']):
            continue
        handled.append(article_id)
        drafts.append(dict(id=article_id, title=f"{a.get('sport', '')} article".strip(),
                           text=article_draft(a), note=f"Published {clock(a['published_at'])}."))
    return drafts, handled, f'{len(drafts)} new drafts.' if drafts else 'Nothing new to draft.'


def subject(drafts):
    card = any(d['id'].startswith('card:') for d in drafts)
    n = sum(d['id'].startswith('article:') for d in drafts)
    parts = (['today’s card'] if card else []) + ([f"{n} article{'s' if n != 1 else ''}"] if n else [])
    return 'Ready to post on X: ' + ' and '.join(parts)


def email_bodies(drafts, now):
    when = now.astimezone(EASTERN)
    intro = (f"Tap a button to open X with the post filled in, edit if you like, then post. "
             f"Built {when:%a}, {when:%b} {when.day} at {clock(now)}.")
    text = [intro, '']
    blocks = []
    for d in drafts:
        url = intent_url(d['text'])
        count = weighted_length(d['text'])
        text += [d['title'], '-' * 40, d['text'], '', f"{d['note']} {count}/{LIMIT} characters.", f'Post on X: {url}', '', '']
        blocks.append(
            f'<div style="margin:0 0 28px">'
            f'<h2 style="font:600 16px/1.3 system-ui,sans-serif;margin:0 0 8px">{html.escape(d["title"])}</h2>'
            f'<div style="white-space:pre-wrap;font:15px/1.45 system-ui,sans-serif;border:1px solid #d0d7de;'
            f'border-radius:8px;padding:12px 14px;background:#f6f8fa;color:#1f2328;overflow-wrap:anywhere">{html.escape(d["text"])}</div>'
            f'<p style="font:13px/1.4 system-ui,sans-serif;color:#57606a;margin:6px 0 10px">'
            f'{html.escape(d["note"])} {count}/{LIMIT} characters.</p>'
            f'<a href="{html.escape(url)}" style="display:inline-block;background:#0f1419;color:#fff;'
            f'text-decoration:none;font:600 15px/1 system-ui,sans-serif;padding:12px 20px;border-radius:999px">'
            f'Post on X</a></div>')
    page = (f'<div style="max-width:560px;margin:0 auto;padding:16px">'
            f'<p style="font:14px/1.45 system-ui,sans-serif;color:#1f2328">{html.escape(intro)}</p>'
            + ''.join(blocks) +
            '<p style="font:12px/1.4 system-ui,sans-serif;color:#57606a">Sent by the X Post Drafts workflow. '
            'Nothing is posted until you tap Post.</p></div>')
    return '\n'.join(text), page


def send(drafts, now):
    import requests
    key, sender = os.getenv('RESEND_API_KEY'), os.getenv('EDITORIAL_NOTIFY_FROM')
    recipient = os.getenv('SOCIAL_DRAFTS_EMAIL') or os.getenv('EDITORIAL_NOTIFY_EMAIL')
    if not all([key, sender, recipient]):
        raise SystemExit('Email not configured: RESEND_API_KEY, EDITORIAL_NOTIFY_FROM and EDITORIAL_NOTIFY_EMAIL are required.')
    text, page = email_bodies(drafts, now)
    ids = hashlib.sha256('|'.join(sorted(d['id'] for d in drafts)).encode()).hexdigest()[:20]
    r = requests.post('https://api.resend.com/emails', timeout=30,
                      headers={'Authorization': 'Bearer ' + key, 'Idempotency-Key': f'fv-x-drafts-{eastern_date(now)}-{ids}'},
                      json={'from': sender, 'to': [recipient], 'subject': subject(drafts), 'text': text, 'html': page})
    if r.status_code == 409:
        print('Resend reports these drafts were already sent.')
    elif not r.ok:
        raise SystemExit(f'Draft email failed: HTTP {r.status_code}')  # Body omitted: it may echo the address.


def load_ledger(path, now):
    try:
        data = json.loads(Path(path).read_text())
        return set(data.get('sent', [])) if data.get('date') == eastern_date(now) else set()
    except (OSError, ValueError, AttributeError):
        return set()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ledger', help='JSON file of ids already emailed today')
    ap.add_argument('--send', action='store_true', help='Email the drafts (otherwise print only)')
    ap.add_argument('--verify-live', action='store_true', help='Hold items until fourthandvalue.com serves them')
    ap.add_argument('--resend-today', action='store_true', help='Ignore today’s ledger and the card hold')
    ap.add_argument('--preview', help='Write the email HTML here')
    ap.add_argument('--now', help='Pretend it is this ISO time (previews and tests)')
    args = ap.parse_args(argv)

    now = parse_time(args.now) if args.now else datetime.now(timezone.utc)
    ledger = load_ledger(args.ledger, now) if args.ledger else set()
    sent = set() if args.resend_today else ledger
    card = json.loads(CARD_FILE.read_text())
    articles = json.loads(ARTICLES_FILE.read_text())
    drafts, handled, message = collect(card, articles, now, sent, is_live=live if args.verify_live else None,
                                       hold=not args.resend_today)
    print(message)
    for d in drafts:  # Public content only: picks and articles already on the site.
        print(f"\n--- {d['title']} ({weighted_length(d['text'])}/{LIMIT})\n{d['text']}")
    if args.preview and drafts:
        Path(args.preview).write_text(email_bodies(drafts, now)[1])
        print(f'\nPreview written to {args.preview}')
    if drafts and args.send:
        send(drafts, now)
        print(f'Emailed {len(drafts)} drafts.')
    changed = bool(handled) and (args.send or not drafts)
    if args.ledger and changed:
        Path(args.ledger).write_text(json.dumps({'date': eastern_date(now), 'sent': sorted(ledger | set(handled))}))
    if os.getenv('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
            out.write(f'ledger_changed={"true" if args.ledger and changed else "false"}\n')


if __name__ == '__main__':
    sys.exit(main())
