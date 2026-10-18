"""Owner email for the morning card: each pick with a suggested stake, the lowest price
that still makes the card, a link to the book and a link that opens Bet Tracker's form.

It reads the published docs/briefing/morning-card.json only. It never places a bet,
changes the card or calls a model. One email per edition: the Resend idempotency key is
the edition id, so recovery starts that re-verify a finished card do not send it again.
Every send honors the recipient's row in public.email_subscriptions
(supabase/email_subscriptions.sql): unsubscribed, or a row that cannot be read, sends
nothing. Without the Resend and Supabase settings it does nothing, like the editorial
notices in scripts/editorial_ideas.py.
"""
import argparse
from datetime import date, datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
from urllib.parse import urlencode

import requests

from display_time import EASTERN

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / 'docs/briefing/morning-card.json'
CONFIG = ROOT / 'config/morning_email.json'
RESEND = 'https://api.resend.com/emails'
LIST = 'morning_card'
SETTINGS = ('RESEND_API_KEY', 'EDITORIAL_NOTIFY_FROM', 'EDITORIAL_NOTIFY_EMAIL',
            'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY')
# A finished edition. research_incomplete is retried by the next recovery start, which
# publishes a new edition, so only its final version is emailed.
FINAL = ('published', 'no_reviewed_candidates')
CONFIDENCE = {'high': 'High confidence', 'moderate': 'Moderate confidence'}
# Brand (docs/assets/brand): ink, green and the wordmark's blue ampersand. Text colors
# on white keep at least 4.5:1 contrast.
INK, GREEN, BLUE = '#0f1418', '#22c55e', '#4c6fff'
TEXT, MUTED, LINE, PAGE, PANEL = '#1d2433', '#5b6475', '#e3e7ee', '#eef1f5', '#f6f8fb'
ACCENT = {'high': GREEN, 'moderate': BLUE}
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def decimal(price):
    return 1 + price / 100 if price > 0 else 1 - 100 / price


def american(price):
    return f'+{price}' if price > 0 else str(price)


def money(value):
    return f'${value:,.2f}'


def clock(value):
    """'6:10 PM EDT': the Eastern time of an ISO timestamp."""
    t = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('A source time zone is required for Eastern conversion')
    t = t.astimezone(EASTERN)
    return f"{t.hour % 12 or 12}:{t:%M} {'AM' if t.hour < 12 else 'PM'} {t:%Z}"


def day_label(decision_date, long=False):
    """'Sun, Oct 18', or 'Sunday, October 18', for the card's Eastern decision date."""
    d = date.fromisoformat(decision_date)
    return f'{d:%A}, {d:%B} {d.day}' if long else f'{d:%a}, {d:%b} {d.day}'


def lowest_price(row):
    """The worst American price at which this pick still clears the card's blend floor.

    The card keeps the blend's final chance, expected value and floor at the offered
    price, so the no-push share is recovered from them rather than re-deriving push
    rules by sport: ev = (1 - push) * (final * decimal - 1).
    """
    b = row.get('blend') or {}
    final, ev, floor, price = b.get('final'), b.get('ev'), b.get('threshold'), row.get('price')
    if not all(finite(v) for v in (final, ev, floor, price)) or abs(price) < 100 or not 0 < final < 1:
        return None
    edge = final * decimal(price) - 1
    if ev <= 0 or edge <= 0 or not 0 < ev / edge <= 1 + 1e-9:
        return None
    need = (floor / (ev / edge) + 1) / final
    if need > decimal(price) + 1e-9:
        return None
    # Round toward the bettor: the result itself must still clear the floor.
    if need >= 2:
        return max(100, math.ceil(round((need - 1) * 100, 6)))
    return -math.floor(round(100 / (need - 1), 6))


def stake(price, config):
    """Risk at most one unit and try to win at most one unit, so long shots stake less."""
    unit, step = config['unit_usd'], config['stake_step_usd']
    raw = unit if price < 100 else unit * 100 / price
    amount = max(config['min_stake_usd'], round(raw / step) * step)
    return amount, amount * (decimal(price) - 1)


def bet_title(row):
    line = row.get('line')
    line = '' if line is None else f'+{line}' if row.get('market') == 'spreads' and line > 0 else str(line)
    side = str(row.get('side') or '')
    return ' '.join(v for v in (row.get('player'), side[:1].upper() + side[1:], line,
                                str(row.get('market_label') or '').lower()) if v)


def review_label(row):
    return str(row.get('review') or '').replace(' · human review needed', '').replace(' · analyst review needed', '')


def picks(card, config):
    """What each email entry shows, in card order."""
    games = {}
    for row in card.get('rows') or []:
        group = row.get('exposure_group') or f"{row.get('sport')}:{row.get('game_id')}"
        games[group] = games.get(group, 0) + 1
    out = []
    for index, row in enumerate(card.get('rows') or []):
        price = row['price']
        amount, win = stake(price, config)
        group = row.get('exposure_group') or f"{row.get('sport')}:{row.get('game_id')}"
        checks = (row.get('qualitative_review') or {}).get('open_checks') or []
        level = (row.get('blend') or {}).get('confidence')
        out.append(dict(
            index=index, sport=row.get('sport'), game=row.get('game'), starts=clock(row['commence_time']),
            start_at=row['commence_time'], title=bet_title(row), price=american(price),
            book=row.get('book_label') or row.get('book'), book_url=config['books'].get(row.get('book')),
            lowest=lowest_price(row), stake=amount, to_win=win, level=level, confidence=CONFIDENCE.get(level),
            review=review_label(row), quoted=clock(row['quoted_at']),
            checks=[c for c in checks if isinstance(c, str)][:config['open_checks_shown']],
            shared_game=games[group] > 1,
            track_url=config['site'] + '/briefing/?' + urlencode(dict(
                edition=card['edition_id'], pick=index, stake=f'{amount:.2f}')) + '#daily-picks'))
    return out


def subject(card, entries, test=False):
    n = len(entries)
    body = (f"Morning Card · {day_label(card['decision_date'])} · {n} pick{'s' if n != 1 else ''}" if n
            else f"Morning Card · {day_label(card['decision_date'])} · No picks today")
    return ('[Test] ' if test else '') + body


def preheader(entries):
    """The inbox preview line."""
    if not entries:
        return 'No reviewed offer cleared today’s card.'
    first = min(entries, key=lambda e: e['start_at'])
    total = sum(e['stake'] for e in entries)
    return (f"{len(entries)} pick{'s' if len(entries) != 1 else ''} · first start {first['starts']} · "
            f"{money(total)} total stake")


def notes(card):
    if card.get('rows'):
        return ['Prices are from the morning card and move during the day. Bet only at the min price or better; '
                'below it the pick no longer clears the card.',
                'Experimental model and research. Analysis, not a guarantee. Bet responsibly.']
    return ['Research finished and no reviewed offer cleared the card. We would rather show nothing than force a pick.',
            'Analysis, not a guarantee. Bet responsibly.']


def text_body(card, entries, config, unsubscribe, test=False):
    lines = []
    if test:
        lines += [f"TEST SEND of the {day_label(card['decision_date'])} card.", '']
    lines += ['FOURTH & VALUE · MORNING CARD',
              f"{day_label(card['decision_date'], long=True)} · published {clock(card['published_at'])}", '']
    if entries:
        lines += [preheader(entries), '']
    for e in entries:
        lines += [f"{e['index'] + 1}. {e['title']}",
                  f"   {e['sport']} · {e['game']} · starts {e['starts']}",
                  f"   {e['price']} at {e['book']} (quoted {e['quoted']})"
                  + (f" · min price {american(e['lowest'])} or better" if e['lowest'] else ''),
                  f"   Suggested stake {money(e['stake'])} to win {money(e['to_win'])}",
                  '   ' + ' · '.join(v for v in (e['confidence'], e['review']) if v)]
        if e['shared_game']:
            lines.append('   Shares a game with another pick; they are not independent.')
        lines += [f'   Before you bet: {c}' for c in e['checks']]
        if e['book_url']:
            lines.append(f"   Open {e['book']}: {e['book_url']}")
        lines += [f"   I placed this: {e['track_url']}", '']
    lines += notes(card)
    lines += ['', f"Full card and research: {config['site']}/briefing/#daily-picks",
              f"Responsible play: {config['site']}/terms.html#responsible-play",
              'U.S. gambling support: call or text 1-800-MY-RESET.', '',
              'You receive this because this address is subscribed to the Fourth & Value Morning Card.',
              f'Unsubscribe: {unsubscribe}']
    return '\n'.join(lines) + '\n'


# ---- HTML: table layout and inline styles, which every major email app renders ----

def p(text, style=''):
    return f'<p style="margin:0;{style}">{text}</p>'


def button(url, label, primary):
    colors = (f'background:{GREEN};color:{INK};border:1px solid {GREEN};' if primary
              else f'background:#ffffff;color:{INK};border:1px solid {INK};')
    return (f'<td style="padding:12px 8px 0 0"><a href="{escape(url)}" style="{colors}display:inline-block;'
            f'padding:11px 14px;border-radius:8px;font:700 14px/1 {FONT};white-space:nowrap;text-decoration:none">'
            f'{escape(label)}</a></td>')


def stat(label, value, detail):
    return (f'<td valign="top" width="33%" style="padding:10px 10px 10px 0">'
            + p(escape(label), f'font:700 11px/1.4 {FONT};letter-spacing:.06em;text-transform:uppercase;color:{MUTED}')
            + p(escape(value), f'font:800 21px/1.3 {FONT};color:{TEXT}')
            + p(escape(detail), f'font:400 13px/1.4 {FONT};color:{MUTED}') + '</td>')


def pick_html(e, total):
    accent = ACCENT.get(e['level'], LINE)
    pill = (f'<span style="display:inline-block;padding:3px 9px;border-radius:999px;background:{accent};'
            f'color:{INK if e["level"] == "high" else "#ffffff"};font:700 12px/1.4 {FONT}">{escape(e["confidence"])}</span> '
            if e['confidence'] else '')
    checks = ''.join(f'<li style="margin:0 0 4px">{escape(c)}</li>' for c in e['checks'])
    return (
        f'<tr><td style="padding:0 24px 14px;background:#ffffff">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border:1px solid {LINE};border-left:4px solid {accent};border-radius:10px;border-collapse:separate">'
        f'<tr><td style="padding:16px 18px">'
        + p(f'<span style="font-weight:700;color:{TEXT}">Pick {e["index"] + 1} of {total}</span> · '
            f'{escape(e["sport"])} · {escape(e["game"])} · {escape(e["starts"])}',
            f'font:400 13px/1.5 {FONT};color:{MUTED}')
        + p(escape(e['title']), f'padding:4px 0 2px;font:700 19px/1.35 {FONT};color:{TEXT}')
        + '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>'
        + stat('Price', e['price'], f"{e['book']} · {e['quoted']}")
        + stat('Min price', american(e['lowest']) if e['lowest'] else '—', 'or better' if e['lowest'] else 'not available')
        + stat('Stake', money(e['stake']), f"to win {money(e['to_win'])}")
        + '</tr></table>'
        + (p(pill + f'<span style="color:{TEXT}">{escape(e["review"])}</span>', f'padding-top:2px;font:400 14px/1.6 {FONT}')
           if pill or e['review'] else '')
        + (p('Shares a game with another pick; they are not independent.',
             f'padding-top:8px;font:400 14px/1.5 {FONT};color:#8a5a00') if e['shared_game'] else '')
        + ((p('Before you bet', f'padding-top:12px;font:700 14px/1.5 {FONT};color:{TEXT}')
            + f'<ul style="margin:4px 0 0;padding-left:20px;font:400 14px/1.5 {FONT};color:#3a4252">{checks}</ul>')
           if checks else '')
        + '<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>'
        + (button(e['book_url'], f'Open {e["book"]}', True) if e['book_url'] else '')
        + button(e['track_url'], 'I placed this', not e['book_url'])
        + '</tr></table></td></tr></table></td></tr>')


def html_body(card, entries, config, unsubscribe, test=False):
    site, e_ = config['site'], escape
    link = 'color:#cfd6e0;text-decoration:underline'
    total = sum(e['stake'] for e in entries)
    first = min(entries, key=lambda e: e['start_at'])['starts'] if entries else None
    summary = ''
    if entries:
        summary = ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-top:16px"><tr>'
                   + ''.join(f'<td width="50%" style="padding:0 {pad}px 0 0"><table role="presentation" width="100%" '
                             f'cellpadding="0" cellspacing="0" border="0"><tr><td style="background:{PANEL};border-radius:8px;padding:10px 12px">'
                             + p(e_(label), f'font:700 11px/1.4 {FONT};letter-spacing:.06em;text-transform:uppercase;color:{MUTED}')
                             + p(e_(value), f'font:800 18px/1.3 {FONT};color:{TEXT}') + '</td></tr></table></td>'
                             for label, value, pad in (('Total stake', money(total), 8), ('First start', first, 0)))
                   + '</tr></table>')
    banner = (f'<tr><td style="padding:10px 24px;background:#fff4d6;color:#5c4400;font:600 14px/1.5 {FONT}">'
              f'Test send of the {e_(day_label(card["decision_date"]))} card.</td></tr>' if test else '')
    count = len(entries)
    headline = f'{count} pick{"s" if count != 1 else ""}' if count else 'No picks today'
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light"><meta name="supported-color-schemes" content="light">'
        f'<title>{e_(subject(card, entries, test))}</title></head>'
        f'<body style="margin:0;padding:0;background:{PAGE}">'
        f'<div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent">{e_(preheader(entries))}'
        + '&#847;&zwnj;&nbsp;' * 60 + '</div>'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:{PAGE}">'
        '<tr><td align="center" style="padding:24px 12px">'
        '<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="width:100%;max-width:600px">'
        # Header: the FV monogram and wordmark from the site's brand, in text so it shows with images off.
        f'<tr><td style="background:{INK};border-radius:12px 12px 0 0;padding:18px 24px">'
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0"><tr>'
        f'<td width="44" valign="middle"><table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>'
        f'<td width="36" height="36" align="center" valign="middle" style="background:{GREEN};border-radius:8px;'
        f'font:800 15px/36px {FONT};color:{INK}">FV</td></tr></table></td>'
        f'<td valign="middle" style="font:700 20px/1.2 {FONT};color:#ffffff">Fourth <span style="color:{BLUE}">&amp;</span> '
        f'<span style="color:{GREEN}">Value</span></td>'
        f'<td align="right" valign="middle" style="font:700 11px/1.2 {FONT};letter-spacing:.12em;color:#9aa4b2">MORNING CARD</td>'
        '</tr></table></td></tr>'
        + banner
        + f'<tr><td style="background:#ffffff;padding:24px 24px 18px">'
        + p(e_(day_label(card['decision_date'], long=True)), f'font:800 26px/1.25 {FONT};color:{TEXT}')
        + p(f'{e_(headline)} · published {e_(clock(card["published_at"]))}', f'padding-top:4px;font:400 15px/1.5 {FONT};color:{MUTED}')
        + summary + '</td></tr>'
        + ''.join(pick_html(e, count) for e in entries)
        + f'<tr><td style="background:#ffffff;padding:4px 24px 22px">'
        + ''.join(p(e_(n), f'padding-top:8px;font:400 14px/1.55 {FONT};color:#3a4252') for n in notes(card))
        + '</td></tr>'
        # Footer: links, support line and the unsubscribe link.
        + f'<tr><td style="background:{INK};border-radius:0 0 12px 12px;padding:20px 24px;font:400 13px/1.6 {FONT};color:#cfd6e0">'
        + p(f'<a href="{e_(site)}/briefing/#daily-picks" style="{link}">Full card and research</a> · '
            f'<a href="{e_(site)}/research/daily-process.html" style="{link}">How picks are chosen</a> · '
            f'<a href="{e_(site)}/tracking/" style="{link}">Bet Tracker</a>')
        + p(f'U.S. gambling support: call or text <a href="tel:+18006973738" style="{link}">1-800-MY-RESET</a> · '
            f'<a href="{e_(site)}/terms.html#responsible-play" style="{link}">Responsible play</a>', 'padding-top:10px')
        + p('You receive this because this address is subscribed to the Fourth &amp; Value Morning Card. '
            f'<a href="{e_(unsubscribe)}" style="{link}">Unsubscribe</a>', 'padding-top:10px')
        + p(f'Fourth &amp; Value · <a href="{e_(site)}" style="{link}">fourthandvalue.com</a>', 'padding-top:10px;color:#9aa4b2')
        + '</td></tr></table></td></tr></table></body></html>')


def unsubscribe_url(config, token):
    return f"{config['site']}/email/unsubscribe.html?" + urlencode(dict(token=token))


def message(card, config, unsubscribe, test=False):
    entries = picks(card, config)
    return dict(subject=subject(card, entries, test), html=html_body(card, entries, config, unsubscribe, test),
                text=text_body(card, entries, config, unsubscribe, test),
                headers={'List-Unsubscribe': f'<{unsubscribe}>'})


def skip_reason(card, now, expected=None):
    """Why this card is not today's finished morning edition, or None to send it."""
    if expected and card.get('edition_id') != expected:
        return 'edition_mismatch'
    if card.get('schema_version') != 1 or card.get('kind') != 'morning':
        return 'not_a_morning_edition'
    if not isinstance(card.get('edition_id'), str) or not card['edition_id']:
        return 'no_edition_id'
    if card.get('status') not in FINAL:
        return 'edition_not_final'
    if card.get('decision_date') != now.astimezone(EASTERN).date().isoformat():
        return 'not_today'
    return None


def subscription(env, recipient, http=requests):
    """The recipient's Morning Card row, created subscribed on first use. Service role."""
    base = env['SUPABASE_URL'].rstrip('/') + '/rest/v1/email_subscriptions'
    key = env['SUPABASE_SERVICE_ROLE_KEY']
    headers = {'apikey': key, 'Authorization': 'Bearer ' + key}
    email = recipient.strip().lower()
    for _ in range(2):
        response = http.get(base, headers=headers, timeout=20,
                            params={'list': 'eq.' + LIST, 'email': 'eq.' + email, 'select': 'token,subscribed'})
        if not response.ok:
            raise RuntimeError(f'Subscription check HTTP {response.status_code}')
        rows = response.json()
        if rows:
            return rows[0]
        response = http.post(base, headers={**headers, 'Prefer': 'return=representation'}, timeout=20,
                             json={'list': LIST, 'email': email})
        if response.status_code == 409:
            continue  # created by a concurrent start; read it
        if not response.ok:
            raise RuntimeError(f'Subscription create HTTP {response.status_code}')
        return response.json()[0]
    raise RuntimeError('Subscription row unavailable')


def send(mail, key, env, post=requests.post):
    """Send through Resend once per key. The recipient address is never printed."""
    response = post(RESEND, headers={'Authorization': 'Bearer ' + env['RESEND_API_KEY'], 'Idempotency-Key': key},
                    json={'from': env['EDITORIAL_NOTIFY_FROM'], 'to': [env['EDITORIAL_NOTIFY_EMAIL']], **mail}, timeout=30)
    # 409: this key was already used (sent, or a concurrent send in flight).
    if response.status_code == 409:
        return 'already_sent'
    if not response.ok:
        raise RuntimeError(f'Morning email HTTP {response.status_code}')
    return 'sent'


def run(now, *, card_path=CARD, config_path=CONFIG, expected=None, test=False, preview=None, env=None,
        post=requests.post, http=requests):
    env = os.environ if env is None else env
    card = json.loads(Path(card_path).read_text())
    config = json.loads(Path(config_path).read_text())
    reason = None if test else skip_reason(card, now, expected)
    if reason:
        return reason
    if preview:
        Path(preview).write_text(message(card, config, unsubscribe_url(config, 'preview'), test)['html'])
        return 'previewed'
    if not all(env.get(name) for name in SETTINGS):
        return 'not_configured'
    # An unsubscribe must hold: a row that cannot be read sends nothing.
    row = subscription(env, env['EDITORIAL_NOTIFY_EMAIL'], http)
    if row.get('subscribed') is not True:
        return 'unsubscribed'
    mail = message(card, config, unsubscribe_url(config, row['token']), test)
    key = (f"morning-card-test-{card.get('edition_id')}-{now:%Y%m%d%H%M}" if test
           else f"morning-card-{card['edition_id']}")
    return send(mail, key, env, post)


MESSAGES = {
    # Resend answers a repeat of the same key with the original result and sends nothing.
    'sent': 'Morning card email accepted by Resend (a repeat for this edition is not resent).',
    'already_sent': 'Morning card email for this edition was already sent.',
    'not_configured': 'Morning email not configured (needs ' + ', '.join(SETTINGS) + '); nothing sent.',
    'unsubscribed': 'The recipient unsubscribed from the Morning Card email; nothing sent.',
    'previewed': 'Preview written; nothing sent.',
    'edition_mismatch': 'The card on disk is not the verified edition; nothing sent.',
    'not_a_morning_edition': 'Not a morning edition; nothing sent.',
    'no_edition_id': 'The card has no edition id; nothing sent.',
    'edition_not_final': 'Research is incomplete; the recovery start that finishes it sends the email.',
    'not_today': "The card is not today's; nothing sent.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edition', help='only send if the card is this verified edition')
    parser.add_argument('--test', action='store_true', help='send the current card now, labeled as a test')
    parser.add_argument('--preview', metavar='HTML', help='write the email to a file instead of sending it')
    args = parser.parse_args()
    try:
        status = run(datetime.now(timezone.utc), expected=args.edition, test=args.test, preview=args.preview)
    except (RuntimeError, requests.RequestException, ValueError, KeyError) as error:
        raise SystemExit(f'Morning email failed: {error}. The card is published; nothing else changed.')
    # A test send exists to prove the setup, so anything short of a send fails it.
    if args.test and status in ('not_configured', 'unsubscribed'):
        raise SystemExit(MESSAGES[status])
    print(MESSAGES[status])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
