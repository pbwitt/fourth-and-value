"""Owner email for the morning card: each pick with a suggested stake, the lowest price
that still makes the card, a link to the book and a link that opens Bet Tracker's form.

It reads the published docs/briefing/morning-card.json only. It never places a bet,
changes the card or calls a model. One email per edition: the Resend idempotency key is
the edition id, so recovery starts that re-verify a finished card do not send it again.
Without RESEND_API_KEY, EDITORIAL_NOTIFY_FROM and EDITORIAL_NOTIFY_EMAIL it does nothing,
like the editorial notices in scripts/editorial_ideas.py.
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
# A finished edition. research_incomplete is retried by the next recovery start, which
# publishes a new edition, so only its final version is emailed.
FINAL = ('published', 'no_reviewed_candidates')
CONFIDENCE = {'high': 'High confidence', 'moderate': 'Moderate confidence'}


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


def day_label(decision_date):
    """'Sunday, Oct 18' for the card's Eastern decision date."""
    d = date.fromisoformat(decision_date)
    return f'{d:%A}, {d:%b} {d.day}'


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
        out.append(dict(
            index=index, sport=row.get('sport'), game=row.get('game'), starts=clock(row['commence_time']),
            title=bet_title(row), price=american(price), book=row.get('book_label') or row.get('book'),
            book_url=config['books'].get(row.get('book')), lowest=lowest_price(row),
            stake=amount, to_win=win, confidence=CONFIDENCE.get((row.get('blend') or {}).get('confidence')),
            review=review_label(row), quoted=clock(row['quoted_at']),
            checks=[c for c in checks if isinstance(c, str)][:config['open_checks_shown']],
            shared_game=games[group] > 1,
            track_url=config['site'] + '/briefing/?' + urlencode(dict(
                edition=card['edition_id'], pick=index, stake=f'{amount:.2f}')) + '#daily-picks'))
    return out


def subject(card, entries, test=False):
    n = len(entries)
    body = (f"Today's picks · {day_label(card['decision_date'])}: {n} pick{'s' if n != 1 else ''}" if n
            else f"Today's card · {day_label(card['decision_date'])}: no picks qualified")
    return ('Test · ' if test else '') + body


def notes(card):
    if card.get('rows'):
        return ['Prices are from the morning card and move during the day. Bet only at the “still a pick at” '
                'price or better.',
                'Experimental model and research. Analysis, not a guarantee. Bet responsibly.']
    return ['Research finished and no reviewed offer cleared the card. We would rather show nothing than force a pick.',
            'Analysis, not a guarantee. Bet responsibly.']


def text_body(card, entries, config, test=False):
    lines = []
    if test:
        lines += [f"TEST SEND of the {card['decision_date']} card.", '']
    lines += [f"Morning card for {day_label(card['decision_date'])} · published {clock(card['published_at'])}", '']
    for e in entries:
        lines += [f"{e['index'] + 1}. {e['title']}",
                  f"   {e['sport']} · {e['game']} · starts {e['starts']}",
                  f"   {e['price']} at {e['book']} (quoted {e['quoted']})"
                  + (f" · still a pick at {american(e['lowest'])} or better" if e['lowest'] else ''),
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
              f"Responsible play: {config['site']}/terms.html#responsible-play"]
    return '\n'.join(lines) + '\n'


def button(url, label, primary):
    style = ('background:#0b6fa4;color:#ffffff;border:1px solid #0b6fa4;' if primary
             else 'background:#ffffff;color:#0b6fa4;border:1px solid #0b6fa4;')
    return (f'<a href="{escape(url)}" style="{style}display:inline-block;padding:10px 16px;margin:4px 8px 0 0;'
            f'border-radius:6px;font-weight:600;text-decoration:none">{escape(label)}</a>')


def html_body(card, entries, config, test=False):
    e_ = escape
    parts = ['<!doctype html><html><body style="margin:0;padding:0;background:#f2f4f7">'
             '<div style="max-width:600px;margin:0 auto;padding:20px 16px;'
             'font:16px/1.5 -apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;color:#1d2433">']
    if test:
        parts.append(f'<p style="background:#fff4d6;border:1px solid #e0b100;border-radius:6px;padding:8px 12px;margin:0 0 12px">'
                     f'Test send of the {e_(card["decision_date"])} card.</p>')
    parts.append('<p style="margin:0;color:#0b6fa4;font-weight:800;letter-spacing:.04em;font-size:13px">FOURTH &amp; VALUE</p>'
                 f'<h1 style="margin:4px 0 2px;font-size:24px">{e_(day_label(card["decision_date"]))}</h1>'
                 f'<p style="margin:0 0 16px;color:#5b6475;font-size:14px">Morning card · published {e_(clock(card["published_at"]))}</p>')
    for e in entries:
        meta = ' · '.join(v for v in (e['confidence'], e['review']) if v)
        lowest = (f'<br><span style="color:#5b6475">Still a pick at <strong style="color:#1d2433">'
                  f'{e_(american(e["lowest"]))}</strong> or better</span>' if e['lowest'] else '')
        parts.append(
            '<div style="background:#ffffff;border:1px solid #dde2ea;border-radius:10px;padding:16px;margin:0 0 12px">'
            f'<p style="margin:0;color:#5b6475;font-size:13px">{e_(e["sport"])} · {e_(e["game"])} · starts {e_(e["starts"])}</p>'
            f'<p style="margin:4px 0 6px;font-size:18px;font-weight:700">{e_(e["title"])}</p>'
            f'<p style="margin:0"><strong>{e_(e["price"])}</strong> at {e_(e["book"])}'
            f' <span style="color:#5b6475;font-size:13px">(quoted {e_(e["quoted"])})</span>{lowest}</p>'
            f'<p style="margin:6px 0 0">Suggested stake <strong>{e_(money(e["stake"]))}</strong> to win {e_(money(e["to_win"]))}</p>'
            + (f'<p style="margin:6px 0 0;color:#5b6475;font-size:14px">{e_(meta)}</p>' if meta else '')
            + ('<p style="margin:6px 0 0;color:#8a5a00;font-size:14px">Shares a game with another pick; they are not independent.</p>'
               if e['shared_game'] else '')
            + (('<p style="margin:10px 0 2px;font-size:14px;font-weight:600">Before you bet</p>'
                '<ul style="margin:0;padding-left:20px;font-size:14px;color:#3a4252">'
                + ''.join(f'<li>{e_(c)}</li>' for c in e['checks']) + '</ul>') if e['checks'] else '')
            + '<p style="margin:10px 0 0">'
            + (button(e['book_url'], f'Open {e["book"]}', True) if e['book_url'] else '')
            + button(e['track_url'], 'I placed this', not e['book_url'])
            + '</p></div>')
    parts += [f'<p style="margin:12px 0 0;font-size:14px;color:#3a4252">{e_(n)}</p>' for n in notes(card)]
    parts.append(f'<p style="margin:12px 0 0;font-size:14px"><a href="{e_(config["site"])}/briefing/#daily-picks" '
                 'style="color:#0b6fa4">Full card and research</a> · '
                 f'<a href="{e_(config["site"])}/terms.html#responsible-play" style="color:#0b6fa4">Responsible play</a></p>'
                 '</div></body></html>')
    return ''.join(parts)


def message(card, config, test=False):
    entries = picks(card, config)
    return dict(subject=subject(card, entries, test), html=html_body(card, entries, config, test),
                text=text_body(card, entries, config, test))


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


def send(mail, key, env=None, post=requests.post):
    """Send through Resend once per key. The recipient address is never printed."""
    env = os.environ if env is None else env
    api, sender, recipient = env.get('RESEND_API_KEY'), env.get('EDITORIAL_NOTIFY_FROM'), env.get('EDITORIAL_NOTIFY_EMAIL')
    if not all([api, sender, recipient]):
        return 'not_configured'
    response = post(RESEND, headers={'Authorization': 'Bearer ' + api, 'Idempotency-Key': key},
                    json={'from': sender, 'to': [recipient], **mail}, timeout=30)
    # 409: this key was already used (sent, or a concurrent send in flight).
    if response.status_code == 409:
        return 'already_sent'
    if not response.ok:
        raise RuntimeError(f'Morning email HTTP {response.status_code}')
    return 'sent'


def run(now, *, card_path=CARD, config_path=CONFIG, expected=None, test=False, preview=None, env=None, post=requests.post):
    card = json.loads(Path(card_path).read_text())
    config = json.loads(Path(config_path).read_text())
    reason = None if test else skip_reason(card, now, expected)
    if reason:
        return reason
    mail = message(card, config, test)
    if preview:
        Path(preview).write_text(mail['html'])
        return 'previewed'
    key = (f"morning-card-test-{card.get('edition_id')}-{now:%Y%m%d%H%M}" if test
           else f"morning-card-{card['edition_id']}")
    return send(mail, key, env, post)


MESSAGES = {
    # Resend answers a repeat of the same key with the original result and sends nothing.
    'sent': 'Morning card email accepted by Resend (a repeat for this edition is not resent).',
    'already_sent': 'Morning card email for this edition was already sent.',
    'not_configured': 'Morning email not configured (RESEND_API_KEY, EDITORIAL_NOTIFY_FROM, EDITORIAL_NOTIFY_EMAIL); nothing sent.',
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
    except (RuntimeError, requests.RequestException) as error:
        raise SystemExit(f'Morning email failed: {error}. The card is published; nothing else changed.')
    # A test send exists to prove the setup, so a missing setting fails it.
    if args.test and status == 'not_configured':
        raise SystemExit(MESSAGES[status])
    print(MESSAGES[status])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
