"""Weekly closing-line value report card for published picks, shared by every sport's recap.

scripts/pick_closes.py registers each verified edition's picks in public.pick_closes and
the closing-lines Edge Function fills their closes in the final minutes before each game:
the median no-vig chance across US books at the pick's exact line. For each pick:

  value at the close = closing no-vig chance x decimal(published price) - 1
  market move        = closing no-vig chance - the chance the pick was decided against

Positive value means the published price paid more than the closing market justified.
Over many picks it is the earliest reliable sign of a real edge, long before win-loss
results settle down. Without the Supabase settings the section says closes are
unavailable; it never blocks a recap.
"""
from html import escape
import math
import os

import requests

SETTINGS = ('SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY')
FIELDS = ('edition_kind,decision_date,pick_index,player,market_type,side,line,book,odds,decision_market_prob,'
          'closing_fair_prob,closing_line,closing_books,clv_ev,clv_status,commence_time')
EDITIONS = (('morning', 'Morning picks'), ('afternoon', 'Afternoon picks'))


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fetch(league, end, env=None, http=requests):
    """Every registered pick for one league through `end` (an ISO date), or None."""
    env = os.environ if env is None else env
    if not all(env.get(name) for name in SETTINGS):
        return None
    key = env['SUPABASE_SERVICE_ROLE_KEY']
    response = http.get(env['SUPABASE_URL'].rstrip('/') + '/rest/v1/pick_closes', timeout=30,
                        headers={'apikey': key, 'Authorization': 'Bearer ' + key},
                        params={'select': FIELDS, 'league': 'eq.' + league, 'decision_date': 'lte.' + str(end),
                                'order': 'decision_date,edition_kind,pick_index'})
    if not response.ok:
        raise RuntimeError(f'Closing prices HTTP {response.status_code}')
    return response.json()


def block(rows):
    """Report-card figures for a set of picks."""
    rows = list(rows)
    closed = [r for r in rows if r.get('clv_status') == 'captured' and number(r.get('clv_ev'))]
    moves = [r['closing_fair_prob'] - r['decision_market_prob'] for r in closed
             if number(r.get('closing_fair_prob')) and number(r.get('decision_market_prob'))]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return dict(picks=len(rows), closed=len(closed), pending=sum(r.get('clv_status') is None for r in rows),
                line_moved=sum(r.get('clv_status') == 'line_moved' for r in rows),
                value=mean([r['clv_ev'] for r in closed]),
                beat=mean([1.0 if r['clv_ev'] > 0 else 0.0 for r in closed]),
                move=mean(moves), moves=len(moves))


def weekly(league, start, end, env=None, http=requests):
    """This week's report card by edition, plus every pick since tracking began."""
    try:
        rows = fetch(league, end, env, http)
    except (RuntimeError, requests.RequestException, ValueError) as error:
        return dict(status='unavailable', reason=str(error))
    if rows is None:
        return dict(status='unavailable', reason='Closing-price settings are not configured.')
    week = [r for r in rows if str(start) <= str(r.get('decision_date')) <= str(end)]
    return dict(status='ready', start=str(start), end=str(end),
                editions={kind: block(r for r in week if r.get('edition_kind') == kind) for kind, _ in EDITIONS},
                week=block(week), to_date=block(rows), first_date=min((str(r['decision_date']) for r in rows), default=None),
                rows=week)


def pct(value, signed=False):
    if not number(value):
        return '—'
    return f'{100 * value:+.1f}%' if signed else f'{100 * value:.0f}%'


def points(value):
    return f'{100 * value:+.1f} pts' if number(value) else '—'


def section_html(report, sport):
    """The recap's report-card section."""
    head = '<h2 id="closing-line-value">Closing-line value report card</h2>'
    explain = ('<p>Each published pick’s price compared with the closing market: the median no-vig chance across US '
               'books at the pick’s exact line, captured in the final minutes before the start. <strong>Value at the '
               'close</strong> is how much more the published price paid than that closing chance justified; '
               '<strong>beat the close</strong> is the share of picks where it was positive; <strong>market move</strong> '
               'is how far the closing chance moved toward our side (positive) or away from it (negative) after the pick. '
               'Over many picks, value at the close is the earliest reliable sign of a real edge, long before win–loss '
               'results settle down. It is a measure of price, not a guarantee.</p>')
    if not report or report.get('status') != 'ready':
        return head + explain + '<p>Closing prices were unavailable for this report.</p>'
    def row(label, b):
        return (f'<tr><td>{escape(label)}</td><td>{b["picks"]}</td><td>{b["closed"]}</td><td>{pct(b["value"], True)}</td>'
                f'<td>{pct(b["beat"])}</td><td>{points(b["move"])}</td></tr>')
    rows = ''.join(row(label, report['editions'][kind]) for kind, label in EDITIONS)
    rows += row('All picks this week', report['week'])
    since = f'All picks since {report["first_date"]}' if report.get('first_date') else 'All picks to date'
    rows += row(since, report['to_date'])
    w = report['week']
    notes = []
    if w['pending']:
        notes.append(f'{w["pending"]} pick{"s" if w["pending"] != 1 else ""} had no close recorded yet.')
    if w['line_moved']:
        notes.append(f'{w["line_moved"]} closed at a different line, so it has no same-line value.')
    empty = '' if report['to_date']['picks'] else f'<p>No published {escape(sport)} picks have been registered for closing prices yet.</p>'
    return (head + explain + '<div class="table-wrap"><table><thead><tr><th>Edition</th><th>Picks</th><th>Closing prices</th>'
            '<th>Value at the close</th><th>Beat the close</th><th>Market move</th></tr></thead><tbody>' + rows
            + '</tbody></table></div>' + empty + (f'<p class="meta">{escape(" ".join(notes))}</p>' if notes else ''))
