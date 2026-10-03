"""Cross-book NHL arbitrage and same-book coherence checks from saved exact quotes.

An arbitrage pairs the best price on each side of one game, market and line across
books. It counts only when both books' settlement rules are verified as the same
profile and both quotes sit inside the pairing window; everything else is labeled
for a rules or timing check instead of being counted. Whole-number lines can push,
which refunds both legs, so the locked return holds on every outcome.
"""
from collections import defaultdict
from datetime import timedelta
import math

from .data import stamp
from .pricing import PAIR_WINDOW, decimal, push_capable

NEAR_MISS = 1.01   # best-of-books implied total shown as context when no arb exists
SAMPLE_STAKE = 100.0


def side_key(row):
    """Same event, player, market and line regardless of which book quoted it."""
    line = row['line']
    if row['market'] == 'spreads' and row['side'] == row['away_team']:
        line = -line
    return (row['event_id'], row.get('player_id') or row['player'], row['market'], line)


def expected_sides(row):
    return {row['home_team'], row['away_team']} if row['market'] in ('h2h', 'spreads') else {'Over', 'Under'}


def _usable(row):
    try:
        return math.isfinite(decimal(row['price'])) and row.get('quoted_at')
    except (TypeError, ValueError, KeyError):
        return False


def find_arbitrage(rows, near_miss=NEAR_MISS, stake=SAMPLE_STAKE):
    """Return (opportunities, near_misses); each is a list of dicts, best first."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if _usable(row):
            groups[side_key(row)][row['side']].append(row)
    found, close = [], []
    for sides in groups.values():
        first = next(iter(sides.values()))[0]
        if set(sides) != expected_sides(first):
            continue
        legs = [min(quotes, key=lambda r: 1 / decimal(r['price'])) for quotes in sides.values()]
        probabilities = [1 / decimal(r['price']) for r in legs]
        total = sum(probabilities)
        if total >= near_miss:
            continue
        times = [stamp(r['quoted_at']) for r in legs]
        profiles = {r.get('settlement_profile') for r in legs}
        verified = all(r.get('settlement_verified') for r in legs) and len(profiles) == 1
        fresh = max(times) - min(times) <= PAIR_WINDOW
        distinct = len({r['book'] for r in legs}) == 2
        if total < 1 and verified and fresh and distinct:
            status = 'arbitrage'
        elif total < 1 and not distinct:
            status = 'same_book_error'
        elif total < 1 and not verified:
            status = 'check_settlement_rules'
        elif total < 1:
            status = 'stale_pair'
        else:
            status = 'near_miss'
        record = dict(
            status=status, game=first['game'], commence_time=first['commence_time'],
            event_id=first['event_id'], market=first['market'],
            market_label=first.get('market_label', first['market']), player=first['player'],
            line=first['line'], implied_total=total, locked_return_pct=100 * (1 / total - 1),
            push_possible=push_capable(first), settlement_verified=verified,
            settlement_profiles=sorted(p for p in profiles if p),
            quote_spread_seconds=(max(times) - min(times)).total_seconds(),
            legs=[dict(side=r['side'], book=r['book'], book_label=r.get('book_label', r['book']),
                       price=r['price'], implied_probability=p, stake=stake * p / total,
                       quoted_at=r['quoted_at'], settlement_profile=r.get('settlement_profile'))
                  for r, p in zip(legs, probabilities)],
            payout=stake / total, stake_total=stake)
        (close if status == 'near_miss' else found).append(record)
    found.sort(key=lambda r: r['implied_total'])
    close.sort(key=lambda r: r['implied_total'])
    return found, close


# Each player count is at least as large as the counts listed against it: every goal is
# recorded as a shot on goal, and points are goals plus assists.
AT_LEAST = {'player_goals': {'player_goals', 'player_points', 'player_shots_on_goal'},
            'player_assists': {'player_assists', 'player_points'},
            'player_points': {'player_points'}, 'player_shots_on_goal': {'player_shots_on_goal'},
            'totals': {'totals'}}


def subject(row):
    """What an offer is about: a player, a team (moneyline and puck line), or the game total."""
    if row['market'].startswith('player_'):
        return row.get('player_id') or row['player']
    return row['side'] if row['market'] in ('h2h', 'spreads') else 'total'


def contains(wide, narrow):
    """True when every result that wins `narrow` also wins `wide`, for the same subject and rules.

    Half lines only for counts, so neither bet can push. Teams: the moneyline contains the
    puck line -1.5, and the puck line +1.5 contains the moneyline.
    """
    if wide is narrow or subject(wide) != subject(narrow) or wide['side'] != narrow['side']:
        return False
    if wide.get('settlement_profile') != narrow.get('settlement_profile'):
        return False
    wm, nm = wide['market'], narrow['market']
    if 'h2h' in (wm, nm) or 'spreads' in (wm, nm):
        return (wm, nm) == ('h2h', 'spreads') and narrow['line'] == -1.5 or (wm, nm) == ('spreads', 'h2h') and wide['line'] == 1.5
    if wide['line'] is None or narrow['line'] is None or float(wide['line']).is_integer() or float(narrow['line']).is_integer():
        return False
    if (wm, wide['line']) == (nm, narrow['line']):
        return False
    if wide['side'] == 'Over':
        return wm in AT_LEAST.get(nm, ()) and wide['line'] <= narrow['line']
    return nm in AT_LEAST.get(wm, ()) and wide['line'] >= narrow['line']


def find_incoherent(rows):
    """Same-book prices that contradict a containment: the wider bet always wins when the narrower one does.

    - Points contain goals and assists; shots on goal contain goals; at the same or a lower line.
    - A lower Over (or higher Under) line contains a higher one in the same market.
    - A team's moneyline contains its puck line -1.5; its puck line +1.5 contains its moneyline.
    A book implying a higher break-even for the narrower bet is underpricing the wider one.
    """
    groups = defaultdict(list)
    for row in rows:
        if _usable(row) and (row['line'] is not None or row['market'] == 'h2h'):
            groups[(row['event_id'], row['book'], subject(row))].append((row, 1 / decimal(row['price'])))
    found = []
    for (event, book, who), offers in groups.items():
        for wide, wide_p in offers:
            for narrow, narrow_p in offers:
                if narrow_p <= wide_p or not contains(wide, narrow):
                    continue
                gap = narrow_p - wide_p
                found.append(dict(
                    game=wide['game'], commence_time=wide['commence_time'], book=book,
                    book_label=wide.get('book_label', book), subject=who,
                    wide=dict(market=wide.get('market_label', wide['market']), side=wide['side'], line=wide['line'],
                              price=wide['price'], implied_probability=wide_p),
                    narrow=dict(market=narrow.get('market_label', narrow['market']), side=narrow['side'], line=narrow['line'],
                                price=narrow['price'], implied_probability=narrow_p),
                    gap=gap, severity='HIGH' if gap >= .05 else 'MEDIUM'))
    found.sort(key=lambda r: -r['gap'])
    return found


def report(state):
    rows = state.get('rows', [])
    arbs, close = find_arbitrage(rows)
    groups = {side_key(r) for r in rows if _usable(r)}
    return dict(sport='icehockey_nhl', checked_at=state.get('checked_at'),
                last_success_at=state.get('last_success_at'), feed_status=state.get('status'),
                snapshot_id=state.get('snapshot_id'), quotes=len(rows), markets_checked=len(groups),
                books=sorted({r['book'] for r in rows}),
                verified_books=sorted({r['book'] for r in rows if r.get('settlement_verified')}),
                arbitrage=[a for a in arbs if a['status'] == 'arbitrage'],
                needs_check=[a for a in arbs if a['status'] != 'arbitrage'],
                near_misses=close[:10], incoherent=find_incoherent(rows),
                pair_window_seconds=PAIR_WINDOW.total_seconds(), near_miss_threshold=NEAR_MISS)


def _odds(price):
    return f'+{price:.0f}' if price > 0 else f'{price:.0f}'


def _when(value):
    return stamp(value).strftime('%b %d, %H:%M UTC') if value else 'not yet checked'


def _offer(row):
    from html import escape
    subject = f"{row['player']} · " if row['player'] else ''
    line = '' if row['line'] is None else f" {row['line']:g}"
    return escape(f"{subject}{row['market_label']}{line}")


def _legs(row):
    from html import escape
    return ''.join(f'<li><strong>{escape(str(leg["side"]))}</strong> {_odds(leg["price"])} at {escape(leg["book_label"])}'
                   f' · stake ${leg["stake"]:.2f}</li>' for leg in row['legs'])


def render(data, extra=''):
    """Static HTML body for docs/nhl/arbitrage.html; no client script required.

    `extra` is a server-rendered section placed before the explanation.
    """
    from html import escape
    arbs, check, close, incoherent = data['arbitrage'], data['needs_check'], data['near_misses'], data['incoherent']
    stale = data['feed_status'] != 'ready'
    parts = [f'''<p class="lead">Risk-free pairs across sportsbooks, and same-book prices that contradict each other. Checked every NHL refresh from the exact quotes on the Props and Game Lines boards.</p>
<div class="grid"><article class="panel"><h3>Locked arbitrage</h3><p class="betline">{len(arbs)}</p><p class="meta">Both books verified, quotes within {data["pair_window_seconds"]/60:.0f} minutes</p></article>
<article class="panel"><h3>Needs a rules check</h3><p class="betline">{len(check)}</p><p class="meta">Under 100% but settlement rules unverified, or quotes too far apart</p></article>
<article class="panel"><h3>Pricing contradictions</h3><p class="betline">{len(incoherent)}</p><p class="meta">One book pricing a narrower bet above the wider one</p></article></div>
<p class="muted">Snapshot {escape(_when(data["last_success_at"]))} · {data["markets_checked"]} markets · {data["quotes"]} quotes from {len(data["books"])} books. Settlement verified: {escape(", ".join(data["verified_books"]) or "none")}.</p>''']
    if stale:
        parts.append('<div class="notice"><strong>Feed is not current.</strong><p>These results come from the last successful snapshot and may no longer be available.</p></div>')

    def table(rows, heading, note, show_status=False):
        if not rows:
            return ''
        body = ''.join(
            f'<tr><td>{escape(r["game"])}<div class="meta">{escape(_when(r["commence_time"]))}</div></td><td>{_offer(r)}'
            + (' <span class="tag">push refunds both</span>' if r['push_possible'] else '')
            + (f' <span class="tag">{escape(r["status"].replace("_", " "))}</span>' if show_status else '')
            + f'</td><td><ul>{_legs(r)}</ul></td><td>{100*r["implied_total"]:.2f}%</td>'
            f'<td class="{"positive" if r["locked_return_pct"] >= .005 else "muted" if r["locked_return_pct"] > -.005 else "negative"}">{r["locked_return_pct"]:+.2f}%</td></tr>'
            for r in rows)
        return (f'<section class="section"><h2>{heading}</h2><p class="muted">{note}</p><div class="table-wrap"><table>'
                '<thead><tr><th>Game</th><th>Offer</th><th>Best price each side ($100 total)</th><th>Implied total</th><th>Return</th></tr></thead>'
                f'<tbody>{body}</tbody></table></div></section>')

    if arbs:
        parts.append(table(arbs, 'Locked arbitrage', 'Stake both legs as shown and every settled outcome returns the same payout. Prices move fast; confirm both quotes before placing.'))
    else:
        parts.append('<section class="section"><h2>Locked arbitrage</h2><div class="empty">No locked arbitrage in this snapshot. That is the normal state of an efficient market; the closest markets are listed below.</div></section>')
    parts.append(table(check, 'Needs a rules check', 'These pairs add to under 100%, but at least one book’s NHL settlement rules are not verified as matching, or the two quotes were captured too far apart. Treat as leads only.', True))
    parts.append(table(close, 'Closest to arbitrage', f'Best-of-books implied totals under {100*data["near_miss_threshold"]:.0f}%. A negative return is the cost of holding both sides.'))
    if incoherent:
        body = ''.join(
            f'<tr><td>{escape(r["game"])}</td><td>{escape(r["book_label"])}</td><td>{escape(str(r["subject"]))}</td>'
            f'<td>{escape(r["narrow"]["market"])} {escape(str(r["narrow"]["side"]))} {_odds(r["narrow"]["price"])} ({100*r["narrow"]["implied_probability"]:.1f}%)</td>'
            f'<td>{escape(r["wide"]["market"])} {escape(str(r["wide"]["side"]))} {_odds(r["wide"]["price"])} ({100*r["wide"]["implied_probability"]:.1f}%)</td>'
            f'<td><span class="tag">{r["severity"]}</span> {100*r["gap"]:.1f} pts</td></tr>' for r in incoherent)
        parts.append('<section class="section"><h2>Pricing contradictions</h2><p class="muted">The wider bet wins every time the narrower one does, so it should never cost less to break even on the narrower bet.</p>'
                     '<div class="table-wrap"><table><thead><tr><th>Game</th><th>Book</th><th>Subject</th><th>Narrower bet</th><th>Wider bet</th><th>Gap</th></tr></thead>'
                     f'<tbody>{body}</tbody></table></div></section>')
    else:
        parts.append('<section class="section"><h2>Pricing contradictions</h2><div class="empty">No book priced a narrower bet above a wider one it sits inside: goals or assists above points, goals above shots on goal, a higher Over line above a lower one, or a puck line −1.5 above the same team’s moneyline.</div></section>')
    parts.append(extra)
    parts.append('''<section class="help section"><h2>How this works</h2><p>For each game, market and line, we take the best price on each side across books. If the two break-even probabilities add up to less than 100%, staking each side in proportion to its probability returns the same amount whichever side wins. On whole-number lines a push refunds both legs.</p><p>A pair is counted only when both books’ NHL settlement rules are verified as the same (overtime, shootouts, player participation), and both quotes were captured within the pairing window. Limits, voids, palpable-error rules and account restrictions are not modeled. This page is separate from Top Picks and Market Watch and does not feed either.</p><p>Cross-market checks are graded research. Every snapshot’s signals are archived under a fixed rule and scored against results; none can affect a pick until that record supports it.</p></section>''')
    return '\n'.join(p for p in parts if p)
