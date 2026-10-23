#!/usr/bin/env python3
"""Build the Arbitrage section's generated files.

  docs/arbitrage/data/rules.json  sportsbook kinds and settlement profiles the engine reads,
                                  from config/arbitrage_settlement.json plus NHL's own
                                  config/nhl_settlement.json, so both arbitrage pages agree.
  docs/arbitrage/plan.html        the arbitrage plan, from scripts/arbitrage/plan.html with the
                                  rules tables and the one-time history replay
                                  (docs/arbitrage/data/replay.json, scripts/arbitrage/replay.cjs).

`--check` fails when a committed file differs from what this would write (PR checks).
"""
import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from display_time import eastern_time  # noqa: E402
from site_metadata import metadata, seo_title  # noqa: E402

CONFIG = ROOT / 'config/arbitrage_settlement.json'
NHL = ROOT / 'config/nhl_settlement.json'
RULES = ROOT / 'docs/arbitrage/data/rules.json'
REPLAY = ROOT / 'docs/arbitrage/data/replay.json'
TEMPLATE = Path(__file__).with_name('plan.html')
PLAN = ROOT / 'docs/arbitrage/plan.html'
SPORTS = ['NFL', 'NBA', 'NHL', 'MLB']


def rules():
    cfg, nhl = json.loads(CONFIG.read_text()), json.loads(NHL.read_text())
    books = {key: dict(label=b['label'], kind=b['kind'], licensed=b['kind'] == 'licensed') for key, b in cfg['books'].items()}
    for key, label in nhl.get('display_names', {}).items():
        books.setdefault(key, dict(label=label, kind=None, licensed=None))
    sports = {sport: dict(profiles=s['profiles'], books=s['books'], pending=s.get('pending', {}))
              for sport, s in cfg['sports'].items()}
    sports['NHL'] = dict(profiles=nhl['profiles'], pending=nhl.get('pending', {}),
                         books={book: {k: v for k, v in rule.items() if k in ('game', 'player', 'assumed', 'source')}
                                for book, rule in nhl['books'].items()})
    return dict(version=f"{cfg['version']}+{nhl['version']}", scope=cfg['scope'], assumed_note=cfg['assumed_note'],
                kinds=cfg['kinds'], books=books, sports={s: sports[s] for s in SPORTS if s in sports})


def money(value):
    return f'−${-value:,.0f}' if value < 0 else f'${value:,.0f}'


def settlement_tables(r):
    groups = {'game': 'Game bets', 'player': 'Player props', 'batter': 'Batter props', 'pitcher': 'Pitcher props'}
    parts = []
    for sport, s in r['sports'].items():
        cols = [g for g in groups if g in ('game', 'player') or any(g in b for b in s['books'].values())]
        if sport == 'MLB':
            cols = ['game', 'batter', 'pitcher']
        def cell(rule, group):
            profile = rule.get(group)
            if not profile:
                return '<span class="muted">Not on file</span>'
            label = escape(s['profiles'].get(profile, {}).get('label', profile))
            return label + (' <span class="tag">assumed</span>' if group in rule.get('assumed', []) else ' <span class="tag ok">checked</span>')
        rows = ''.join(f'<tr><td>{escape(r["books"].get(book, {}).get("label") or book)}</td>'
                       + ''.join(f'<td>{cell(rule, g)}</td>' for g in cols) + '</tr>'
                       for book, rule in s['books'].items())
        pending = ''.join(f'<li><strong>{escape(k)}:</strong> {escape(v)}</li>' for k, v in s.get('pending', {}).items())
        parts.append(f'<h3>{sport}</h3><div class="table-wrap"><table><thead><tr><th>Sportsbook</th>'
                     + ''.join(f'<th>{groups[g]}</th>' for g in cols) + f'</tr></thead><tbody>{rows}</tbody></table></div>'
                     + (f'<ul class="meta">{pending}</ul>' if pending else ''))
    profiles = ''.join(f'<li><strong>{escape(p["label"])}</strong> ({sport}): {escape(p["description"])}</li>'
                       for sport, s in r['sports'].items() for p in s['profiles'].values())
    kinds = ''.join(f'<li><strong>{escape(k.title())}:</strong> {escape(v)}</li>' for k, v in r['kinds'].items())
    return ''.join(parts) + f'<h3>What the profiles mean</h3><ul>{profiles}</ul><h3>Kinds of sportsbook</h3><ul>{kinds}</ul>'


def replay_sections(replay):
    if not replay:
        none = '<p class="muted">The history replay has not been run.</p>'
        return dict(REPLAY_WINDOW='no replay yet', REPLAY_TABLE=none, REPLAY_SPORTS='', REPLAY_TAX='', REPLAY_EXAMPLES='',
                    REPLAY_LASTED='')
    window = (f"{eastern_time(replay['window']['from'], '%B %-d, %Y, %-I:%M %p %Z')} to "
              f"{eastern_time(replay['window']['to'], '%B %-d, %Y, %-I:%M %p %Z')}")
    rows = ''.join(
        f'<tr><td>{escape(s["label"])}<div class="meta">{escape(s["note"])}</div></td><td class="num">{s["pairs"]}</td>'
        f'<td class="num">{money(s["profit"]["250"])}</td><td class="num">{money(s["profit"]["1000"])}</td>'
        f'<td class="num">{money(s["profit_no_mlb"]["1000"])}</td><td class="num">{s["mean_return_pct"]:.2f}%</td></tr>'
        for s in replay['scenarios'])
    table = ('<div class="table-wrap"><table><thead><tr><th>Scenario</th><th>Pairs</th><th>$250 per pair</th>'
             '<th>$1,000 per pair</th><th>$1,000, no MLB</th><th>Average return</th></tr></thead>'
             f'<tbody>{rows}</tbody></table></div>')
    sports = ''.join(
        f'<tr><td>{sport}</td><td class="num">{s["snapshots"]}</td><td class="num">{s["snapshots_with_pairs"]}</td>'
        f'<td class="num">{s["pairs"]}</td><td class="num">{s["locked"]}</td><td class="num">{s["persisted"]}</td>'
        f'<td>{escape(", ".join(f"{a} + {b} ({n})" for a, b, n in s["top_book_pairs"][:3]))}</td></tr>'
        for sport, s in replay['sports'].items())
    sports = ('<div class="table-wrap"><table><thead><tr><th>League</th><th>Snapshots</th><th>With a pair</th><th>Pairs</th>'
              '<th>Rules on file</th><th>Seen twice or more</th><th>Most frequent books</th></tr></thead>'
              f'<tbody>{sports}</tbody></table></div>')
    tax = replay['tax']
    tax_rows = ''.join(f'<tr><td>{escape(t["label"])}</td><td class="num">{money(t["profit"])}</td><td class="num">{t["pairs_positive"]} of {t["pairs"]}</td></tr>'
                       for t in tax['cases'])
    tax_table = ('<div class="table-wrap"><table><thead><tr><th>Scenario B at $1,000 per pair</th><th>Profit</th>'
                 f'<th>Pairs still profitable</th></tr></thead><tbody>{tax_rows}</tbody></table></div>')
    def leg(l):
        line = '' if l['line'] is None else f" {l['line']:g}"
        price = f"+{l['price']:.0f}" if l['price'] > 0 else f"{l['price']:.0f}"
        return f"{l['side']}{line} {price} at {l['book_label']}"
    examples = ''.join(
        f'<tr><td>{e["sport"]}</td><td>{escape(e["game"])}<div class="meta">{escape(e["subject"])}</div></td>'
        f'<td>{" + ".join(escape(leg(l)) for l in e["legs"])}</td>'
        f'<td class="num">{e["return_pct"]:.2f}%</td><td class="num">{e["snapshots"]}</td>'
        f'<td>{"Locked" if e["status"] == "locked" else "Check first"}{"" if e["licensed"] else " · offshore book"}</td>'
        f'<td>{escape(eastern_time(e["first_seen"], "%b %-d, %-I:%M %p %Z"))}</td></tr>'
        for e in replay['examples'])
    examples = ('<div class="table-wrap"><table><thead><tr><th>League</th><th>Game and bet</th><th>Pair</th><th>Return</th>'
                f'<th>Snapshots</th><th>Status</th><th>First seen</th></tr></thead><tbody>{examples}</tbody></table></div>')
    mix = replay['persisted_mix']
    lasted = (f'Of the {sum(mix.values())} pairs seen in two or more snapshots, {mix["offshore"]} involved an offshore book and '
              f'{mix["mlb_props"]} were MLB props, whose settlement rules are not on file; {mix["licensed_other"]} were neither.')
    return dict(REPLAY_WINDOW=escape(window), REPLAY_TABLE=table, REPLAY_SPORTS=sports, REPLAY_TAX=tax_table,
                REPLAY_EXAMPLES=examples, REPLAY_LASTED=escape(lasted))


def plan(r):
    replay = json.loads(REPLAY.read_text()) if REPLAY.exists() else None
    title, description = 'Sports Arbitrage Plan', ('How Fourth & Value finds sportsbook arbitrage across NFL, NBA, NHL and MLB, '
                                                   'what a month of history shows, and the rules, taxes and risks before you bet.')
    html = TEMPLATE.read_text()
    values = dict(TITLE=escape(seo_title(title)), META=metadata(PLAN, seo_title(title), description),
                  SETTLEMENT=settlement_tables(r), RULES_VERSION=escape(r['version']),
                  ASSUMED_NOTE=escape(r['assumed_note']), **replay_sections(replay))
    for key, value in values.items():
        html = html.replace('{{' + key + '}}', value)
    if '{{' in html:
        raise ValueError('Unfilled placeholder in plan template')
    from site_notices import apply
    return apply(html, 'arbitrage/plan.html')


def outputs():
    r = rules()
    return {RULES: json.dumps(r, indent=1, ensure_ascii=False) + '\n', PLAN: plan(r)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = []
    for path, text in outputs().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    if stale:
        print('Out of date; run python scripts/arbitrage/build.py: ' + ', '.join(stale))
        return 1
    print('Arbitrage rules and plan are current.' if args.check else 'Wrote arbitrage rules and plan.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
