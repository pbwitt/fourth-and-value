"""Static NHL player prop pages for search, rebuilt by every refresh.

One stable URL per player (/nhl/players/<slug>/) with the next game, every book's price
at each line, the market's fair price, the experimental model estimate and recent games
against today's lines; plus an index of today's players by game and every player A-Z.
A player without posted props keeps the page with recent games only, so indexed URLs
never break, and an existing player keeps his slug. The section has its own sitemap
(docs/nhl/players/sitemap.xml, listed in robots.txt). Pages are rewritten only when
their content changes; nothing here selects or ranks picks.
"""
from collections import defaultdict
from datetime import datetime, timezone
import gzip
from html import escape
import json
from pathlib import Path
import re
import statistics
import sys
import unicodedata
from zoneinfo import ZoneInfo

from nba.pipeline import normal_name, timestamp
from site_metadata import SITE, seo_description, seo_title, social_tags

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/nhl/players'
EASTERN = ZoneInfo('America/New_York')
MARKETS = {'player_shots_on_goal': ('Shots on goal', 'shots', 'SOG'), 'player_goals': ('Goals', 'goals', 'G'),
           'player_assists': ('Assists', 'assists', 'A'), 'player_points': ('Points', 'points', 'P')}
RECENT = 10
# Fields Bet Tracker's shared dialog reads (docs/assets/offer-tracker.js), copied from the offer.
TICKET = ['event_id', 'commence_time', 'game', 'home_team', 'away_team', 'player', 'market', 'market_label', 'side',
          'line', 'book', 'book_label', 'price', 'quoted_at', 'settlement_profile', 'settlement_scope',
          'model_data_checked_at', 'model_withheld', 'independent_probability', 'final_probability', 'push_probability']


def slugify(name):
    text = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', text).strip('-')


def odds(price):
    price = round(float(price))
    return f'+{price}' if price > 0 else str(price)


def fair(probability):
    if not probability or not 0 < probability < 1:
        return None
    dec = 1 / probability
    return odds(100 * (dec - 1) if dec >= 2 else -100 / (dec - 1))


def eastern(value, pattern='%a, %b %-d, %-I:%M %p ET'):
    return timestamp(value).astimezone(EASTERN).strftime(pattern)


def history(season):
    """Per-game player records and games: the frozen training history plus this season's cache."""
    try:
        with gzip.open(ROOT / 'models/nhl/v2/history.json.gz', 'rt') as f:
            data = json.load(f)
        from nhl.v2.data import load
        games, players, _ = load(ROOT / 'data/nhl/v2/history', [season])
        return ([g for g in data['games'] if g['season'] != season] + games,
                [r for r in data['players'] if r['season'] != season] + players)
    except Exception as error:
        print(f'NHL player pages: history unavailable ({type(error).__name__}); recent games omitted', file=sys.stderr)
        return [], []


def read_registry(out):
    try:
        return json.loads((Path(out) / 'players.json').read_text())['players']
    except (OSError, ValueError, KeyError):
        return {}


def assign(registry, name, pid):
    """An existing player keeps his slug; a new namesake gets his id appended."""
    key = normal_name(name)
    for slug, entry in registry.items():
        if pid is not None and entry.get('player_id') == pid:
            return slug
    for slug, entry in registry.items():
        if normal_name(entry['name']) == key and (entry.get('player_id') is None or pid is None):
            return slug
    base = slugify(name) or 'player'
    return base if base not in registry else f'{base}-{pid if pid is not None else len(registry)}'


def offers(state, now):
    """Upcoming prop offers grouped by player, then by market and exact line and side."""
    players = {}
    for r in state.get('rows', []):
        if r.get('market') not in MARKETS or not r.get('player'):
            continue
        start = timestamp(r.get('commence_time'))
        if not start or start <= now:
            continue
        p = players.setdefault(normal_name(r['player']), dict(name=r['player'], player_id=None, games={}, lines=defaultdict(list)))
        p['player_id'] = p['player_id'] or r.get('player_id')
        p['games'][r.get('nhl_game_id') or r.get('event_id')] = r
        p['lines'][(r['market'], float(r['line']), r['side'])].append(r)
    return players


def summarize(quotes):
    best = min(quotes, key=lambda q: (q['book_probability'], q.get('quoted_at') or ''))
    paired = [q['fair_probability'] for q in quotes if q.get('fair_probability') is not None]
    model = next((q['conditional_probability'] for q in quotes if q.get('conditional_probability') is not None), None)
    return dict(best=best, books=len({q['book'] for q in quotes}), fair=statistics.median(paired) if paired else None,
                model=model, projection=next((q['projected_mean'] for q in quotes if q.get('projected_mean') is not None), None))


def ticket(row):
    out = {k: row.get(k) for k in TICKET}
    out.update(sport='NHL', market_label=MARKETS[row['market']][0])
    return out


def recent_games(records, games, teams):
    """Last regular-season games, newest first, with the opponent as seen from the player's side."""
    by_game = {g['game_id']: g for g in games}
    out = []
    for r in sorted(records, key=lambda r: r['game_date'], reverse=True):
        g = by_game.get(r['game_id'])
        if not g or g.get('game_type', 2) != 2:
            continue
        opponent = g['away_id'] if r.get('home') else g['home_id']
        out.append(dict(r, opponent=('vs ' if r.get('home') else '@ ') + teams.get(opponent, '?')))
        if len(out) == RECENT:
            break
    return out


def head(title, description, url, crumbs):
    ld = json.dumps({'@context': 'https://schema.org', '@type': 'BreadcrumbList', 'itemListElement': [
        {'@type': 'ListItem', 'position': i + 1, 'name': n, 'item': SITE + u} for i, (n, u) in enumerate(crumbs)]}).replace('</', '<\\/')
    t, d = escape(title, quote=True), escape(description, quote=True)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{t}</title><meta name="description" content="{d}"><link rel="canonical" href="{url}">'
            f'<meta property="og:type" content="website"><meta property="og:title" content="{t}"><meta property="og:description" content="{d}">'
            f'<meta property="og:url" content="{url}">{social_tags()}<script type="application/ld+json">{ld}</script>'
            '<link rel="stylesheet" href="/assets/site.css"><link rel="stylesheet" href="/assets/offer-tracker.css?v=1">'
            '<link rel="stylesheet" href="/assets/player-pages.css?v=1"><link rel="icon" href="/assets/logo.svg"></head>'
            '<body><a class="skip-link" href="#main">Skip to content</a><div id="nav-root"></div><script src="/nav.js?v=47"></script>')


def links(current):
    here = ' aria-current="page"' if current == 'index' else ''
    return ('<nav class="subnav" aria-label="NHL sections"><a href="/nhl/">NHL overview</a>'
            f'<a href="/nhl/players/"{here}>Player props</a>'
            '<a href="/nhl/props/">Compare every prop</a><a href="/nhl/methods.html">Methods</a></nav>')


POSITION_NAMES = {'C': 'Center', 'L': 'Left wing', 'R': 'Right wing', 'D': 'Defense'}


def player_page(slug, entry, offer, recent, seasons, mates, team_names):
    name, url = entry['name'], f'{SITE}/nhl/players/{slug}/'
    team = team_names.get(entry.get('team')) or entry.get('team')
    position = POSITION_NAMES.get(entry.get('position'))
    lines, sections, tickets = (offer or {}).get('lines', {}), [], []
    game = next(iter(sorted((offer or {}).get('games', {}).values(), key=lambda r: r['commence_time'])), None)
    if game:
        start = eastern(game['commence_time'])
        today = timestamp(game['commence_time']).astimezone(EASTERN).date() == datetime.now(EASTERN).date()
        when = ' Today' if today else ''
        title = seo_title(f'{name} Props{when}: Shots, Goals & Points Odds')
        day = eastern(game['commence_time'], '%b %-d')
        description = seo_description(f"{name} props for {game['away_team']} at {game['home_team']}, {day}: shots, goals, "
                                      'assists and points at every sportsbook, with best prices and fair odds.')
        lead = f"{escape(game['away_team'])} at {escape(game['home_team'])} · {escape(start)}"
        newest = max(q['quoted_at'] for qs in lines.values() for q in qs)
        body = [f'<p class="muted">Prices as of {escape(eastern(newest))}. Best price is the highest payout among books at that exact line; '
                'fair odds remove the books’ margin from paired prices. The model column is an experimental estimate, not a pick. '
                'Confirm the price and the rules at your sportsbook before betting.</p>']
        for market, (label, unit, _) in MARKETS.items():
            keys = sorted(k for k in lines if k[0] == market)
            if not keys:
                continue
            stats = [summarize(lines[k]) for k in keys]
            projection = next((s['projection'] for s in stats if s['projection'] is not None), None)
            rows = []
            for (_, line, side), s in zip(keys, stats):
                tickets.append(ticket(s['best']))
                push = ' · can push' if line.is_integer() else ''
                book = escape(s['best'].get('book_label') or s['best']['book'])
                model = '—' if s['model'] is None else f"{s['model']:.0%}"
                rows.append(f'<tr><td>{side} {line:g}{push}</td><td class="num">{odds(s["best"]["price"])}<span class="sub">{book}</span></td>'
                            f'<td class="num">{s["books"]}</td><td class="num">{fair(s["fair"]) or "—"}</td><td class="num">{model}</td>'
                            f'<td><span data-offer="{len(tickets) - 1}"></span></td></tr>')
            note = f'<p class="muted">Experimental model projection: {projection:.1f} {unit}.</p>' if projection is not None else ''
            body.append(f'<h3>{label}</h3>{note}<div class="table-wrap"><table><thead><tr><th>Bet</th><th>Best price</th><th>Books</th>'
                        f'<th>Fair odds</th><th>Model</th><th>Track</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
        sections.append(f'<section class="panel" id="lines"><h2>{escape(name)} lines and prices</h2>{"".join(body)}</section>')
    else:
        title = seo_title(f'{name} Props: Shots, Goals & Points Odds')
        description = seo_description(f'{name} NHL player props: shots on goal, goals, assists and points lines at every sportsbook '
                                      'when books post them, plus recent games against the lines.')
        lead = 'No props posted right now'
        sections.append('<section class="panel" id="lines"><h2>No props posted right now</h2><p>Books usually post NHL player props within '
                        'a day or two of puck drop. This page updates with every refresh of our NHL board.</p></section>')
    if recent:
        main = {}
        for (market, line, side) in lines:
            if side == 'Over':
                main.setdefault(market, set()).add(line)
        rates = []
        for market, values in main.items():
            label, unit, _ = MARKETS[market]
            for line in sorted(values):
                hits = sum(r[unit] > line for r in recent)
                rates.append(f'{escape(unit)} over {line:g} in {hits} of {len(recent)}')
        rows = ''.join(f'<tr><td>{escape(r["game_date"])}</td><td>{escape(r["opponent"])}</td><td class="num">{clock(r["toi"])}</td>'
                       f'<td class="num">{r["shots"]}</td><td class="num">{r["goals"]}</td><td class="num">{r["assists"]}</td><td class="num">{r["points"]}</td></tr>'
                       for r in recent)
        sections.append(f'<section class="panel" id="recent"><h2>Last {len(recent)} games</h2>'
                        + (f'<p>Against the current lines: {"; ".join(rates)}.</p>' if rates else '')
                        + '<div class="table-wrap"><table><thead><tr><th>Date</th><th>Opponent</th><th>Ice time</th><th>Shots</th><th>Goals</th>'
                        f'<th>Assists</th><th>Points</th></tr></thead><tbody>{rows}</tbody></table></div>'
                        '<p class="muted">Regular-season games from the official NHL record. Past results do not make a future outcome certain.</p></section>')
    if seasons:
        rows = ''.join(f'<tr><td>{s[:4]}–{s[6:]}</td><td class="num">{v["games"]}</td><td class="num">{v["shots"] / v["games"]:.1f}</td>'
                       f'<td class="num">{v["goals"]}</td><td class="num">{v["assists"]}</td><td class="num">{v["points"]}</td></tr>'
                       for s, v in seasons)
        sections.append('<section class="panel" id="seasons"><h2>Season totals</h2><div class="table-wrap"><table><thead><tr><th>Season</th>'
                        f'<th>Games</th><th>Shots per game</th><th>Goals</th><th>Assists</th><th>Points</th></tr></thead><tbody>{rows}</tbody></table></div></section>')
    if mates:
        sections.append('<section id="same-game"><h2>Also in this game</h2><ul class="players">'
                        + ''.join(f'<li><a href="/nhl/players/{m}/">{escape(n)}</a></li>' for m, n in mates) + '</ul></section>')
    data = json.dumps(tickets, separators=(',', ':')).replace('</', '<\\/')
    html = (head(title, description, url, [('NHL', '/nhl/'), ('Player props', '/nhl/players/'), (name, f'/nhl/players/{slug}/')])
            + f'<main class="wrap player-page" id="main">{links("player")}<header><p class="eyebrow">NHL player props{f" · {escape(team)}" if team else ""}'
            f'{f" · {position}" if position else ""}</p>'
            f'<h1>{escape(name)} props</h1><p class="lead">{lead}</p></header>{"".join(sections)}'
            '<p><a href="/nhl/players/">Every NHL player’s props →</a> · <a href="/nhl/props/">Compare every NHL prop →</a> · '
            '<a href="/nhl/methods.html">How we price NHL props →</a></p>'
            '<footer>Fourth &amp; Value · <a href="/terms.html">Terms &amp; privacy</a></footer></main>'
            f'<script type="application/json" id="offers">{data}</script>'
            '<script src="/assets/offer-tracker.js?v=2" defer></script><script src="/assets/player-props.js?v=1" defer></script></body></html>')
    return html, title, description


def index_page(registry, today, team_names):
    title = seo_title('NHL Player Props Today: Shots, Goals & Points by Player')
    description = seo_description('Every NHL player with props posted, grouped by game, plus an A–Z index of player pages with '
                                  'shots, goals, assists and points lines and recent games.')
    games = []
    for key, (label, start, players) in sorted(today.items(), key=lambda kv: kv[1][1]):
        items = ''.join(f'<li><a href="/nhl/players/{s}/">{escape(registry[s]["name"])}</a> <span class="muted">{escape(summary)}</span></li>'
                        for s, summary in sorted(players, key=lambda p: registry[p[0]]['name']))
        games.append(f'<section class="panel"><h3>{escape(label)}</h3><p class="muted">{escape(eastern(start))}</p><ul class="roster">{items}</ul></section>')
    letters = defaultdict(list)
    for slug, entry in registry.items():
        letters[(slugify(entry['name'])[:1] or '#').upper()].append((entry['name'], slug))
    az = ''.join(f'<h3>{letter}</h3><ul class="players">' + ''.join(f'<li><a href="/nhl/players/{s}/">{escape(n)}</a></li>' for n, s in sorted(v)) + '</ul>'
                 for letter, v in sorted(letters.items()))
    body = (f'<section id="today"><h2>Players with props posted</h2>'
            + (''.join(games) if games else '<p>No NHL player props are posted right now. Books usually post them within a day or two of puck drop.</p>')
            + f'</section><section id="all-players"><h2>All players A–Z</h2>{az}</section>')
    return (head(title, description, f'{SITE}/nhl/players/', [('NHL', '/nhl/'), ('Player props', '/nhl/players/')])
            + f'<main class="wrap player-page" id="main">{links("index")}<header><p class="eyebrow">NHL</p><h1>NHL player props today</h1>'
            '<p class="lead">Shots on goal, goals, assists and points lines for every player books have posted, with best prices, fair odds and recent games.</p></header>'
            f'{body}<p><a href="/nhl/props/">Compare every NHL prop side by side →</a></p>'
            '<footer>Fourth &amp; Value · <a href="/terms.html">Terms &amp; privacy</a></footer></main></body></html>')


def clock(minutes):
    seconds = round(minutes * 60)
    return f'{seconds // 60}:{seconds % 60:02d}'


def write(path, html, rel):
    """Write only real changes, so unchanged pages keep their sitemap date and add no history."""
    from site_notices import apply
    html = apply(html, rel)   # the site's responsible-use notice and footer
    if path.exists() and path.read_text() == html:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return True


def build(state, out=OUT, now=None, data=None):
    if state.get('status') not in ('ready', 'waiting_for_markets'):
        return dict(skipped=f"feed status {state.get('status')}", written=0, players=0)
    out, now = Path(out), now or datetime.now(timezone.utc)
    stamp_day = now.astimezone(EASTERN).date().isoformat()
    games, records = data if data is not None else history(state['season'])
    teams = {}
    team_names = {}
    for g in games:
        team_names.setdefault(g['home_id'], g['home_team']); team_names.setdefault(g['away_id'], g['away_team'])
    by_player, ids_by_name = defaultdict(list), defaultdict(set)
    regular = {g['game_id'] for g in games if g.get('game_type', 2) == 2}
    for r in records:
        by_player[r['player_id']].append(r)
        ids_by_name[normal_name(r['player'])].add(r['player_id'])
        teams[r['team_id']] = r.get('team_abbrev') or teams.get(r['team_id'])
    names = {abbrev: team_names.get(tid) for tid, abbrev in teams.items()}
    registry = read_registry(out)
    current = offers(state, now)
    for key, offer in list(current.items()):
        ids = ids_by_name.get(key, set())
        offer['player_id'] = offer['player_id'] or (next(iter(ids)) if len(ids) == 1 else None)
        if offer['player_id'] is None and len(ids) > 1:
            del current[key]   # namesakes without a stable identity: no page rather than a mixed one
            continue
        slug = assign(registry, offer['name'], offer['player_id'])
        entry = registry.setdefault(slug, dict(name=offer['name'], player_id=offer['player_id']))
        entry['player_id'] = entry.get('player_id') or offer['player_id']
        entry['last_props'] = stamp_day
        offer['slug'] = slug
    by_game = defaultdict(list)
    for offer in current.values():
        for game in offer['games']:
            by_game[game].append((offer['slug'], offer['name']))
    written, today = 0, {}
    for slug, entry in sorted(registry.items()):
        pid = entry.get('player_id')
        mine = sorted(by_player.get(pid, []), key=lambda r: r['game_date'])
        if mine:
            entry['team'] = mine[-1].get('team_abbrev')
            position = next((r['position'] for r in reversed(mine) if r.get('position') in POSITION_NAMES), None)
            if position:
                entry['position'] = position
        offer = next((o for o in current.values() if o['slug'] == slug), None)
        seasons = defaultdict(lambda: dict(games=0, shots=0, goals=0, assists=0, points=0))
        for r in mine:
            if r['game_id'] not in regular:
                continue
            s = seasons[str(r['season'])]
            s['games'] += 1
            for k in ('shots', 'goals', 'assists', 'points'):
                s[k] += r[k]
        mates = []
        if offer:
            for game, row in offer['games'].items():
                mates += [m for m in by_game[game] if m[0] != slug]
                summary = ', '.join(f"{MARKETS[m][2]} {l:g}" for m, l in sorted({(k[0], k[1]) for k in offer['lines'] if k[2] == 'Over'}))
                label = f"{row['away_team']} at {row['home_team']}"
                today.setdefault(game, (label, row['commence_time'], []))[2].append((slug, summary))
        html, _, _ = player_page(slug, entry, offer, recent_games(mine, games, teams),
                                 sorted(seasons.items(), reverse=True)[:2], sorted(set(mates), key=lambda m: m[1]), names)
        if write(out / slug / 'index.html', html, f'nhl/players/{slug}/index.html'):
            written += 1
            entry['updated'] = stamp_day
        entry.setdefault('updated', stamp_day)
    if write(out / 'index.html', index_page(registry, today, names), 'nhl/players/index.html'):
        written += 1
    (out / 'players.json').write_text(json.dumps(dict(version=1, players=registry), indent=1, sort_keys=True) + '\n')
    newest = max([e['updated'] for e in registry.values()] or [stamp_day])
    urls = [(f'{SITE}/nhl/players/', newest)] + [(f'{SITE}/nhl/players/{s}/', e['updated']) for s, e in sorted(registry.items())]
    sitemap = ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
               + ''.join(f'  <url><loc>{escape(u)}</loc><lastmod>{d}</lastmod></url>\n' for u, d in urls) + '</urlset>\n')
    if not (out / 'sitemap.xml').exists() or (out / 'sitemap.xml').read_text() != sitemap:
        (out / 'sitemap.xml').write_text(sitemap)
    return dict(written=written, players=len(registry), with_props=len(current))
