"""Current-season NHL pages; no embedded historical betting cards."""
from html import escape
import re
import sys
from pathlib import Path
from nhl.refresh import ROOT
from site_metadata import metadata
from nba.pipeline import save_json
from nhl.v2.arbitrage import report, render


def settlement_rules():
    """Publish the sportsbook rules the pipeline relies on, from the same config it reads."""
    import json
    cfg = json.loads((Path(__file__).resolve().parents[2] / 'config/nhl_settlement.json').read_text())
    books, profiles, names = cfg.get('books', {}), cfg.get('profiles', {}), cfg.get('display_names', {})
    def cell(profile):
        return escape(profiles.get(profile, {}).get('label', profile)) if profile else 'Not verified'
    from datetime import date
    try: checked = date.fromisoformat(cfg['checked_at']).strftime('%B %-d, %Y')
    except (KeyError, ValueError): checked = 'recently'
    definitions = ''.join(f"<li><strong>{escape(p['label'])}:</strong> {escape(p['description'])}</li>" for p in profiles.values())
    rows = ''.join(
        f"<tr><td>{escape(names.get(book, book))}</td><td>{cell(rule.get('game'))}</td><td>{cell(rule.get('player'))}</td>"
        f"<td><a href=\"{escape(rule['source'])}\" rel=\"nofollow noopener\">Published rules</a></td></tr>"
        for book, rule in books.items())
    pending = ''.join(f"<li>{escape(book.title())}: {escape(note)}</li>" for book, note in cfg.get('pending', {}).items())
    return (f'<section class="panel" id="settlement-rules"><h2>Sportsbook rules we rely on</h2>'
            f'<p>Two prices are only comparable when the books grade the bet the same way. This is our reading of each '
            f'book&rsquo;s published hockey rules, last checked {checked}. It covers standard '
            f'full-game markets only. Rules can differ by state and by market name, so confirm them with your sportsbook before betting.</p>'
            f'<ul>{definitions}</ul>'
            f'<div class="table-wrap"><table><thead><tr><th>Sportsbook</th><th>Game bets</th><th>Player props</th><th>Source</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
            f'<p>&ldquo;Not verified&rdquo; and any sportsbook not listed: its prices still appear on our boards, but they are not '
            f'combined with other books into a market estimate and cannot become a model pick.</p>'
            f'<p>Why it matters, with examples: <a href="/blog/settlement-rules.html">Same bet, different rules</a>.</p>'
            + (f'<p>Being checked:</p><ul>{pending}</ul>' if pending else '') + '</section>')


def build(state):
    season = str(state['season'])
    label = season[:4] + '–' + season[-2:]
    pages = [('index.html', 'NHL Overview', 'overview'), ('props/index.html', 'Player Props', 'props'),
             ('totals/index.html', 'Game Lines', 'lines'), ('picks.html', 'Top Picks', 'candidates'), ('top.html', 'Market Watch', 'watch'),
             ('arbitrage.html', 'Arbitrage', 'arbitrage'),
             ('methods.html', 'NHL Methods', 'methods')]
    for filename, title, page in pages:
        path = ROOT / 'docs/nhl' / filename
        rel = '../..' if '/' in filename else '..'
        nhl = '..' if '/' in filename else '.'
        links = '<nav class="subnav" aria-label="NHL sections">' + ''.join(
            f'<a href="{nhl}/{f}"' + (' aria-current="page"' if f == filename else '') + f'>{t}</a>'
            for f,t,_ in pages if f != 'methods.html') + f'<a href="{nhl}/players/">Players</a></nav>'
        if page == 'overview':
            intro = '''<p class="lead">Fresh hockey markets, clear prices, and regular-season context.</p>
<div class="actions"><a class="button primary" href="totals/">Compare game lines →</a><a class="button" href="props/">Compare player props →</a></div>
<div class="grid"><article class="panel"><h2>Player props</h2><p>Shots on goal, goals, assists and points. Compare books at the same line.</p><a href="props/">Open props →</a></article><article class="panel"><h2>Game lines</h2><p>Totals, puck lines and moneylines with margin removed from paired prices.</p><a href="totals/">Open game lines →</a></article><article class="panel"><h2>Market Watch</h2><p>Find prices that differ from at least three other books, then examine the evidence.</p><a href="top.html">Explore market differences →</a></article></div>
<section class="panel"><h2>Ready for the regular season</h2><p>Only games confirmed as regular-season fixtures by the NHL schedule enter these boards. Preseason and playoff markets are excluded. Player props are checked within 48 hours of puck drop.</p><p>Independent forecasts use game-level history and chronological validation. Analyst context, uncertainty and model status accompany eligible offers. Recommendations remain disabled until executable-price evidence supports them.</p><a href="methods.html">How to read the NHL numbers →</a></section>
<section class="section"><h2>Upcoming regular-season games</h2><p class="muted">Official NHL schedule, next 45 days. Odds may appear closer to game day.</p><div id="schedule" class="grid"></div></section>'''
        elif page == 'methods':
            intro = (Path(__file__).parent / 'v2/methods.html').read_text() + settlement_rules()
        elif page == 'candidates':
            intro = (Path(__file__).parent / 'v2/candidates.html').read_text()
        elif page == 'arbitrage':
            arb = report(state)
            save_json(ROOT / 'docs/nhl/data/arbitrage.json', arb)
            intro = render(arb)
        else:
            lead = {'props':'Compare shots on goal, goals, assists and points. Props appear as books post them near puck drop.',
                    'lines':'Regular-season totals, puck lines and moneylines. Historical scoring references are labeled separately.',
                    'watch':'Offers that differ from at least three other books at the same line. These are research leads, not validated model picks.'}[page]
            intro = f'''<p class="lead">{lead}</p>
<div class="filters"><label>Player or matchup<input id="search" type="search" placeholder="Search NHL…"></label><label>Market<select id="market"><option value="">All markets</option></select></label><label>Sportsbook<select id="book"><option value="">All books</option></select></label><label>Game<select id="game"><option value="">All games</option></select></label></div>
<div class="checks"><label><input type="checkbox" id="best" checked> Best price at each line</label><button id="reset">Reset filters</button></div><p id="result-count" role="status"></p><div id="results" class="prop-grid"></div><button id="more" hidden>Show more</button>
<section class="help section"><h2>Read the comparison</h2><p>Book probability is the break-even rate at that price. Paired fair probability removes the book’s margin. Consensus uses distinct books at the same line. Historical references are uncalibrated and do not qualify model picks.</p><a href="{nhl}/methods.html">NHL methods and limitations →</a></section>'''
        search_title = title if title.startswith('NHL') else 'NHL ' + title
        descriptions = {'overview': 'Explore NHL regular-season matchups, sportsbook lines, player props and experimental forecasts with links to research and model methods.', 'props': 'Compare NHL player props with recent shots, scoring, ice time and model inputs alongside exact sportsbook lines and prices.', 'lines': 'Compare NHL moneylines, puck lines and game totals with exact sportsbook prices, independent forecasts and settlement context.', 'candidates': 'Review NHL research candidates with model probabilities, recent player form, ice time, source analysis and current quote checks.', 'watch': 'Compare NHL offers against other sportsbooks at the same line, with player statistics and clearly labeled market price differences.', 'arbitrage': 'Inspect NHL cross-book price combinations and settlement assumptions, with exact lines, potential returns and quote timestamps.', 'methods': 'Understand NHL count models, recency weighting, projected ice time, uncertainty, settlement rules and experimental forecast limits.'}
        body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{search_title} | Fourth &amp; Value</title>
{metadata(path, search_title+' | Fourth & Value', descriptions[page])}
<link rel="stylesheet" href="{rel}/assets/site.css"><link rel="stylesheet" href="{rel}/assets/player-context.css?v=3"><script src="{rel}/assets/player-context.js?v=3" defer></script><link rel="icon" href="{rel}/assets/logo.svg"></head><body><a class="skip-link" href="#main">Skip to content</a><div id="nav-root"></div><script src="{rel}/nav.js?v=47"></script>
<main class="wrap" id="main" data-nhl-page="{page}" data-feed="{nhl}/data/latest.json">{links}<p class="eyebrow">NHL · {label}</p><h1>{title}</h1>
<div class="notice" id="feed-status" role="status"><strong>NHL regular-season market snapshot</strong><p>Last successful check: {escape(str(state.get('last_success_at') or 'Not yet checked'))}. Enable JavaScript to view current quote availability.</p></div>
<p class="muted" id="history-status"></p><p class="muted" id="model-status"></p>{intro}<footer>Fourth &amp; Value · <a href="{rel}/terms.html">Terms &amp; privacy</a> · <a href="{rel}/videos/">Videos</a></footer></main><script src="{rel}/assets/nhl.js?v=3" defer></script></body></html>'''
        path.parent.mkdir(parents=True, exist_ok=True)
        if page == 'arbitrage':
            # Server-rendered from the same snapshot; no feed script to overwrite it.
            body = body.replace(f'<script src="{rel}/assets/nhl.js?v=3" defer></script>', '')
            body = re.sub(r'<div class="notice" id="feed-status".*?</div>\n<p class="muted" id="history-status"></p><p class="muted" id="model-status"></p>', '', body, flags=re.S)
        if page == 'candidates':
            body = body.replace('assets/nhl.js?v=3', 'assets/nhl-candidates.js?v=7')
            body = body.replace('<script src="../assets/nhl-candidates', '<script src="../assets/injury-context.js?v=1" defer></script><script src="../assets/nhl-candidates')
            body = body.replace('</head>', f'<link rel="stylesheet" href="{rel}/assets/nhl-candidates.css"></head>')
        if page in ('props', 'lines', 'watch', 'candidates'):
            body = body.replace('</head>', f'<link rel="stylesheet" href="{rel}/assets/offer-tracker.css?v=1"></head>')
            body = body.replace(f'<script src="{rel}/assets/nhl', f'<script src="{rel}/assets/offer-tracker.js?v=1" defer></script><script src="{rel}/assets/nhl', 1)
            body = body.replace('assets/nhl.js?v=3', 'assets/nhl.js?v=7').replace('assets/nhl-candidates.js?v=7', 'assets/nhl-candidates.js?v=11')
        path.write_text(body)
    try:
        # The pop-up's track record for the running model version; display only.
        from nhl.v2.track import write as write_track
        write_track()
    except Exception as error:
        print(f'NHL track record skipped ({type(error).__name__}: {error})', file=sys.stderr)
    try:
        # One search page per player (scripts/nhl/players.py); never blocks the board.
        from nhl.players import build as build_players
        result = build_players(state)
        print(f"NHL player pages: {result.get('players', 0)} players, {result.get('written', 0)} pages updated"
              + (f" (skipped: {result['skipped']})" if result.get('skipped') else ''))
    except Exception as error:
        print(f'NHL player pages skipped ({type(error).__name__}: {error})', file=sys.stderr)
