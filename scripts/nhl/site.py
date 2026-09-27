"""Current-season NHL pages; no embedded historical betting cards."""
from html import escape
from pathlib import Path
from nhl.refresh import ROOT
from site_metadata import metadata


def build(state):
    season = str(state['season'])
    label = season[:4] + '–' + season[-2:]
    pages = [('index.html', 'NHL Overview', 'overview'), ('props/index.html', 'Player Props', 'props'),
             ('totals/index.html', 'Game Lines', 'lines'), ('picks.html', 'Top Picks', 'candidates'), ('top.html', 'Market Watch', 'watch'),
             ('methods.html', 'NHL Methods', 'methods')]
    for filename, title, page in pages:
        path = ROOT / 'docs/nhl' / filename
        rel = '../..' if '/' in filename else '..'
        nhl = '..' if '/' in filename else '.'
        links = '<nav class="subnav" aria-label="NHL sections">' + ''.join(
            f'<a href="{nhl}/{f}"' + (' aria-current="page"' if f == filename else '') + f'>{t}</a>'
            for f,t,_ in pages if f != 'methods.html') + '</nav>'
        if page == 'overview':
            intro = '''<p class="lead">Fresh hockey markets, clear prices, and regular-season context.</p>
<div class="actions"><a class="button primary" href="totals/">Compare game lines →</a><a class="button" href="props/">Compare player props →</a></div>
<div class="grid"><article class="panel"><h2>Player props</h2><p>Shots on goal, goals, assists and points. Compare books at the same line.</p><a href="props/">Open props →</a></article><article class="panel"><h2>Game lines</h2><p>Totals, puck lines and moneylines with margin removed from paired prices.</p><a href="totals/">Open game lines →</a></article><article class="panel"><h2>Market Watch</h2><p>Find prices that differ from at least three other books, then examine the evidence.</p><a href="top.html">Explore market differences →</a></article></div>
<section class="panel"><h2>Ready for the regular season</h2><p>Only games confirmed as regular-season fixtures by the NHL schedule enter these boards. Preseason and playoff markets are excluded. Player props are checked within 48 hours of puck drop.</p><p>Independent forecasts use game-level history and chronological validation. Analyst context, uncertainty and model status accompany eligible offers. Recommendations remain disabled until executable-price evidence supports them.</p><a href="methods.html">How to read the NHL numbers →</a></section>
<section class="section"><h2>Upcoming regular-season games</h2><p class="muted">Official NHL schedule, next 45 days. Odds may appear closer to game day.</p><div id="schedule" class="grid"></div></section>'''
        elif page == 'methods':
            intro = (Path(__file__).parent / 'v2/methods.html').read_text()
        elif page == 'candidates':
            intro = (Path(__file__).parent / 'v2/candidates.html').read_text()
        else:
            lead = {'props':'Compare shots on goal, goals, assists and points. Props appear as books post them near puck drop.',
                    'lines':'Regular-season totals, puck lines and moneylines. Historical scoring references are labeled separately.',
                    'watch':'Offers that differ from at least three other books at the same line. These are research leads, not validated model picks.'}[page]
            intro = f'''<p class="lead">{lead}</p>
<div class="filters"><label>Player or matchup<input id="search" type="search" placeholder="Search NHL…"></label><label>Market<select id="market"><option value="">All markets</option></select></label><label>Sportsbook<select id="book"><option value="">All books</option></select></label><label>Game<select id="game"><option value="">All games</option></select></label></div>
<div class="checks"><label><input type="checkbox" id="best" checked> Best price at each line</label><button id="reset">Reset filters</button></div><p id="result-count" role="status"></p><div id="results" class="prop-grid"></div><button id="more" hidden>Show more</button>
<section class="help section"><h2>Read the comparison</h2><p>Book probability is the break-even rate at that price. Paired fair probability removes the book’s margin. Consensus uses distinct books at the same line. Historical references are uncalibrated and do not qualify model picks.</p><a href="{nhl}/methods.html">NHL methods and limitations →</a></section>'''
        body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} | Fourth &amp; Value</title>
{metadata(path, title+' | Fourth & Value', 'NHL regular-season odds, player props, game lines and market consensus from Fourth & Value.')}
<link rel="stylesheet" href="{rel}/assets/site.css"><link rel="icon" href="{rel}/assets/logo.svg"></head><body><a class="skip-link" href="#main">Skip to content</a><div id="nav-root"></div><script src="{rel}/nav.js?v=44"></script>
<main class="wrap" id="main" data-nhl-page="{page}" data-feed="{nhl}/data/latest.json">{links}<p class="eyebrow">NHL · {label}</p><h1>{title}</h1>
<div class="notice" id="feed-status" role="status"><strong>NHL regular-season market snapshot</strong><p>Last successful check: {escape(str(state.get('last_success_at') or 'Not yet checked'))}. Enable JavaScript to view current quote availability.</p></div>
<p class="muted" id="history-status"></p><p class="muted" id="model-status"></p>{intro}<footer>Fourth &amp; Value · <a href="{rel}/terms.html">Terms &amp; privacy</a> · <a href="{rel}/videos/">Videos</a></footer></main><script src="{rel}/assets/nhl.js?v=3" defer></script></body></html>'''
        path.parent.mkdir(parents=True, exist_ok=True)
        if page == 'candidates':
            body = body.replace('assets/nhl.js?v=3', 'assets/nhl-candidates.js?v=7')
            body = body.replace('<script src="../assets/nhl-candidates', '<script src="../assets/injury-context.js?v=1" defer></script><script src="../assets/nhl-candidates')
            body = body.replace('</head>', f'<link rel="stylesheet" href="{rel}/assets/nhl-candidates.css"></head>')
        path.write_text(body)
