"""Build NBA pages without importing sport-specific NFL templates."""
from html import escape
from pathlib import Path

from nba.pipeline import ROOT, timestamp
from site_metadata import metadata


def build(state):
    pages = [('index.html', 'NBA Overview'), ('props/index.html', 'Player Props'),
             ('totals/index.html', 'Game Lines'), ('top.html', 'Market Watch'),
             ('methods.html', 'NBA Methods')]
    for filename, title in pages:
        path = ROOT / 'docs/nba' / filename
        rel = '../..' if '/' in filename else '..'
        nba = '..' if '/' in filename else '.'
        links = '<nav class="subnav" aria-label="NBA sections">' + ''.join(
            f'<a href="{nba}/{f}"' + (' aria-current="page"' if f == filename else '') + f'>{t}</a>' for f,t in pages if f != 'methods.html') + '</nav>'
        if filename == 'index.html':
            intro = '''<p class="lead">Follow the schedule, compare sportsbooks and inspect NBA player props in one place.</p>
<div class="actions"><a class="button primary" href="props/">Compare player props →</a><a class="button" href="totals/">Compare game lines →</a></div>
<div class="grid"><article class="panel"><h2>Player props</h2><p>Points, rebounds, assists, threes, combinations, blocks, steals and turnovers. Match the player, market and line before comparing prices.</p><a href="props/">Open props →</a></article>
<article class="panel"><h2>Game lines</h2><p>Compare totals, spreads and moneylines. See paired fair probabilities and the books behind the consensus.</p><a href="totals/">Open game lines →</a></article>
<article class="panel"><h2>Market watch</h2><p>See quotes that differ from at least three other books at the same line. These are market disagreements to research.</p><a href="top.html">Explore disagreements →</a></article></div>
<section class="panel"><h2>Ready for opening night</h2><p>The regular season begins October 20. The board checks for new markets daily; player props are collected within 48 hours of tipoff. Early game lines can appear sooner.</p><p>NBA predictions are not validated yet. Historical baselines appear only when enough player game logs are available. Minutes, lineup changes and injury effects need separate review.</p><a href="methods.html">What goes into the NBA numbers →</a></section>
<section class="section"><h2>Upcoming games</h2><p class="muted">Upcoming events from our odds provider. This is not the complete NBA schedule.</p><div id="schedule" class="grid"></div></section>'''
            page = 'overview'
        elif filename == 'methods.html':
            intro = '''<div class="help"><p class="lead">Separate the market's estimate from a basketball forecast.</p>
<h2>What is available now</h2><p>We collect NBA odds from The Odds API, compare identical lines across books, remove margin within each book, and show the median fair probability. Missing prices or unpaired outcomes stay unavailable.</p>
<h2>Consensus and market watch</h2><p>Over and under must belong to the same game, player, market, line and book. Spread pairs use opposite signed handicaps. Duplicate quotes never create extra book votes. Best price means the best payout on the exact same outcome and line.</p><p>Market Watch compares an offer against the median fair probability from at least three <em>other</em> books at that line. Its price-based return estimate is not a validated win probability or a recommendation. Integer lines can push; the screen compares probabilities conditional on a decision and does not estimate a cash return including pushes.</p>
<h2>Historical player baselines</h2><p>Where NBA Stats game logs are available, we use up to 30 prior regular-season appearances, with at least 20 required. Games with zero minutes, future dates, duplicate player/game IDs and ambiguous names are excluded. We do not fill missing rookies with a 50% guess.</p><p>The mean is the average observed box score. The historical probability is a smoothed hit rate, (wins + 1) ÷ (non-push games + 2). A combination uses that player's points, rebounds and assists from the same game. Equal results on whole-number lines are pushes. Last season's games may be used as historical context; they are not presented as an injury-adjusted prediction for a new rotation.</p>
<h2>Before model picks go live</h2><p>A production NBA model needs expected minutes, role and usage, current roster identities, opponent pace and efficiency, home court, rest and back-to-backs, and confirmed availability. We also need chronological out-of-sample testing, calibration and price-based evaluation. Until then, the site publishes market comparisons and labeled historical references, without a model-driven best-bet list.</p>
<h2>Freshness and season handling</h2><p>Started games and quotes older than 24 hours are hidden. Failed refreshes are reported as feed errors, not as an empty slate. The last successful snapshot time remains visible. Preseason prices, if supplied, do not enter regular-season history. Opening-night markets can appear weeks before player props; an empty prop board is normal before the season.</p>
<h2>Sources</h2><p><a href="https://the-odds-api.com/sports/nba-odds.html">The Odds API: NBA coverage</a> · <a href="https://www.nba.com/stats/players/boxscores">NBA player box scores</a> · <a href="https://www.nba.com/news/2026-27-nba-regular-season-schedule">2026–27 NBA schedule</a></p></div>'''
            page = 'methods'
        else:
            page = {'props/index.html':'props', 'totals/index.html':'lines', 'top.html':'watch'}[filename]
            lead = {'props':'Compare the same NBA player prop across sportsbooks. Historical baselines, when available, are labeled separately.',
                    'lines':'Totals, spreads and moneylines from the same saved market snapshot.',
                    'watch':'Prices that differ from at least three other books. A market disagreement is a research lead, not a validated model pick.'}[page]
            intro = f'''<p class="lead">{lead}</p>
<div class="filters"><label>Player or matchup<input id="search" type="search" placeholder="Search NBA…"></label><label>Market<select id="market"><option value="">All markets</option></select></label><label>Sportsbook<select id="book"><option value="">All books</option></select></label><label>Game<select id="game"><option value="">All games</option></select></label></div>
<div class="checks"><label><input type="checkbox" id="best" checked> Best price at each line</label><button id="reset">Reset filters</button></div>
<p id="result-count" role="status"></p><div id="results" class="prop-grid"></div><button id="more" hidden>Show more</button>
<section class="help section"><h2>Read the comparison</h2><p>Book probability is the break-even rate at the quoted price. Fair probability removes that book's margin using both sides. Consensus uses distinct books at the exact same line. A historical baseline is not a calibrated game prediction.</p><a href="{nba}/methods.html">NBA methods and limitations →</a></section>'''
        body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} | Fourth &amp; Value</title>
{metadata(path, title+' | Fourth & Value', 'NBA odds, player props, game lines and market consensus from Fourth & Value.')}
<link rel="stylesheet" href="{rel}/assets/site.css"><link rel="icon" href="{rel}/assets/logo.svg"></head>
<body><a class="skip-link" href="#main">Skip to content</a><div id="nav-root"></div><script src="{rel}/nav.js?v=42"></script>
<main class="wrap" id="main" data-nba-page="{page}" data-feed="{nba}/data/latest.json">{links}<p class="eyebrow">NBA · 2026–27</p><h1>{title}</h1>
<div class="notice" id="feed-status" role="status"><strong>NBA market snapshot</strong><p>Last successful check: {escape(str(state.get('last_success_at') or 'Not yet checked'))}. Enable JavaScript to view current quote availability.</p></div>
{intro}<footer>Fourth &amp; Value · <a href="{rel}/terms.html">Terms &amp; privacy</a> · <a href="{rel}/videos/">Videos</a></footer></main>
<script src="{rel}/assets/nba.js?v=1" defer></script></body></html>'''
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
