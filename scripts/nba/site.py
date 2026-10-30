"""Build NBA pages without importing sport-specific NFL templates."""
from html import escape
from pathlib import Path

from nba.pipeline import ROOT, timestamp
from site_metadata import metadata
from display_time import eastern_time


def build(state):
    pages = [('index.html', 'NBA Overview'), ('props/index.html', 'Player Props'),
             ('totals/index.html', 'Game Lines'), ('top.html', 'Market Watch'),
             ('methods.html', 'NBA Methods')]
    for filename, title in pages:
        path = ROOT / 'docs/nba' / filename
        rel = '../..' if '/' in filename else '..'
        nba = '..' if '/' in filename else '.'
        links = '<nav class="subnav" aria-label="NBA sections">' + ''.join(
            f'<a href="{nba}/{f.removesuffix("index.html")}"' + (' aria-current="page"' if f == filename else '') + f'>{t}</a>' for f,t in pages if f != 'methods.html') + '</nav>'
        if filename == 'index.html':
            intro = '''<p class="lead">Follow the schedule, compare sportsbooks and inspect NBA player props in one place.</p>
<div class="actions"><a class="button primary" href="props/">Compare player props →</a><a class="button" href="totals/">Compare game lines →</a></div>
<div class="grid"><article class="panel"><h2>Player props</h2><p>Points, rebounds, assists, threes, combinations, blocks, steals and turnovers. Match the player, market and line before comparing prices.</p><a href="props/">Open props →</a></article>
<article class="panel"><h2>Game lines</h2><p>Compare totals, spreads and moneylines. See paired fair probabilities and the books behind the consensus.</p><a href="totals/">Open game lines →</a></article>
<article class="panel"><h2>Market watch</h2><p>See quotes that differ from at least three other books at the same line. These are market disagreements to research.</p><a href="top.html">Explore disagreements →</a></article>
<article class="panel"><h2>Market analytics</h2><p>Every NBA prop and game line we track, graded against what the books’ own prices implied. Are overs or unders hitting this season?</p><a href="../markets/?sport=nba">See NBA market analytics →</a></article></div>
<section class="panel"><h2>Ready for opening night</h2><p>The regular season begins October 20. The board checks for new markets daily; player props are collected within 48 hours of tipoff. Early game lines can appear sooner.</p><p>Experimental NBA v1 forecasts appear in shadow mode: shown for research, not yet graded against prices and not used for picks. Historical baselines appear only when a player has enough box scores. Lineup changes and teammate injuries are not yet adjusted for.</p><a href="methods.html">What goes into the NBA numbers →</a></section>
<section class="section"><h2>Upcoming games</h2><p class="muted">Upcoming events from our odds provider. This is not the complete NBA schedule.</p><div id="schedule" class="grid"></div></section>'''
            page = 'overview'
        elif filename == 'methods.html':
            intro = '''<div class="help"><p class="lead">Separate the market's estimate from a basketball forecast.</p>
<h2>What is available now</h2><p>We collect NBA odds from The Odds API, compare identical lines across books, remove margin within each book, and show the median fair probability. Missing prices or unpaired outcomes stay unavailable.</p>
<h2>Consensus and market watch</h2><p>Over and under must belong to the same game, player, market, line and book. Spread pairs use opposite signed handicaps. Duplicate quotes never create extra book votes. Best price means the best payout on the exact same outcome and line.</p><p>Market Watch compares an offer against the median fair probability from at least three <em>other</em> books at that line. Its price-based return estimate is not a validated win probability or a recommendation. Integer lines can push; the screen compares probabilities conditional on a decision and does not estimate a cash return including pushes.</p>
<h2>Historical player baselines</h2><p>Box scores come from ESPN, starting with the 2023–24 season and updated after each game day. We use up to 30 prior regular-season appearances, with at least 20 required; play-in, playoff and NBA Cup final games are left out, matching the NBA’s official regular-season totals. Games with zero minutes, future dates, duplicate player/game IDs and ambiguous names are excluded. We do not fill missing rookies with a 50% guess.</p><p>The mean is the average observed box score. The historical probability is a smoothed hit rate, (wins + 1) ÷ (non-push games + 2). A combination uses that player's points, rebounds and assists from the same game. Equal results on whole-number lines are pushes. Last season's games may be used as historical context; they are not presented as an injury-adjusted prediction for a new rotation.</p><p>Select a player’s name for a snapshot: the last ten games against the line, how the hit rate is computed, how his recent games fell around the line and the opponent’s points or stats allowed over its last 20 games. NBA snapshots have no model and no track record yet.</p>
<h2>Experimental NBA v1 forecasts</h2><p>Each game line and player prop can show an NBA v1 probability and projection (its average and median), labeled experimental. The model runs in shadow mode: it is not used for Top Picks or articles until its forecasts have been graded against 2026–27 prices. Player Props and Game Lines can sort by Best value, which blends it with the market the way Today’s Picks does (10% model, 90% the other books’ no-vig price, the rule the shadow grade scores) and shows that blend’s expected value. The model’s own expected value is not shown.</p><p><strong>Games:</strong> pace and points per possession come from each team’s recent offense and defense, carried over from last season with shrinkage. Home court, rest and back-to-backs then adjust them. One score distribution prices the moneyline, spread and total together, including overtime and pushes on whole-number lines.</p><p><strong>Players:</strong> projected minutes (recent role, starts, rest, team changes and blowout risk) × per-minute production, adjusted for pace and the opponent’s allowed rates and blended with recent averages. Combinations such as points + rebounds + assists are simulated from the same minutes and correlated outcomes, never added up as if they were independent. Each forecast shows its average and its median, the value with half the model’s chance at or below it. Prop lines sit near the median, and counts are skewed upward, so an average above the line can still favor the Under. Forecasts assume the player plays, because books void props for players who don’t. Players listed out on ESPN’s injury report get no forecast. Other injury statuses are shown but not adjusted for, and teammate absences don’t change v1’s minutes.</p><p><strong>Testing:</strong> fit on 2023–24 and 2024–25, then tested once on the 2025–26 regular season. v1 beat a home-court-only baseline on moneyline log loss (0.603 vs. 0.688). On every player market it beat our historical hit rate, at lines set from each player’s last ten games. Points forecasts remain somewhat overconfident at the extremes. Early-season forecasts don’t know about offseason roster moves, so large October disagreements with the market are expected and are not evidence of value.</p>
<h2>Freshness and season handling</h2><p>Started games and quotes older than 24 hours are hidden. Failed refreshes are reported as feed errors, not as an empty slate. The last successful snapshot time remains visible. Preseason prices, if supplied, do not enter regular-season history. Opening-night markets can appear weeks before player props; an empty prop board is normal before the season.</p>
<h2>Sources</h2><p><a href="https://the-odds-api.com/sports/nba-odds.html">The Odds API: NBA coverage</a> · <a href="https://www.nba.com/stats/players/boxscores">NBA player box scores</a> · <a href="https://www.nba.com/news/2026-27-nba-regular-season-schedule">2026–27 NBA schedule</a></p></div>'''
            page = 'methods'
        else:
            page = {'props/index.html':'props', 'totals/index.html':'lines', 'top.html':'watch'}[filename]
            lead = {'props':'Compare the same NBA player prop across sportsbooks. Historical baselines, when available, are labeled separately.',
                    'lines':'Totals, spreads and moneylines from the same saved market snapshot.',
                    'watch':'Prices that differ from at least three other books. A market disagreement is a research lead, not a validated model pick.'}[page]
            # Player Props and Game Lines can show started games' last pregame quotes for lookup.
            started = '<label><input type="checkbox" id="history"> Include started games</label>' if page in ('props', 'lines') else ''
            # Best value sorts by the Today's Picks market blend (assets/briefing-picks.js).
            sort = '<label>Sort<select id="sort"><option value="value">Best value (our blend)</option><option value="time">Game time</option></select></label>' if page in ('props', 'lines') else ''
            sort_help = ('<p>Best value orders offers by the market blend Today’s Picks uses: 10% NBA v1 forecast and 90% the other books’ no-vig price, '
                         'then expected value at the listed price. Offers at 1% or more come first. NBA v1 is not yet graded against prices and NBA stays off '
                         'Today’s Picks, so this order is a research screen, not a recommendation.</p>') if sort else ''
            intro = f'''<p class="lead">{lead}</p>
<div class="filters"><label>Player or matchup<input id="search" type="search" placeholder="Search NBA…"></label><label>Market<select id="market"><option value="">All markets</option></select></label><label>Sportsbook<select id="book"><option value="">All books</option></select></label><label>Game<select id="game"><option value="">All games</option></select></label>{sort}</div>
<div class="checks"><label><input type="checkbox" id="best" checked> Best price at each line</label>{started}<button id="reset">Reset filters</button></div>
<p id="result-count" role="status"></p><div id="results" class="prop-grid"></div><button id="more" hidden>Show more</button>
<section class="help section"><h2>Read the comparison</h2><p>Book probability is the break-even rate at the quoted price. Fair probability removes that book's margin using both sides. Consensus uses distinct books at the exact same line. A historical baseline is not a calibrated game prediction.</p>{sort_help}<a href="{nba}/methods.html">NBA methods and limitations →</a></section>'''
        # Offer boards load the shared Track bet dialog before the board script.
        tracking = page in ('props', 'lines', 'watch')
        tracker_css = f'<link rel="stylesheet" href="{rel}/assets/offer-tracker.css?v=1">' if tracking else ''
        tracker_js = f'<script src="{rel}/assets/offer-tracker.js?v=3" defer></script>' if tracking else ''
        picks_js = f'<script src="{rel}/assets/briefing-picks.js?v=27" defer></script>' if page in ('props', 'lines') else ''
        # Player prop cards add a Share button that draws the offer as an image.
        share_css = f'<link rel="stylesheet" href="{rel}/assets/offer-share.css?v=1">' if page == 'props' else ''
        share_js = f'<script src="{rel}/assets/offer-share.js?v=1" defer></script>' if page == 'props' else ''
        search_title = title if title.startswith('NBA') else 'NBA ' + title
        descriptions = {'overview': 'Explore NBA matchups, player props and sportsbook comparisons, with links to historical player statistics and current model limitations.', 'props': 'Compare NBA player prop prices with recent scoring, rebounding, assists and minutes when available, using clearly labeled historical data.', 'lines': 'Compare NBA moneylines, spreads and totals at the same line across sportsbooks, with paired prices and market consensus.', 'watch': 'Find NBA prices that differ from other sportsbooks at the same line, and inspect historical player context before researching a bet.', 'methods': 'Learn how NBA sportsbook prices are compared, how historical player averages are calculated and why validated model picks remain unavailable.'}
        body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{search_title} | Fourth &amp; Value</title>
{metadata(path, search_title+' | Fourth & Value', descriptions[page])}
<link rel="stylesheet" href="{rel}/assets/site.css"><link rel="stylesheet" href="{rel}/assets/player-context.css?v=7"><script src="{rel}/assets/player-context.js?v=8" defer></script><link rel="icon" href="{rel}/assets/logo.svg">{tracker_css}{share_css}</head>
<body><a class="skip-link" href="#main">Skip to content</a><div id="nav-root"></div><script src="{rel}/nav.js?v=47"></script>
<main class="wrap" id="main" data-nba-page="{page}" data-feed="{nba}/data/latest.json">{links}<p class="eyebrow">NBA · 2026–27</p><h1>{title}</h1>
<div class="notice" id="feed-status" role="status"><strong>NBA market snapshot</strong><p>Last successful check: {escape(eastern_time(state['last_success_at']) if state.get('last_success_at') else 'Not yet checked')}. Enable JavaScript to view current quote availability.</p></div>
{intro}<footer>Fourth &amp; Value · <a href="{rel}/terms.html">Terms &amp; privacy</a> · <a href="{rel}/videos/">Videos</a></footer></main>
{tracker_js}{share_js}{picks_js}<script src="{rel}/assets/nba.js?v=8" defer></script></body></html>'''
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
