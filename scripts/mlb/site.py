"""Current-season MLB pages; no embedded historical betting cards."""
from html import escape
from mlb.refresh import ROOT
from site_metadata import metadata


def build(state):
    season = str(state['season'])
    label = season
    pages = [('index.html', 'MLB Overview', 'overview'), ('props/index.html', 'Player Props', 'props'),
             ('totals/index.html', 'Game Lines', 'lines'), ('top.html', 'Market Watch', 'watch'),
             ('methods.html', 'MLB Methods', 'methods')]
    for filename, title, page in pages:
        path = ROOT / 'docs/mlb' / filename
        rel = '../..' if '/' in filename else '..'
        mlb = '..' if '/' in filename else '.'
        links = '<nav class="subnav" aria-label="MLB sections">' + ''.join(
            f'<a href="{mlb}/{f}"' + (' aria-current="page"' if f == filename else '') + f'>{t}</a>'
            for f,t,_ in pages) + '</nav>'
        if page == 'overview':
            intro = '''<p class="lead">Follow the pennant race into October. Compare baseball prices with the matchup in view.</p>
<div class="actions"><a class="button primary" href="totals/">Compare game lines →</a><a class="button" href="props/">Compare player props →</a></div>
<div class="grid"><article class="panel"><h2>Pitcher and batter props</h2><p>Strikeouts, outs, hits, total bases, home runs and RBIs. Shop the same outcome across books.</p><a href="props/">Open props →</a></article><article class="panel"><h2>Game lines</h2><p>Moneylines, run lines and totals with de-vigged book prices and same-line consensus.</p><a href="totals/">Open game lines →</a></article><article class="panel"><h2>Market Watch</h2><p>Find prices that disagree with at least three other books at the same line.</p><a href="top.html">Explore disagreements →</a></article></div>
<section class="panel"><h2>Built for the postseason</h2><p>Regular-season games, Wild Card, Division Series, Championship Series and World Series games have distinct labels. Playoff markets appear when MLB confirms the matchup and time and sportsbooks post prices. Conditional games remain labeled “if necessary.”</p><p>Probable starters are provisional. Batting lineups are not verified here. Playoff pitching workloads, bullpen use and batting opportunities can differ sharply from regular-season averages.</p><a href="methods.html">How to read the baseball numbers →</a></section>
<section class="section"><h2>Upcoming games</h2><p class="muted">Official MLB schedule, next 45 days. Doubleheaders are matched by game time.</p><label>Season phase <select id="schedule-phase"><option value="">All games</option><option value="R">Regular season</option><option value="postseason">Postseason</option></select></label><div id="schedule" class="grid"></div></section>'''
        elif page == 'methods':
            intro = '''<div class="help"><p class="lead">A baseball market board with explicit postseason context.</p>
<h2>Which games qualify</h2><p>We match The Odds API events to MLB’s official schedule using both teams and a start time within ten minutes. MLB game IDs distinguish doubleheaders. Only regular season (R), Wild Card (F), Division Series (D), Championship Series (L) and World Series (W) qualify. Spring training, exhibitions, All-Star games, postponed or suspended games, delayed starts, and unknown start times are excluded. Unconfirmed playoff matchups are not invented.</p>
<h2>Price comparisons</h2><p>Moneylines, run lines and totals refer to full games. We do not mix first-five-inning markets into them. Props cover pitcher strikeouts and outs, plus batter hits, total bases, home runs and RBIs. Margin removal needs both sides from the same book, event, player, market and line. Run-line pairs use opposite handicaps. Each book gets one vote in same-line consensus; duplicate conflicting quotes are withheld.</p><p>Market Watch compares an offer with at least three other paired books, excluding the quoted book itself. It identifies disagreement, not a validated expected profit. Integer lines can push; de-vigged market probabilities are conditional on a non-push result. Confirm your book’s listed-pitcher, participation, postponement and settlement rules before using a quote.</p>
<h2>Pitchers, lineups and baseball statistics</h2><p>Probable pitchers come from MLB’s schedule and can change. We show those names on matchup cards without calling them confirmed. Batting lineups are not verified by this feed. Pitcher prop context identifies whether that player is listed as a probable starter.</p><p>Statistical context uses MLB regular-season totals through yesterday’s Eastern date. Current-day and postseason results are excluded. Pitcher K/9 and ERA use recorded outs: 5.2 innings means 17 outs, not 5.2 decimal innings. Batter context includes plate appearances, average, OPS and counting stats. These are descriptive statistics, not prop win probabilities; ambiguous player matches are withheld.</p>
<h2>Postseason adjustments still required</h2><p>A shorter starter outing or earlier bullpen switch can change strikeout and outs props. Confirmed batting order, platoon matchups, pinch-hit risk, park, weather and bullpen availability affect hitter opportunities and totals. A production model must estimate those inputs and pass chronological validation against outcomes and timestamped prices. Until then, MLB model probabilities remain unavailable and Market Watch is a research board.</p>
<h2>Freshness and coverage</h2><p>Markets refresh twice daily; quotes expire after 12 hours or when the game begins. Open boards refresh every five minutes. Failed updates hide saved offers while preserving the last successful timestamp. Prop requests cover the next 24 hours, capped at 20 games per run; skipped games are reported. Official season statistics refresh daily and their context is hidden if more than 36 hours old or the latest statistics refresh failed.</p>
<h2>Sources</h2><p><a href="https://www.mlb.com/schedule">MLB schedule</a> · <a href="https://www.mlb.com/postseason">MLB postseason</a> · <a href="https://www.mlb.com/stats/">MLB statistics</a> · <a href="https://the-odds-api.com/sports/mlb-odds.html">The Odds API MLB coverage</a></p></div>'''
        else:
            lead = {'props':'Compare pitcher strikeouts and outs, plus batter hits, total bases, home runs and RBIs.',
                    'lines':'Full-game totals, run lines and moneylines with probable-pitcher context and postseason labels.',
                    'watch':'Offers that differ from at least three other books at the same line. These are research leads, not validated model picks.'}[page]
            intro = f'''<p class="lead">{lead}</p>
<div class="filters"><label>Player or matchup<input id="search" type="search" placeholder="Search MLB…"></label><label>Market<select id="market"><option value="">All markets</option></select></label><label>Sportsbook<select id="book"><option value="">All books</option></select></label><label>Game<select id="game"><option value="">All games</option></select></label><label>Season phase<select id="phase"><option value="">All phases</option><option value="R">Regular season</option><option value="postseason">Postseason</option><option value="F">Wild Card</option><option value="D">Division Series</option><option value="L">Championship Series</option><option value="W">World Series</option></select></label></div>
<div class="checks"><label><input type="checkbox" id="best" checked> Best price at each line</label><button id="reset">Reset filters</button></div><p id="result-count" role="status"></p><div id="results" class="prop-grid"></div><button id="more" hidden>Show more</button>
<section class="help section"><h2>Read the comparison</h2><p>Book probability is the break-even rate at that price. Paired fair probability removes the book’s margin. Consensus uses distinct books at the same line. Season statistics describe past performance and do not qualify model picks.</p><a href="{mlb}/methods.html">MLB methods and limitations →</a></section>'''
        body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} | Fourth &amp; Value</title>
{metadata(path, title+' | Fourth & Value', 'MLB regular-season and postseason odds, player props, game lines and market consensus from Fourth & Value.')}
<link rel="stylesheet" href="{rel}/assets/site.css"><link rel="icon" href="{rel}/assets/logo.svg"></head><body><a class="skip-link" href="#main">Skip to content</a><div id="nav-root"></div><script src="{rel}/nav.js?v=40"></script>
<main class="wrap" id="main" data-mlb-page="{page}" data-feed="{mlb}/data/latest.json">{links}<p class="eyebrow">MLB · {label}</p><h1>{title}</h1>
<div class="notice" id="feed-status" role="status"><strong>MLB baseball market snapshot</strong><p>Last successful check: {escape(str(state.get('last_success_at') or 'Not yet checked'))}. Enable JavaScript to view current quote availability.</p></div>
<p class="muted" id="history-status"></p>{intro}<footer>Fourth &amp; Value · <a href="{rel}/terms.html">Terms &amp; privacy</a> · <a href="{rel}/videos/">Videos</a></footer></main><script src="{rel}/assets/mlb.js?v=1" defer></script></body></html>'''
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
