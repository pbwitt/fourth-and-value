#!/usr/bin/env python3
"""Build a crawlable NFL board with paginated filtering and explicit model coverage."""
import argparse
import json
from html import escape
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from market_math import add_market_comparisons, expected_profit
from site_metadata import metadata, nfl_links, root_relative
from site_common import pretty_market, kickoff_et

LABELS = {'rush_yds':'Rushing yards', 'recv_yds':'Receiving yards', 'pass_yds':'Passing yards',
          'rush_attempts':'Rushing attempts', 'receptions':'Receptions', 'pass_attempts':'Passing attempts',
          'pass_completions':'Passing completions', 'pass_tds':'Passing touchdowns',
          'pass_interceptions':'Interceptions thrown', 'interceptions':'Interceptions thrown', 'anytime_td':'Anytime touchdown'}


def prepare_records(path):
    d = pd.read_csv(path, low_memory=False)
    d = add_market_comparisons(d)
    for col in ['model_prob', 'mu', 'push_prob']:
        if col not in d:
            d[col] = np.nan
    if 'model_status' not in d:
        # Old exported probabilities predate provenance and cannot be promoted.
        d['model_status'] = 'Legacy estimate; not verified'
    d['edge_bps'] = (d['model_prob'] - d['mkt_prob']) * 10000
    d['ev_per_100'] = d.apply(lambda r: expected_profit(r['model_prob'], r['price'], r['push_prob']), axis=1)
    d['market_label'] = d['market_std'].map(LABELS).fillna(d['market_std'].map(pretty_market))
    d['book_label'] = d.get('bookmaker_title', d.get('bookmaker', '')).fillna(d.get('bookmaker', ''))
    d['kick_et'] = d['commence_time'].map(kickoff_et)
    d = d[d['player'].str.lower().ne('no scorer')].copy()
    d = d.drop_duplicates(['game_id', 'player', 'market_std', 'name', 'point', 'bookmaker'] if 'game_id' in d else ['commence_time', 'player', 'market_std', 'name', 'point', 'bookmaker'])
    cols = ['game_id','game','player','bookmaker','book_label','market_std','market_label','name','point','price',
            'mu','model_prob','push_prob','mkt_prob','prob_devig','consensus_prob','consensus_line','book_count',
            'edge_bps','ev_per_100','model_status','last_update','commence_time','kick_et','home_team','away_team']
    for c in cols:
        if c not in d:
            d[c] = None
    return json.loads(d[cols].to_json(orient='records'))


def static_card(r):
    """Useful content without JavaScript; the full board is progressively enhanced."""
    line = '' if r['point'] is None else f"{r['point']:g}"
    odds = '—' if r['price'] is None else f"{r['price']:+g}"
    return f'''<article class="panel prop-card"><p class="meta">{escape(r['kick_et'] or '')} · {escape(r['game'] or '')}</p>
+<h2>{escape(r['player'])}</h2><p>{escape(r['market_label'])}</p>
+<p class="betline">{escape(r['name'].title())} {line} · {odds}</p><p>{escape(r['book_label'])}</p>
+<p class="meta">{escape(r['model_status'])}</p></article>'''.replace('\n+', '\n')


def build_page(args, top_only=False):
    records = prepare_records(args.merged_csv)
    now = datetime.now(timezone.utc)
    future = [r for r in records if r['commence_time'] and pd.to_datetime(r['commence_time'], utc=True) > now]
    # Do not label a build time as the time the sportsbook price was checked.
    updated = [pd.to_datetime(r['last_update'], utc=True, errors='coerce') for r in future]
    verified = bool(updated) and all(pd.notna(t) and 0 <= (now-t).total_seconds() <= 172800 for t in updated)
    status = ('No upcoming NFL games in this snapshot.' if not future else
              'Sportsbook quote times are within the last 48 hours. Confirm the current line before using an estimate.' if verified else
              'Quote freshness is unverified or older than 48 hours. These are saved prices; check your sportsbook.')
    title = 'NFL Top Picks' if top_only else 'NFL Player Props & Odds Comparison'
    rel = root_relative(args.out)
    context = f'{args.season} · Week {args.week}' if args.season and args.week else 'NFL odds comparison'
    description = 'Compare NFL player prop lines across sportsbooks, inspect model probabilities and understand the evidence behind each estimate.'
    initial = [r for r in future if not top_only or (r['model_status'].startswith('Calibration fitted') and (r['edge_bps'] or 0) > 0 and r['last_update'] and 0 <= (now-pd.to_datetime(r['last_update'],utc=True)).total_seconds() <= 172800)][:24]
    empty = '<div class="empty"><h2>No qualifying picks in this snapshot</h2><p>Fresh quotes, player evidence and a fitted calibration curve are required for Top Picks.</p><a href="index.html">Compare all sportsbook lines</a></div>' if top_only else '<div class="empty">No upcoming props in this snapshot. Check back after the next data refresh.</div>'
    # Dictionary-encoded columns avoid repeating team names and field names on
    # every offer. The shortlist ships only its qualifying candidates.
    published = [r for r in future if r['model_status'].startswith('Calibration fitted')
                 and (r['edge_bps'] or 0) > 0 and r['last_update']
                 and 0 <= (now-pd.to_datetime(r['last_update'],utc=True)).total_seconds() <= 172800] if top_only else records
    fields = list(records[0]) if records else []
    dictionary = {}
    for field in fields:
        values = [r[field] for r in published]
        if values and all(v is None or isinstance(v, str) for v in values):
            dictionary[field] = list(dict.fromkeys(values))
    indexes = {field: {v:i for i,v in enumerate(values)} for field,values in dictionary.items()}
    packed = [[indexes[field][r[field]] if field in indexes else r[field] for field in fields] for r in published]
    payload = json.dumps({'fields':fields, 'dictionary':dictionary, 'rows':packed, 'topOnly':top_only, 'root':rel, 'snapshotUpcoming':len(future), 'snapshotVerified':verified, 'lastKickoff':max((r['commence_time'] for r in future), default=None)}, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} | Fourth &amp; Value</title>{metadata(args.out,title+' | Fourth & Value',description)}
<link rel="icon" href="{rel}/assets/logo.svg" type="image/svg+xml"><link rel="stylesheet" href="{rel}/assets/site.css"></head>
<body><a class="skip-link" href="#main">Skip to props</a><div id="nav-root"></div><script src="{rel}/nav.js?v=33"></script>
<main id="main" class="wrap">{nfl_links(rel, 'Top picks' if top_only else 'Player props')}
<p class="eyebrow">{context}</p><h1>{title}</h1>
<p class="lead">{'A shortlist of positive model edges with player data, fitted calibration and recent quotes.' if top_only else 'Find a player, compare the same line across books, and see what supports the model estimate.'}</p>
<div class="notice" role="status"><strong id="freshness">{status}</strong><p class="meta">Page built <time datetime="{now.isoformat()}">{now.strftime('%b %d, %Y at %H:%M UTC')}</time>. Model estimates are experimental; calibration is not proof of profitability.</p></div>
<details class="help"><summary>How to read prices, probabilities and edges</summary>
<p><strong>Book probability</strong> is the break-even probability implied by the offered odds, including margin. <strong>Paired fair probability</strong> removes margin using both sides at the same book, game and line; it is unavailable without a matching opposite side.</p>
<p><strong>Model probability</strong> is conditional on the bet settling without a push. <strong>Edge</strong> is model probability minus book probability: 100 basis points = 1 percentage point. <strong>Expected profit / $100</strong> accounts for pushes and is an uncertain estimate, not a payout promise.</p>
<p>Consensus probability uses paired prices at this exact line. The median line is a separate descriptive comparison. Model means are not calibrated betting thresholds. <a href="{rel}/methods.html#edge-nfl">Read the full method and limitations</a>.</p></details>
<section class="panel" aria-label="Filter props" id="filters" hidden>
<div class="filters"><label>Search player or team<input id="q" type="search" placeholder="Player or team name"></label>
<label>Market<select id="market"><option value="">All markets</option></select></label>
<label>Game<select id="game"><option value="">All games</option></select></label>
<label>Sort by<select id="sort"><option value="kickoff">Kickoff</option><option value="edge">Model edge</option><option value="ev">Expected profit</option><option value="player">Player name</option></select></label></div>
<details><summary>Choose sportsbooks <span id="book-count"></span></summary><div class="actions"><button id="all-books" type="button">Select all</button><button id="no-books" type="button">Clear all</button></div><div id="books" class="books"></div></details>
<div class="checks"><label><input id="best" type="checkbox" checked>Best price per line</label><label><input id="positive" type="checkbox">Positive model edges</label><label><input id="history" type="checkbox">Include started games</label></div>
<div class="actions"><button id="reset" type="button">Reset filters</button><button id="share" type="button">Copy filtered link</button><span id="feedback" role="status"></span></div></section>
<div class="results-bar"><p id="count" role="status">{len(future):,} upcoming offers in this snapshot</p><a href="{'index.html' if top_only else 'top.html'}">{'Compare all props' if top_only else 'View qualifying top picks'} →</a></div>
<div id="results" class="prop-grid">{''.join(static_card(r) for r in initial) or empty}</div>
<div class="pager" id="pager" hidden><button id="previous">Previous</button><span id="page-info" aria-live="polite"></span><button id="next">Next</button></div>
<noscript><p>The first 24 upcoming offers are shown. Enable JavaScript to filter and compare the full snapshot.</p></noscript>
<footer><a href="{rel}/nfl/">NFL overview</a> · <a href="{rel}/methods.html">Methods &amp; limitations</a> · <a href="{rel}/terms.html">Terms &amp; privacy</a><p>Free sports analysis from Fourth &amp; Value. No guaranteed outcomes.</p></footer></main>
<script type="application/json" id="props-data">{payload}</script><script src="{rel}/assets/props.js" defer></script></body></html>'''
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(html)
    print(f'[props] wrote {args.out}: {len(records):,} offers, {len(future):,} upcoming; quote freshness verified: {verified}')


def main(top_only=False):
    ap=argparse.ArgumentParser()
    ap.add_argument('--merged_csv',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--season',type=int); ap.add_argument('--week',type=int); ap.add_argument('--title')
    # Retained for existing callers; the browser now paginates rather than silently truncating.
    ap.add_argument('--limit',type=int); ap.add_argument('--min_prob',type=float)
    ap.add_argument('--drop_no_scorer',action='store_true'); ap.add_argument('--show_unmodeled',action='store_true')
    build_page(ap.parse_args(),top_only)

if __name__ == '__main__':
    main()
