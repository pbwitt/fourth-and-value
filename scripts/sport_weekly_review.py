#!/usr/bin/env python3
"""Auditable MLB/NHL/NBA weekly reviews from saved pregame evidence.

Never reconstruct picks from today's selector. Published cards pass the canonical
editionRows reader; full-board comparisons are a separate population. Results
missing from official feeds remain unresolved. No paid models or odds calls.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import gzip
from html import escape
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
ET = ZoneInfo('America/New_York')
LABELS = {'player_rebounds': 'Rebounds', 'player_threes': 'Three-pointers', 'player_points_rebounds_assists': 'Points, rebounds and assists', 'player_points_rebounds': 'Points and rebounds', 'player_points_assists': 'Points and assists', 'player_rebounds_assists': 'Rebounds and assists', 'player_blocks': 'Blocks', 'player_steals': 'Steals', 'player_turnovers': 'Turnovers', 'totals': 'Game total', 'h2h': 'Moneyline', 'spreads': 'Spread / run line',
          'pitcher_outs': 'Pitcher outs', 'pitcher_strikeouts': 'Pitcher strikeouts',
          'batter_hits': 'Batter hits', 'batter_total_bases': 'Batter total bases',
          'batter_home_runs': 'Batter home runs', 'batter_rbis': 'Batter RBIs',
          'player_assists': 'Assists', 'player_points': 'Points',
          'player_goals': 'Goals', 'player_shots_on_goal': 'Shots on goal'}


def stamp(value):
    try:
        t = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return t.astimezone(timezone.utc) if t.tzinfo else None
    except (ValueError, TypeError):
        return None


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def decimal(price):
    return 1 + (price / 100 if price > 0 else 100 / abs(price))


def gid(row, sport):
    value = row.get(f'{sport}_game_id')
    if sport == 'nba' and value is None:
        value = row.get('event_id') or row.get('id')
    return str(value) if value is not None else None


def name(value):
    from build_market_results import fold
    return fold(value or '')


def load_runs(sport, root=ROOT):
    runs = []
    for path in sorted((Path(root) / f'artifacts/{sport}/runs').glob('*.json.gz')):
        with gzip.open(path, 'rt') as f:
            run = json.load(f)
        snap = run.get('snapshot', run)
        if stamp(snap.get('checked_at')):
            runs.append((path, run, snap))
    return sorted(runs, key=lambda r: stamp(r[2]['checked_at']))


def pregame(runs, sport):
    """Last captured snapshot per game, strictly before official AND offered start."""
    chosen = {}
    for path, _, snap in runs:
        at = stamp(snap['checked_at'])
        if at is None or snap.get('local_replay'):
            continue
        events = {gid(e, sport): e for e in snap.get('events', [])}
        grouped = defaultdict(list)
        for row in snap.get('rows', []):
            game = gid(row, sport)
            start = stamp(row.get('commence_time'))
            official = stamp(events.get(game, {}).get('commence_time'))
            quoted = stamp(row.get('quoted_at'))
            cutoff = min(start, official) if start and official else start
            # API responses may be a few seconds newer than checked_at (request start),
            # but neither their quote nor ingestion time may reach kickoff.
            recorded = [stamp(record[key]) for record, keys in (
                (snap, ('captured_at', 'model_prediction_at', 'model_checked_at')),
                (row, ('ingested_at', 'forecast_at', 'decision_at')))
                for key in keys if record.get(key) is not None]
            if (not game or not cutoff or at >= cutoff or not quoted or quoted >= cutoff or
                    any(t is None or t >= cutoff for t in recorded)):
                continue
            if not number(row.get('price')) or abs(row['price']) < 100:
                continue
            grouped[game].append(dict(row, pregame_cutoff=cutoff.isoformat(), evidence_at=max([at,quoted]+recorded).isoformat()))
        for game, rows in grouped.items():
            chosen[game] = dict(captured_at=at.isoformat(), source=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
                                rows=rows, event=events.get(game, {}))
    return chosen


def published_picks(sport, start, end, root=ROOT):
    """Use the same historical-card reader as the website, not today's ranking."""
    program = r'''const fs=require('fs'); const path=require('path');
const root=process.argv[1],sport=process.argv[2],start=process.argv[3],end=process.argv[4];
const {editionRows,day,SHORTLIST_LIMIT}=require(path.join(root,'docs/assets/briefing-picks.js')); const days={};
const directory=path.join(root,'docs/briefing/cards');
for(const f of (fs.existsSync(directory)?fs.readdirSync(directory):[]).sort()){
 if(!f.endsWith('.json'))continue;
 const c=JSON.parse(fs.readFileSync(path.join(root,'docs/briefing/cards',f)));
 const at=Date.parse(c.published_at);
 if(c.kind!=='morning'||c.schema_version!==1||c.decision_date<start||c.decision_date>end||
    !/(Z|[+-]\d\d:\d\d)$/.test(c.published_at)||!Number.isFinite(at)||day(at)!==c.decision_date||
    !Array.isArray(c.rows)||c.rows.length>SHORTLIST_LIMIT)continue;
 const old=days[c.decision_date];if(!old||at>Date.parse(old.published_at))days[c.decision_date]=c;
}
const rows=[];for(const c of Object.values(days))for(const r of editionRows(c,Date.parse(c.published_at)+1)){
 if(r.sport?.toLowerCase()===sport)rows.push({...r,edition_id:c.edition_id,published_at:c.published_at,policy_version:c.policy_version||'unversioned'});
}process.stdout.write(JSON.stringify({rows,editions:Object.values(days).map(c=>({id:c.edition_id,date:c.decision_date,policy:c.policy_version||'unversioned'}))}));'''
    result = subprocess.run(['node', '-e', program, str(Path(root).resolve()), sport, str(start), str(end)],
                            check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def fetch_json(url):
    response = requests.get(url, timeout=35)
    response.raise_for_status()
    return response.json()


def mlb_results(start, end, game_ids, root=ROOT, fetch=fetch_json, offline=False):
    cache = Path(root) / 'artifacts/mlb/results'
    cache.mkdir(parents=True, exist_ok=True)
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start}&endDate={end}'
    game_ids = {str(g) for g in game_ids}
    if offline:
        records = [json.loads(path.read_text()) for key in sorted(game_ids)
                   if (path := cache / f'{key}.json').exists()]
    else:
        schedule = fetch(url)
        records = []
        for day in schedule.get('dates', []):
            for game in day.get('games', []):
                key = str(game['gamePk'])
                if key not in game_ids or game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                try:
                    box = fetch(f'https://statsapi.mlb.com/api/v1/game/{key}/boxscore')
                except requests.RequestException:
                    box = {}  # Team scores can be settled; missing player stats stay unresolved.
                record = dict(game=game, boxscore=box)
                (cache / f'{key}.json').write_text(json.dumps(record, sort_keys=True) + '\n')
                records.append(record)
    games, players = {}, {}
    for record in records:
        game, box = record['game'], record['boxscore']
        key = str(game['gamePk'])
        if key not in game_ids or game.get('status', {}).get('abstractGameState') != 'Final':
            continue
        home, away = game['teams']['home'], game['teams']['away']
        if not number(home.get('score')) or not number(away.get('score')):
            continue
        games[key] = dict(home_score=home['score'], away_score=away['score'],
                          home_team=home['team']['name'], away_team=away['team']['name'])
        for side in ('home', 'away'):
            for p in box.get('teams', {}).get(side, {}).get('players', {}).values():
                stats = p.get('stats') or {}
                person = p.get('person') or {}
                if person.get('id') is not None:
                    players[(key, str(person['id']))] = dict(player=person.get('fullName', ''), **stats)
    return games, players


def nhl_results(runs, years, root=ROOT):
    from build_market_results import nhl_results as official_results
    games, players = {}, {}
    for year in years:
        g, p = official_results([r[1] for r in runs], int(f'{year}{year+1}'), Path(root) / 'artifacts/nhl')
        games.update({str(k): v for k, v in g.items()})
        players.update({(str(k[0]), str(k[1])): v for k, v in p.items()})
    return games, players


def actual(row, sport, games, players):
    if sport == 'nba':
        from nba_weekly_results import get_actual
        official = stamp(row.get('official_commence_time'))
        times = [stamp(row.get(k)) for k in ('evidence_at','comparison_evidence_at','published_at','quoted_at','ingested_at','forecast_at','decision_at') if row.get(k)]
        if official and any(t is None or t >= official for t in times):
            return None, 'evidence_at_or_after_official_tipoff'
        return get_actual(row, games, players)
    game_id = gid(row, sport)
    game = games.get(game_id)
    if not game:
        return None, 'official_result_unavailable'
    market = row.get('market')
    if not number(game.get('home_score')) or not number(game.get('away_score')):
        return None, 'official_result_unavailable'
    if market == 'totals':
        return game['home_score'] + game['away_score'], 'graded'
    if market in ('h2h', 'spreads'):
        if row.get('side') not in (row.get('home_team'), row.get('away_team')):
            return None, 'unknown_team_side'
        margin = game['home_score'] - game['away_score']
        return margin if row['side'] == row['home_team'] else -margin, 'graded'
    pid = row.get('player_id') if sport == 'nhl' else row.get('model_player_id')
    player = players.get((game_id, str(pid))) if pid is not None else None
    if player is None and pid is None and row.get('player'):
        found = [p for (g, _), p in players.items() if g == game_id and name(p.get('player')) == name(row.get('player'))]
        if len(found) == 1:
            player = found[0]
    if player is None:
        return None, 'participation_unconfirmed'
    if sport == 'nhl':
        stat = {'player_shots_on_goal': 'shots', 'player_goals': 'goals', 'player_assists': 'assists', 'player_points': 'points'}.get(market)
        value = player.get(stat) if stat else None
    else:
        pitching, batting = player.get('pitching', {}), player.get('batting', {})
        if str(market).startswith('pitcher_'):
            if pitching.get('gamesStarted') != 1:
                return None, 'starting_pitcher_unconfirmed'
            stat = {'pitcher_outs': 'outs', 'pitcher_strikeouts': 'strikeOuts'}.get(market)
            value = pitching.get(stat) if stat else None
        else:
            if not number(batting.get('plateAppearances')) or batting['plateAppearances'] <= 0:
                return None, 'participation_unconfirmed'
            stat = {'batter_hits': 'hits', 'batter_total_bases': 'totalBases', 'batter_home_runs': 'homeRuns', 'batter_rbis': 'rbi'}.get(market)
            value = batting.get(stat) if stat else None
            if market == 'batter_total_bases' and value is None and all(number(batting.get(k)) for k in ['hits', 'doubles', 'triples', 'homeRuns']):
                value = batting['hits'] + batting['doubles'] + 2 * batting['triples'] + 3 * batting['homeRuns']
    return (value, 'graded') if number(value) else (None, 'stat_or_market_unavailable')


def grade(row, sport, games, players):
    profile = row.get('settlement_profile')
    if sport == 'nhl' and (row.get('settlement_verified') is False or
            (profile and profile != ('nhl_player_ot_no_so_participation'
             if str(row.get('market')).startswith('player_') else 'nhl_full_game_ot_so'))):
        return dict(row, actual=None, grading_status='settlement_unconfirmed', outcome=None, units=None)
    value, status = actual(row, sport, games, players)
    out = dict(row, actual=value, grading_status=status, outcome=None, units=None)
    if value is None:
        return out
    market, line, side = row.get('market'), row.get('line'), str(row.get('side', '')).lower()
    if market == 'h2h':
        delta = value
    elif market == 'spreads' and number(line):
        delta = value + line
    elif side in ('over', 'under') and number(line):
        delta = (value-line) * (1 if side == 'over' else -1)
    else:
        return dict(out, grading_status='unsupported_contract')
    outcome = 'win' if delta > 0 else 'loss' if delta < 0 else 'push'
    if not number(row.get('price')) or abs(row['price']) < 100:
        return dict(out, grading_status='invalid_price')
    return dict(out, outcome=outcome, units=decimal(row['price'])-1 if delta > 0 else -1.0 if delta < 0 else 0.0)


def representatives(quotes):
    """Most-offered paired line, median tie-break; one observation per contract.

    O/U uses Over, team markets use home. Model from the best executable side-A
    quote; market is average same-book de-vig at that same line. No selection by result.
    """
    groups = defaultdict(list)
    for q in quotes:
        if not q.get('book') or not number(q.get('price')) or abs(q['price']) < 100:
            continue
        profile = q.get('settlement_profile')
        expected = ('nhl_player_ot_no_so_participation' if str(q.get('market')).startswith('player_')
                    else 'nhl_full_game_ot_so')
        if q.get('settlement_verified') is False or profile and profile != expected:
            continue
        groups[(q.get('market'), q.get('player') or '')].append(q)
    output = []
    for (market, _), rows in sorted(groups.items()):
        is_team = market in ('h2h', 'spreads')
        side_a = rows[0].get('home_team') if is_team else 'Over'
        side_b = rows[0].get('away_team') if is_team else 'Under'
        paired = {}
        comparison_times = []
        for a in rows:
            if str(a.get('side')).lower() != str(side_a).lower():
                continue
            line = 0.0 if market == 'h2h' else a.get('line')
            if not number(line):
                continue
            opposite = -line if market == 'spreads' else line
            b = next((b for b in rows if b.get('book') == a.get('book') and
                      str(b.get('side')).lower() == str(side_b).lower() and
                      (market == 'h2h' or b.get('line') == opposite)), None)
            if not b:
                continue
            # NHL settlement mismatches cannot establish a comparable market.
            if a.get('settlement_verified') is False or b.get('settlement_verified') is False:
                continue
            if a.get('settlement_profile') != b.get('settlement_profile'):
                continue
            pa, pb = 1/decimal(a['price']), 1/decimal(b['price'])
            paired[(line, a.get('book'))] = (a, pa/(pa+pb))
            # Every paired quote influences line choice, not only the winning side-A quote.
            comparison_times.extend(t for q in (a,b) for k in ('evidence_at','quoted_at','ingested_at','forecast_at','decision_at') if (t := stamp(q.get(k))))
        if not paired:
            continue
        counts = Counter(k[0] for k in paired)
        median = statistics.median(k[0] for k in paired)
        line = min(counts, key=lambda x: (-counts[x], abs(x-median), x))
        at = [v for k, v in paired.items() if k[0] == line]
        row = dict(max(at, key=lambda v: (decimal(v[0]['price']), str(v[0].get('book'))))[0])
        row.update(market_probability=statistics.mean(v[1] for v in at), paired_books=len(at), reference_line=line)
        if comparison_times:
            row['comparison_evidence_at'] = max(comparison_times).isoformat()
        output.append(row)
    return output


def conditional(row):
    p = row.get('model_probability')
    push = row.get('model_push_probability', row.get('push_probability'))
    if not number(p) or not 0 <= p <= 1:
        return None
    # With no recorded push estimate, only half-point O/U and moneyline are safe.
    if push is None:
        line = row.get('line')
        if row.get('market') != 'h2h' and (not number(line) or line % 1 != .5):
            return None
        push = 0
    return p/(1-push) if number(push) and 0 <= push < 1 and p <= 1-push+1e-9 else None


def summary(rows):
    resolved = [r for r in rows if r.get('outcome')]
    counts = Counter(r['outcome'] for r in resolved)
    return dict(selected=len(rows), graded=len(resolved), unresolved=len(rows)-len(resolved),
                wins=counts['win'], losses=counts['loss'], pushes=counts['push'],
                units=sum(r['units'] for r in resolved), roi=statistics.mean(r['units'] for r in resolved) if resolved else None)


def metrics(rows, sport):
    paired = []
    for row in rows:
        p = conditional(row)
        market = row.get('market_probability')
        if row.get('outcome') not in ('win', 'loss') or p is None or not number(market) or not 0 <= market <= 1:
            continue
        y = int(row['outcome'] == 'win')
        paired.append((gid(row, sport), (p-y)**2, (market-y)**2))
    quality = None
    if paired:
        blocks = defaultdict(list)
        for game, a, b in paired:
            blocks[game].append(a-b)
        ci = None
        if len(blocks) >= 2:
            rng = np.random.default_rng(20261007)
            groups = list(blocks.values())
            samples = rng.integers(0, len(groups), (2000, len(groups)))
            totals = np.array([sum(g) for g in groups])
            sizes = np.array([len(g) for g in groups])
            draws = totals[samples].sum(axis=1) / sizes[samples].sum(axis=1)
            ci = [float(x) for x in np.quantile(draws, [.025, .975])]
        quality = dict(n=len(paired), games=len(blocks), model_brier=statistics.mean(p[1] for p in paired),
                       market_brier=statistics.mean(p[2] for p in paired), difference_ci95=ci)
    errors = []
    for r in rows:
        mean = r.get('model_mean', r.get('projected_mean'))
        if r.get('market') in ('h2h', 'spreads') or r.get('actual') is None or not number(mean) or not number(r.get('line')):
            continue
        errors.append((abs(mean-r['actual']), abs(r['line']-r['actual'])))
    return dict(probability=quality, mean_error=dict(n=len(errors), model_mae=statistics.mean(x[0] for x in errors),
                  line_mae=statistics.mean(x[1] for x in errors)) if errors else None)


def review(sport, end, root=ROOT, fetch=fetch_json, offline=False):
    root = Path(root)
    start = end-timedelta(days=6)
    if end >= datetime.now(ET).date():
        raise ValueError('Review end must be a completed Eastern calendar day')
    runs = load_runs(sport, root)
    selected = pregame(runs, sport)
    chosen = {g: s for g, s in selected.items() if start <= stamp(s['rows'][0]['pregame_cutoff']).astimezone(ET).date() <= end}
    cards = published_picks(sport, start, end, root)
    game_ids = set(chosen) | {gid(r, sport) for r in cards['rows'] if gid(r, sport)}
    if not game_ids:
        return dict(sport=sport, status='waiting_for_pregame_evidence', start=str(start), end=str(end))
    starts = [stamp(s['rows'][0]['pregame_cutoff']) for s in chosen.values()] + [stamp(r['commence_time']) for r in cards['rows']]
    years = {t.year if t.month >= 7 else t.year-1 for t in starts if t}
    board_quotes = [r for s in chosen.values() for r in representatives(s['rows'])]
    ticket_quotes = cards['rows']
    if sport == 'nba':
        from nba_weekly_results import fetch_results
        games, players, resolved = fetch_results(start, end, board_quotes+ticket_quotes, root, fetch, offline)
        board_quotes, ticket_quotes = resolved[:len(board_quotes)], resolved[len(board_quotes):]
    else:
        games, players = (mlb_results(start, end, game_ids, root, fetch, offline) if sport == 'mlb' else nhl_results(runs, years, root))
    board = [grade(r, sport, games, players) for r in board_quotes]
    tickets = [grade(r, sport, games, players) for r in ticket_quotes]
    if not any(r.get('outcome') for r in board+tickets):
        return dict(sport=sport, status='waiting_for_official_results', start=str(start), end=str(end), games=len(game_ids))
    by_market = {}
    for market in sorted({r['market'] for r in board}):
        rows = [r for r in board if r['market'] == market]
        by_market[market] = dict(**summary(rows), **metrics(rows, sport))
    policies = {}
    for row in tickets:
        key = row['policy_version']
        policies.setdefault(key, []).append(row)
    policy_summary = {p: dict(total=summary(rs), markets={m: summary([r for r in rs if r['market'] == m]) for m in sorted({r['market'] for r in rs})}) for p, rs in policies.items()}
    now = datetime.now(timezone.utc)
    preview = []
    for s in selected.values():
        for r in representatives(s['rows']):
            kickoff = stamp(r['pregame_cutoff'])
            mean = r.get('model_mean', r.get('projected_mean'))
            if not now < kickoff <= now+timedelta(days=7) or now-stamp(s['captured_at']) > timedelta(hours=36):
                continue
            if r['market'] == 'totals' and number(mean) and number(r.get('line')):
                preview.append(dict(game=r['game'], start=r['commence_time'], mean=mean, line=r['line'], gap=mean-r['line'], captured=s['captured_at']))
    preview.sort(key=lambda r: (-abs(r['gap']), r['game']))
    all_dates = [stamp(s['rows'][0]['pregame_cutoff']).astimezone(ET).date() for s in selected.values()]
    result = dict(schema=1, sport=sport, status='published', start=str(start), end=str(end), generated_at=now.isoformat(),
                  first_archived_game=str(min(all_dates)) if all_dates else None,
                  partial_coverage=not all_dates or min(all_dates)>start,
                  games_with_quotes=len(chosen), games_with_official_results=len({gid(r,sport) for r in board if gid(r,sport) in games}),
                  published_pick_games=len({gid(r, sport) for r in tickets}),
                  published_pick_games_with_results=len({gid(r, sport) for r in tickets} & set(games)),
                  board=by_market, policies=policy_summary, tickets=summary(tickets), editions=cards['editions'],
                  probability=metrics(board, sport)['probability'], preview=preview[:5],
                  unresolved_reasons=dict(Counter(r['grading_status'] for r in board+tickets if not r.get('outcome'))),
                  sources=[dict(game=g, captured_at=s['captured_at'], path=s['source']) for g,s in sorted(chosen.items())])
    keep = {'comparison_evidence_at','result_source_url','official_schedule_path','official_result_path','espn_player_id','nba_game_id','official_commence_time','evidence_at','resolution_status','result_source','espn_event_id','event_id', 'mlb_game_id', 'nhl_game_id', 'player_id', 'model_player_id', 'game', 'home_team', 'away_team', 'commence_time', 'pregame_cutoff', 'market', 'player', 'side', 'line', 'price', 'book', 'quoted_at', 'ingested_at', 'forecast_at', 'decision_at', 'model_probability', 'model_mean', 'projected_mean', 'push_probability', 'model_push_probability', 'market_probability', 'paired_books', 'reference_line', 'model_version', 'model_policy', 'model_status', 'settlement_profile', 'settlement_verified', 'actual', 'grading_status', 'outcome', 'units', 'edition_id', 'published_at', 'policy_version'}
    public = dict(summary=result, board=[{k:v for k,v in r.items() if k in keep} for r in board], tickets=[{k:v for k,v in r.items() if k in keep} for r in tickets])
    publish(public, root)
    return result


def article_head(title, desc, slug, published, modified=None):
    from site_metadata import seo_title, social_tags
    url = 'https://fourthandvalue.com/blog/'+slug
    schema = dict(**{'@context':'https://schema.org','@type':'Article'}, headline=title, description=desc,
                  datePublished=published, dateModified=modified or published, image='https://fourthandvalue.com/assets/social-card.png',
                  author={'@type':'Organization','name':'Fourth & Value'}, publisher={'@type':'Organization','name':'Fourth & Value'})
    return (f'<title>{escape(seo_title(title))}</title><meta name="description" content="{escape(desc, quote=True)}">'
            f'<link rel="canonical" href="{url}"><meta property="og:type" content="article">'
            f'<meta property="og:title" content="{escape(title, quote=True)}"><meta property="og:description" content="{escape(desc, quote=True)}">'
            f'<meta property="og:url" content="{url}">{social_tags()}'
            f'<meta property="article:published_time" content="{published}"><meta property="article:section" content="Weekly review">'
            f'<script type="application/ld+json">{json.dumps(schema).replace("<", chr(92)+"u003c")}</script>')


def discovery(path, title, desc, root=ROOT):
    blog = Path(root)/'docs/blog/index.html'
    text = blog.read_text()
    href = './'+path.name
    if href not in text:
        marker = '<!-- editorial-managed:end -->'
        if marker not in text:
            raise ValueError('Missing blog discovery marker')
        card = f'<li class="post" data-title="{escape(title, quote=True)}" data-excerpt="{escape(desc, quote=True)}"><h2><a href="{href}">{escape(title)}</a></h2><p class="excerpt">{escape(desc)}</p></li>'
        blog.write_text(text.replace(marker, marker+card, 1))
    sitemap = Path(root)/'docs/sitemap.xml'
    text = sitemap.read_text()
    url = 'https://fourthandvalue.com/blog/'+path.name
    if url not in text:
        sitemap.write_text(text.replace('</urlset>', f'<url><loc>{url}</loc></url>\n</urlset>'))


def publish(public, root):
    s = public['summary']; sport = s['sport'].upper()
    slug = f'{s["sport"]}-recap-{s["end"]}'
    title = f'{sport} weekly recap: {s["start"]} to {s["end"]}'
    desc = f'{sport} saved picks and forecasts graded against official results, with market comparisons, unresolved outcomes and the next watchlist.'
    blog = root/'docs/blog'; blog.mkdir(parents=True, exist_ok=True)
    data = blog/f'{slug}.json'
    if data.exists():
        old = json.loads(data.read_text())
        s['published_at'] = old['summary'].get('published_at', old['summary']['generated_at'])
    else:
        s['published_at'] = s['generated_at']
    data.write_text(json.dumps(public, indent=2, allow_nan=False)+'\n')
    pct = lambda x: '—' if x is None else f'{x*100:.1f}%'
    ticket_rows = ''.join(f'<tr><td>{escape(p)}</td><td>{escape(LABELS.get(m,m))}</td><td>{v["wins"]}–{v["losses"]}–{v["pushes"]}</td><td>{v["units"]:+.2f}</td><td>{pct(v["roi"])}</td><td>{v["unresolved"]}</td></tr>' for p,r in s['policies'].items() for m,v in r['markets'].items())
    board_rows = ''.join(f'<tr><td>{escape(LABELS.get(m,m))}</td><td>{v["wins"]} / {v["losses"]} / {v["pushes"]}</td><td>{v["unresolved"]}</td><td>{v["probability"]["model_brier"]:.4f} / {v["probability"]["market_brier"]:.4f} ({v["probability"]["n"]})</td><td>{v["mean_error"]["model_mae"]:.2f} / {v["mean_error"]["line_mae"]:.2f} ({v["mean_error"]["n"]})</td></tr>' if v['probability'] and v['mean_error'] else board_row(m,v) for m,v in s['board'].items())
    quality = s['probability']
    qtext = 'No matched probability comparison is available.'
    if quality:
        qtext = f'Across {quality["n"]} matched non-push observations in {quality["games"]} games, model Brier was {quality["model_brier"]:.4f}, versus market {quality["market_brier"]:.4f}. Lower is better.'
        if quality['difference_ci95']:
            qtext += ' The game-block 95% interval for model minus market Brier is '+ ' to '.join(f'{x:+.4f}' for x in quality['difference_ci95'])+'.'
        else:
            qtext += ' Too few games for a game-block uncertainty interval.'
    preview = ''.join(f'<li>{escape(r["game"])}: saved model {r["mean"]:.2f}, line {r["line"]:g}, gap {r["gap"]:+.2f}. Snapshot {escape(r["captured"])}.</li>' for r in s['preview'])
    evidence = ('Partial full-board archive coverage: saved game snapshots begin '+str(s['first_archived_game'])+'. '
                if s['partial_coverage'] and s['first_archived_game'] else '')
    evidence += (f'The full-board sample has {s["games_with_quotes"]} games with saved pregame quotes; '
                 f'{s["games_with_official_results"]} have official results available. '
                 f'Separately, {s["tickets"]["selected"]} published morning picks cover {s["published_pick_games"]} games; '
                 f'{s["published_pick_games_with_results"]} of those games have official results. '
                 'These are archived samples, not a claim to cover every league game.')
    reasons = ', '.join(f'{k.replace("_"," ")}: {v}' for k,v in s['unresolved_reasons'].items()) or 'None in the supported archived sample.'
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{article_head(title,desc,slug+'.html',s['published_at'],s['generated_at'])}<link rel="stylesheet" href="/assets/site.css"><style>main{{max-width:980px;margin:auto;padding:32px 20px 80px}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px 10px;text-align:left;border-bottom:1px solid #555}}img{{max-width:100%}}p,li{{line-height:1.65}}</style></head><body><div id="nav-root"></div><script src="/nav.js?v=47"></script><main><p>Fourth &amp; Value · Weekly review</p><h1>{escape(title)}</h1><p>{escape(desc)}</p><h2>What this week can tell us</h2><p>{escape(evidence)}</p><p>{escape(takeaway(s))}</p><h2 id="scorecard">Published morning picks</h2><p>Only the final morning edition for each Eastern date is counted. One unit risked at each published price; pushes return the stake. Unresolved picks are excluded from ROI. Policy versions stay separate.</p><div class="table-wrap"><table><thead><tr><th>Policy</th><th>Market</th><th>W–L–push</th><th>Net units</th><th>ROI</th><th>Unresolved</th></tr></thead><tbody>{ticket_rows}</tbody></table></div>{'<p>No eligible published morning picks for this sport were present in the archived editions. Full-board forecasts below are not published picks.</p>' if not ticket_rows else ''}<h2>Full-board market results</h2><p>One most-offered paired line per game, player and market from the final saved pregame snapshot. O/U columns count over / under / push; team markets count home win or cover / away / push. These are market outcomes, not a strategy return.</p><div class="table-wrap"><table><thead><tr><th>Market</th><th>Over or home / other / push</th><th>Unresolved</th><th>Brier model / market (n)</th><th>Mean error model / line (n)</th></tr></thead><tbody>{board_rows}</tbody></table></div><h2>Forecast quality</h2><p>{escape(qtext)}</p><p>Probabilities are compared on the same non-push outcomes after conditioning the model on non-push. Missing forecasts stay missing. Error comparisons use matched observations only. Small samples and correlated markets limit what a strong week can prove.</p><h2>Next week's watchlist</h2><ul>{preview or '<li>No fresh archived future total disagreements are available. Recheck the current board after its next refresh.</li>'}</ul><p>These are dated model-market disagreements to research, not validated edges. Check lineups, availability and current prices.</p><h2>Evidence and limitations</h2><p>{escape(reasons)}</p><p>Official nonparticipants, unsupported contracts and missing statistics are unresolved, never automatic zeroes or winning unders. Standard full-game settlement is assumed; sportsbook exceptions require separate review. MLB pitcher props require an official starting appearance. Team totals are not part of the supported archive. Results may be revised when official feeds are corrected.</p><p><a href="{slug}.json">Download every graded row and source snapshot</a> · <a href="/{s['sport']}/">Current {sport} board</a> · <a href="/research/daily-process.html">How updates work</a> · <a href="/blog/">All recaps</a></p></main></body></html>'''
    path = blog/f'{slug}.html';path.write_text(content+'\n')
    discovery(path,title,desc,root)
    from nfl_weekly_review import chart
    chart(blog/f'{slug}.svg',f'{sport}: published picks by market','One unit risked per published offer; policies kept separate',
          [(f'{LABELS.get(m,m)} [{p.replace("morning-edition-", "v")}]',v['units'],f'{v["units"]:+.2f}u') for p,r in s['policies'].items() for m,v in r['markets'].items()])
    if ticket_rows:
        path.write_text(path.read_text().replace('<h2>Full-board market results</h2>',f'<figure><img src="{slug}.svg" alt="Published picks net units by market and policy"></figure><h2>Full-board market results</h2>'))


def board_row(m,v):
    p,e=v['probability'],v['mean_error']
    ps=f'{p["model_brier"]:.4f} / {p["market_brier"]:.4f} ({p["n"]})' if p else '—'
    es=f'{e["model_mae"]:.2f} / {e["line_mae"]:.2f} ({e["n"]})' if e else '—'
    return f'<tr><td>{escape(LABELS.get(m,m))}</td><td>{v["wins"]} / {v["losses"]} / {v["pushes"]}</td><td>{v["unresolved"]}</td><td>{ps}</td><td>{es}</td></tr>'


def takeaway(s):
    markets=[(m,v['probability']) for m,v in s['board'].items() if v['probability']]
    if not markets:
        return 'The saved evidence does not support a model-versus-market probability comparison this week.'
    best=min(markets,key=lambda x:x[1]['model_brier']-x[1]['market_brier'])
    worst=max(markets,key=lambda x:x[1]['model_brier']-x[1]['market_brier'])
    return f'The strongest relative probability score was {LABELS.get(best[0],best[0]).lower()} ({best[1]["n"]} observations); the weakest was {LABELS.get(worst[0],worst[0]).lower()} ({worst[1]["n"]} observations). These rankings describe this sample; even the strongest market may trail sportsbook probabilities.'


def rebuild_home():
    from editorial import render_home
    data=json.loads((ROOT/'docs/briefing/latest.json').read_text())
    render_home(data,datetime.now(timezone.utc))
    from site_notices import apply
    for rel in ['index.html','editorial/index.html']:
        path=ROOT/'docs'/rel
        path.write_text(apply(path.read_text(),rel))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sport',choices=['mlb','nhl','nba','all'],default='all')
    parser.add_argument('--end',type=date.fromisoformat,help='Optional completed period end; otherwise follows each sport delivery weekday')
    parser.add_argument('--no-home',action='store_true')
    parser.add_argument('--offline',action='store_true',help='Use saved official MLB responses; NHL always uses archived results')
    args=parser.parse_args()
    from recap_schedule import expected_end
    now=datetime.now(timezone.utc)
    statuses=[];failures=[]
    for sport in (['mlb','nhl','nba'] if args.sport=='all' else [args.sport]):
        end=args.end or expected_end(sport,now)
        try:
            end=args.end or expected_end(sport,now)
            s=review(sport,end,offline=args.offline)
            statuses.append({k:s[k] for k in ['sport','status','start','end']})
        except Exception as exc:
            failures.append(f'{sport}: {exc}')
            statuses.append(dict(sport=sport,status='failed',end=str(end),error=str(exc)))
    status_path=ROOT/'docs/recaps/status.json';status_path.parent.mkdir(parents=True,exist_ok=True)
    status_path.write_text(json.dumps(dict(checked_at=datetime.now(timezone.utc).isoformat(),sports=statuses),indent=2)+'\n')
    if not args.no_home:rebuild_home()
    print(json.dumps(statuses,indent=2))
    if failures:raise SystemExit('; '.join(failures))


if __name__=='__main__':main()
