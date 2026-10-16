"""Build the game tables for the low-scoring study (Technical Report FV-2026-04).

Runs in GitHub Actions (open network): python scripts/research/low_scoring/fetch.py <nfl|nhl|nba|mlb>
Writes reports/low-scoring-2026/tables/<sport>.jsonl.gz and <sport>-coverage.json. Free public sources
only; no paid odds request is made.

Sources
- NFL: nflverse schedules (closing totals, rest, roof, temperature, wind, referee, starting QBs) and
  nflverse play-by-play (the game-time weather description).
- NHL: NHL schedule, game-center officials and per-game goalie starts (api-web.nhle.com,
  api.nhle.com/stats); closing totals from the SportsbookReviewsOnline archive as preserved by the
  Internet Archive (2007-08 to 2022-23) and ESPN's sportsbook odds feed afterwards.
- NBA: ESPN schedule and officials; closing totals as for the NHL.
- MLB: MLB Stats API schedule (weather, wind, home-plate umpire, day/night, doubleheaders);
  ESPN's sportsbook odds feed for totals.
"""
import gzip
import io
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/low-scoring-2026/tables'
CACHE_DIR = ROOT / '.low-scoring-cache'
ET = ZoneInfo('America/New_York')
WAYBACK = 'http://web.archive.org/web/2023id_/'
SBRO = 'https://sportsbookreviewsonline.com/scoresoddsarchives/'
CORE = 'https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{lg}/events/{eid}/competitions/{eid}/{what}'
BOARD = 'https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/scoreboard'
EXCLUDE_BOOK = re.compile(r'live|opening|\(old\)|numberfire|teamrankings', re.I)

_local = threading.local()


def session():
    if not hasattr(_local, 's'):
        _local.s = requests.Session()
        _local.s.headers['User-Agent'] = 'Mozilla/5.0 (research; fourthandvalue.com)'
    return _local.s


def fetch(url, params=None, kind='json', tries=5):
    for attempt in range(tries):
        try:
            r = session().get(url, params=params, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r.json() if kind == 'json' else r.content
        except Exception as e:  # noqa: BLE001 - retried, then reported
            if attempt == tries - 1:
                print('FAILED', url, params, e, flush=True)
                return None
            time.sleep(2 ** attempt)


class Cache:
    """Extracted (small) results keyed by request, persisted between Actions runs."""

    def __init__(self, name):
        self.path = CACHE_DIR / f'{name}.json.gz'
        self.data = json.loads(gzip.decompress(self.path.read_bytes())) if self.path.exists() else {}
        self.lock = threading.Lock()

    def get(self, key, compute):
        if key in self.data:
            return self.data[key]
        value = compute()
        if value is not None:
            with self.lock:
                self.data[key] = value
        return value

    def save(self):
        CACHE_DIR.mkdir(exist_ok=True)
        self.path.write_bytes(gzip.compress(json.dumps(self.data).encode()))


def pmap(fn, items, workers=8, label=''):
    items = list(items)
    out, t0 = [None] * len(items), time.time()

    def safe(x):
        try:
            return fn(x)
        except Exception as e:  # noqa: BLE001 - one bad record must not end a multi-hour fetch
            print('ERROR', label, repr(x)[:120], repr(e)[:200], flush=True)
            return None
    with ThreadPoolExecutor(workers) as ex:
        for i, v in enumerate(ex.map(safe, items)):
            out[i] = v
            if (i + 1) % 2000 == 0:
                print(f'  {label} {i + 1}/{len(items)} {time.time() - t0:.0f}s', flush=True)
    return out


def write(sport, rows, coverage):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r['date'], str(r['gid'])))
    with gzip.open(OUT / f'{sport}.jsonl.gz', 'wt', compresslevel=9) as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(',', ':'), sort_keys=True) + '\n')
    (OUT / f'{sport}-coverage.json').write_text(json.dumps(coverage, indent=1, sort_keys=True) + '\n')
    print(sport, 'rows', len(rows), json.dumps(coverage)[:3000], flush=True)


def season_of(d):
    """Season label = the year the season starts (NBA/NHL)."""
    return d.year if d.month >= 8 else d.year - 1


def et_date(iso):
    return datetime.fromisoformat(iso.replace('Z', '+00:00')).astimezone(ET).date().isoformat()


# ---------------------------------------------------------------- ESPN helpers

def espn_board(cache, sport, lg, day):
    def compute():
        j = fetch(BOARD.format(sport=sport, lg=lg), dict(dates=day.strftime('%Y%m%d'), limit=200))
        if j is None:
            return None
        evs = []
        for ev in j.get('events', []):
            try:
                c = ev['competitions'][0]
                side = {x['homeAway']: x for x in c['competitors']}
            except (KeyError, IndexError, TypeError):
                continue
            if 'home' not in side or 'away' not in side:
                continue
            evs.append(dict(
                eid=ev['id'], start=ev['date'], stype=ev.get('season', {}).get('type'),
                completed=c['status']['type'].get('completed'), state=c['status']['type'].get('name'),
                neutral=c.get('neutralSite'), venue=c.get('venue', {}).get('fullName'),
                city=c.get('venue', {}).get('address', {}).get('city'),
                home=side['home']['team'].get('abbreviation'), away=side['away']['team'].get('abbreviation'),
                home_name=side['home']['team'].get('displayName'), away_name=side['away']['team'].get('displayName'),
                home_short=side['home']['team'].get('name') or side['home']['team'].get('shortDisplayName'),
                hs=side['home'].get('score'), as_=side['away'].get('score'),
                notes=[n.get('headline') for n in c.get('notes', []) if n.get('headline')]))
        return evs
    return cache.get(f'{lg}:{day.isoformat()}', compute) or []


def espn_odds(cache, sport, lg, eid):
    def compute():
        j = fetch(CORE.format(sport=sport, lg=lg, eid=eid, what='odds'), dict(limit=100))
        if j is None:
            return []
        books = []
        for it in j.get('items', []):
            name = (it.get('provider') or {}).get('name') or ''
            close = (((it.get('close') or {}).get('total') or {}).get('value'))
            books.append([name, it.get('overUnder'), close, it.get('overOdds'), it.get('underOdds')])
        return books
    return cache.get(f'{lg}:odds:{eid}', compute)


def consensus_total(books):
    vals = []
    for name, ou, close, *_ in books or []:
        v = close if isinstance(close, (int, float)) and close > 0 else ou
        if EXCLUDE_BOOK.search(name) or not isinstance(v, (int, float)) or v <= 0:
            continue
        vals.append(float(v))
    return (median(vals), len(vals)) if vals else (None, 0)


def espn_officials(cache, sport, lg, eid):
    def compute():
        j = fetch(CORE.format(sport=sport, lg=lg, eid=eid, what='officials'), dict(limit=50))
        if j is None:
            return []
        return [[(o.get('displayName') or o.get('fullName') or '').strip(), (o.get('position') or {}).get('name'), o.get('order')]
                for o in j.get('items', [])]
    return cache.get(f'{lg}:officials:{eid}', compute)


# ---------------------------------------------------------------- SportsbookReviewsOnline archive

def sbro_rows(sport, season):
    import openpyxl
    label = f'{season}-{str(season + 1)[2:]}'
    names = [f'{sport}%20odds%20{label}.xlsx']
    if sport == 'nhl' and season == 2020:
        names.insert(0, 'nhl%20odds%202021.xlsx')
    for name in names:
        content = fetch(WAYBACK + SBRO + f'{sport}/{name}', kind='bytes')
        if content and content[:2] == b'PK':
            ws = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True).worksheets[0]
            rows = list(ws.iter_rows(values_only=True))
            print(f'  SBRO {sport} {season}: {len(rows)} rows, header {rows[0]}', flush=True)
            return rows
    print(f'  SBRO {sport} {season}: not found', flush=True)
    return []


def num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().replace('½', '.5')
    if s in ('pk', 'pick', 'nl', '', '-'):
        return 0.0 if s in ('pk', 'pick') else None
    try:
        return float(s)
    except ValueError:
        return None


def sbro_games(sport, season):
    """Pairs of rows (visitor then home) -> date, teams, final, closing total."""
    rows = sbro_rows(sport, season)
    if not rows:
        return [], Counter()
    head = [re.sub(r'[^a-z0-9]', '', str(h).lower()) if h is not None else '' for h in rows[0]]
    col = {h: i for i, h in enumerate(head) if h}
    issues, games = Counter(), []
    body = [r for r in rows[1:] if r and r[0] is not None and r[col['vh']] is not None]
    pairs, i = [], 0
    # Rows normally alternate visitor/home, but some files list home first or drop a row: pair two
    # consecutive rows only when they share a date and are not both visitors or both home teams.
    while i < len(body) - 1:
        a, b = body[i], body[i + 1]
        fa, fb = str(a[col['vh']]).strip().upper(), str(b[col['vh']]).strip().upper()
        if a[col['date']] != b[col['date']] or (fa == fb and fa != 'N'):
            issues['resync'] += 1
            i += 1
            continue
        pairs.append((b, a) if (fa, fb) == ('H', 'V') else (a, b))
        i += 2
    for v, h in pairs:
        try:
            md = int(num(v[col['date']]))
        except (TypeError, ValueError):
            issues['bad date'] += 1
            continue
        month, day = divmod(md, 100)
        year = season if month >= 8 else season + 1
        if sport == 'nhl' and season == 2019 and month in (8, 9):      # 2020 bubble playoffs
            year = 2020
        if sport == 'nba' and season == 2019 and month in (7, 8, 9, 10):
            year = 2020
        if sport in ('nhl', 'nba') and season == 2020 and month in (12,):
            year = 2020
        try:
            d = date(year, month, day)
        except ValueError:
            issues['bad date'] += 1
            continue
        fv, fh = num(v[col['final']]), num(h[col['final']])
        if sport == 'nhl':
            total = num(h[col['closeou']]) if 'closeou' in col else None
            if total is None and 'closeou' in col:
                total = num(v[col['closeou']])
            opening = num(h[col['openou']]) if 'openou' in col else None
        else:
            # NBA: the larger of the two closing numbers is the total, the smaller the spread.
            cv, ch = num(v[col['close']]), num(h[col['close']])
            cands = [x for x in (cv, ch) if x is not None and x > 50]
            total = max(cands) if cands else None
            ov, oh = num(v[col['open']]), num(h[col['open']])
            ocands = [x for x in (ov, oh) if x is not None and x > 50]
            opening = max(ocands) if ocands else None
        if total is None:
            issues['no total'] += 1
        games.append(dict(date=d.isoformat(), away_name=str(v[col['team']]).strip(), home_name=str(h[col['team']]).strip(),
                          away_final=fv, home_final=fh, line=total, line_open=opening,
                          neutral=str(v[col['vh']]).strip().upper() == 'N'))
    return games, issues


# ---------------------------------------------------------------- NFL

def build_nfl():
    import pandas as pd
    games = pd.read_csv('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv')
    games = games[games.season >= 2006]
    weather = {}
    for year in sorted(games.season.unique()):
        url = f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet'
        content = fetch(url, kind='bytes')
        if not content:
            continue
        pbp = pd.read_parquet(io.BytesIO(content), columns=['game_id', 'weather'])
        for gid, w in pbp.dropna().drop_duplicates('game_id').itertuples(index=False):
            weather[gid] = w
        print('  nfl pbp', year, len(pbp), flush=True)
    keep = ['game_id', 'season', 'game_type', 'week', 'gameday', 'weekday', 'gametime', 'away_team', 'away_score',
            'home_team', 'home_score', 'location', 'total', 'overtime', 'away_rest', 'home_rest', 'total_line',
            'under_odds', 'over_odds', 'div_game', 'roof', 'surface', 'temp', 'wind', 'away_qb_id', 'home_qb_id',
            'away_qb_name', 'home_qb_name', 'referee', 'stadium']
    rows = []
    for r in games[keep].to_dict('records'):
        r = {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
        gid = r.pop('game_id')
        rows.append(dict(r, gid=gid, date=r.pop('gameday'), away=r.pop('away_team'), home=r.pop('home_team'),
                         actual=r.pop('total'), line=r.pop('total_line'), weather_text=weather.get(gid)))
    cov = Counter((r['season'], r['line'] is not None and r['actual'] is not None, r['weather_text'] is not None) for r in rows)
    write('nfl', rows, {f'{s}|line={a}|weather={b}': n for (s, a, b), n in sorted(cov.items())})


# ---------------------------------------------------------------- NHL

NHL_SBRO = {  # SportsbookReviewsOnline team names -> NHL abbreviations
    'anaheim': 'ANA', 'arizona': 'ARI', 'phoenix': 'PHX', 'boston': 'BOS', 'buffalo': 'BUF', 'calgary': 'CGY',
    'carolina': 'CAR', 'chicago': 'CHI', 'colorado': 'COL', 'columbus': 'CBJ', 'dallas': 'DAL', 'detroit': 'DET',
    'edmonton': 'EDM', 'florida': 'FLA', 'losangeles': 'LAK', 'minnesota': 'MIN', 'montreal': 'MTL', 'nashville': 'NSH',
    'newjersey': 'NJD', 'nyislanders': 'NYI', 'nyrangers': 'NYR', 'ottawa': 'OTT', 'philadelphia': 'PHI',
    'pittsburgh': 'PIT', 'sanjose': 'SJS', 'seattle': 'SEA', 'stlouis': 'STL', 'tampabay': 'TBL', 'toronto': 'TOR',
    'vancouver': 'VAN', 'vegas': 'VGK', 'washington': 'WSH', 'winnipeg': 'WPG', 'atlanta': 'ATL', 'utah': 'UTA',
    'arizonacoyotes': 'ARI', 'tampa': 'TBL', 'sanjosesharks': 'SJS', 'islanders': 'NYI', 'rangers': 'NYR',
    'stlouisblues': 'STL', 'losangeleskings': 'LAK',
}
NHL_SAME = {'PHX': ('ARI', 'UTA'), 'ARI': ('PHX', 'UTA'), 'UTA': ('ARI',), 'ATL': ('WPG',), 'WPG': ('ATL',)}
NHL_ESPN = {'LA': 'LAK', 'NJ': 'NJD', 'SJ': 'SJS', 'TB': 'TBL', 'UTAH': 'UTA', 'WSH': 'WSH', 'MON': 'MTL', 'CLB': 'CBJ',
            'NAS': 'NSH', 'PHO': 'PHX', 'VEG': 'VGK'}


def key_name(s):
    return re.sub(r'[^a-z]', '', s.lower())


def team_code(name, table):
    """Archive team name -> code; tolerates suffixes such as 'SeattleKraken' or 'Arizonas'."""
    k = key_name(name)
    if k in table:
        return table[k]
    for key in sorted(table, key=len, reverse=True):
        if k.startswith(key):
            return table[key]
    return None


def build_nhl():
    cache = Cache('nhl')
    # Schedule, one week per call, deduplicated by game id.
    start, stop = date(2007, 9, 24), datetime.now(ET).date()
    weeks = [start + timedelta(days=7 * i) for i in range((stop - start).days // 7 + 1)]

    def week(d):
        def compute():
            j = fetch(f'https://api-web.nhle.com/v1/schedule/{d.isoformat()}')
            if j is None:
                return None
            out = []
            for gw in j.get('gameWeek', []):
                for g in gw.get('games', []):
                    if g.get('gameType') not in (2, 3):
                        continue
                    out.append(dict(gid=g['id'], season=int(str(g['season'])[:4]), game_type=g['gameType'],
                                    date=gw['date'], start=g.get('startTimeUTC'), venue=(g.get('venue') or {}).get('default'),
                                    neutral=g.get('neutralSite'), venue_offset=g.get('venueUTCOffset'), state=g.get('gameState'),
                                    away=g['awayTeam']['abbrev'], home=g['homeTeam']['abbrev'],
                                    away_score=g['awayTeam'].get('score'), home_score=g['homeTeam'].get('score'),
                                    last_period=(g.get('gameOutcome') or {}).get('lastPeriodType')))
            return out
        return cache.get(f'week:{d.isoformat()}', compute) or []

    games = {}
    for chunk in pmap(week, weeks, 6, 'nhl weeks'):
        for g in chunk:
            games[g['gid']] = g
    games = {k: g for k, g in games.items() if g['state'] in ('OFF', 'FINAL') and g['home_score'] is not None}
    print('nhl games', len(games), flush=True)

    # Referees from the game centre.
    def refs(gid):
        def compute():
            j = fetch(f'https://api-web.nhle.com/v1/gamecenter/{gid}/right-rail')
            if j is None:
                return []
            return [(r.get('fullName') or {}).get('default') for r in (j.get('gameInfo') or {}).get('referees', [])]
        return cache.get(f'refs:{gid}', compute)
    for gid, rr in zip(games, pmap(refs, list(games), 12, 'nhl refs')):
        games[gid]['referees'] = rr or []
    cache.save()

    # Goalie starts per game.
    starters = defaultdict(dict)
    for season in sorted({g['season'] for g in games.values()}):
        sid = f'{season}{season + 1}'
        for gtype in (2, 3):
            def page(start_at, sid=sid, gtype=gtype):
                def compute():
                    j = fetch('https://api.nhle.com/stats/rest/en/goalie/summary', {
                        'isAggregate': 'false', 'isGame': 'true', 'start': start_at, 'limit': 100,
                        'sort': json.dumps([{'property': 'gameId', 'direction': 'ASC'}, {'property': 'playerId', 'direction': 'ASC'}]),
                        'cayenneExp': f'seasonId={sid} and gameTypeId={gtype}'})
                    if j is None:
                        return None
                    return dict(total=j.get('total'), rows=[[x['gameId'], x.get('teamAbbrev'), x.get('playerId'), x.get('goalieFullName'), x.get('gamesStarted')] for x in j.get('data', [])])
                return cache.get(f'goalies:{sid}:{gtype}:{start_at}', compute)
            first = page(0)
            if not first:
                continue
            pages = [first] + pmap(page, range(100, first['total'] or 0, 100), 6, 'goalies')
            for p in pages:
                for gid, team, pid, name, gs in (p or {}).get('rows', []):
                    if gs:
                        starters[gid][team] = [pid, name]
    cache.save()

    # Closing totals: archive through 2022-23, ESPN odds afterwards.
    by_key = defaultdict(list)
    for g in games.values():
        by_key[(g['date'], g['home'])].append(g)
    cov, matched = Counter(), set()
    for season in range(2007, 2023):
        sb, issues = sbro_games('nhl', season)
        cov[f'{season}|sbro_issues'] = dict(issues)
        for s in sb:
            home = team_code(s['home_name'], NHL_SBRO)
            if home is None:
                cov[f'unmapped:{s["home_name"]}'] = cov.get(f'unmapped:{s["home_name"]}', 0) + 1
                continue
            away = team_code(s['away_name'], NHL_SBRO)
            d = date.fromisoformat(s['date'])
            cands = []
            for dd in (d, d - timedelta(days=1), d + timedelta(days=1)):
                for h in (home,) + NHL_SAME.get(home, ()):
                    cands = [g for g in by_key.get((dd.isoformat(), h), []) if dd == d or g['away'] in (away,) + NHL_SAME.get(away, ())]
                    if cands:
                        break
                if cands:
                    break
            if not cands:
                cov[f'{season}|sbro_unmatched'] = cov.get(f'{season}|sbro_unmatched', 0) + 1
                continue
            g = cands[0]
            if s['line'] is not None:
                g.update(line=s['line'], line_open=s['line_open'], line_source='sbro', sbro_final=(s['away_final'], s['home_final']))
                matched.add(g['gid'])
    espn_cache = Cache('nhl-espn')
    # ESPN's odds feed starts in 2019-20; it supplies later seasons and archive games left unmatched.
    recent = sorted({g['date'] for g in games.values() if g['season'] >= 2023 or (g['season'] >= 2019 and g['gid'] not in matched)})
    boards = pmap(lambda d: espn_board(espn_cache, 'hockey', 'nhl', date.fromisoformat(d)), recent, 6, 'nhl boards')
    espn_cache.save()
    events = []
    for d, evs in zip(recent, boards):
        for ev in evs:
            home = NHL_ESPN.get(ev['home'], ev['home'])
            cands = by_key.get((et_date(ev['start']), home)) or by_key.get((d, home))
            if cands:
                events.append((cands[0], ev['eid']))
    odds = pmap(lambda x: espn_odds(espn_cache, 'hockey', 'nhl', x[1]), events, 8, 'nhl odds')
    espn_cache.save()
    for (g, eid), books in zip(events, odds):
        total, n = consensus_total(books)
        if total is not None and g['gid'] not in matched:
            g.update(line=total, line_source='espn', books=n, espn_id=eid)
    rows = []
    for g in games.values():
        g['away_goalie'] = (starters.get(g['gid'], {}).get(g['away']) or [None, None])
        g['home_goalie'] = (starters.get(g['gid'], {}).get(g['home']) or [None, None])
        g['actual'] = g['away_score'] + g['home_score']
        rows.append(g)
    c2 = Counter((g['season'], g.get('line_source'), bool(g['referees']), g['away_goalie'][0] is not None) for g in rows)
    cov.update({f'{s}|line={a}|refs={b}|goalies={c}': n for (s, a, b, c), n in sorted(c2.items(), key=str)})
    write('nhl', rows, {k: v for k, v in cov.items()})
    cache.save()


# ---------------------------------------------------------------- NBA

NBA_SBRO = {
    'atlanta': 'ATL', 'boston': 'BOS', 'brooklyn': 'BKN', 'newjersey': 'NJ', 'charlotte': 'CHA', 'chicago': 'CHI',
    'cleveland': 'CLE', 'dallas': 'DAL', 'denver': 'DEN', 'detroit': 'DET', 'goldenstate': 'GS', 'houston': 'HOU',
    'indiana': 'IND', 'laclippers': 'LAC', 'lalakers': 'LAL', 'losangeles': 'LAL', 'memphis': 'MEM', 'miami': 'MIA',
    'milwaukee': 'MIL', 'minnesota': 'MIN', 'neworleans': 'NO', 'newyork': 'NY', 'oklahomacity': 'OKC',
    'orlando': 'ORL', 'philadelphia': 'PHI', 'phoenix': 'PHX', 'portland': 'POR', 'sacramento': 'SAC',
    'sanantonio': 'SA', 'seattle': 'SEA', 'toronto': 'TOR', 'utah': 'UTAH', 'washington': 'WSH', 'okcity': 'OKC',
    'lalakers.': 'LAL', 'goldenstatewarriors': 'GS', 'losangeleslakers': 'LAL', 'losangelesclippers': 'LAC',
}
NBA_SAME = {'NJ': ('BKN',), 'BKN': ('NJ',), 'SEA': ('OKC',), 'OKC': ('SEA',)}
NBA_ALIAS = {'NOH': 'NO', 'NOK': 'NO', 'NOP': 'NO', 'GSW': 'GS', 'SAS': 'SA', 'NYK': 'NY', 'UTA': 'UTAH', 'BRK': 'BKN',
             'NJN': 'NJ', 'PHO': 'PHX', 'WAS': 'WSH', 'CHO': 'CHA', 'CHH': 'CHA'}


def build_nba():
    cache = Cache('nba')
    days = []
    for season in range(2007, 2026):
        d, end = date(season, 10, 1), date(season + 1, 7, 15)
        if season == 2019:
            end = date(2020, 10, 15)
        if season == 2020:
            d, end = date(2020, 12, 1), date(2021, 7, 25)
        while d <= end:
            days.append((season, d))
            d += timedelta(days=1)
    boards = pmap(lambda sd: espn_board(cache, 'basketball', 'nba', sd[1]), days, 8, 'nba boards')
    cache.save()
    games = {}
    for (season, d), evs in zip(days, boards):
        for ev in evs:
            if ev['stype'] not in (2, 3) or not ev['completed']:
                continue
            if ev['home'] in ('EAST', 'WEST', 'USA', 'WORLD', 'LEB', 'GIA', 'STARS') or ev['away'] in ('EAST', 'WEST'):
                continue
            ed = date.fromisoformat(et_date(ev['start']))
            games[ev['eid']] = dict(gid=ev['eid'], date=ed.isoformat(), season=season,
                                    game_type=ev['stype'], start=ev['start'], neutral=ev['neutral'], venue=ev['venue'],
                                    city=ev['city'], away=NBA_ALIAS.get(ev['away'], ev['away']), home=NBA_ALIAS.get(ev['home'], ev['home']),
                                    away_score=int(ev['as_']), home_score=int(ev['hs']), notes=ev['notes'])
    print('nba games', len(games), flush=True)
    offs = pmap(lambda gid: espn_officials(cache, 'basketball', 'nba', gid), list(games), 10, 'nba officials')
    for gid, o in zip(list(games), offs):
        o = sorted(o or [], key=lambda x: (x[2] if isinstance(x[2], int) else 99))
        games[gid]['officials'] = [x[0] for x in o]
    cache.save()
    by_key = defaultdict(list)
    for g in games.values():
        by_key[(g['date'], g['home'])].append(g)
    cov, matched = Counter(), set()
    for season in range(2007, 2023):
        sb, issues = sbro_games('nba', season)
        cov[f'{season}|sbro_issues'] = dict(issues)
        for s in sb:
            home = team_code(s['home_name'], NBA_SBRO)
            if home is None:
                cov[f'unmapped:{s["home_name"]}'] = cov.get(f'unmapped:{s["home_name"]}', 0) + 1
                continue
            away = team_code(s['away_name'], NBA_SBRO)
            d = date.fromisoformat(s['date'])
            cands = []
            for dd in (d, d - timedelta(days=1), d + timedelta(days=1)):
                for h in (home,) + NBA_SAME.get(home, ()):
                    cands = [g for g in by_key.get((dd.isoformat(), h), []) if dd == d or g['away'] in (away,) + NBA_SAME.get(away, ())]
                    if cands:
                        break
                if cands:
                    break
            if not cands:
                cov[f'{season}|sbro_unmatched'] = cov.get(f'{season}|sbro_unmatched', 0) + 1
                continue
            g = cands[0]
            if s['line'] is not None and g['gid'] not in matched:
                g.update(line=s['line'], line_open=s['line_open'], line_source='sbro', sbro_final=(s['away_final'], s['home_final']))
                matched.add(g['gid'])
    recent = [g for g in games.values() if g['season'] >= 2023 or (g['gid'] not in matched and g['season'] >= 2014)]
    odds = pmap(lambda g: espn_odds(cache, 'basketball', 'nba', g['gid']), recent, 8, 'nba odds')
    cache.save()
    for g, books in zip(recent, odds):
        total, n = consensus_total(books)
        if total is not None and g['gid'] not in matched:
            g.update(line=total, line_source='espn', books=n)
    rows = []
    for g in games.values():
        g['actual'] = g['away_score'] + g['home_score']
        rows.append(g)
    c2 = Counter((g['season'], g.get('line_source'), bool(g['officials'])) for g in rows)
    cov.update({f'{s}|line={a}|officials={b}': n for (s, a, b), n in sorted(c2.items(), key=str)})
    write('nba', rows, dict(cov))


# ---------------------------------------------------------------- MLB

def parse_wind(w):
    if not w:
        return None, None
    m = re.match(r'\s*(\d+)\s*mph,?\s*(.*)', w)
    if not m:
        return None, w
    return int(m.group(1)), m.group(2).strip()


def build_mlb():
    cache = Cache('mlb')
    games = {}
    for year in range(2010, 2027):
        for month in range(3, 12):
            a = date(year, month, 1)
            b = (date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1))

            def compute(a=a, b=b):
                for hydrate in ('weather,officials,linescore,venue(timezone),team', 'weather,officials,linescore,venue,team'):
                    j = fetch('https://statsapi.mlb.com/api/v1/schedule', dict(
                        sportId=1, startDate=a.isoformat(), endDate=b.isoformat(), gameType='R,F,D,L,W', hydrate=hydrate))
                    if j is not None:
                        break
                if j is None:
                    return None
                out = []
                for d in j.get('dates', []):
                    for g in d.get('games', []):
                        st = g.get('status', {})
                        if st.get('codedGameState') != 'F' and st.get('detailedState') not in ('Final', 'Game Over', 'Completed Early'):
                            continue
                        hp = [o['official']['fullName'] for o in g.get('officials', []) if o.get('officialType') == 'Home Plate']
                        tz = ((g.get('venue') or {}).get('timeZone') or {}).get('id')
                        ls = g.get('linescore') or {}
                        out.append(dict(
                            gid=g['gamePk'], date=g.get('officialDate') or d['date'], start=g.get('gameDate'),
                            game_type=g.get('gameType'), dh=g.get('doubleHeader'), game_number=g.get('gameNumber'),
                            day_night=g.get('dayNight'), venue=(g.get('venue') or {}).get('name'), venue_id=(g.get('venue') or {}).get('id'),
                            tz=tz, away=g['teams']['away']['team'].get('abbreviation'), home=g['teams']['home']['team'].get('abbreviation'),
                            away_name=g['teams']['away']['team'].get('teamName'), home_name=g['teams']['home']['team'].get('teamName'),
                            away_score=g['teams']['away'].get('score'), home_score=g['teams']['home'].get('score'),
                            innings=len(ls.get('innings', [])) or None, scheduled_innings=g.get('scheduledInnings'),
                            weather=g.get('weather') or {}, hp_umpire=hp[0] if hp else None,
                            series_game=g.get('seriesGameNumber'), games_in_series=g.get('gamesInSeries')))
                return out
            for g in cache.get(f'sched:{a.isoformat()}', compute) or []:
                games[g['gid']] = g
        print('  mlb schedule', year, len(games), flush=True)
    cache.save()
    games = {k: g for k, g in games.items() if g['home_score'] is not None and g['away_score'] is not None}
    days = sorted({g['date'] for g in games.values()})
    boards = pmap(lambda d: espn_board(cache, 'baseball', 'mlb', date.fromisoformat(d)), days, 8, 'mlb boards')
    cache.save()
    by_key = defaultdict(list)
    for g in games.values():
        by_key[(g['date'], key_name(g['home_name'] or ''))].append(g)
    for v in by_key.values():
        v.sort(key=lambda g: (g['start'] or '', g['game_number'] or 0))
    synonyms = {'indians': 'guardians', 'guardians': 'indians', 'diamondbacks': 'dbacks', 'dbacks': 'diamondbacks'}
    pairs, unmatched = [], Counter()
    for d, evs in zip(days, boards):
        evs = sorted(evs, key=lambda e: e['start'])
        used = Counter()
        for ev in evs:
            nick = key_name(ev['home_short'] or '')
            k = (d, nick)
            cands = by_key.get(k) or by_key.get((d, synonyms.get(nick, '')))
            if not cands:
                unmatched[d[:4]] += 1
                continue
            i = used[nick]
            used[nick] += 1
            if i < len(cands):
                pairs.append((cands[i], ev['eid']))
    odds = pmap(lambda x: espn_odds(cache, 'baseball', 'mlb', x[1]), pairs, 10, 'mlb odds')
    cache.save()
    for (g, eid), books in zip(pairs, odds):
        total, n = consensus_total(books)
        if total is not None:
            g.update(line=total, books=n, espn_id=eid, line_source='espn')
    rows = []
    for g in games.values():
        speed, direction = parse_wind((g['weather'] or {}).get('wind'))
        g.update(actual=g['away_score'] + g['home_score'], temp=num((g['weather'] or {}).get('temp')),
                 condition=(g['weather'] or {}).get('condition'), wind_mph=speed, wind_dir=direction, season=int(g['date'][:4]))
        del g['weather']
        rows.append(g)
    c2 = Counter((g['season'], g.get('line') is not None, g['hp_umpire'] is not None, g['temp'] is not None) for g in rows)
    cov = {f'{s}|line={a}|ump={b}|weather={c}': n for (s, a, b, c), n in sorted(c2.items(), key=str)}
    cov['espn_unmatched_by_year'] = dict(unmatched)
    write('mlb', rows, cov)


if __name__ == '__main__':
    {'nfl': build_nfl, 'nhl': build_nhl, 'nba': build_nba, 'mlb': build_mlb}[sys.argv[1]]()
