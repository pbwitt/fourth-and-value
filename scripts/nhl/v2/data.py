"""Official per-game data with immutable raw snapshots and explicit availability."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import requests

UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[3]
ET = ZoneInfo('America/New_York')


def history_day(now):
    """Exclusive game-date boundary: completed dates before today Eastern."""
    return now.astimezone(ET).date().isoformat()


def iso(dt):
    return dt.astimezone(UTC).isoformat().replace('+00:00', 'Z')


def stamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timezone required')
    return dt.astimezone(UTC)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fetch_report(kind, season, root, through=None, after=None):
    """Fetch paginated report; never put credentials in requests or manifests."""
    rows, pages = [], []
    start = 0
    while True:
        params = dict(isAggregate='false', isGame='true', start=start, limit=-1,
                      sort=json.dumps([dict(property='gameId',direction='ASC'), dict(property='playerId' if kind in ('skater', 'goalie') else 'teamId',direction='ASC')]),
                      cayenneExp=f'seasonId={season} and gameTypeId=2')
        if through:
            params['cayenneExp'] += f' and gameDate<"{through}"'
        if after:
            params['cayenneExp'] += f' and gameDate>="{after}"'
        url = f'https://api.nhle.com/stats/rest/en/{kind}/summary'
        payload = None
        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=60)
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError):
                if attempt == 2:
                    raise RuntimeError(f'Official {kind} report unavailable for {season}') from None
                time.sleep(2 ** attempt)
        if not isinstance(payload.get('data'), list) or 'total' not in payload:
            raise ValueError('Unexpected NHL report schema')
        if payload['total'] >= 10000:
            raise ValueError('NHL report hits 10,000 row cap; use smaller date partitions')
        received = iso(datetime.now(UTC))
        sha = digest(payload)
        path = Path(root) / 'raw' / f'{kind}-{season}-{start}-{sha[:16]}.json'
        if not path.exists():
            write_json(path, dict(source_url=url, query=params, source_published_at=None,
                                 ingested_at=received, sha256=sha, payload=payload))
        pages.append(dict(path=str(path.relative_to(root)), sha256=sha, ingested_at=received))
        rows.extend(payload['data'])
        start = len(rows)
        if start >= int(payload['total']):
            break
        if not payload['data']:
            raise ValueError('Incomplete NHL pagination')
        time.sleep(.15)
    return rows, pages


def normalize(team_rows, player_rows, season):
    """Statistics dates describe completed events, not original publication times."""
    groups = {}
    for row in team_rows:
        key = int(row['gameId'])
        groups.setdefault(key, []).append(row)
    games = []
    for gid, pair in groups.items():
        if len(pair) != 2 or {r['homeRoad'] for r in pair} != {'H', 'R'}:
            raise ValueError(f'Incomplete/duplicate team pair: {gid}')
        h, a = sorted(pair, key=lambda r:r['homeRoad'])
        if h['goalsFor'] != a['goalsAgainst'] or a['goalsFor'] != h['goalsAgainst']:
            raise ValueError(f'Inconsistent final score: {gid}')
        so = bool(h['winsInShootout'] or a['winsInShootout'])
        if h['goalsFor'] == a['goalsFor'] and not so:
            raise ValueError(f'Nonfinal tied NHL game: {gid}')
        # Reconstructed evaluation assumption, NOT the NHL's publication time.
        # Live inference may use a verified observation before this assumed cutoff.
        available = iso(datetime.fromisoformat(h['gameDate']).replace(tzinfo=UTC) + timedelta(days=1, hours=12))
        if h['wins']+a['wins'] != 1:
            raise ValueError(f'Nonfinal winner state: {gid}')
        # An OT empty-net loss can be recorded as a regulation loss in standings.
        extra = h['winsInRegulation']+a['winsInRegulation'] == 0
        game = dict(game_id=gid, season=season, game_type=2, game_date=h['gameDate'],
                    available_at=available, source_published_at=None,
                    availability_basis='reconstructed_next_day_12UTC',
                    home_id=int(h['teamId']), away_id=int(a['teamId']),
                    home_team=h['teamFullName'], away_team=a['teamFullName'],
                    home_score=int(h['goalsFor'])+int(so and h['wins']),
                    away_score=int(a['goalsFor'])+int(so and a['wins']),
                    extra_time=extra, shootout=so)
        for side, r in [('home', h), ('away', a)]:
            score = int(r['goalsFor'])
            game[side + '_reg_goals'] = score - int(extra and not so and r['wins'] == 1)
            game[side + '_shots'] = float(r['shotsForPerGame'])
            game[side + '_pp_pct'] = r.get('powerPlayPct')
            game[side + '_pk_pct'] = r.get('penaltyKillPct')
        games.append(game)
    by_game = {g['game_id']:g for g in games}
    players, seen = [], set()
    for row in player_rows:
        gid, pid = int(row['gameId']), int(row['playerId'])
        if (gid, pid) in seen:
            raise ValueError(f'Duplicate player-game: {gid}/{pid}')
        seen.add((gid, pid))
        g = by_game.get(gid)
        if not g:
            raise ValueError(f'Player without team game: {gid}')
        # Zero-TOI rows cannot establish participation. DNP is a void, not a zero outcome.
        toi = float(row.get('timeOnIcePerGame') or 0) / 60
        if toi <= 0:
            continue
        if int(row['points']) != int(row['goals']) + int(row['assists']):
            raise ValueError('Inconsistent player points')
        players.append(dict(game_id=gid, player_id=pid, player=row['skaterFullName'],
                            position=row.get('positionCode', 'U'), season=season,
                            game_date=g['game_date'], available_at=g['available_at'],
                            source_published_at=None, toi=toi,
                            home=row['homeRoad'] == 'H', team_abbrev=row.get('teamAbbrev'),
                            team_id=g['home_id'] if row['homeRoad'] == 'H' else g['away_id'],
                            shots=int(row['shots']), goals=int(row['goals']),
                            assists=int(row['assists']), points=int(row['points'])))
    return sorted(games, key=lambda g:(g['game_date'],g['game_id'])), sorted(players, key=lambda r:(r['game_date'],r['game_id'],r['player_id']))


def collect(seasons, root, through=None):
    root = Path(root)
    for season in seasons:
        teams, tp = fetch_report('team', season, root, through)
        players, pp = [], []
        year = season // 10000
        # The API silently caps reports at 10,000 rows. Monthly partitions avoid truncation.
        for month in range(9, 19):
            y, m = year + (month-1)//12, (month-1)%12+1
            lo = f'{y}-{m:02}-01'
            hi = f'{y+int(m==12)}-{m%12+1:02}-01'
            if through and lo >= through:
                continue
            part, pages = fetch_report('skater', season, root, min(through,hi) if through else hi,lo)
            players.extend(part); pp.extend(pages)
        games, skaters = normalize(teams, players, season)
        manifest = dict(schema_version=1, season=season, ingested_at=iso(datetime.now(UTC)),
                        through=through, pages=tp+pp, games=len(games), players=len(skaters),
                        data_sha256=digest([games, skaters]), vintage_verified=False)
        write_json(root / f'{season}.json', dict(manifest=manifest, games=games, players=skaters))
        print(f'{season}: {len(games)} games; {len(skaters)} skater appearances; {len(tp)+len(pp)} source pages', flush=True)


def load(root, seasons=None):
    games, players, manifests = [], [], []
    for path in sorted(Path(root).glob('20??????.json')):
        if seasons and int(path.stem) not in seasons:
            continue
        data = json.loads(path.read_text())
        if digest([data['games'],data['players']]) != data['manifest']['data_sha256']:
            raise ValueError(f'History checksum mismatch: {path.name}')
        games.extend(data['games']); players.extend(data['players']); manifests.append(data['manifest'])
    return games, players, manifests


def observed_history(games, players, manifests, asof):
    """Use already retrieved final results in live inference without rewriting backtests.

    Stored history keeps its reconstructed timestamps. Only when that assumption
    lies after a real observation do these live copies use the observation instead.
    Never claim an original publication time or make a later retrieval available
    to an earlier decision. The conservative chronology for older games is retained
    for the reconstructed player forecast ledger.
    """
    observed = {m['season']: stamp(m['ingested_at']) for m in manifests}
    if any(when > asof for when in observed.values()):
        raise ValueError('History observed after the live decision')
    copies, by_game = [], {}
    for game in games:
        when = observed[game['season']]
        if game['game_date'] >= history_day(when):
            raise ValueError('Live history must contain only earlier completed dates')
        current = dict(game, observed_at=iso(when))
        if (game.get('availability_basis') == 'reconstructed_next_day_12UTC'
                and stamp(game['available_at']) > when):
            current.update(available_at=iso(when),
                           reconstructed_available_at=game['available_at'],
                           availability_basis='observed_final_report')
        copies.append(current)
        by_game[game['game_id']] = current
    appearances = []
    for row in players:
        game = by_game[row['game_id']]
        current = dict(row, observed_at=game['observed_at'])
        if game.get('availability_basis') == 'observed_final_report':
            current.update(available_at=game['available_at'],
                           reconstructed_available_at=row['available_at'],
                           availability_basis='observed_final_report')
        appearances.append(current)
    return copies, appearances


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons', nargs='+', type=int, default=[20222023,20232024,20242025,20252026])
    parser.add_argument('--root', type=Path, default=ROOT/'data/nhl/v2/history')
    parser.add_argument('--through', help='Exclusive game-date cutoff, YYYY-MM-DD')
    args = parser.parse_args()
    collect(args.seasons, args.root, args.through)
