"""Cached official game-level observations for chronological MLB modeling.

Only per-game stats are retained. Box-score seasonStats are deliberately ignored.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from nba.pipeline import save_json, iso

DIRECTORY = ROOT / 'data/mlb/model_data'
TYPES = {'R', 'F', 'D', 'L', 'W'}
BAT_KEYS = ['plateAppearances', 'atBats', 'hits', 'doubles', 'triples', 'homeRuns',
            'totalBases', 'rbi', 'runs', 'strikeOuts', 'baseOnBalls']
PITCH_KEYS = ['outs', 'battersFaced', 'strikeOuts', 'baseOnBalls', 'hits', 'homeRuns',
              'runs', 'earnedRuns', 'numberOfPitches', 'gamesStarted']


def get(endpoint, **params):
    for attempt in range(3):
        try:
            r = requests.get('https://statsapi.mlb.com/api/v1/' + endpoint, params=params, timeout=30)
            if r.status_code in [429, 500, 502, 503, 504]:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 2:
                raise RuntimeError('MLB history request failed') from None
            time.sleep(2 ** attempt)
    raise RuntimeError('MLB history request failed after retries')


def numbers(stats, keys):
    return {key: float(stats.get(key) or 0) for key in keys}


def normalize(meta, box):
    game = dict(meta)
    game['teams'] = {}
    for side in ['home', 'away']:
        team = box['teams'][side]
        players = team['players']
        batters, pitchers = [], []
        for player in players.values():
            person = player['person']
            batting = player.get('stats', {}).get('batting', {})
            pitching = player.get('stats', {}).get('pitching', {})
            order = int(player.get('battingOrder') or 0)
            if batting.get('plateAppearances', 0) > 0 or order:
                batters.append(dict(id=person['id'], name=person['fullName'],
                    slot=order//100 if order and order%100 == 0 else 0,
                    **numbers(batting, BAT_KEYS)))
            if pitching.get('gamesStarted', 0) or pitching.get('battersFaced', 0) > 0:
                pitchers.append(dict(id=person['id'], name=person['fullName'], **numbers(pitching, PITCH_KEYS)))
        starters = [p for p in pitchers if p['gamesStarted'] == 1]
        if len(starters) != 1:
            raise ValueError('Completed box score lacks a unique starter')
        batting = numbers(team['teamStats']['batting'], BAT_KEYS)
        pitching = numbers(team['teamStats']['pitching'], PITCH_KEYS)
        if batting['runs'] != meta[side + '_score'] or batting['plateAppearances'] < 10:
            raise ValueError('Box score does not reconcile with final result')
        game['teams'][side] = dict(id=team['team']['id'], name=team['team']['name'],
            starter=starters[0]['id'], batting=batting, pitching=pitching, batters=batters, pitchers=pitchers)
    return game


def manifest(now):
    now = now.astimezone(ZoneInfo('America/New_York'))
    # Two seasons include an actual prior postseason and a full current-season test.
    seasons = [(now.year-1, f'{now.year-1}-08-01', f'{now.year-1}-11-15'),
               (now.year, f'{now.year}-03-01', (now-timedelta(days=1)).date().isoformat())]
    result = {}
    for year, start, end in seasons:
        if start > end:
            continue
        payload = get('schedule', sportId=1, startDate=start, endDate=end, hydrate='linescore')
        if not isinstance(payload.get('dates'), list):
            raise ValueError('Incomplete MLB historical schedule')
        for day in payload['dates']:
            for g in day.get('games', []):
                if (g.get('gameType') not in TYPES or g.get('status', {}).get('abstractGameState') != 'Final'
                    or g.get('scheduledInnings', 9) != 9 or g.get('resumeDate') or g.get('resumedFrom')
                    or g.get('linescore', {}).get('currentInning', 0) < 9):
                    continue
                home, away = g['teams']['home'], g['teams']['away']
                if home.get('score') is None or away.get('score') is None or home['score'] == away['score']:
                    continue
                result[g['gamePk']] = dict(id=g['gamePk'], date=g.get('officialDate', day['date']),
                    start=g['gameDate'], season=int(g['season']), game_type=g['gameType'],
                    venue=g['venue']['id'], home_id=home['team']['id'], away_id=away['team']['id'],
                    home_score=home['score'], away_score=away['score'])
    return sorted(result.values(), key=lambda x:(x['date'], x['id']))


def update(now=None):
    now = now or datetime.now(timezone.utc)
    now = now.astimezone(ZoneInfo('America/New_York'))
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    entries = manifest(now)
    needed = [g for g in entries if not (DIRECTORY/f"{g['id']}.json").exists()]
    print(f'MLB history: {len(entries)} completed games; fetching {len(needed)} missing box scores', flush=True)
    failures = []
    def fetch(meta):
        try:
            game = normalize(meta, get(f"game/{meta['id']}/boxscore"))
            save_json(DIRECTORY/f"{meta['id']}.json", game)
            time.sleep(.12)
            return None
        except (RuntimeError, KeyError, TypeError, ValueError) as error:
            return dict(game_id=meta['id'], reason=str(error))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, g) for g in needed]
        for count, future in enumerate(as_completed(futures), 1):
            failure = future.result()
            if failure:
                failures.append(failure)
            if count%100 == 0:
                print(f'MLB history: {count}/{len(needed)} fetched; {len(failures)} unavailable', flush=True)
    report = dict(fetched_at=iso(now), expected_games=len(entries), failures=failures,
                  game_ids=[g['id'] for g in entries], through_date=(now-timedelta(days=1)).date().isoformat())
    save_json(DIRECTORY/'manifest.json', report)
    print(f'MLB history complete: {len(entries)-len(failures)} games, {len(failures)} missing', flush=True)
    return report


def load():
    report = json.loads((DIRECTORY/'manifest.json').read_text())
    games = []
    for game_id in report['game_ids']:
        path = DIRECTORY/f'{game_id}.json'
        if path.exists():
            games.append(json.loads(path.read_text()))
    return sorted(games, key=lambda g:(g['date'], g['id'])), report


if __name__ == '__main__':
    update()
