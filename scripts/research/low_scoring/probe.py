"""One-off source probe for the low-scoring study: prints coverage and field names, writes nothing."""
import json
import requests

S = requests.Session()
S.headers['User-Agent'] = 'Mozilla/5.0 (research; fourthandvalue.com)'


def get(url, **kw):
    try:
        r = S.get(url, timeout=40, **kw)
        print('GET', r.status_code, url, len(r.content))
        return r
    except Exception as e:
        print('ERR', url, e)


def keys(d, depth=0, maxd=2):
    if isinstance(d, dict) and depth <= maxd:
        return {k: keys(v, depth + 1, maxd) for k, v in list(d.items())[:40]}
    if isinstance(d, list):
        return [keys(d[0], depth + 1, maxd)] if d else []
    return type(d).__name__


print('== Wayback CDX: SBRO archives')
r = get('http://web.archive.org/cdx/search/cdx', params=dict(url='sportsbookreviewsonline.com/scoresoddsarchives/*', output='json', filter='statuscode:200', collapse='urlkey', limit=2000))
if r is not None and r.ok:
    rows = r.json()
    for row in rows[1:]:
        if any(x in row[2].lower() for x in ('.xls', '.xlsx')):
            print(row[1], row[2])

for sport, lg in [('hockey', 'nhl'), ('basketball', 'nba'), ('baseball', 'mlb')]:
    print('== ESPN scoreboard', lg)
    for d in ['20121215', '20160115', '20190115', '20200815', '20221215', '20250115', '20250715', '20260115']:
        r = get(f'https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/scoreboard', params=dict(dates=d, limit=50))
        if r is None or not r.ok:
            continue
        ev = r.json().get('events', [])
        odds = [e['competitions'][0].get('odds') for e in ev]
        with_ou = [o for o in odds if o and o[0].get('overUnder') is not None]
        print(lg, d, 'events', len(ev), 'with overUnder', len(with_ou), (with_ou[0][0].get('provider', {}).get('name'), with_ou[0][0].get('overUnder')) if with_ou else None)
        if ev and d in ('20250115', '20250715', '20190115'):
            c = ev[0]['competitions'][0]
            print(' comp keys', list(c.keys()))
            if c.get('odds'):
                print(' odds0', json.dumps(c['odds'][0])[:900])
            print(' venue', c.get('venue', {}).get('fullName'), 'neutral', c.get('neutralSite'), 'weather', ev[0].get('weather'))
            eid = ev[0]['id']
            s = get(f'https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/summary', params=dict(event=eid))
            if s is not None and s.ok:
                j = s.json()
                print(' summary keys', list(j.keys()))
                gi = j.get('gameInfo', {})
                print(' gameInfo', json.dumps(gi)[:700])
                print(' pickcenter', json.dumps(j.get('pickcenter', [])[:1])[:900])
            o = get(f'https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{lg}/events/{eid}/competitions/{eid}/odds')
            if o is not None and o.ok:
                print(' core odds', json.dumps(o.json())[:1500])

print('== NHL API')
for gid in ['2015020500', '2024020500']:
    for path in ['right-rail', 'boxscore', 'landing']:
        r = get(f'https://api-web.nhle.com/v1/gamecenter/{gid}/{path}')
        if r is not None and r.ok:
            j = r.json()
            print(path, json.dumps(keys(j, maxd=1))[:900])
            gi = j.get('gameInfo')
            if gi:
                print(' gameInfo', json.dumps(gi)[:600])
r = get('https://api.nhle.com/stats/rest/en/goalie/summary', params={'isAggregate': 'false', 'isGame': 'true', 'start': 0, 'limit': 3, 'cayenneExp': 'seasonId=20152016 and gameTypeId=2'})
if r is not None and r.ok:
    print(json.dumps(r.json())[:1500])
r = get('https://api-web.nhle.com/v1/schedule/2016-01-15')
if r is not None and r.ok:
    print(json.dumps(r.json())[:1200])

print('== MLB StatsAPI')
r = get('https://statsapi.mlb.com/api/v1/schedule', params=dict(sportId=1, date='2015-07-04'))
if r is not None and r.ok:
    pk = r.json()['dates'][0]['games'][0]['gamePk']
    f = get(f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live', params=dict(fields='gameData,weather,datetime,dayNight,venue,name,liveData,boxscore,officials,official,fullName,officialType,linescore,teams,home,away,runs'))
    if f is not None and f.ok:
        print(json.dumps(f.json())[:2000])

print('== NBA')
r = get('https://stats.nba.com/stats/boxscoresummaryv2', params=dict(GameID='0021500500'), headers={'Referer': 'https://www.nba.com/', 'Origin': 'https://www.nba.com', 'x-nba-stats-origin': 'stats', 'x-nba-stats-token': 'true'})
if r is not None and r.ok:
    print(json.dumps(r.json())[:1500])
for gid in ['0021500500', '0022400500']:
    r = get(f'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json')
    if r is not None and r.ok:
        print(json.dumps(r.json()['game'].get('officials'))[:500])
