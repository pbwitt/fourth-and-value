"""Rebuild 2026-27 official results from the run archive's content-addressed raw pages."""
import gzip, json, glob, sys
sys.path.insert(0, 'scripts')
from nhl.v2.data import normalize, digest, write_json
runs = sorted(glob.glob('artifacts/nhl/runs/*.json.gz'), key=lambda p: json.load(gzip.open(p,'rt'))['checked_at'])
rec = json.load(gzip.open(runs[-1], 'rt'))
by_source = {r['source']: r for r in rec['input_objects']}
for manifest in rec['history_manifests']:
    teams, players = [], []
    for page in manifest['pages']:
        src = 'data/nhl/v2/history/' + page['path']
        ref = by_source[src]
        raw = json.load(gzip.open('artifacts/nhl/' + ref['path'], 'rt'))
        assert digest(raw['payload']) == page['sha256'], 'raw checksum'
        (teams if '/team/' in raw['source_url'] else players).extend(raw['payload']['data'])
    games, skaters = normalize(teams, players, manifest['season'])
    assert digest([games, skaters]) == manifest['data_sha256'], 'normalized checksum'
    write_json(f'data/nhl/v2/history/{manifest["season"]}.json', dict(manifest=manifest, games=games, players=skaters))
    print(manifest['season'], 'through', manifest['through'], len(games), 'games', len(skaters), 'skater rows',
          min(g['game_date'] for g in games), max(g['game_date'] for g in games))
