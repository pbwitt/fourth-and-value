"""Production-code live check: archived 2026-27 snapshots re-annotated with the installed model artifact.

Each snapshot is re-forecast at its own decision time with only results available by then.
Reports model/market mean ratio, Over/Under balance of the current NHL screen, and log loss on
settled props (Over side, first snapshot per outcome) against the market consensus.
"""
import copy, glob, gzip, json, math, sys
from collections import defaultdict
from datetime import timedelta
import numpy as np
sys.path.insert(0, 'scripts')
from nhl.v2.data import load, stamp, iso
from nhl.v2.inference import annotate, bundle
from nhl.v2.candidates import exclusion
from nhl.v2.coherence import implied_mean
from nhl.v2.grading import settle

out_path = sys.argv[1]
models, manifest = bundle()
frozen = json.load(gzip.open('models/nhl/v2/history.json.gz', 'rt'))
cur_games, cur_players, _ = load('data/nhl/v2/history', [20262027])
games = [g for g in frozen['games'] if g['season'] != 20262027] + cur_games
players = [p for p in frozen['players'] if p['season'] != 20262027] + cur_players
by_game = {g['game_id']: g for g in cur_games}
by_player = {(p['game_id'], p['player_id']): p for p in cur_players}
config = json.load(open('config/nhl_analyst.json'))
ALPHA = {'player_shots_on_goal': .0767, 'player_goals': .0252, 'player_assists': .0252, 'player_points': .0252}

snaps = []
for path in glob.glob('artifacts/nhl/runs/*.json.gz'):
    s = json.load(gzip.open(path, 'rt'))['snapshot']
    if s.get('model_prediction_at') and any(r['market'].startswith('player_') for r in s['rows']):
        snaps.append(s)
snaps.sort(key=lambda s: s['checked_at'])

first, ratios, screen = {}, defaultdict(list), None
for s in snaps:
    now = stamp(s['model_prediction_at'])
    rows = annotate(copy.deepcopy(s['rows']), games, players, s['events'], models, manifest, now, iso(now))
    for r in rows:
        if not r['market'].startswith('player_') or r.get('conditional_probability') is None:
            continue
        if r['side'] == 'Over' and r.get('consensus_probability') is not None:
            k = (r['nhl_game_id'], r.get('player_id'), r['market'], r['line'])
            first.setdefault(k, r)
            m = implied_mean(r['consensus_probability'], r['line'], ALPHA[r['market']])
            if m and r.get('projected_mean'):
                ratios[r['market']].append(r['projected_mean'] / m)
    if s is snaps[-1]:
        decision = now + timedelta(seconds=1)
        best = {}
        for r in rows:
            if r['market'].startswith('player_') and exclusion(r, decision, config) is None:
                key = (r['nhl_game_id'], r.get('player_id'), r['market'], r['side'], r['line'])
                if key not in best or r['price'] > best[key]['price']:
                    best[key] = r
        screen = defaultdict(lambda: [0, 0])
        for r in best.values():
            screen[r['market']][r['side'] == 'Under'] += 1

ll = defaultdict(lambda: [[], [], []])
for k, r in first.items():
    result = settle(r, by_game.get(r['nhl_game_id']), by_player.get((r['nhl_game_id'], r.get('player_id'))))
    if result not in ('won', 'lost'):
        continue
    y = result == 'won'
    for i, p in enumerate([r['conditional_probability'], r['consensus_probability']]):
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll[r['market']][i].append(-math.log(p if y else 1 - p))
    ll[r['market']][2].append(r['nhl_game_id'])

report = dict(model_version=manifest['version'], artifact_sha256=manifest['artifact_sha256'], snapshots=len(snaps),
              median_model_over_market_mean={m: float(np.median(v)) for m, v in ratios.items()},
              latest_screen={m: dict(overs=v[0], unders=v[1]) for m, v in screen.items()},
              settled={m: dict(n=len(v[0]), games=len(set(v[2])), model_log_loss=float(np.mean(v[0])),
                               market_log_loss=float(np.mean(v[1]))) for m, v in ll.items()})
tot = [sum(v[i] for v in screen.values()) for i in (0, 1)]
report['latest_screen_total'] = dict(overs=tot[0], unders=tot[1], under_share=tot[1] / max(1, sum(tot)))
json.dump(report, open(out_path, 'w'), indent=1)
print(json.dumps(report, indent=1))
