"""v2.1 vs v2.2 on the same 2025-26 final-test rows: paired count log loss and level ratios."""
import gzip, json, sys
from collections import defaultdict
import numpy as np

OLD, NEW = sys.argv[1], sys.argv[2]
hist = json.load(gzip.open('models/nhl/v2/history.json.gz', 'rt'))
prev = defaultdict(lambda: [0, 0])
for r in hist['players']:
    if r['season'] == 20242025:
        prev[r['player_id']][0] += r['shots']; prev[r['player_id']][1] += 1


def tier(pid):
    s, n = prev.get(int(pid), (0, 0))
    if not n: return 'no previous season'
    rate = s / n
    return 'prop (>=2.5)' if rate >= 2.5 else 'mid (1.5-2.5)' if rate >= 1.5 else 'low (<1.5)'


def rows(path):
    out = {}
    for line in gzip.open(path, 'rt'):
        r = json.loads(line)
        if r['model'] == 'opportunity_nb':
            out[(r['game_id'], r['player_id'], r['market'])] = r
    return out


a, b = rows(OLD), rows(NEW)
assert a.keys() == b.keys(), 'different final-test rows'
result = dict(rows=len(a), markets={})
rng = np.random.default_rng(20261003)
for market in ['shots', 'goals', 'assists', 'points']:
    keys = [k for k in a if k[2] == market]
    games = defaultdict(list)
    for k in keys:
        games[k[0]].append(float(b[k]['count_log_loss']) - float(a[k]['count_log_loss']))
    sums = np.array([sum(v) for v in games.values()]); counts = np.array([len(v) for v in games.values()])
    idx = rng.integers(0, len(sums), (2000, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    level = defaultdict(lambda: [0., 0., 0.])
    for k in keys:
        month = a[k]['game_date'][5:7]
        phase = 'Oct' if month == '10' else 'Nov' if month == '11' else 'Dec-Apr'
        for group in ['all', tier(k[1]), tier(k[1]) + ' ' + phase, 'all ' + phase]:
            level[group][0] += float(a[k]['mean']); level[group][1] += float(b[k]['mean']); level[group][2] += float(a[k]['actual'])
    result['markets'][market] = dict(
        n=len(keys), games=len(games),
        log_loss_v21=float(np.mean([float(a[k]['count_log_loss']) for k in keys])),
        log_loss_v22=float(np.mean([float(b[k]['count_log_loss']) for k in keys])),
        difference=float(sums.sum() / counts.sum()), interval=np.quantile(boot, [.025, .975]).tolist(),
        level={g: dict(v21=v[0] / v[2], v22=v[1] / v[2]) for g, v in sorted(level.items())})
json.dump(result, open(sys.argv[3], 'w'), indent=1)
for m, r in result['markets'].items():
    print(m, f"LL {r['log_loss_v21']:.5f} -> {r['log_loss_v22']:.5f} diff {r['difference']:+.5f} [{r['interval'][0]:+.5f}, {r['interval'][1]:+.5f}]")
    for g in ['all', 'prop (>=2.5)', 'prop (>=2.5) Oct', 'prop (>=2.5) Dec-Apr', 'low (<1.5)', 'no previous season']:
        if g in r['level']:
            print(f"   {g:24s} {r['level'][g]['v21']:.3f} -> {r['level'][g]['v22']:.3f}")
