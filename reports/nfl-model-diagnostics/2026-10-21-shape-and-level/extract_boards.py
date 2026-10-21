"""Extract every archived NFL props board (docs/props/index.html) that carries the model mean.

Writes one row per snapshot, event, player, market, line, side and book, with the commit time
as the time the snapshot was public. Boards before September 9, 2026 do not record the mean.
History must be complete (not a shallow clone). usage: extract_boards.py REPO OUT_CSV [REV]
"""
import json
import re
import subprocess
import sys

import pandas as pd

REPO, OUT = sys.argv[1], sys.argv[2]
REV = sys.argv[3] if len(sys.argv) > 3 else 'origin/main'
PATH = 'docs/props/index.html'
PATTERN = re.compile(r'<script type="application/json" id="props-data">(.*?)</script>', re.S)


def git(*args):
    return subprocess.run(['git', '-C', REPO, *args], check=True, capture_output=True, text=True).stdout


commits = [line.split(' ', 1) for line in git('log', REV, '--format=%H %cI', '--', PATH).splitlines()]
frames, skipped = [], 0
for sha, committed in commits:
    m = PATTERN.search(git('show', f'{sha}:{PATH}'))
    if not m:
        skipped += 1
        continue
    d = json.loads(m.group(1))
    fields, dictionary = d['fields'], d.get('dictionary') or {}
    if 'mu' not in fields:
        skipped += 1
        continue
    rows = [{f: (dictionary[f][v] if f in dictionary and v is not None else v) for f, v in zip(fields, r)}
            for r in d['rows']]
    df = pd.DataFrame(rows)
    df['commit'], df['committed_at'] = sha[:10], committed
    if 'projection_diagnostics' not in df:
        df['projection_diagnostics'] = None
    sigma = df['projection_diagnostics'].map(lambda s: json.loads(s).get('sigma') if isinstance(s, str) else None)
    stages = df['projection_diagnostics'].map(lambda s: json.loads(s).get('mean_stages') if isinstance(s, str) else None)
    df['sigma'] = sigma
    for k in ['before_adjustments', 'after_defense', 'after_venue', 'final']:
        df[f'stage_{k}'] = stages.map(lambda x, k=k: x.get(k) if isinstance(x, dict) else None)
    frames.append(df.drop(columns=['projection_diagnostics']))
out = pd.concat(frames, ignore_index=True)
out.to_csv(OUT, index=False)
print(f'{len(commits)} commits, {len(frames)} with model means, {skipped} without; {len(out)} rows -> {OUT}')
print(out.groupby(out.committed_at.str[:10]).size().to_string())
