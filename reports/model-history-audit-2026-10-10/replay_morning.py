"""Replay the actual morning NHL artifact with only live availability corrected.

Run with the downloaded nhl-snapshot-38047149608 ZIP. No network or paid requests.
Historical forecasts and outcomes remain untouched; this measures input sensitivity,
not betting profitability or out-of-sample improvement.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from nhl.v2.data import digest, observed_history, stamp, write_json
from nhl.v2.inference import annotate, bundle, MODEL_DIR


def run(archive, output):
    with zipfile.ZipFile(archive) as z:
        current = json.loads(z.read('data/nhl/v2/history/20262027.json'))
        snapshot = json.loads(z.read('data/nhl/snapshots/20261010T110632Z.json'))
    assert digest([current['games'], current['players']]) == current['manifest']['data_sha256']
    models, manifest = bundle()
    assert manifest['artifact_sha256'] == snapshot['model_manifest']['artifact_sha256']
    frozen = MODEL_DIR/'history.json.gz'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == manifest['history_archive_sha256']
    with gzip.open(frozen, 'rt') as f:
        past = json.load(f)
    season = current['manifest']['season']
    old_games = [g for g in past['games'] if g['season'] != season] + current['games']
    old_players = [r for r in past['players'] if r['season'] != season] + current['players']
    when = stamp(snapshot['model_prediction_at'])
    observed_games, observed_players = observed_history(current['games'], current['players'], [current['manifest']], when)
    new_games = [g for g in past['games'] if g['season'] != season] + observed_games
    new_players = [r for r in past['players'] if r['season'] != season] + observed_players
    def predict(games, players):
        return annotate(copy.deepcopy(snapshot['rows']), games, players, snapshot['events'],
                        models, manifest, when, snapshot['model_data_checked_at'])
    before, after = predict(old_games, old_players), predict(new_games, new_players)
    errors = [abs(r['independent_probability']-s['independent_probability'])
              for r, s in zip(before, snapshot['rows']) if r.get('independent_probability') is not None]
    assert max(errors, default=0) < 1e-10, 'Original forecasts must reproduce before comparing the fix'
    changes, seen = [], set()
    for a, b in zip(before, after):
        if a.get('independent_probability') is None or not a.get('player') or a['side'] != 'Over':
            continue
        key = (a['nhl_game_id'], a['player_id'], a['market'], a['line'])
        if key in seen:
            continue
        seen.add(key)
        changes.append(dict(game_id=key[0], player_id=key[1], player=a['player'], market=a['market'], line=a['line'],
            last_game_before=a['model_inputs']['last_game'], last_game_after=b['model_inputs']['last_game'],
            mean_before=a['projected_mean'], mean_after=b['projected_mean'],
            probability_before=a['independent_probability'], probability_after=b['independent_probability'],
            probability_change_pp=100*(b['independent_probability']-a['independent_probability']),
            toi_before=a['projected_toi'], toi_after=b['projected_toi']))
    shots = [r for r in changes if r['market'] == 'player_shots_on_goal']
    affected = [r for r in shots if abs(r['mean_after']-r['mean_before']) > 1e-10]
    missed = [g for g in current['games'] if stamp(g['available_at']) > when]
    summary = dict(source_run='https://github.com/pbwitt/fourth-and-value/actions/runs/38047149608',
        source_artifact_id=11668675237, source_zip_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        model_version=manifest['version'], model_sha256=manifest['artifact_sha256'],
        decision_at=snapshot['model_prediction_at'], results_ingested_at=current['manifest']['ingested_at'],
        original_probability_max_error=max(errors, default=0),
        completed_games_excluded=[dict(game_id=g['game_id'], date=g['game_date'], home=g['home_team'], away=g['away_team']) for g in missed],
        shots=dict(unique_over_player_game_lines=len(shots), changed_lines=len(affected),
            lines_with_new_player_game=sum(r['last_game_before'] != r['last_game_after'] for r in shots),
            mean_absolute_probability_change_pp=sum(abs(r['probability_change_pp']) for r in shots)/len(shots),
            max_absolute_probability_change_pp=max(abs(r['probability_change_pp']) for r in shots),
            largest_changes=sorted(affected, key=lambda r:abs(r['probability_change_pp']), reverse=True)[:10]),
        copp=[r for r in changes if r['player_id'] == 8477429],
        interpretation='Controlled replay of the actual archived inputs and offers, changing availability only. This measures forecast sensitivity; it does not establish improved calibration or explain season-long losses.')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'nhl-morning-replay.json', summary)
    with (output/'nhl-prop-forecast-changes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(changes[0]))
        writer.writeheader(); writer.writerows(changes)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = p.parse_args()
    run(args.archive, args.output)
