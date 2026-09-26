"""Content-addressed durable inputs and forecast snapshots, separate from public pages."""
import gzip
import json
from pathlib import Path

from .data import ROOT,digest,write_json


def archive_run(state,root=ROOT/'artifacts/nhl'):
    root=Path(root)
    references=[]
    manifests=[]
    paths=[]
    for normalized in (ROOT/'data/nhl/v2/history').glob('20??????.json'):
        if normalized.stem != str(state['season']): continue
        manifest=json.loads(normalized.read_text())['manifest'];manifests.append(manifest)
        # Keep the original source pages, deduplicated by content. Re-normalization is
        # deterministic; copying full normalized seasons at each refresh is unnecessary.
        paths.extend(normalized.parent/page['path'] for page in manifest['pages'])
    paths+=list((ROOT/'data/nhl/v2/raw_odds').glob('*.json'))
    # Include current reference summaries as well as the new model's game records.
    summary=ROOT/'data/nhl/history/current.json'
    if summary.exists(): paths.append(summary)
    for path in paths:
        value=json.loads(path.read_text());sha=digest(value)
        destination=root/'objects'/f'{sha}.json.gz'
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            with gzip.GzipFile(filename=str(destination),mode='wb',mtime=0) as f:
                f.write(json.dumps(value,separators=(',',':'),allow_nan=False).encode())
        references.append(dict(source=str(path.relative_to(ROOT)),sha256=sha,path=str(destination.relative_to(root))))
    record=dict(snapshot=state,input_objects=references,history_manifests=manifests,model_manifest=state.get('model_manifest'),
                checked_at=state.get('checked_at'))
    sha=digest(record);destination=root/'runs'/f'{state["snapshot_id"]}-{sha[:12]}.json.gz'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():
        with gzip.GzipFile(filename=str(destination),mode='wb',mtime=0) as f:
            f.write(json.dumps(record,separators=(',',':'),allow_nan=False).encode())
    return str(destination)
