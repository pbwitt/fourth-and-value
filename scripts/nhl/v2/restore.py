"""Reconstruct normalized historical data from the checked-in raw source archive."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2])); __package__='nhl.v2'
from .data import ROOT,normalize,digest,write_json


def restore(archive,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        # Do not extract paths or links from an archive into the filesystem.
        members={m.name:m for m in tar.getmembers() if m.isfile()}
        for name in sorted(members):
            if not name.endswith('-manifest.json'): continue
            manifest=json.load(tar.extractfile(members[name]));teams=[];players=[]
            for ref in manifest['pages']:
                raw=json.load(tar.extractfile(members[ref['path']]))
                if digest(raw['payload'])!=ref['sha256']: raise ValueError('Raw source checksum mismatch')
                (teams if '/team/' in raw['source_url'] else players).extend(raw['payload']['data'])
                # Only the known basename within our raw directory is written.
                write_json(root/'raw'/Path(ref['path']).name,raw)
            games,skaters=normalize(teams,players,manifest['season'])
            if digest([games,skaters])!=manifest['data_sha256']: raise ValueError('Normalized history checksum mismatch')
            write_json(root/f'{manifest["season"]}.json',dict(manifest=manifest,games=games,players=skaters))
            print(f"Restored {manifest['season']}: {len(games)} games; {len(skaters)} skaters")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'artifacts/nhl/training-sources.tar.gz')
    p.add_argument('--root',type=Path,default=ROOT/'data/nhl/v2/history')
    a=p.parse_args();restore(a.archive,a.root)
