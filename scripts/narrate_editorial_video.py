"""Generate approved narration with cached WAV segments and a timed manifest.

Explicit --approved flag required. No key or response body is logged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import wave
from concurrent.futures import ThreadPoolExecutor
import requests
from dotenv import load_dotenv

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('spec',type=Path)
    ap.add_argument('--approved',action='store_true')
    args=ap.parse_args()
    if not args.approved: ap.error('Owner approval is required before narration generation.')
    load_dotenv()
    key=os.getenv('OPENAI_API_KEY')
    if not key: raise SystemExit('OPENAI_API_KEY is not configured; no requests made.')
    spec=json.loads(args.spec.read_text())
    out=Path('docs/videos')/spec['slug'];out.mkdir(parents=True,exist_ok=True)
    def generate(item):
        i,s=item
        payload={'model':spec['model'],'voice':spec['voice'],'input':s['narration'],
                 'instructions':spec['instructions'],'response_format':'wav'}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        wav=out/f'scene-{i:02d}.wav';stamp=wav.with_suffix('.sha256')
        if not(wav.exists() and wav.stat().st_size>44 and stamp.exists() and stamp.read_text().strip()==digest):
            response=requests.post('https://api.openai.com/v1/audio/speech',headers={'Authorization':'Bearer '+key},json=payload,timeout=180)
            if not response.ok: raise RuntimeError(f'Narration scene {i}: HTTP {response.status_code}; response omitted to protect account details.')
            wav.write_bytes(response.content)
            stamp.write_text(digest+'\n')
        with wave.open(str(wav),'rb') as w:
            params=w.getparams()
            pcm=w.readframes(w.getnframes())
            duration=len(pcm)/(w.getframerate()*w.getnchannels()*w.getsampwidth())
        # Streaming WAV responses may use a placeholder frame count.
        # Write a finalized header before browsers and duration tools read it.
        with wave.open(str(wav),'wb') as w:
            w.setparams(params._replace(nframes=0))
            w.writeframes(pcm)
        print(f'Scene {i:02d}: {duration:.2f}s',flush=True)
        return dict(s,audio=wav.name,speech_duration=duration,duration=duration+.65)
    with ThreadPoolExecutor(max_workers=3) as pool:
        scenes=list(pool.map(generate,enumerate(spec['scenes'],1)))
    start=0
    for s in scenes:s['start']=start;start+=s['duration']
    spec.update(scenes=scenes,duration=start,production_status='Owner approved generation September 22, 2026')
    (out/'timeline.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(f'Total duration: {start:.2f}s')

if __name__=='__main__':main()
