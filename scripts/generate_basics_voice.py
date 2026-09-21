#!/usr/bin/env python3
"""Explicit, cached TTS generation for one educational video. Never part of weekly runs."""
import argparse
import hashlib
import json
import sys
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-voice', action='store_true', help='Opt in to paid OpenAI speech requests')
    parser.add_argument('--spec', default='content/videos/01-why-we-devig.json', help='JSON scene specification')
    parser.add_argument('--output-dir', help='Audio/timeline directory (defaults beside the spec slug)')
    args = parser.parse_args()
    spec_path = ROOT / args.spec
    spec = json.loads(spec_path.read_text())
    if not args.generate_voice:
        print('Preview only. No API calls. Add --generate-voice to opt in.')
        print(json.dumps(spec, indent=2))
        return
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / '.env')
    client = OpenAI(max_retries=0, timeout=90)
    out = ROOT / (args.output_dir or 'docs/videos/why-we-devig')
    out.mkdir(parents=True, exist_ok=True)
    timeline = []
    start = 0.0
    for i, scene in enumerate(spec['scenes']):
        payload = dict(model=spec['model'], voice=spec['voice'], input=scene['narration'],
                       instructions=spec['instructions'], response_format='wav')
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        audio = out / f'scene-{i+1:02}.wav'
        stamp = out / f'scene-{i+1:02}.sha256'
        if not (audio.exists() and stamp.exists() and stamp.read_text() == digest):
            try:
                # Write completely before replacing an existing cached asset.
                with client.audio.speech.with_streaming_response.create(**payload) as response:
                    response.stream_to_file(audio.with_suffix('.tmp'))
                audio.with_suffix('.tmp').replace(audio)
                stamp.write_text(digest)
            except Exception as error:
                # Do not print requests, headers, environment values, or keys.
                print(f'Speech generation stopped at scene {i+1}: {type(error).__name__}', file=sys.stderr)
                raise SystemExit(1)
        # Normalize the streaming WAV sentinel so browsers can seek and report
        # the correct duration. The generated payload is PCM16 mono at 24 kHz.
        payload_bytes = audio.stat().st_size - 44
        with audio.open('r+b') as wav:
            wav.seek(4); wav.write(struct.pack('<I', 36 + payload_bytes))
            wav.seek(40); wav.write(struct.pack('<I', payload_bytes))
        # OpenAI's streaming WAV may retain the RF64/streaming frame sentinel
        # (2^31-1) in its header. These files are PCM16 mono at 24 kHz; derive
        # duration from the actual payload so the timeline stays accurate.
        seconds = max(0, audio.stat().st_size - 44) / (2 * 1 * 24000)
        timeline.append(dict(scene, audio=audio.name, start=round(start, 3), speech_duration=seconds,
                             duration=seconds + 0.55))
        start += seconds + 0.55
        print(f'Scene {i+1}: {seconds:.1f}s; saved/cached', flush=True)
    spec.update(scenes=timeline, duration=round(start, 3))
    (out / 'timeline.json').write_text(json.dumps(spec, indent=2) + '\n')
    print(f'Total: {start:.1f}s. No recurring job was created.')


if __name__ == '__main__':
    main()
