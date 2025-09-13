#!/usr/bin/env python3
import argparse, sys, io, gzip, pathlib, requests, pandas as pd

URLS_TPL = [
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{SEASON}.parquet",
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{SEASON}.csv.gz",
]

def fetch(season:int, out_parquet:pathlib.Path):
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    for u in [u.format(SEASON=season) for u in URLS_TPL]:
        print(f"[fetch] trying {u}")
        r = requests.get(u, timeout=90); r.raise_for_status()
        if u.endswith(".parquet"):
            out_parquet.write_bytes(r.content)
            print(f"[ok] wrote {out_parquet} ({out_parquet.stat().st_size:,} bytes)")
            return
        else:
            # csv.gz → read → write parquet at the expected path
            buf = io.BytesIO(r.content)
            with gzip.GzipFile(fileobj=buf) as gz:
                df = pd.read_csv(gz, low_memory=False)
            df.to_parquet(out_parquet, index=False)
            print(f"[ok] converted csv.gz → {out_parquet} rows={len(df):,}")
            return
    raise RuntimeError("Could not fetch weekly stats in parquet or csv.gz form")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", type=str, default=None,
                    help="Output parquet path (default: data/weekly_player_stats_{SEASON}.parquet)")
    args = ap.parse_args()
    out = pathlib.Path(args.out or f"data/weekly_player_stats_{args.season}.parquet")
    fetch(args.season, out)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[err] {e}", file=sys.stderr); sys.exit(2)
