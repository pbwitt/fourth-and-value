#!/usr/bin/env python3
"""Render the injury-versus-total-market research screen."""
from pathlib import Path
import argparse
import html
import pandas as pd


def render(df: pd.DataFrame, season: int, week: int) -> str:
    rows = []
    for _, r in df.iterrows():
        impact = f"{r.model_injury_impact_points:+.1f}"
        residual = "—" if pd.isna(r.reaction_residual) else f"{r.reaction_residual:+.1f}"
        move = "—" if pd.isna(r.market_total_move) else f"{r.market_total_move:+.1f}"
        rows.append(f"<tr><td>{html.escape(str(r.game))}</td><td>{r.baseline_total_model:.1f}</td>"
                    f"<td>{r.injury_adjusted_total_model:.1f}</td><td>{impact}</td>"
                    f"<td>{move}</td><td>{residual}</td><td>{html.escape(str(r.signal_status))}</td></tr>")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Injury vs Totals Market — Week {week} | Fourth &amp; Value</title><meta name="description" content="Research screen comparing NFL injury impacts with captured totals market movement."><link rel="canonical" href="https://fourthandvalue.com/nfl/injuries/"><link rel="stylesheet" href="../../assets/site.css"></head><body><main class="wrap"><p class="eyebrow">NFL · {season} Week {week}</p><h1>Injury vs totals market</h1><p class="lead">A research screen comparing conservative injury impacts with captured total-line movement.</p><div class="notice"><strong>Research only.</strong> Position effects are priors awaiting historical calibration. A signal is not a pick, and missing line history is shown explicitly.</div><div class="table-wrap"><table><thead><tr><th>Game</th><th>Base model</th><th>Injury model</th><th>Injury impact</th><th>Market move</th><th>Residual</th><th>Status</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p><a href="../../nfl/totals/">Back to totals</a> · <a href="../../nfl/">NFL overview</a> · <a href="https://github.com/pbwitt/fourth-and-value/blob/main/docs/INJURY_AUTOMATION.md">Method documentation</a></p></main></body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(pd.read_csv(a.input), a.season, a.week))
    print(f"[injury-page] wrote {out}")


if __name__ == "__main__":
    main()
