# scripts/build_game_page.py
# Usage:
#   python3 scripts/build_game_page.py --season 2025 --week 3 \
#     --csv data/games/model_preds_week3.csv --out docs/games/index.html
#
# If you have odds merged later, you can add columns and they'll render automatically.

import argparse, html
import pandas as pd
from pathlib import Path

def df_to_html_table(df: pd.DataFrame) -> str:
    # Minimal dark-friendly table (inherits your site css if present)
    cols = df.columns.tolist()
    thead = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>"
    rows = []
    for _, r in df.iterrows():
        tds = "".join(f"<td>{html.escape(str(r[c]))}</td>" for c in cols)
        rows.append(f"<tr>{tds}</tr>")
    return f"<table class='fv-table'><thead>{thead}</thead><tbody>{''.join(rows)}</tbody></table>"

def main(season:int, week:int, csv_path:Path, out_path:Path):
    df = pd.read_csv(csv_path)
    # Nice column order if present
    prefer = [c for c in ["game_id","team_home","team_away","model_spread","model_total"] if c in df.columns]
    other  = [c for c in df.columns if c not in prefer]
    df = df[prefer + other]

    table_html = df_to_html_table(df)

    # Page shell with unified nav.js
    # NOTE: games/ is one level under docs/, so src="../nav.js?v=6"
    title = f"Fourth & Value — Game Predictions (Week {week})"
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 0; background: #0b0f14; color: #e6edf3; }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 16px; }}
    h1 {{ font-size: 1.4rem; margin: 12px 0 8px 0; }}
    .meta {{ color: #8b949e; margin-bottom: 12px; }}
    table.fv-table {{ width: 100%; border-collapse: collapse; }}
    table.fv-table th, table.fv-table td {{ padding: 8px 10px; border-bottom: 1px solid #1f2630; font-size: 0.95rem; }}
    table.fv-table th {{ text-align: left; color: #a8b3c1; }}
    .note {{ margin: 12px 0; font-size: 0.9rem; color: #9fb0c4; }}
  </style>
</head>
<body>
  <div id="nav-root"></div>
  <script src="../nav.js?v=6"></script>

  <div class="wrap">
    <h1>Game Predictions — Week {week}</h1>
    <div class="meta">Season {season}. Model = Ridge baselines on rolling team stats (player → team → game).</div>

    {table_html}

    <div class="note">Tip: We’ll add market lines & edges next, plus filters and a Top Picks view.</div>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"[OK] wrote {out_path.as_posix()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/games/index.html"))
    args = ap.parse_args()
    main(args.season, args.week, args.csv, args.out)
