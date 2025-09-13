#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, re, math
from html import escape
from pathlib import Path
from site_common import inject_nav
from site_common import write_with_nav_raw
 # add this
# shared helpers
from site_common import (
    nav_html, pretty_market, fmt_odds_american, fmt_pct,
    american_to_prob, kickoff_et, BRAND
)

# -------- helpers local to this script --------
LINE_CANDIDATES = [
    "line_disp","point","line","market_line","prop_line","number","threshold","total","line_number",
    "handicap","spread","yards","receptions","receiving_yards","rushing_yards","passing_yards","prop_total"
]
GAME_CANDIDATES = ["game","Game","matchup","matchup_name","matchup_display"]

def _num(x):
    with np.errstate(all="ignore"):
        return pd.to_numeric(x, errors="coerce")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def _first_nonnull(row, cols):
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return row[c]
    return np.nan

def _is_numeric_total_market(mkt) -> bool:
    m = pretty_market(mkt or "").lower()
    if not m: return False
    if "yards" in m: return True
    return m in {"receptions","rush attempts","pass attempts","completions","passing touchdowns","rushing attempts"}

def mk_line_disp(r):
    # keep if already nice
    if "line_disp" in r and str(r["line_disp"]).strip():
        return str(r["line_disp"]).strip()
    raw_line = _first_nonnull(r, LINE_CANDIDATES)
    line_txt = ""
    if pd.notna(raw_line):
        try: line_txt = f"{float(raw_line):g}"
        except Exception: line_txt = str(raw_line).strip()
    side_raw = str(r.get("name") or r.get("side") or "").strip()  # Over/Under/Yes/No
    side = side_raw
    # totals: Yes/No → Over/Under
    if line_txt and _is_numeric_total_market(r.get("market")):
        if side_raw.lower() == "yes": side = "Over"
        elif side_raw.lower() == "no": side = "Under"
    # Anytime TD special "No Scorer"
    mkt_lbl = pretty_market(r.get("market","")).lower()
    player  = str(r.get("player","")).strip().lower()
    if ("anytime td" in mkt_lbl or "anytime touchdown" in mkt_lbl) and player in {"no scorer","no td scorer","no touchdown scorer"}:
        return "No Scorer"
    # assemble
    if side and line_txt: return f"{side} {line_txt}"
    return side or line_txt or ""

def kickoff_col(df):
    if "kick_et" in df.columns: return "kick_et"
    if "commence_time" in df.columns: return "commence_time"
    return None

def read_df(path):
    df = pd.read_csv(path)

    # normalize
    if "book" not in df.columns and "bookmaker" in df.columns:
        df["book"] = df["bookmaker"]

    for col in ["price"]:
        if col in df.columns: df[col] = _num(df[col])

    # kickoff
    kc = kickoff_col(df)
    if kc == "commence_time": df["kick_et"] = df["commence_time"]

    # ensure presence
    for col in ["player","market","book","home_team","away_team","name"]:
        if col not in df.columns: df[col] = np.nan
    df["name"] = df["name"].fillna("")

    # line display & game label
    df["line_disp"] = df.apply(mk_line_disp, axis=1)

    def mk_game(row):
        for c in GAME_CANDIDATES:
            if c in df.columns:
                v = row.get(c)
                if pd.notna(v) and str(v).strip(): return str(v).strip()
        away = str(row.get("away_team") or "").strip()
        home = str(row.get("home_team") or "").strip()
        return f"{away} vs {home}" if away and home else (away or home or "")
    df["game_disp"] = df.apply(mk_game, axis=1)

    # implied prob from book price
    df["imp_prob"] = df["price"].apply(american_to_prob) if "price" in df.columns else np.nan

    # consensus prob by (player, market, game, line_disp)
    key = ["player","market","game_disp","line_disp"]
    grp = df.groupby(key, dropna=False)["imp_prob"].median().rename("consensus_prob")
    df = df.merge(grp, on=key, how="left")

    # edge vs consensus (positive if book is better than consensus)
    df["consensus_edge_bps"] = 10000.0 * (df["consensus_prob"] - df["imp_prob"])

    # best book per bet (max edge)
    df["_edge_sort"] = df["consensus_edge_bps"].astype(float).where(df["consensus_edge_bps"].notna(), -1e15)
    df = (df.sort_values(key + ["_edge_sort"], ascending=[True,True,True,True,False])
            .drop_duplicates(subset=key, keep="first")
            .drop(columns=["_edge_sort"])
            .copy())

    # normalized attrs for filters if you add later
    df["_mkt_norm"]  = df["market"].apply(lambda m: _norm(pretty_market(m)))
    df["_game_norm"] = df["game_disp"].apply(_norm)
    df["_book_norm"] = df["book"].apply(_norm)

    return df

# -------- rendering --------
def row_html(r):
    time = kickoff_et(r.get("kick_et",""))
    game = str(r.get("game_disp",""))
    player = str(r.get("player",""))
    market = pretty_market(r.get("market",""))
    bet = []
    ld = str(r.get("line_disp","")).strip()
    if ld: bet.append(ld)
    odds = fmt_odds_american(r.get("price"))
    if odds: bet.append(f"@ {odds}")
    book = str(r.get("book",""))
    if book: bet.append(f"on {book}")
    bet_txt = " ".join(bet)

    cons_p = fmt_pct(r.get("consensus_prob"))
    edge   = r.get("consensus_edge_bps")
    edge_txt = "" if (edge is None or (isinstance(edge,float) and math.isnan(edge))) else f"{edge:,.0f} bps"

    return f"""<tr>
      <td>{escape(str(time))}</td>
      <td class="game">{escape(game)}</td>
      <td class="player">{escape(player)}</td>
      <td>{escape(market)}</td>
      <td class="bet">{escape(bet_txt)}</td>
      <td class="cons">{escape(cons_p)}</td>
      <td class="edge">{escape(edge_txt)}</td>
    </tr>"""

def html_page(rows_html, title):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{escape(title)}</title>
<link rel="icon" href="data:,">
<style>
:root {{ color-scheme: dark }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#0b0b0c; color:#e7e7ea; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Inter,Roboto,Ubuntu,Helvetica,Arial,sans-serif; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 18px 16px 32px; }}
.h1 {{ font-size: clamp(22px,3.5vw,28px); font-weight:900; color:#fff; margin: 4px 0 10px; }}

.tablewrap {{ overflow:auto; border:1px solid #1f1f22; border-radius:14px; }}
table {{ width:100%; border-collapse: collapse; min-width: 900px; }}
thead th {{ text-align:left; font-weight:700; font-size:12px; color:#b7b7bb; padding:10px 12px; background:#111113; position:sticky; top:0; }}
tbody td {{ border-top:1px solid #1f1f22; padding:10px 12px; font-size:13px; }}
tbody tr:hover {{ background:#0f0f11; }}
td.game {{ color:#c8c8cd; }}
td.player {{ font-weight:700; color:#fff; }}
td.bet {{ color:#e3e3e6; }}
td.cons, td.edge {{ white-space:nowrap; }}

.note {{ margin:10px 0 16px; color:#b7b7bb; font-size:13px; }}
</style>
</head>
<body>
__NAV__
<main class="container">
  <div class="h1">{escape(title)}</div>
  <div class="note">Market consensus = median implied probability across books for the same bet. “Edge” shows how favorable the best book is versus that market consensus.</div>
  <div class="tablewrap">
    <table>
      <thead>
        <tr><th>Kick</th><th>Game</th><th>Player</th><th>Market</th><th>Best book bet</th><th>Market consensus</th><th>Edge</th></tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</main>
</body>
</html>
"""
# ... after you finish assembling the final HTML into `html`

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--title", default=f"{BRAND} — Consensus vs Best Book")
    ap.add_argument("--limit", type=int, default=3000)
    args = ap.parse_args()

    # 1) Build the table
    df = read_df(args.merged_csv)
    df = df.sort_values(by="consensus_edge_bps", ascending=False).head(args.limit)
    rows = "\n".join(row_html(r) for _, r in df.iterrows())

    # 2) Build the full page HTML (no manual nav replace)
    page = html_page(rows, args.title)

    # 3) Inject nav permanently, then write once
    from site_common import inject_nav
    page = inject_nav(page, active="Consensus")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_with_nav_raw(
        out_path.as_posix(),
        (getattr(args, "title", None) or f"Fourth & Value — Consensus (Week {args.week})"),
        page,                     # ← this is your final HTML string
        active="Consensus",
    )




    print(f"[consensus] wrote {args.out} with {len(df)} rows (from {len(read_df(args.merged_csv))} source rows)")

if __name__ == "__main__":
    main()
