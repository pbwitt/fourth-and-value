#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_insights_page.py
Generates docs/props/insights.html from weekly AI commentary JSON.

Features:
- Dark theme consistent with the site (#0b1220).
- Game selector: pick a single game to view just that game's summary.
- "Week at a glance" summary (visible when All games is selected).
- Mobile-friendly layout.
- Share links: when a game is selected, a Share button and copy link appear
  (uses Web Share API on mobile; clipboard fallback elsewhere).
- Robust JSON loading (docs/data/ai → data/ai → data/ai cache), tolerant
  to slightly different JSON shapes.
"""

import argparse, json, sys, html
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOCS = ROOT / "docs"
AI_DATA_DOCS = DOCS / "data" / "ai"
AI_DATA_LOCAL = ROOT / "data" / "ai"

# ---------- JSON LOADING ----------

def guess_json_paths(season: int, week: int) -> List[Path]:
    return [
        p for p in [
            AI_DATA_DOCS / f"insights_week{week}.json",
            AI_DATA_LOCAL / f"insights_week{week}.json",
            AI_DATA_LOCAL / f"insights_cache_week{week}.json",
        ] if p.exists()
    ]

def coerce_items(obj: Any) -> List[Dict[str, Any]]:
    """
    Normalize to: [{"game": str, "summary": str, "picks": [str, ...]}, ...]
    """
    items: List[Dict[str, Any]] = []

    def pick_list(d: Dict[str, Any]) -> List[str]:
        # common variants
        for k in ("picks", "bullets", "cards", "top", "top_picks"):
            v = d.get(k)
            if isinstance(v, list):
                out: List[str] = []
                for x in v:
                    if isinstance(x, str):
                        s = x.strip()
                        if s:
                            out.append(s)
                    elif isinstance(x, dict):
                        txt = x.get("text") or x.get("line") or x.get("desc") or ""
                        txt = str(txt).strip()
                        if txt:
                            out.append(txt)
                return out
        # fallback
        v = d.get("lines")
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    # normalize top-level shapes
    if isinstance(obj, dict) and isinstance(obj.get("games"), list):
        src = obj["games"]
    elif isinstance(obj, list):
        src = obj
    elif isinstance(obj, dict):
        src = [{"game": g, **(v if isinstance(v, dict) else {"summary": str(v)})} for g, v in obj.items()]
    else:
        src = []

    for it in src:
        if not isinstance(it, dict):
            continue
        game = str(it.get("game") or it.get("matchup") or it.get("title") or "").strip()
        if not game:
            for k in ("home_vs_away", "game_name"):
                if k in it:
                    game = str(it[k]).strip()
                    break
        summary = str(it.get("summary") or it.get("commentary") or it.get("synopsis") or "").strip()
        picks = pick_list(it)
        if game or summary or picks:
            items.append({"game": game, "summary": summary, "picks": picks})

    return items

def load_insights_json(season: int, week: int, override: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[Path]]:
    paths: List[Path] = []
    if override:
        p = Path(override)
        if p.exists():
            paths = [p]
        else:
            print(f"[warn] override JSON not found: {p}", file=sys.stderr)
    if not paths:
        paths = guess_json_paths(season, week)
    if not paths:
        raise FileNotFoundError(
            "No insights JSON found. Looked for:\n"
            f"  {AI_DATA_DOCS}/insights_week{week}.json\n"
            f"  {AI_DATA_LOCAL}/insights_week{week}.json\n"
            f"  {AI_DATA_LOCAL}/insights_cache_week{week}.json"
        )
    src = paths[0]
    with src.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    week_overview = raw.get("week_overview", "") if isinstance(raw, dict) else ""
    items = coerce_items(raw)
    return items, week_overview, src

# ---------- HTML RENDERING ----------

STYLE = """
  :root { color-scheme: dark; }
  body {
    margin: 0;
    background: #0b0b0b;
    color: #e5e7eb;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
  }
  .wrap { max-width: 980px; margin: 0 auto; padding: 16px; }
  h1, h2, h3 { color: #fff; }
  .muted { color: #94a3b8; }
  .filterbar { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin: 12px 0 10px; }
  .filterbar label { font-weight:600; }
  .filterbar select {
    background:#0f172a; color:#e5e7eb; border:1px solid #27324a;
    border-radius:10px; padding:8px 10px; min-width: 220px;
  }
  .share {
    display:flex; gap:8px; align-items:center; flex-wrap:wrap;
  }
  .share button, .share input {
    background:#0f172a; color:#e5e7eb; border:1px solid #27324a;
    border-radius:10px; padding:8px 10px;
  }
  .share input {
    min-width: 220px;
  }
  .share button {
    cursor:pointer;
  }
  .week {
    border: 1px solid #27324a; background:#0f172a; border-radius:12px;
    padding:12px; margin: 6px 0 12px;
  }
  .week ul { margin: 8px 0 0 18px; }
  .game { margin-top: 16px; }
  .game p { line-height: 1.55; color: #cbd5e1; }
  .game ul { margin: 10px 0 0 18px; }
  .game li { margin: 6px 0; }
  @media (max-width: 720px) {
    .wrap { padding: 12px; }
    .filterbar select { min-width: 180px; }
    .share input { min-width: 180px; }
  }

  /* Minimal nav styles (for injected nav.js) */
  .site-header{position:sticky;top:0;z-index:50;background:rgba(11,11,11,.85);
    backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
    border-bottom:1px solid #27324a}
  .navbar{max-width:1100px;margin:0 auto;padding:10px 16px;display:flex;align-items:center;gap:12px}
  .nav-links{margin-left:auto;display:flex;flex-wrap:wrap;gap:8px}
  .nav-link{display:inline-flex;align-items:center;gap:6px;padding:8px 12px;border-radius:10px;text-decoration:none;color:#e5e7eb}
  .nav-link:hover{background:rgba(76,117,255,.14)}
  .nav-link.active{border:1px solid #4c74ff;background:rgba(76,117,255,.10)}
"""
SCRIPT = r"""
document.addEventListener('DOMContentLoaded', () => {
  const main = document.querySelector('main') || document.body;
  if (!document.querySelector('.wrap')) {
    const wrap = document.createElement('div');
    wrap.className = 'wrap';
    while (main.firstChild) wrap.appendChild(main.firstChild);
    main.appendChild(wrap);
  }
  const wrap = document.querySelector('.wrap');

  // Filter bar: Game select + counts + share controls
  const bar = document.createElement('div');
  bar.className = 'filterbar';
  bar.innerHTML = `
    <label class="muted" for="gameSel">Game:</label>
    <select id="gameSel" aria-label="Select a game"><option value="__all__">All games</option></select>
    <span id="count" class="muted"></span>
    <span class="share" id="shareBox" style="display:none;">
      <button id="shareBtn" type="button" title="Share this game">Share</button>
      <input id="shareUrl" readonly value="">
      <button id="copyBtn" type="button" title="Copy link">Copy</button>
    </span>
  `;
  const h1 = wrap.querySelector('h1');
  wrap.insertBefore(bar, h1 ? h1.nextSibling : wrap.firstChild);

  // Week-at-a-glance paragraph (shown only for "All games")
  const weekBox = document.createElement('section');
  weekBox.className = 'week';
  weekBox.id = 'weekBox';
  weekBox.innerHTML = `
    <strong>Week at a glance</strong>
    <p class="muted" id="weekLine"></p>
  `;
  wrap.insertBefore(weekBox, bar.nextSibling);
  const weekLine = weekBox.querySelector('#weekLine');

  // Identify game blocks and populate dropdown
  const blocks = Array.from(wrap.querySelectorAll('.game[data-key]'));
  const sel = document.getElementById('gameSel');
  blocks.forEach(b => {
    const opt = document.createElement('option');
    opt.value = b.dataset.key;
    opt.textContent = (b.querySelector('h3')?.textContent || b.dataset.key).trim();
    sel.appendChild(opt);
  });

  // Precompute totals
  const totalGames = blocks.length;
  const totalPicks = blocks.reduce((acc, b) => acc + b.querySelectorAll('li').length, 0);

  // Counts + Share controls
  const countSpan = document.getElementById('count');
  const shareBox = document.getElementById('shareBox');
  const shareBtn = document.getElementById('shareBtn');
  const copyBtn  = document.getElementById('copyBtn');
  const shareUrl = document.getElementById('shareUrl');

  function updateShare(slug, title) {
    const url = new URL(location.href);
    if (!slug || slug === '__all__') {
      url.searchParams.delete('game');
      shareUrl.value = '';
      shareBox.style.display = 'none';
      return;
    }
    url.searchParams.set('game', slug);
    shareUrl.value = url.toString();
    shareBox.style.display = '';
    shareBtn.onclick = async () => {
      try {
        if (navigator.share) {
          await navigator.share({ title: title || document.title, url: shareUrl.value });
        } else {
          await navigator.clipboard.writeText(shareUrl.value);
          shareBtn.textContent = 'Copied!';
          setTimeout(() => shareBtn.textContent = 'Share', 1200);
        }
      } catch (_) {}
    };
    copyBtn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(shareUrl.value);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => copyBtn.textContent = 'Copy', 1200);
      } catch (_) {}
    };
  }

  function applyFilter() {
  const v = sel.value;

  if (v === '__all__') {
    blocks.forEach(b => { b.style.display = 'none'; });
    weekBox.style.display = '';
    const weekOverview = wrap.dataset.weekOverview || '';
    if (weekOverview) {
      weekLine.textContent = weekOverview;
    } else {
      weekLine.textContent = `${totalGames} games • ${totalPicks} total picks. Select a game to view its summary.`;
    }
    countSpan.textContent = `${totalGames} games`;
    updateShare(null, null);
    return;
  }

  let shownName = '', shownPicks = 0;
  blocks.forEach(b => {
    const show = (b.dataset.key === v);
    b.style.display = show ? '' : 'none';
    if (show) {
      shownName = (b.querySelector('h3')?.textContent || '').trim();
      shownPicks = b.querySelectorAll('li').length;
    }
  });
  weekBox.style.display = 'none';
  countSpan.textContent = shownPicks > 0 ? `${shownPicks} picks` : ''; // hide when zero
  updateShare(v, shownName);
}


  sel.addEventListener('change', applyFilter);

  // Deep-link support: ?game=slug
  const params = new URLSearchParams(location.search);
  const want = params.get('game');
  if (want && Array.from(sel.options).some(o => o.value === want)) sel.value = want;

  applyFilter();
});
"""


def slugify(s: str) -> str:
    return (
        s.lower().strip()
         .replace("&", "and")
         .replace("@", " at ")
    ).replace("  ", " ").strip().replace(" ", "-")

def render_html(title: str, items: List[Dict[str, Any]], week_overview: str = "") -> str:
    total_picks = sum(len(it.get("picks", [])) for it in items)

    game_sections: List[str] = []
    for it in items:
        game = it.get("game", "").strip() or "Game"
        summary = it.get("summary", "").strip()
        picks = [p for p in it.get("picks", []) if str(p).strip()]
        key = slugify(game) or "game"

        parts = [f'<section class="game" data-key="{html.escape(key)}">', f'  <h3>{html.escape(game)}</h3>']
        if summary:
            parts.append(f'  <p>{html.escape(summary)}</p>')
        if picks:
            parts.append("  <ul>")
            parts.extend(f'    <li>{html.escape(str(p))}</li>' for p in picks)
            parts.append("  </ul>")
        parts.append("</section>")
        game_sections.append("\n".join(parts))

    sections_html = "\n".join(game_sections)

    # Add weekly overview as a hidden attribute for JavaScript
    week_overview_escaped = html.escape(week_overview) if week_overview else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>{STYLE}</style>
</head>
<body>
  <div id="nav-root"></div>

  <div class="wrap" data-week-overview="{week_overview_escaped}">
    <h1>{html.escape(title)}</h1>
    <p class="muted">AI-assisted commentary per game. Choose a matchup to view just that game's summary. Total picks: {total_picks}.</p>

    {sections_html}
  </div>

  <script src="../nav.js?v=25"></script>
  <script>{SCRIPT}</script>
</body>
</html>
"""

# ---------- CLI ----------

def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", help="Optional override path to insights JSON")
    args = ap.parse_args(argv)

    items, week_overview, src = load_insights_json(args.season, args.week, args.json)
    if not items:
        print("[warn] No items to render; page will say 0 picks.", file=sys.stderr)

    html_str = render_html(args.title, items, week_overview)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_str, encoding="utf-8")

    print(f"[ok] loaded commentary from {src.name if src else '(unknown)'} with {len(items)} games.")
    print(f"[ok] wrote {out_path}.")

if __name__ == "__main__":
    main(sys.argv[1:])
