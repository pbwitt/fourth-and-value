#!/usr/bin/env python3
# Shared helpers for site pages (branding, nav, labels, formatters)

from datetime import datetime, timezone
import math

import re  # add this near the top with your other imports

# scripts/site_common.py
from pathlib import Path

import re
from pathlib import Path
import re

def _rel_to_nav(out_path: str) -> str:
    return '..' if '/props/' in Path(out_path).as_posix() else '.'

def write_with_nav(out_path: str, title: str, main_html: str,
                   active: str = None, extra_head: str = ""):
    rel = _rel_to_nav(out_path)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
{extra_head}<!-- preserved page-specific styles/links -->
</head>
<body>
  <div id="nav-root"></div>
  <script src="{rel}/nav.js?v=25"></script>
  {main_html}
</body>
</html>"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")





def _rel_to_nav(out_path: str) -> str:
    # If we are writing under docs/ (root pages) → './'
    # If under docs/*/* (e.g., docs/props/) → '../'
    p = Path(out_path).as_posix()
    return '..' if '/props/' in p else '.'

def write_with_nav_raw(out_path: str, title: str, full_html: str,
                       active: str = None):
    # --- keep important bits from the original <head> ---
    extra_head = ""
    mhead = re.search(r"<head[^>]*>(?P<h>[\s\S]*?)</head>", full_html, re.I)
    if mhead:
        h = mhead.group("h")
        keep = []
        # inline styles
        keep += re.findall(r"<style[^>]*>[\s\S]*?</style>", h, re.I)
        # css/icon links
        keep += re.findall(
            r"<link[^>]+rel=[\"'](?:stylesheet|icon|shortcut icon)[\"'][^>]*>",
            h, re.I
        )
        # color-scheme meta (dark mode)
        keep += re.findall(r"<meta[^>]+name=[\"']color-scheme[\"'][^>]*>", h, re.I)
        extra_head = "\n".join(keep)

    # --- extract body content (fallback: whole doc) ---
    mbody = re.search(r"<body[^>]*>(?P<b>[\s\S]*?)</body>", full_html, re.I)
    main_html = mbody.group("b") if mbody else full_html

    # --- remove any old nav hook to avoid duplicates ---
    main_html = re.sub(r'<div id="nav-root"></div>\s*', '', main_html, flags=re.I)
    main_html = re.sub(r'<script[^>]+nav\.js[^>]*></script>\s*', '', main_html, flags=re.I)

    # --- write with our standard shell (adds correct ./ or ../ nav.js) ---
    write_with_nav(out_path, title, main_html, active=active, extra_head=extra_head)






def normalize_display(x) -> str:
    """
    Make a value safe/nice for UI without changing intended casing:
    - convert underscores to spaces
    - collapse repeated whitespace
    - trim ends
    - normalize lone 'v' to 'vs' (common feed quirk)
    """
    if x is None:
        return ""
    s = str(x).replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    # normalize ' v ' → ' vs '
    s = re.sub(r"\bv\b", "vs", s, flags=re.IGNORECASE)
    return s


# ---------- Branding & Nav ----------
BRAND = "Fourth & Value"

def nav_html(active: str = "") -> str:
    def li(label, href, is_active):
        klass = "font-semibold text-white" if is_active else "text-gray-300 hover:text-white"
        return f'<a class="{klass} px-3 py-2 rounded-lg" href="{href}">{label}</a>'

    NAV_LINKS = [
           ("Home", "/index.html"),
        ("Insights", "/props/insights.html"),
        ("Props", "/props/index.html"),
       ("Top Picks", "/props/top.html"),
     ("Consensus", "/props/consensus.html"),
     ("Arbitrage", "/props/arbitrage.html"),
    ]
    items = "".join(li(lbl, href, active.lower()==lbl.lower()) for (lbl, href) in NAV_LINKS)

    # JS shim: if we’re on a GitHub Pages project site, prefix links with '/<repo>'
    return f"""
<header class="w-full sticky top-0 z-20 bg-neutral-900/85 backdrop-blur border-b border-neutral-800">
  <div class="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
    <div class="text-xl font-black tracking-tight text-white">{BRAND}</div>
    <nav class="flex gap-1">{items}</nav>
  </div>
</header>
<script>
(function() {{
  try {{
    var parts = location.pathname.split('/').filter(Boolean);
    // project sites look like /<repo>/...; user/org sites don't need this
    if (!parts.length) return;
    var base = '/' + parts[0];  // '/NFL-2025'
    // Only rewrite if link starts with '/' and isn't already prefixed
    document.querySelectorAll('header nav a[href^="/"]').forEach(function(a) {{
      var href = a.getAttribute('href');
      if (!href) return;
      if (href.indexOf(base + '/') === 0) return;  // already has prefix
      a.setAttribute('href', base + href);
    }});
  }} catch (e) {{}}
}})();
</script>
"""


# ---------- Market labels ----------
PRETTY_MAP = {
    "player_rush_yds": "Rushing Yards",
    "player_passing_yds": "Passing Yards",
    "player_pass_yds": "Passing Yards",
    "player_rec_yds": "Receiving Yards",
    "player_receptions": "Receptions",
    "player_anytime_td": "Anytime TD",
    "anytime_td": "Anytime TD",
    # feed variants
    "player reception yds": "Receiving Yards",
    "player reception yards": "Receiving Yards",
    "player rushing yds": "Rushing Yards",
    "player passing yds": "Passing Yards",
}

def pretty_market(m):
    if not m: return ""
    s = str(m).strip()
    key = s.lower().replace("_"," ")
    return PRETTY_MAP.get(key, s.replace("_"," ").title())

# ---------- Odds/percent helpers ----------
def fmt_odds_american(x):
    """Format American odds with sign (+150 / -110)."""
    if x is None or (isinstance(x, float) and math.isnan(x)): return ""
    try:
        v = int(round(float(x)))
        return f"{v:+d}"
    except Exception:
        return str(x)

# alias for older scripts
def fmt_odds(x):  # noqa
    return fmt_odds_american(x)

def american_to_prob(o):
    """Implied probability from American odds (no vig). Returns float in [0,1] or NaN."""
    try:
        v = float(o)
    except Exception:
        return float("nan")
    if math.isnan(v): return float("nan")
    return 100.0/(v+100.0) if v > 0 else (-v)/((-v)+100.0)

def prob_to_american(p):
    """
    Convert implied probability (0–1) to American odds.
    Returns int odds (e.g. +150, -120), or None if invalid.
    Clamps extreme probabilities to avoid mathematical overflow.
    """
    try:
        v = float(p)
    except Exception:
        return None
    if math.isnan(v):
        return None
    # Clamp to safe range: (0.01, 0.99) to prevent extreme odds
    # 0.01 → +9900, 0.99 → -9900 (reasonable max odds)
    v = max(min(v, 0.99), 0.01)
    # Underdog (prob < 0.5) → positive odds
    if v < 0.5:
        return int(round(100 * (1 - v) / v))
    # Favorite (prob > 0.5) → negative odds
    return int(round(-v * 100 / (1 - v)))

def fmt_pct(x):
    """Accepts 0–1 or 0–100 and returns '74.1%' style string; '' if invalid."""
    try:
        v = float(x)
    except Exception:
        return ""
    if math.isnan(v): return ""
    if 0.0 <= v <= 1.0:
        p = v * 100.0
    elif 1.0 < v <= 100.0:
        p = v
    else:
        return ""
    return f"{p:.1f}%"

# ---------- Kickoff formatter ----------
def kickoff_et(iso_or_str):
    """Best-effort: format to 'Sun 1:00 PM ET'. Safe if already formatted."""
    if not iso_or_str: return ""
    s = str(iso_or_str)
    # If already pretty, return as-is
    if (" ET" in s and any(ch.isalpha() for ch in s[:3])) or ("M ET" in s):
        return s
    try:
        dt = datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(timezone.utc)
        # Show day-of-week and time; label as ET for consistency across pages
        return dt.strftime("%a %-I:%M %p ET")
    except Exception:
        return s

# alias for older scripts
def to_kick_et(x):  # noqa
    return kickoff_et(x)


# ---- Permanent nav injector (append this to site_common.py) ----
import re
# Matches your existing <header> nav block generated by nav_html
_NAV_HEADER_RE = re.compile(r"<header[^>]*backdrop-blur[^>]*>[\s\S]*?</header>", re.I)

def inject_nav(html: str, active: str) -> str:
    """
    Permanently inject or replace the site's nav into an HTML document.
    - Replaces __NAV__ placeholder if present.
    - Otherwise replaces any existing <header ...>...</header> that looks like our nav.
    - Otherwise inserts right after <body>.
    """
    nav = nav_html(active=active)

    # 1) Explicit placeholder
    if "__NAV__" in html:
        return html.replace("__NAV__", nav)

    # 2) Replace an existing header that matches our nav structure
    if _NAV_HEADER_RE.search(html):
        return _NAV_HEADER_RE.sub(nav, html)

    # 3) Inject after <body> tag
    m = re.search(r"<body[^>]*>", html, re.I)
    if m:
        i = m.end()
        return html[:i] + "\n" + nav + html[i:]

    # Fallback: prepend
    return nav + "\n" + html
