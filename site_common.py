"""Compatibility wrapper to expose site_common utilities at the repository root."""
BRAND = "Fourth & Value"

__all__ = [
    "BRAND",
    "american_to_prob",
    "fmt_odds_american",
    "fmt_pct",
    "kickoff_et",
    "pretty_market",
    "write_with_nav_raw",
]

import math

from pathlib import Path as _Path

def _nav_script_for(out_path: str) -> str:
    p = _Path(out_path)
    rel = "../nav.js?v=6" if p.parent.name == "props" else "./nav.js?v=6"
    return f'<script src="{rel}"></script>'

def write_with_nav_raw(out_path: str, inner_html: str, active: str = "") -> None:
    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Fourth & Value</title>
<style>
  body {{ background:#0f1418; color:#e5e7eb; margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
  a {{ color: inherit; }}
</style>
</head><body data-active="{active}">
  <div id="nav-root"></div>
  {inner_html}
  {_nav_script_for(out_path)}
</body></html>"""
    _Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _Path(out_path).write_text(html, encoding="utf-8")

def fmt_odds_american(x):
    try:
        if x is None:
            return "—"
        v = float(x)
    except Exception:
        return "—"
    if v != v:  # NaN
        return "—"
    v_int = int(round(v))
    if v_int > 0:
        return f"+{v_int}"
    if v_int < 0:
        return str(v_int)
    # 0 shouldn't occur; treat as EVEN
    return "EVEN"


from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

def kickoff_et(iso_or_str):
    if not iso_or_str:
        return ""
    s = str(iso_or_str).strip()
    # tolerate 'Z' suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        # last-ditch: try no tz
        try:
            dt = datetime.fromisoformat(s.split('.')[0])
        except Exception:
            return str(iso_or_str)
    try:
        if dt.tzinfo is None:
            # assume UTC if naive
            dt = dt.replace(tzinfo=ZoneInfo("UTC")) if ZoneInfo else dt
        dt_et = dt.astimezone(ZoneInfo("America/New_York")) if ZoneInfo else dt
    except Exception:
        dt_et = dt
    # Example: "Sun 1 p.m"
    day = dt_et.strftime("%a")
    hour = dt_et.strftime("%-I") if "%" in "%-I" else dt_et.strftime("%I").lstrip("0") or "12"
    ampm = dt_et.strftime("%p").lower().replace("am","a.m").replace("pm","p.m")
    return f"{day} {hour} {ampm}"


def fmt_pct(x, digits=1):
    try:
        v = float(x)
        if v != v:
            return "—"
        return f"{v*100:.{digits}f}%"
    except Exception:
        return "—"


_PRETTY_MAP = {
    "recv_yds": "Receiving Yards",
    "rush_yds": "Rushing Yards",
    "pass_yds": "Passing Yards",
    "receptions": "Receptions",
    "rush_attempts": "Rush Attempts",
    "pass_attempts": "Pass Attempts",
    "pass_completions": "Pass Completions",
    "pass_tds": "Passing TDs",
    "pass_interceptions": "Interceptions",
    "anytime_td": "Anytime TD",
}

def pretty_market(m):
    if not m:
        return ""
    key = str(m).strip().lower()
    return _PRETTY_MAP.get(key, m.replace("_", " ").title())


BRAND = "Fourth & Value"
def american_to_prob(odds):
    """
    Convert American odds to implied probability.
    Returns float in [0,1], or None if odds invalid.
    """
    try:
        o = float(odds)
    except Exception:
        return None
    if math.isnan(o):
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    elif o < 0:
        return -o / (-o + 100.0)
    else:
        return None

def prob_to_american(p):
    """
    Convert implied probability (0–1) to American odds.
    Returns int odds (e.g. +150, -120), or None if invalid.
    """
    try:
        v = float(p)
    except Exception:
        return None
    if v <= 0 or v >= 1 or math.isnan(v):
        return None
    # Underdog (prob < 0.5) → positive odds
    if v < 0.5:
        return int(round(100 * (1 - v) / v))
    # Favorite (prob > 0.5) → negative odds
    return int(round(-v * 100 / (1 - v)))
BRAND = "Fourth & Value"
