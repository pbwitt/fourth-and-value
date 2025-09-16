# scripts/common_markets.py
# Safe, side-effect-free helpers shared across params/edges/site.

import re
import unicodedata
from typing import Optional

# -------- Aliases: bookmaker/feed keys -> canonical keys we use
ALIAS_MAP = {
    # Touchdowns
    "player_anytime_td":        "anytime_td",
    "player_1st_td":            "first_td",        # (unmodeled unless you add a model)
    "player_last_td":           "last_td",         # (unmodeled unless you add a model)

    # Receiving
    "player_reception_yds":     "recv_yds",
    "player_receptions":        "receptions",
    "player_reception_longest": "reception_longest",  # (unmodeled for now)

    # Rushing
    "player_rush_yds":          "rush_yds",
    "player_rush_attempts":     "rush_attempts",
    "player_rush_longest":      "rush_longest",       # (unmodeled for now)

    # Passing
    "player_pass_yds":          "pass_yds",
    "player_pass_attempts":     "pass_attempts",
    "player_pass_completions":  "pass_completions",
    "player_pass_tds":          "pass_tds",

    "player_pass_interceptions": "interceptions",
    # add these so params map too:


      "pass_interceptions": "interceptions",
      "interceptions_thrown": "interceptions",
      "ints": "interceptions",
      "interception": "interceptions",


}

PRETTY_MAP = {
    "rush_yds": "Rushing Yards",
    "recv_yds": "Receiving Yards",
    "receptions": "Receptions",
    "pass_yds": "Passing Yards",
    "rush_attempts": "Rushing Attempts",
    "pass_attempts": "Pass Attempts",
    "pass_completions": "Pass Completions",
    "pass_tds": "Passing TDs",
    "pass_interceptions": "Interceptions Thrown",
    "anytime_td": "Anytime TD",
    "reception_longest": "Longest Reception",
    "rush_longest": "Longest Rush",
    "1st_td": "First TD",
    "last_td": "Last TD",
}

# --- Modeled sets
OU_MARKETS_NORMAL = {
    "rush_yds", "recv_yds", "receptions", "pass_yds",
    "rush_attempts", "pass_attempts", "pass_completions",
}
POISSON_MARKETS = {"pass_tds", "pass_interceptions", "anytime_td"}
MODELED_MARKETS = OU_MARKETS_NORMAL | POISSON_MARKETS

def std_market(m: str) -> str:
    m = (m or "").strip().lower()
    if not m:
        return ""
    # first try explicit aliases
    if m in ALIAS_MAP:
        return ALIAS_MAP[m]
    # generic cleanup: many feeds just prefix with "player_"
    if m.startswith("player_"):
        m = m[len("player_"):]
    # normalize a few common synonyms
    if m in {"receiving_yds","reception_yds"}:
        return "recv_yds"
    if m in {"ints","interception","interceptions_thrown"}:
        return "interceptions"
    return m

def pretty_market(m: str) -> str:
    """Human-friendly label for UI."""
    m_std = std_market(m)
    return PRETTY_MAP.get(m_std, m_std.replace("_", " ").title())

# --- Player name normalization (for merge fallbacks)
_name_clean_re = re.compile(r"[^a-z0-9]+")

def std_player_name(s: Optional[str]) -> Optional[str]:
    """Lowercase, strip accents/punct/spaces for safer merges."""
    if s is None:
        return None
    s = (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    s = _name_clean_re.sub("", s)
    return s or None

# --- Over/Under label normalizer
def _norm_side(x: Optional[str]) -> Optional[str]:
    """Normalize Over/Under style labels to 'Over'/'Under'."""
    if x is None:
        return None
    t = str(x).strip().lower()
    if t in ("o", "over", "ovr", "+"):
        return "Over"
    if t in ("u", "under", "und", "-"):
        return "Under"
    return t.title()

# --- Frame normalizer used by multiple scripts
def standardize_input(
    df,
    market_col_candidates=("market_std", "market", "market_name", "market_key", "key", "market_slug"),
    name_col_candidates=("name", "side", "selection", "bet_name", "over_under", "bet_side"),
    point_col_candidates=("point", "line", "odds_point", "bet_line", "points"),
    player_col_candidates=("player", "player_name", "athlete", "athlete_name", "name_player"),
):
    """
    Light-touch frame normalizer used by params/edges.
    Ensures (if possible): market_std, name (OU/Yes/No), point (float), name_std, player_key.
    """
    import pandas as pd
    import re

    df = df.copy()

    # --- market_std (canonical, set ONCE) ---
    src = next((c for c in market_col_candidates if c in df.columns), None)
    if src:
        df["market_std"] = df[src].astype(str).map(lambda x: std_market(x.strip().lower()))
    else:
        df["market_std"] = ""

    # --- name (Over/Under/Yes/No) ---
    ncol = next((c for c in name_col_candidates if c in df.columns), None)
    if ncol:
        df["name"] = df[ncol].astype(str).map(_norm_side)
    elif "name" not in df.columns:
        df["name"] = pd.NA

    # --- point numeric ---
    pcol = next((c for c in point_col_candidates if c in df.columns), None)
    if pcol:
        df["point"] = pd.to_numeric(df[pcol], errors="coerce")
    elif "point" not in df.columns:
        df["point"] = pd.NA

    # --- player identifiers ---
    psrc = next((c for c in player_col_candidates if c in df.columns), None)
    if psrc and "name_std" not in df.columns:
        s = df[psrc].astype(str).str.lower()
        s = s.str.replace(r"[^a-z0-9\\s]", "", regex=True).str.replace(r"\\s+", " ", regex=True).str.strip()
        df["name_std"] = s
    if "name_std" not in df.columns:
        df["name_std"] = pd.NA

    if "player_key" not in df.columns and "name_std" in df.columns:
        df["player_key"] = df["name_std"].astype(str).str.replace(" ", "-", regex=False)

    return df


# --- Priors (conservative, season-agnostic) and helpers
PRIORS = {
    # Normal O/U
    "rush_yds": {"dist": "normal", "mu": 40.0, "sigma": 28.0},
    "recv_yds": {"dist": "normal", "mu": 35.0, "sigma": 25.0},
    "receptions": {"dist": "normal", "mu": 3.6, "sigma": 2.0},
    "pass_yds": {"dist": "normal", "mu": 235.0, "sigma": 45.0},
    "rush_attempts": {"dist": "normal", "mu": 10.0, "sigma": 4.0},
    "pass_attempts": {"dist": "normal", "mu": 33.0, "sigma": 6.0},
    "pass_completions": {"dist": "normal", "mu": 21.0, "sigma": 5.0},
    # Poisson
    "pass_tds": {"dist": "poisson", "lam": 1.4},
    "pass_interceptions": {"dist": "poisson", "lam": 0.7},
    "anytime_td": {"dist": "poisson", "lam": 0.35},
}
DEFAULT_PRIOR_NORMAL = {"dist": "normal", "mu": 25.0, "sigma": 20.0}
DEFAULT_PRIOR_POISSON = {"dist": "poisson", "lam": 0.30}

def ensure_param_schema(df):
    """Guarantee columns exist for params CSV."""
    import pandas as pd
    import numpy as np

    want = [
        "season",
        "week",
        "player",
        "player_key",
        "name_std",
        "market_std",
        "dist",
        "mu",
        "sigma",
        "lam",
        "used_logs",
    ]
    for col in want:
        if col not in df.columns:
            if col in ("mu", "sigma", "lam"):
                df[col] = np.nan
            elif col == "used_logs":
                df[col] = False
            else:
                df[col] = pd.NA

    # Fill dist from market if missing
    mask_na = df["dist"].isna()
    if mask_na.any():
        m = df.loc[mask_na, "market_std"].astype(str)
        df.loc[mask_na & m.isin(list(OU_MARKETS_NORMAL)), "dist"] = "normal"
        df.loc[mask_na & m.isin(list(POISSON_MARKETS)), "dist"] = "poisson"
    return df

def apply_priors_if_missing(df):
    """Fill missing mu/sigma/lam using market priors; keep existing values."""
    import numpy as np

    df = ensure_param_schema(df)

    # Determine dist per row if still missing
    markets = df["market_std"].astype(str).fillna("")
    need_dist = df["dist"].isna()
    if need_dist.any():
        df.loc[need_dist & markets.isin(list(OU_MARKETS_NORMAL)), "dist"] = "normal"
        df.loc[need_dist & markets.isin(list(POISSON_MARKETS)), "dist"] = "poisson"

    # Normal: fill mu/sigma
    mask_norm = df["dist"].astype(str).str.lower().eq("normal")
    if mask_norm.any():
        for mk in df.loc[mask_norm, "market_std"].dropna().unique():
            prior = PRIORS.get(mk, DEFAULT_PRIOR_NORMAL)
            m = mask_norm & (df["market_std"] == mk)
            if "mu" in prior:
                df.loc[m & df["mu"].isna(), "mu"] = prior["mu"]
            if "sigma" in prior:
                # floor sigma to avoid degenerate normals
                df.loc[m & df["sigma"].isna(), "sigma"] = max(0.1, float(prior["sigma"]))

    # Poisson: fill lam
    mask_poi = df["dist"].astype(str).str.lower().eq("poisson")
    if mask_poi.any():
        for mk in df.loc[mask_poi, "market_std"].dropna().unique():
            prior = PRIORS.get(mk, DEFAULT_PRIOR_POISSON)
            m = mask_poi & (df["market_std"] == mk)
            if "lam" in prior:
                # floor lambda to avoid zero-prob issues
                df.loc[m & df["lam"].isna(), "lam"] = max(0.01, float(prior["lam"]))

    # used_logs NaN -> False for clarity
    if "used_logs" in df.columns:
        df["used_logs"] = df["used_logs"].fillna(False)

    return df
