# NHL Integration — Implementation Plan

**Status:** Design Complete → Ready for Phase A Implementation
**Author:** Claude (Lead Dev)
**Date:** 2025-10-08
**Approval:** Pending user review

---

## Executive Summary

Add NHL player props (SOG, Goals, Assists, Points) and game lines (Moneyline, Totals, Spread) to Fourth & Value with **zero NFL regressions**. Isolated data/build paths, shared navigation, phased rollout.

---

## Architecture Decisions

### 1. Directory Structure (Approved)

```
data/
  nhl/
    raw/           # Odds API dumps, NHL API responses
    processed/     # Normalized skater/goalie logs, schedule
    props/         # props_with_model_{date}.csv
    games/         # games_with_model_{date}.csv
    consensus/     # consensus_{date}.csv
scripts/
  nhl/
    fetch_nhl_odds.py
    fetch_nhl_stats.py
    make_nhl_params.py
    make_nhl_consensus.py
    make_nhl_edges.py
    build_nhl_props_site.py
docs/
  nhl/
    props/
      index.html
      top.html
      consensus.html
    games/
      index.html
data/ai/calibration/nhl/{market}/
```

### 2. Canonical Schemas

#### Props Table (`props_with_model_nhl_{date}.csv`)
```python
# Shared keys
sport, game_id, game_date, puck_et, team, opp, home_away
player, player_key, name_std, name_slug
market_std, side, point, book, bookmaker_title, price

# Model outputs
mu, sigma, lam  # Distribution parameters
model_prob, mkt_prob, fair_odds, edge_bps
model_line  # μ for Normal, λ for Poisson
model_meta  # JSON: {model, features_used, calibration_id, flags}

# Consensus
consensus_line, consensus_prob, book_count

# Display
line_disp, model_line_disp, consensus_line_disp
mkt_odds, mkt_pct, model_pct, consensus_pct
kick_et, book_count_disp
is_consensus  # bool (for UI filtering)
```

#### Game Lines Table (`games_with_model_nhl_{date}.csv`)
```python
# Game keys
sport, game_id, game_date, puck_et, home_team, away_team, venue

# Market: moneyline
market, side  # ("moneyline", "home"|"away")
price, mkt_prob, model_prob, fair_odds, edge_bps

# Market: totals
market, side  # ("total", "over"|"under")
total_goals, price, mkt_prob, model_prob, edge_bps
model_total_mu, model_total_sigma  # From bivariate Poisson

# Market: spread (puckline)
market, side  # ("spread", "home"|"away")
spread, price, mkt_prob, model_prob, edge_bps

# Consensus
consensus_price, consensus_line, book_count

# Model internals
model_meta  # JSON: {model, elo_home, elo_away, goalie_home, goalie_away}
```

### 3. NHL Market Codes (Odds API)

**Phase A (MVP):**
- `player_shots_on_goal` → `sog`
- `player_goal` → `goals`
- `player_assists` → `assists`
- `player_points` → `points`
- `h2h` → `moneyline`
- `totals` → `total`
- `spreads` → `spread`

**Phase B:**
- `player_goalie_saves` → `goalie_saves`
- Team props (team_total_goals, etc.)

---

## Phase A: MVP (Week 1)

### Deliverables

1. ✅ **NHL Odds Adapter** (`scripts/nhl/fetch_nhl_odds.py`)
   - Fetch from Odds API (`sport_key="icehockey_nhl"`)
   - Markets: `player_shots_on_goal, player_goal, player_assists, player_points, h2h, totals, spreads`
   - Output: `data/nhl/raw/odds_{date}.csv` (raw) → `data/nhl/processed/odds_normalized_{date}.csv` (canonical schema)
   - Rate limiting: 0.5s between requests, cache responses 24hr

2. ✅ **NHL Stats Adapter** (`scripts/nhl/fetch_nhl_stats.py`)
   - Fetch skater logs from NHL Stats API (last 30 days)
   - Fetch schedule with rest_days/b2b computation
   - Supplement with MoneyPuck CSVs (xG, pace, goalie sv%)
   - Output: `data/nhl/processed/skater_logs_{date}.parquet`, `goalie_logs_{date}.parquet`, `schedule_{date}.csv`

3. ✅ **SOG Model** (`scripts/nhl/models/sog_model.py`)
   - **Type:** Normal distribution (μ/σ regression)
   - **Features:** `toi_ev_60, toi_pp_60, icf60, line_slot, pp_unit, opp_sog_allowed60, b2b_flag, goalie_ev_sv_pct`
   - **Training:** Last 3 seasons, EMA (α=0.30) on last 10 games
   - **Output:** `mu, sigma` per player-game
   - **Coverage target:** ≥80%

4. ✅ **Params Builder** (`scripts/nhl/make_nhl_params.py`)
   - Load skater logs + opponent/goalie features
   - Predict SOG params (μ/σ) using trained model
   - Write: `data/nhl/processed/params_{date}.csv`

5. ✅ **Consensus Calculator** (`scripts/nhl/make_nhl_consensus.py`)
   - Group by `(name_std, market_std, side)` for props
   - Group by `(game_id, market, side)` for games
   - Compute median line, median price-implied prob, book_count
   - Write: `data/nhl/consensus/consensus_props_{date}.csv`, `consensus_games_{date}.csv`

6. ✅ **Edges Calculator** (`scripts/nhl/make_nhl_edges.py`)
   - Join params + odds + consensus (100% match or fail)
   - Compute `model_prob` from μ/σ or λ
   - Compute `fair_odds = prob_to_american(model_prob)`
   - Compute `edge_bps = (model_prob - mkt_prob) * 10000`
   - Write: `data/nhl/props/props_with_model_{date}.csv`

7. ✅ **Props Page** (`scripts/nhl/build_nhl_props_site.py`)
   - Clone `build_props_site.py` structure
   - Table view (desktop) + card view (mobile)
   - Columns: Player, Market, Side, Model Line, Book Line, Cons. Line, Model %, Edge, Book/Odds, Puck Drop
   - Add 🔥 indicator when Model % ≥85%
   - Write: `docs/nhl/props/index.html`

8. ✅ **QC Report** (`scripts/nhl/qc_nhl.py`)
   - Check: freshness (≤24hr), coverage (SOG ≥80%), joins (100%), edge sanity (p95 <3500bps)
   - Write: `data/qc/run_nhl_{date}.json`
   - Block publish if `overall.status != "pass"`

9. ✅ **Makefile Targets**
   ```makefile
   nhl_daily:
   	python3 scripts/nhl/fetch_nhl_odds.py --date $(DATE)
   	python3 scripts/nhl/fetch_nhl_stats.py --date $(DATE)
   	python3 scripts/nhl/make_nhl_params.py --date $(DATE)
   	python3 scripts/nhl/make_nhl_consensus.py --date $(DATE)
   	python3 scripts/nhl/make_nhl_edges.py --date $(DATE)
   	python3 scripts/nhl/qc_nhl.py --date $(DATE)
   	python3 scripts/nhl/build_nhl_props_site.py --date $(DATE)
   ```

### Acceptance Criteria (Phase A)

- [ ] QC report shows: SOG coverage ≥80%, all joins 100%, edge p95 <3500bps
- [ ] `/nhl/props/index.html` loads with ≥500 rows
- [ ] No diffs in `/docs/props/` (NFL untouched)
- [ ] Mobile view shows Model Line first, 🔥 on high confidence
- [ ] Consensus filter works (same as NFL)

---

## Phase B: Goals + Game Lines (Week 2)

### Deliverables

1. **Goals Model** (Zero-Inflated Poisson)
2. **Assists/Points Models** (Poisson)
3. **Moneyline Model** (Logistic baseline with Elo + rest features)
4. **Calibration System** (isotonic for SOG, Platt for moneyline)
5. **Top Picks Page** (`/nhl/props/top.html`)
6. **Consensus Page** (`/nhl/props/consensus.html`)

---

## Phase C: Advanced Models + Totals (Week 3)

### Deliverables

1. **Totals Model** (Bivariate Poisson with learned ρ)
2. **Spread Model** (Skellam on score diff)
3. **Game Lines Page** (`/nhl/games/index.html`)
4. **Drift Detection** (weekly KS test on residuals)
5. **Model Cards** (SOG, Goals, ML, Totals assumptions/limits)

---

## Calibration System (All Phases)

### Storage Format

**Path:** `data/ai/calibration/nhl/{market}/{method}_{version}.json`

**Example:** `data/ai/calibration/nhl/sog/sog_iso_2025w06.json`

**Schema:**
```json
{
  "schema_version": "1.0",
  "sport": "nhl",
  "market": "sog",
  "method": "isotonic",
  "model_id": "sog_mvp_v03",
  "trained_on": {"start": "2025-01-01", "end": "2025-02-15"},
  "ts_utc": "2025-02-16T03:12:45Z",
  "sample_size": 1823,
  "bins": [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0],
  "mapping": [[0.12, 0.14], [0.22, 0.24], ...],
  "bin_counts": [210, 195, 188, ...],
  "metrics": {"ece_pre": 0.089, "ece_post": 0.032},
  "bounds": {"out_of_range": "clip", "min": 0.001, "max": 0.999},
  "interpolation": "step",
  "valid_from": "2025-02-16",
  "valid_to": "2025-03-16"
}
```

### Application Logic

```python
def apply_calibration(raw_prob, calib_file):
    calib = json.load(open(calib_file))
    bins = calib["bins"]
    mapping = calib["mapping"]

    # Find bin
    idx = bisect.bisect_right(bins, raw_prob) - 1
    idx = max(0, min(idx, len(mapping) - 1))

    # Apply mapping
    if calib["interpolation"] == "step":
        return mapping[idx][1]
    elif calib["interpolation"] == "linear":
        # Linear interpolate between surrounding bins
        ...

    # Handle out-of-range
    if raw_prob < bins[0] or raw_prob > bins[-1]:
        return np.clip(raw_prob, calib["bounds"]["min"], calib["bounds"]["max"])
```

---

## QC JSON Schema (Blocking Rules)

**Path:** `data/qc/run_nhl_{date}.json`

**Structure:**
```json
{
  "run_id": "nhl_2025-10-08_am",
  "sport": "nhl",
  "ts_utc": "2025-10-08T20:05:12Z",
  "version": {"pipeline": "2025.10.08-claude-a1", "models": {"sog": "sog_mvp_v03"}},
  "inputs": {
    "odds_dump": {"path": "...", "mtime": "2025-10-08T18:59:31Z"},
    "skater_logs": {"path": "...", "mtime": "2025-10-08T18:40:02Z"}
  },
  "coverage": {
    "props": {"rows_total": 1000, "rows_modeled": 850, "share_modeled": 0.85}
  },
  "joins": {
    "params_join": {"attempted": 1000, "matched": 1000, "match_rate": 1.0}
  },
  "edges": {
    "props": {"edge_bps_p95": 2800, "outlier_count_over_4000": 0}
  },
  "overall": {"status": "pass", "errors": [], "warnings": []}
}
```

**Blocking Rules:**
- Fail if `inputs[*].mtime` > 24hr stale
- Fail if `coverage.props.share_modeled` < 0.80 for SOG
- Fail if `joins.*.match_rate` < 1.0
- Fail if `edges.props.edge_bps_p95` > 3500

---

## Navigation Updates

### Shared Nav (`docs/nav.js`)

Add NHL links:
```javascript
const pages = [
  { href: `${base}/index.html`, label: 'Home' },
  { href: `${base}/props/insights.html`, label: 'Insights' },
  { href: `${base}/props/index.html`, label: 'NFL Props' },  // Clarify
  { href: `${base}/nhl/props/index.html`, label: 'NHL Props' },  // NEW
  { href: `${base}/props/top.html`, label: 'Top Picks' },
  { href: `${base}/arbitrage.html`, label: 'Arbitrage' },
  { href: `${base}/methods.html`, label: 'Methods' },
  { href: `${base}/blog/`, label: 'Blog' },
];
```

### Homepage Update

Change tagline:
```html
<h1>Sharper NFL & NHL bets, minus the noise.</h1>
<p class="lead">
  We build probability models from historical sports data—tracking player performance,
  matchups, and game contexts—to derive fair odds and identify edges against sportsbook lines.
</p>
```

---

## Risk Mitigation

### Risk 1: NHL API Rate Limits
- **Mitigation:** Cache all responses 24hr, 0.5s delay between requests
- **Fallback:** Use MoneyPuck if NHL API throttles

### Risk 2: Missing Linemate Data
- **Mitigation:** Fall back to team PP stats, flag as `linemates_confidence: low`
- **QC:** Report linemate coverage separately

### Risk 3: Goalie Confirmation Timing
- **Mitigation:** Run pipeline 6am (projected) + 12pm (confirmed refresh)
- **UI:** Show "Projected starter" disclaimer if `goalie_confirmed=False`

### Risk 4: Model Overconfidence (99% predictions)
- **Mitigation:** Cap displayed probs at 95% (log raw in backend)
- **UI:** Add 🔥 indicator + "Strong signal—no guarantees" tooltip

---

## Success Metrics

### Phase A (MVP)
- **Launch:** /nhl/props/ live with ≥500 modeled props
- **Coverage:** SOG ≥80%, Goals ≥70%
- **Calibration:** ECE ≤0.05 (holdout test)
- **User feedback:** Collect via X for 1 week

### Phase B
- **Coverage:** All 4 markets ≥70%
- **Moneyline Brier:** ≤0.22 (vs coin flip 0.25)
- **Top Picks:** ≥20 consensus picks daily

### Phase C
- **Totals RMSE:** ≤1.2 goals
- **Spread Calibration:** P(Cover) within ±5% of empirical
- **Drift:** No KS warnings for 2 weeks

---

## Next Steps

1. **Review this plan** - Confirm approach, flag any concerns
2. **Implement Phase A** - Start with `fetch_nhl_odds.py` adapter
3. **Test with live data** - Run pipeline on today's slate
4. **Iterate QC thresholds** - Tune coverage/edge caps based on real data
5. **Launch soft preview** - Share `/nhl/props/` link on X for feedback

---

**Ready to proceed? Confirm approval and I'll start with the NHL odds adapter.**
