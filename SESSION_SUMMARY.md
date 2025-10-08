# Session Summary - Oct 7, 2025

## What We Fixed Today

### 1. Consensus Calculation Bug ✅
**Problem:** Consensus was counting Over+Under as separate books, showing "3 books" when only 1 book offered both sides.

**Fix:**
- Calculate consensus per `player + market + SIDE` (not just player + market)
- Count unique books per side using `pd.Series.nunique()`
- De-vig probabilities within each book×player×market group

**Result:** Cam Ward now correctly shows 1 book, Breece Hall shows 4 books with median line of 15.5

### 2. Added Side (Over/Under) Column ✅
**Problem:** Player Props page was missing the Side column, making it unclear which direction the bet was.

**Fix:**
- Added `name` column (contains "Over", "Under", "Yes") to props page
- Updated table headers and row rendering
- Added to data keep list in `build_props_site.py`

### 3. Blog Post: "Why Sharps Bet Early" ✅
**Created:** New blog post explaining the timing edge:
- Early lines are softer (books still learning)
- Lower vig on Tuesday vs Sunday
- Optionality to hedge/middle
- Live example: Saquon Barkley Over 2.5 receptions
- Best price callout: Fanatics -110

**URL:** https://fourthandvalue.com/blog/early-lines-sharp-advantage.html

### 4. Added Model Line to Player Props ✅
**Problem:** Users couldn't see the model's predicted line, only probabilities.

**Fix:**
- Added `model_line_disp` column (formatted mu from params)
- Reordered columns: **Book Line | Model Line | Consensus Line**
- Makes it easy to compare all three at a glance

**Example:** Saquon 2.5 (book) | 3.8 (model) | 2.5 (consensus)

### 5. Fixed Insights Page Background ✅
**Problem:** Background was `#0b1220` instead of site standard `#0b0b0b`

**Fix:** Updated both `docs/props/insights.html` and `scripts/build_insights_page.py` template

### 6. CRITICAL: Fixed L4 Data Source ✅
**Problem:** Model was using 2024 playoff data instead of 2025 current season L4!

**Root cause:**
- Missing `targets` column in `fetch_recent_game_logs()` needed columns
- This caused receive family to fall back to default CR=0.65
- Would have incorrectly used 2024 data for early season

**Fix:**
- Added `"targets"` to needed columns list
- Removed logic that would use previous season
- **Now correctly uses 2025 Weeks 1-4 for Week 6 predictions**

**Impact on Saquon:**
- **Before:** mu=3.803, implied_cr=0.65 (fallback), using 2024 playoff games
- **After:** mu=3.855, implied_cr=0.943 (actual 2025 data), using weeks 1-4

This is a 94.3% catch rate vs 65% default - huge difference!

## Current Model Status

### What's Working ✅
- ✅ L4 (last 4 games) from **current season 2025**
- ✅ Exponential weighting (alpha=0.4): [10.3%, 25.6%, 64.1%]
- ✅ Family-based modeling (receptions = targets × catch_rate)
- ✅ Targets column loaded correctly
- ✅ Catch rates calculated from actual game data
- ✅ Consensus calculation (per side, de-vigged)
- ✅ Normal distributions (mu, sigma)
- ✅ Poisson distributions (lam)
- ✅ Bounded efficiency metrics (CR: 0.5-0.95, YPR: 6.0-18.0)

### What's Missing ❌
- ❌ **Defensive adjustments** (the function exists but is never called)
- ❌ Opponent team lookup
- ❌ Defensive ratings calculation

## Defensive Adjustments - Why They Matter

**Methods page promises:** ±15% adjustment based on opponent defense (0.5-2.0 scale)

**Current Saquon projection (NO defense):**
- 3.855 receptions → P(>2.5) = 96.0%
- Edge: +4,387 bps

**Estimated with defense (if NYG has tough pass D):**
- If NYG rating = 1.85 → 12% penalty → 3.39 receptions
- P(>2.5) = ~88%
- Edge: ~+3,500 bps (887 bps lower)

**This could explain why our edges are so high** - the market is pricing in defensive strength, but we're not adjusting for it yet.

## Files Modified Today

### Code Changes
- `scripts/make_player_prop_params.py` - Added targets, fixed L4 to use current season
- `scripts/build_props_site.py` - Added model_line_disp, reordered columns
- `scripts/build_insights_page.py` - Fixed background color
- `docs/props/insights.html` - Background color fix

### Data Regenerated
- `data/props/params_week6.csv` - With 2025 L4 data, actual catch rates
- `data/props/props_with_model_week6.csv` - Edges with corrected params
- `docs/props/index.html` - Props page with model line column

### New Content
- `docs/blog/early-lines-sharp-advantage.html` - Blog post on timing edge
- `docs/blog/index.html` - Updated with new post
- `DEFENSIVE_ADJUSTMENTS_TODO.md` - Implementation plan (this repo)
- `SESSION_SUMMARY.md` - This file

## Next Steps (Priority Order)

### 1. Implement Defensive Adjustments (HIGH PRIORITY)
See `DEFENSIVE_ADJUSTMENTS_TODO.md` for detailed plan.

**Steps:**
1. Calculate defensive ratings from 2025 weeks 1-5 data
2. Add opponent lookup (player → team → opponent)
3. Apply `apply_defensive_adjustment()` in `build_params()`
4. Rebuild params and edges
5. Verify Saquon numbers change appropriately

**Estimated time:** 2-3 hours

### 2. Update Blog Post with Corrected Numbers
After defensive adjustments are implemented:
- Update Saquon example with final numbers
- Verify edge calculations are accurate
- Add note about defensive adjustments in methodology

### 3. Verify All Markets
Once defense is implemented:
- Check rush_yds (uses rush_def_rating)
- Check pass_yds (QBs use pass_def_rating)
- Verify adjustments are realistic (±15% range)

### 4. Documentation
- Update methods page if needed
- Add defensive rating columns to params CSV for transparency
- Document team abbreviation mappings

## Key Learnings

1. **Always verify data source matches methodology** - We were unknowingly using 2024 data when methods said "L4 from current season"

2. **Missing columns fail silently** - The targets column was missing, but code fell back to defaults without warning

3. **Consensus needs to be per-side** - Over and Under are separate bets, not the same book twice

4. **Defensive adjustments are significant** - ±15% can swing edges by 800+ bps

5. **Family-based modeling requires all components** - Without targets, the whole catch_rate calculation breaks

## Commits Today
1. `3ffca69` - Fix consensus calculation and add Side column
2. `a5fdae8` - Add blog post: Why Sharps Bet Early — The Timing Edge
3. `99baf7f` - Add Model Line to Player Props page and align lines for comparison
4. `1591bc5` - Fix player modeling to use current season L4 data with targets

## Token Usage
Used ~121K / 200K tokens (60.5% of context)
