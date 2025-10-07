# Weekly Refresh Workflow & QC Checklist

## Overview

This document defines the standard weekly workflow for Fourth & Value, including all QC checks that must pass before pushing to production.

---

## Weekly Workflow (Run Every Tuesday/Wednesday)

### Step 1: Fetch Fresh Data
```bash
# Fetch latest props from The Odds API
python3 scripts/fetch_all_player_props.py --season 2025 --week 6

# Verify props were fetched
ls -lh data/props/latest_all_props.csv
```

**Expected output:** 1,500-2,000 props across 8-10 books

---

### Step 2: Generate Model Parameters
```bash
python3 scripts/make_player_prop_params.py \
    --season 2025 \
    --week 6 \
    --props_csv data/props/latest_all_props.csv \
    --out data/props/params_week6.csv
```

**Key checks in output:**
- ✓ `[defense] Calculated ratings for 32 teams`
- ✓ `[opponent] Mapped opponents for XXX players`
- ✓ `[defense] Applied XXX defensive adjustments`
- ✓ `[family] Derived params for 7 Normal markets from latents`
- ✓ `[diagnostics] Added implied_ypc, implied_ypr, implied_cr...`

---

### Step 3: Calculate Edges
```bash
python3 scripts/make_props_edges.py \
    --season 2025 \
    --week 6 \
    --props_csv data/props/latest_all_props.csv \
    --params_csv data/props/params_week6.csv \
    --out data/props/props_with_model_week6.csv
```

**Key checks in output:**
- ✓ `[merge] wrote data/props/props_with_model_week6.csv with XXXX rows`
- ✓ Backup created in `data/preds_historical/`

---

### Step 4: Build Site Pages
```bash
# Props page
python3 scripts/build_props_site.py \
    --merged_csv data/props/props_with_model_week6.csv \
    --out docs/props/index.html \
    --season 2025 \
    --week 6

# Consensus page
python3 scripts/build_consensus_page.py \
    --merged_csv data/props/props_with_model_week6.csv \
    --out docs/consensus/index.html \
    --season 2025 \
    --week 6
```

---

### Step 5: RUN QC CHECKS ⚠️ CRITICAL
```bash
python3 scripts/weekly_qc_checks.py \
    --season 2025 \
    --week 6 \
    --props data/props/latest_all_props.csv \
    --params data/props/params_week6.csv \
    --edges data/props/props_with_model_week6.csv
```

**Must see:** `✓ ALL CHECKS PASSED` or `⚠ PASSED WITH X WARNINGS`

**Do NOT proceed if you see:** `✗ FAILED WITH X ERRORS`

---

### Step 6: Manual Spot Checks

#### 6.1 Check a Known Player (e.g., Saquon Barkley)
```bash
python3 -c "
import pandas as pd
edges = pd.read_csv('data/props/props_with_model_week6.csv')
saquon = edges[edges['player'] == 'Saquon Barkley']
print(saquon[['market_std', 'name', 'point', 'model_prob', 'mkt_prob', 'edge_bps']].head(10))
"
```

**Sanity check:**
- Receptions Over 2.5 should be high probability (90%+)
- Rush yards should align with rush attempts
- Edges should be in reasonable range (-5000 to +5000 bps)

#### 6.2 Check Methods Page Still Matches Code
```bash
# Open methods page and verify:
open docs/methods.html
```

**Verify sections:**
- [ ] L4 exponential weighting (α=0.4) mentioned
- [ ] Defensive adjustment formula: `1.0 - (def_rating - 1.0) × 0.3`
- [ ] Family-based modeling described
- [ ] Adjustment ranges: ±15% for defense

#### 6.3 Visually Inspect Props Page
```bash
# Open in browser
open docs/props/index.html
```

**Check:**
- [ ] Book Line, Model Line, Consensus Line all show
- [ ] Side (Over/Under) column present
- [ ] Edges display correctly
- [ ] Filters work (market, book)
- [ ] Consensus shows "X books" accurately

---

### Step 7: Commit and Push
```bash
# Stage changes
git add data/props/params_week6.csv \
        data/props/props_with_model_week6.csv \
        docs/props/index.html \
        docs/consensus/index.html

# Commit with descriptive message
git commit -m "Week 6 refresh: params, edges, site pages

- 987 defensive adjustments applied
- XXX props modeled across X markets
- All QC checks passed

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to production
git push origin main
```

---

## QC Checks Reference

### 1. Historical Data Availability
- ✓ All weeks 1 through (current_week - 1) present
- ✓ Data is current (max week = current_week - 1)
- ✓ No missing weeks in parquet file

### 2. Props Data Quality
- ✓ 1,500-2,000 props fetched
- ✓ 150-200 unique players
- ✓ 8-10 unique books
- ✓ All required columns present (player, market, price, point)
- ✓ No critical NaN values
- ⚠️ Modeled market coverage >50% (may be 0% if market_std not added yet)

### 3. Params Quality & Join Coverage
- ✓ 200-300 param rows (player × market combinations)
- ✓ No NaN mu values in normal distributions
- ✓ Family diagnostic columns present (implied_ypc, implied_cr, etc.)
- ✓ Join coverage (props → params) >80%

### 4. Edge Calculations & Distributions
- ✓ All model probabilities in [0, 1]
- ✓ Mean edge around -300 to +300 bps (slightly negative is normal)
- ✓ No extreme edges >10,000 bps (indicates model bug)
- ✓ Consensus coverage >50%
- ⚠️ Watch for >50 props with edge > +5,000 bps (potential overconfidence)

### 5. Model Calibration
- ✓ Over/Under pairs sum to 1.0
- ⚠️ <50 props with 99%+ model confidence but <60% market (indicates disagreement)
- 🛑 If >100 extreme disagreements, investigate before pushing

### 6. Methodology Consistency
- ✓ Params tagged with correct season/week
- ✓ Rush YPC in realistic range [2.5, 6.5]
- ✓ Catch rates in realistic range [0.5, 0.95]
- ✓ Completion % in realistic range [0.5, 0.75]
- ✓ Defensive adjustment diagnostics present

---

## Red Flags (Do NOT Push to Production)

🛑 **STOP if you see any of these:**

1. **No defensive adjustments applied** - Check output shows `[defense] Applied 0 defensive adjustments`
2. **Join coverage <50%** - Most props have no model predictions
3. **>100 extreme disagreements** - Model is wildly off from market
4. **Over/Under asymmetry** - Model probabilities don't sum to 1.0
5. **Efficiency metrics out of bounds** - YPC >6.5, catch rate >0.95, etc.
6. **Missing historical weeks** - Can't calculate L4 properly
7. **Model probabilities outside [0,1]** - Math error in probability calculation

---

## Common Issues & Fixes

### Issue: "Applied 0 defensive adjustments"
**Cause:** Opponent map failed or defensive ratings not calculated
**Fix:**
```bash
# Check logs for opponent mapping errors
# Verify team abbreviations match between props and defensive ratings
```

### Issue: "Low join coverage: 20%"
**Cause:** market_std normalization failed or props have markets we don't model
**Fix:**
```bash
# Check props markets
python3 -c "
import pandas as pd
props = pd.read_csv('data/props/latest_all_props.csv')
print(props['market'].value_counts())
"
# Add missing markets to common_markets.py MODELED_MARKETS
```

### Issue: "100+ extreme disagreements"
**Cause:** Model is overconfident or missing context (injury, weather, game script)
**Fix:**
- Review top disagreements manually
- Check if there's news we missed (injuries, suspensions)
- Consider if defensive adjustments are too aggressive
- May need to implement confidence shrinkage (see CALIBRATION_STRATEGY.md)

### Issue: "Rush YPC = 12.5"
**Cause:** Family coherence bug - likely rush_yds not matching rush_attempts
**Fix:**
```bash
# Check params for the affected player
python3 -c "
import pandas as pd
params = pd.read_csv('data/props/params_week6.csv')
bad_player = params[params['implied_ypc'] > 6.5]
print(bad_player[['player', 'market_std', 'mu', 'implied_ypc']])
"
# Debug derive_market_from_latents() in make_player_prop_params.py
```

---

## Week-to-Week Consistency Checks

Before each week's refresh, verify these stay consistent:

### Methods Page
- [ ] L4 methodology unchanged
- [ ] Exponential weighting (α=0.4) unchanged
- [ ] Defensive adjustment formula unchanged
- [ ] Family bounds unchanged (YPC: 2.5-6.5, CR: 0.5-0.95, etc.)

### Code Constants
```python
# In make_player_prop_params.py
ALPHA = 0.4  # Exponential weighting
DEVIG_FACTOR = 0.9524  # ~5% vig removal

# Bounds (should NOT change week-to-week)
YPC_BOUNDS = (2.5, 6.5)
CR_BOUNDS = (0.50, 0.95)
YPR_BOUNDS = (6.0, 18.0)
COMP_PCT_BOUNDS = (0.50, 0.75)
```

### Defensive Adjustment
```python
# Formula should stay consistent
adjustment = 1.0 - (def_rating - 1.0) * 0.3
# Range: 0.85 to 1.15 (±15%)
```

---

## Future Enhancements

### Add to QC Script (Priority Order):
1. **Game script indicators** - Flag when spread >10 (blowout expected)
2. **Weather alerts** - Flag when wind >15mph or precip >50%
3. **Injury tracking** - Auto-check injury reports from NFL API
4. **Line movement tracking** - Flag when line moved significantly since fetch
5. **Consensus drift** - Flag when our model diverges from consensus trend

### Add to Modeling:
1. **Game script adjustments** - Reduce volume for losing teams
2. **Weather adjustments** - Reduce passing in bad weather
3. **Rest/schedule** - Account for short weeks, travel
4. **Coaching tendencies** - Historical play-calling patterns

---

## Emergency Rollback Procedure

If you push bad data to production:

```bash
# Revert to previous commit
git revert HEAD

# Or rollback to specific good commit
git reset --hard <commit-hash>
git push origin main --force

# Rebuild from last good data
cp data/preds_historical/props_with_model_week5_YYYYMMDD_HHMMSS.csv \\
   data/props/props_with_model_week6.csv

# Rebuild pages
python3 scripts/build_props_site.py \\
    --merged_csv data/props/props_with_model_week6.csv \\
    --out docs/props/index.html \\
    --season 2025 --week 6
```

---

## Contact & Escalation

If QC fails and you're not sure why:
1. Check this document for common issues
2. Review git history: `git log --oneline -10`
3. Compare to last week's successful run
4. Don't push if in doubt - better to skip a week than publish bad data
