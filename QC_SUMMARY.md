# QC Framework Summary

## What We Built Today

### 1. Automated QC Script (`scripts/weekly_qc_checks.py`)
Comprehensive weekly quality control that checks:

✅ **Historical data availability** - All weeks 1-(N-1) present
✅ **Props data quality** - Correct columns, no critical NaNs, market coverage
✅ **Params quality** - Join coverage, no NaN mu values, family diagnostics present
✅ **Edge calculations** - Valid probabilities [0,1], reasonable edge distribution
✅ **Model calibration** - Over/Under symmetry, high-confidence disagreement count
✅ **Methodology consistency** - Season/week tags, efficiency bounds, defensive diagnostics

**Usage:**
```bash
python3 scripts/weekly_qc_checks.py \
    --season 2025 --week 6 \
    --props data/props/latest_all_props.csv \
    --params data/props/params_week6.csv \
    --edges data/props/props_with_model_week6.csv
```

**Exit codes:**
- `0` = All checks passed (may have warnings)
- `1` = Failed with errors (DO NOT PUSH)

---

## Your Questions Answered

### Q1: What global checks should we print with each run?

**Answer:** The QC script now prints 6 categories of checks:

1. **Historical Data** - Verify weeks 1-(N-1) are all available
2. **Props Markets** - Check market distribution, column completeness
3. **Params Quality** - Verify joins work, no NaN values
4. **Edge Distribution** - Check probability validity, edge reasonableness
5. **Calibration** - Count extreme disagreements, verify Over/Under symmetry
6. **Methodology** - Ensure efficiency bounds, defensive adjustments applied

### Q2: Make sure all previous weeks are modeled

**Answer:** Check #1 verifies historical data exists:
- Loads `data/weekly_player_stats_2025.parquet`
- Checks weeks 1, 2, 3, ..., (current_week - 1) all present
- Fails if any week is missing

### Q3: Check the props markets present

**Answer:** Check #2 shows market distribution:
```
Market distribution:
  player_anytime_td      : 437 props
  player_receptions      : 152 props
  player_rush_yds        : 92 props
  ...
```
Plus warns if modeled market coverage <50%

### Q4: Check that joins are working

**Answer:** Check #3 calculates join coverage:
- Merges props (player, market) → params (player, market)
- Reports percentage matched
- Shows top 10 unmatched pairs if coverage <80%

### Q5: Check all markets have been modeled

**Answer:** Check #3 also verifies:
- No NaN mu values for normal distributions
- Family diagnostic columns present (implied_ypc, implied_cr, etc.)
- Warns if defensive adjustment diagnostics missing

### Q6: Make sure methods are intact

**Answer:** Check #6 validates methodology consistency:
- ✅ Season/week tags match
- ✅ Rush YPC in bounds [2.5, 6.5]
- ✅ Catch rate in bounds [0.5, 0.95]
- ✅ Completion % in bounds [0.5, 0.75]
- ✅ Defensive diagnostics present

### Q7: Should we adjust high-confidence bets toward market consensus?

**Answer:** NOT YET - See `CALIBRATION_STRATEGY.md` for full analysis.

**Recommendation:**
1. **Weeks 6-8:** Track outcomes, don't adjust
2. **After Week 8:** Backtest if shrinkage helps
3. **Week 9+:** Implement calibrated shrinkage if validated

**Current stance:** 39 props with 99%+ confidence is moderate. Watch for:
- 🛑 If >100 extreme disagreements → investigate
- ⚠️ If market consistently right → consider shrinkage
- ✅ If model well-calibrated → keep as-is

---

## Files Created

1. **`scripts/weekly_qc_checks.py`** - Automated QC script (run every week)
2. **`CALIBRATION_STRATEGY.md`** - Deep dive on when/how to adjust confidence
3. **`WEEKLY_WORKFLOW.md`** - Step-by-step refresh procedure with all checks
4. **`QC_SUMMARY.md`** - This file (quick reference)

---

## What Else Should We Check?

### Currently Checking ✅
- Historical data completeness
- Props data quality
- Join coverage
- Edge distributions
- Probability validity
- Over/Under symmetry
- Efficiency bound violations
- Defensive adjustment application

### Not Yet Checking ⚠️
- **Game script alignment** - Does spread/total align with volume bets?
- **Weather conditions** - Flag rain/wind for passing props
- **Line movement** - Did line move significantly since fetch?
- **Book availability** - Is prop pulled at some books? (red flag)
- **Injury news** - Are we missing recent injury reports?
- **Consensus drift** - Is our model diverging from consensus over time?
- **Historical accuracy** - Are 90% bets actually hitting 90%? (need more weeks)

### Recommended Additions (Priority Order)

#### High Priority (Add by Week 8):
1. **Calibration tracking** - Log all bets with model_prob, actual outcome
2. **Line movement alerts** - Flag when line moved >1.5 points since fetch
3. **Missing data warnings** - Flag players with <3 games in L4

#### Medium Priority (Add by Week 10):
4. **Game script indicators** - Warn when spread >10 (blowout risk)
5. **Weather integration** - Flag wind >15mph, precip >50%
6. **Consensus drift monitoring** - Track if model systematically diverges

#### Low Priority (Nice to have):
7. **Injury report scraping** - Auto-check official NFL injury reports
8. **Sharp line movement** - Detect when "sharp" books move first
9. **Historical calibration curves** - Plot model_prob vs actual outcomes

---

## How to Use This Framework

### Every Week (Tuesday/Wednesday):
1. Run weekly refresh (fetch → params → edges → site)
2. **Run QC script** - Must pass before pushing
3. Review warnings, investigate any failures
4. Commit and push if all checks pass

### Every Month:
1. Review `CALIBRATION_STRATEGY.md`
2. Check if 99% bets are actually hitting 99%
3. Adjust confidence shrinkage if needed

### Every Quarter:
1. Review `WEEKLY_WORKFLOW.md` - Update if process changes
2. Add new QC checks as model evolves
3. Document methodology changes in `docs/methods.html`

---

## Red Flags (Stop and Investigate)

🛑 **Do NOT push if you see:**
- Join coverage <50%
- >100 extreme disagreements (99% model vs <60% market)
- Efficiency metrics out of bounds (YPC >6.5, CR >0.95)
- Applied 0 defensive adjustments
- Over/Under pairs don't sum to 1.0
- Missing historical weeks

⚠️ **Investigate but may be OK:**
- 30-50 extreme disagreements (monitor)
- Join coverage 50-80% (check which markets missing)
- Modeled coverage 40-60% (longest/1st/last TD not modeled)

✅ **Green light:**
- All weeks present
- Join coverage >80%
- <50 extreme disagreements
- All probabilities in [0, 1]
- Efficiency bounds respected
- Defensive adjustments applied

---

## Commit This Framework

```bash
git add scripts/weekly_qc_checks.py \
        CALIBRATION_STRATEGY.md \
        WEEKLY_WORKFLOW.md \
        QC_SUMMARY.md

git commit -m "Add comprehensive QC framework for weekly refreshes

- Automated QC script checks 6 categories of data quality
- Calibration strategy for handling high-confidence disagreements
- Weekly workflow documentation with all checks
- Red flag guide for when NOT to push

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git push origin main
```

---

## Next Steps

1. **This week (Week 6):** Use the QC script, start logging outcomes
2. **Week 7-8:** Continue tracking, build calibration database
3. **Week 9:** Review calibration, decide on confidence shrinkage
4. **Week 10+:** Add game script, weather, line movement checks

**The goal:** Clean, consistent week-to-week refreshes with confidence that our data is accurate and our methods are working as designed.
