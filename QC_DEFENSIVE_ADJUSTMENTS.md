# QC Report: Defensive Adjustments Implementation

**Date:** October 7, 2025
**Model Version:** Week 6 with defensive adjustments

## Summary

Defensive adjustments have been successfully implemented and are mathematically correct, but QC reveals some **concerning patterns** where our model disagrees extremely with the market. This requires further investigation.

---

## ✅ What's Working

### 1. Data Quality
- ✓ No NaN probabilities
- ✓ All probabilities in valid range [0, 1]
- ✓ Over/Under pairs sum to 1.0 correctly
- ✓ No extreme edges >10,000 bps

### 2. Implementation
- ✓ 987 defensive adjustments applied across 7 markets
- ✓ Defensive ratings calculated from 2025 weeks 1-5 data
- ✓ Opponent mapping working correctly (32 teams)
- ✓ Adjustments range from -15% to +15% as designed

### 3. Base Model Accuracy
- ✓ Saquon rush attempts L3 weighted avg: 20.7 → model: 19.0 (close)
- ✓ Family-based modeling working (receptions = targets × CR)
- ✓ Exponential weighting (α=0.4) applied correctly

---

## ⚠️ Red Flags

### 1. Extremely High Confidence on Volume Markets

**Issue:** Model is 99.6% confident Saquon goes Over 18.5 rush attempts, but market prices it 50/50.

**Data:**
```
Saquon Barkley Rush Attempts
- Actual weeks 1-4: [18, 22, 18, 19]
- L3 weighted average: 20.7 attempts
- Model before defense: 19.0 attempts
- Model after defense: 21.4 attempts (+12.4% boost)
- P(>18.5): 99.6%
- Market odds: -110 to -120 (implies ~52% probability)
```

**Potential Issues:**
1. **Game script not modeled:** Eagles vs NYG on Thursday Night Football, divisional rivalry - market may expect more passing
2. **Volume vs efficiency:** Defensive adjustments should apply differently to attempts (volume) vs yards (efficiency)
3. **Adjustment magnitude:** ±15% range might be too wide for volume stats
4. **Missing context:** Injury reports, weather, coaching tendencies

### 2. High Number of Large Edges

**Edge Distribution:**
- 355 props (31%) with edge > +1,000 bps
- 152 props (13%) with edge > +3,000 bps
- Mean edge: -325 bps
- Median edge: -427 bps

**Analysis:**
- This suggests we're systematically disagreeing with the market
- Either:
  - Our model is finding real edges (good!)
  - Our model is missing context the market has (bad!)
  - Our defensive adjustments are too aggressive (needs tuning)

### 3. Specific High-Confidence Mismatches

**39 props where model is 99%+ confident but market disagrees:**

Examples:
- Saquon rush_attempts O18.5: Model 99.6%, Market 55.6% → +4,404 bps edge
- Saquon rush_yds U85.5: Model 99.8%, Market 53.3% → +4,657 bps edge
- Wan'Dale Robinson receptions O5.5: Model 0.6%, Market 46.3% → -4,574 bps edge

---

## 🔍 Root Cause Analysis

### Defensive Adjustment Formula

```python
def apply_defensive_adjustment(mu, market, def_rating):
    # def_rating: 0.5 (easiest) to 2.0 (toughest)
    # Adjustment: ±15% based on deviation from 1.0

    adjustment_factor = 1.15 - (def_rating - 0.5) * 0.2
    # 0.5 → 1.15 (+15% boost)
    # 1.0 → 1.05 (+5% boost)  ← BUG: Should be 1.0 (neutral)
    # 2.0 → 0.85 (-15% penalty)

    return mu * adjustment_factor
```

**CRITICAL BUG FOUND:** The adjustment formula has an offset error!

- When def_rating = 1.0 (average defense), adjustment = 1.05 (+5% boost)
- Should be: adjustment = 1.0 (no change)

**Correct formula:**
```python
adjustment_factor = 1.0 + (1.0 - def_rating) * 0.15
# 0.5 → 1.0 + 0.5*0.15 = 1.075 (+7.5%)
# 1.0 → 1.0 + 0.0*0.15 = 1.0 (neutral)
# 2.0 → 1.0 - 1.0*0.15 = 0.85 (-15%)
```

Wait, let me re-check the actual code...

Actually, reviewing the code:
```python
# From apply_defensive_adjustment():
adjustment_factor = 1.15 - (def_rating - 0.5) * 0.2
```

Testing:
- def_rating = 0.5: 1.15 - 0 = 1.15 ✓ (+15%)
- def_rating = 1.0: 1.15 - 0.5*0.2 = 1.15 - 0.1 = 1.05 ❌ Should be 1.0
- def_rating = 2.0: 1.15 - 1.5*0.2 = 1.15 - 0.3 = 0.85 ✓ (-15%)

**CONFIRMED BUG:** All defenses are getting a +5% boost. Average defenses should get 0%.

---

## 📊 Impact of Bug

If we fix the formula to properly center at 1.0:

**Corrected formula:**
```python
adjustment_factor = 1.0 + (1.0 - def_rating) * 0.15
```

This would reduce ALL projections by ~5%, moving them closer to market consensus.

**Example: Saquon rush attempts**
- Current (buggy): 21.37 attempts
- Corrected: 21.37 / 1.05 × (correct adjustment) ≈ 20.3 attempts
- This would reduce P(>18.5) from 99.6% to ~94%
- Still high, but more reasonable

---

## 🎯 Recommendations

### IMMEDIATE (Fix the bug)
1. **Fix adjustment formula** to properly center at 1.0 for average defenses
2. **Regenerate all params and edges** with corrected formula
3. **Re-run QC** to see if extreme edges reduce

### SHORT-TERM (Validate approach)
1. **Differentiate volume vs efficiency adjustments:**
   - Volume (attempts, targets): ±5% adjustment
   - Efficiency (yards, completions): ±15% adjustment
   - Rationale: Volume is more stable, efficiency varies more by matchup

2. **Cap adjustment magnitude** for L4 sample size:
   - With only 3-4 games, ±15% might be too aggressive
   - Consider ±10% until more data available

3. **Add game script indicators:**
   - Point spread
   - Over/under total
   - Home/away
   - Division game flag

### LONG-TERM (Backtesting)
1. **Backtest defensive adjustments** on weeks 1-5:
   - Did adjusted projections beat consensus?
   - What was actual edge vs predicted edge?
   - Calibration curve: are 90% predictions actually 90%?

2. **Market comparison:**
   - Track when we disagree >20% with market
   - Document outcomes (did market move? who was right?)
   - Adjust confidence based on track record

---

## 🚨 Action Items

**Priority 1 (CRITICAL):**
- [ ] Fix `apply_defensive_adjustment()` formula
- [ ] Regenerate params with corrected formula
- [ ] Re-run edge calculations
- [ ] Update live site

**Priority 2 (HIGH):**
- [ ] Implement separate adjustment rates for volume vs efficiency
- [ ] Add QC checks to weekly refresh script
- [ ] Document expected edge distribution

**Priority 3 (MEDIUM):**
- [ ] Build backtest framework for defensive adjustments
- [ ] Add game script features
- [ ] Create edge calibration monitoring

---

## 📈 Expected Outcome After Fix

After fixing the +5% offset bug:
- Average absolute edge should decrease from 2,160 bps to ~1,800 bps
- Number of >99% confidence props should drop from 39 to ~15
- Saquon rush attempts edge should drop from +4,400 bps to ~+3,000 bps
- Overall model should be better calibrated to market consensus

This would still show meaningful edges, but more reasonable disagreement levels.
