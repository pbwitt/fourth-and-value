# Model Calibration Strategy

## The Question: Should We Adjust High-Confidence Bets Toward Market Consensus?

When our model shows 99% confidence but the market prices it at 50%, we have two interpretations:

1. **We found an edge** - The model knows something the market doesn't
2. **We're overconfident** - The model is missing context that the market has

## Arguments FOR Adjusting Toward Consensus

### 1. Market Efficiency
- **The market aggregates information from many sources** including:
  - Injury reports we might miss
  - Weather forecasts
  - Coaching tendencies
  - Game script expectations (spread, over/under)
  - Late-breaking news
  - Sharp money movement

### 2. Model Limitations
Our model currently uses:
- ✅ L4 player performance (exponentially weighted)
- ✅ Family-based multi-market modeling
- ✅ Defensive adjustments (±15% based on opponent)
- ❌ Game script (spread, total, home/away)
- ❌ Weather conditions
- ❌ Injury probability
- ❌ Coaching tendencies
- ❌ Rest days / schedule spot

**When we're 99% confident and market is 50%, we're likely missing something.**

### 3. Risk Management
- Extreme confidence creates **concentration risk**
- If we're wrong on a 99% bet, it's catastrophic for bankroll
- Shrinking toward consensus **reduces variance** without killing edge

### 4. Historical Evidence
- Market closing lines beat opening lines ~55% of the time
- Sharp bettors move markets, but rarely to extremes
- When model and market disagree wildly, market is usually right

## Arguments AGAINST Adjusting Toward Consensus

### 1. That's Where the Edge Is
- If we always agree with consensus, we have no edge
- The whole point of modeling is to **find market inefficiencies**
- Early-week lines (our focus) are softer by design

### 2. Self-Fulfilling Mediocrity
- Adjusting toward consensus makes us a worse model
- We'd never beat the market if we always defer to it
- **Our job is to be right when others are wrong**

### 3. Selection Bias
- We only notice disagreements when they're large
- We don't track the thousands of times we agree with market
- **Extreme disagreements might be our best bets**

### 4. We Built This Model for a Reason
- If we don't trust it at 99%, why trust it at 65%?
- Either the model is calibrated or it isn't
- **Shrinking is admitting the model doesn't work**

---

## Recommended Approach: Tiered Confidence Adjustment

Instead of blind shrinkage, use **context-aware calibration**:

### Tier 1: Keep Full Confidence (No Adjustment)
When model is 99%+ confident AND:
- ✅ Multiple books agree on similar line
- ✅ Consensus is close to our model (within 10%)
- ✅ Player's L4 data is stable (low variance)
- ✅ No injury/weather concerns

**Example:** Saquon O2.5 receptions when consensus is O3.5

### Tier 2: Moderate Shrinkage (20-30%)
When model is 99%+ confident BUT:
- ⚠️ Market consensus disagrees by 20-40%
- ⚠️ Player has high variance in L4 data
- ⚠️ Game script factors not in our model (big spread, bad weather)

**Formula:**
```python
adjusted_prob = model_prob * 0.7 + consensus_prob * 0.3
```

**Example:** Model 99%, consensus 55% → adjusted 81%

### Tier 3: Heavy Shrinkage (50%)
When model is 99%+ confident BUT:
- 🚨 Market consensus disagrees by >40%
- 🚨 Limited sample size (player played <3 games)
- 🚨 Known context we're missing (injury report, weather alert)

**Formula:**
```python
adjusted_prob = model_prob * 0.5 + consensus_prob * 0.5
```

**Example:** Model 99%, consensus 50% → adjusted 75%

### Tier 4: Flag for Review (Don't Bet)
When model is 99%+ confident BUT:
- 🛑 Market moved AWAY from us (we liked O, now line is higher)
- 🛑 Bet not available at multiple books (pulled = books know something)
- 🛑 Extreme disagreement with zero explanation

**Action:** Manual review or skip the bet

---

## Implementation Strategy

### Phase 1: Track, Don't Adjust (Weeks 6-8)
- **Goal:** Build calibration data
- **Action:** Log all 99%+ disagreements with actual outcomes
- **Metrics:** Win rate, average edge realized, ROI

### Phase 2: Backtest Shrinkage (After Week 8)
- **Goal:** Test if shrinkage improves results
- **Action:** Simulate different shrinkage factors on historical data
- **Metrics:** Sharpe ratio, max drawdown, hit rate

### Phase 3: Implement If Validated (Week 9+)
- **Goal:** Deploy calibrated shrinkage if it improves metrics
- **Action:** Add shrinkage logic to edge calculations
- **Metrics:** Monitor ongoing performance

---

## Specific Checks for High-Confidence Bets

Before betting a 99%+ confident prop, ask:

### 1. Data Quality Check
- [ ] Player has ≥3 games in L4 window
- [ ] No recent injury/suspension news
- [ ] Defensive adjustment is reasonable (opponent exists)

### 2. Market Sanity Check
- [ ] Line available at ≥3 books
- [ ] Line hasn't moved significantly since we pulled data
- [ ] Consensus exists (not a one-book outlier)

### 3. Context Check
- [ ] Game script makes sense (spread/total align with volume bet)
- [ ] Weather is normal (no rain/wind for pass-heavy bets)
- [ ] No coaching/role changes this week

### 4. Model Coherence Check
- [ ] If betting rush attempts, check rush yards aligns
- [ ] If betting receptions, check receiving yards aligns
- [ ] Family markets should agree on implied efficiency

---

## Recommendation for Week 6 Forward

**DO NOT adjust yet.** Instead:

1. **Add logging** to track 99%+ disagreements and outcomes
2. **Add QC flags** to weekly reports showing extreme disagreements
3. **Build a database** of model_prob vs actual_outcome over 4-6 weeks
4. **Then decide** if shrinkage is needed based on evidence

**If after 4-6 weeks:**
- Model is well-calibrated (99% bets hit ~99%) → Keep as-is ✓
- Model is overconfident (99% bets hit ~75%) → Implement shrinkage ⚠️
- Model is underconfident (99% bets hit 100%) → Increase confidence 📈

---

## The Real Answer

**Your model is only as good as your ability to know when it's wrong.**

Right now, with defensive adjustments just implemented:
- We don't have enough track record to know if 99% is real or overfit
- The market probably has information we don't (game script, late injuries)
- **Conservative play:** Flag for review, but don't auto-adjust yet

**Build the tracking system first. Adjust second.**
