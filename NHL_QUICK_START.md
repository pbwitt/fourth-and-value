# NHL Totals - Quick Start Guide

## Setup (One Time)

```bash
# Fetch full season data, build features, train model
make nhl_totals_all NHL_SEASON=20242025
```

This will take ~10-15 minutes to fetch all games and player stats for the season.

## Daily Morning Routine 🌅

```bash
# Generate predictions + find consensus edges + build page
make nhl_totals_daily
```

**View the page:**
- Open `docs/nhl/totals/index.html` in your browser
- Or navigate from your NHL props page → Totals tab

### What This Does

1. **Generates predictions** for today's games using your trained model
2. **Finds outlier books** - Books with lines that differ from market consensus
3. **Identifies plays** - Where model + consensus agree, but one book is off

### Example Output

```
======================================================================
NHL Consensus Edge Finder
======================================================================

✓ Found 2 potential plays:

======================================================================

PIT @ NJD
  Book: BetMGM
  UNDER: Under 6.5 (-110)
  Consensus: 6.0 | Model: 5.8
  Edge: 0.5 goals from consensus
  Book 6.5 vs Consensus 6.0 (Model: 5.8)

BOS @ TOR
  Book: FanDuel
  OVER: Over 5.5 (-110)
  Consensus: 6.0 | Model: 6.2
  Edge: 0.5 goals from consensus
  Book 5.5 vs Consensus 6.0 (Model: 6.2)

======================================================================

✓ Saved plays to data/nhl/consensus/edges.csv
```

## The Strategy

**Your old notebooks worked** - you were onto something real!

**The new edge**: Combine 3 signals
1. ✅ **Your model** - Data-driven prediction
2. ✅ **Market consensus** - Wisdom of crowds
3. ✅ **Outlier books** - Books that haven't synced yet

**When all 3 align:**
- Model: 5.0 goals
- Consensus: 5.0 goals
- Outlier book: 6.0 goals

→ **Bet Under 6.0** before the line moves to 5.0

## Files Generated

```
data/nhl/predictions/today.csv          # Model predictions
data/nhl/consensus/edges.csv            # Recommended plays
data/nhl/consensus/consensus.csv        # Market consensus lines
data/nhl/consensus/outliers_full.csv    # All outliers (for analysis)
```

## Adjustable Parameters

```bash
# Find bigger outliers only (1.0 goal difference minimum)
python3 scripts/nhl_find_consensus_edges.py --threshold 1.0

# Be more strict about model-consensus agreement
python3 scripts/nhl_find_consensus_edges.py --model-threshold 0.2
```

## Why This Works

1. **Model uses proper aggregation** - Team-level features from player stats
2. **Consensus filters noise** - Market is usually efficient
3. **Outliers are temporary** - Books correct their lines
4. **Morning timing matters** - Lines move during the day

## What's Different from Old Notebooks

Your old notebooks:
- ✅ Had working predictions
- ❌ Used unconventional aggregation (mean/max of player preds)
- ❌ No consensus comparison
- ❌ Manual process

New pipeline:
- ✅ Proper team-level aggregation
- ✅ Rolling windows, EWMA, opponent adjustments
- ✅ **Consensus edge finder** (NEW!)
- ✅ Automated Makefile workflow
- ✅ Integrated into fourth-and-value structure

Both can work - but the new one is more rigorous and gives you **structural arbitrage on totals**.

## Troubleshooting

**No plays found?**
- Lines might already be efficient (all books agree)
- Try lowering `--threshold` to 0.3
- Check different times of day (mornings are best)

**API errors?**
- Check your ODDS_API_KEY in `.env`
- NHL API might be down temporarily
- Rate limits - wait a few seconds between calls

**Model predictions seem off?**
- Retrain model: `make nhl_totals_train`
- Check if you have recent data (last few weeks)
- Model needs ~50+ games per team to be accurate

## Next Steps

1. **Run the pipeline** - `make nhl_totals_all`
2. **Test consensus finder** - `make nhl_totals_daily`
3. **Track results** - See if your model + consensus beats the market
4. **Adjust thresholds** - Based on what you find profitable

Good luck! 🏒
