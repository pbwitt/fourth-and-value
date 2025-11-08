# NHL Team Totals Prediction Pipeline

A proper implementation of NHL totals modeling using player-level stats aggregated to team level, following the same methodology as the NFL props pipeline.

## Overview

Unlike the old notebooks which incorrectly predicted individual player scores and averaged them, this pipeline:

1. **Aggregates player stats to team level** - Creates team-level features from individual player performance
2. **Predicts team totals directly** - One prediction per team per game
3. **Uses rolling windows & exponential weighting** - Recent games matter more
4. **Includes opponent adjustments** - Accounts for defensive strength
5. **Home/away splits** - Teams perform differently at home vs away

## Quick Start

### Full Pipeline (Recommended for first run)

```bash
make nhl_totals_all NHL_SEASON=20242025
```

This will:
1. Fetch all games and player stats for 2024-25 season
2. Build team-level features
3. Train the model
4. Generate predictions for today's games

### Daily Updates (Morning Routine - Before Lines Move!)

Once you have historical data, run daily predictions + consensus edge finder:

```bash
make nhl_totals_daily
```

This will:
1. Generate model predictions for today's games
2. Find books with lines out of sync with consensus
3. Show you plays to make **before the lines move**

### Just Predictions (No Consensus)

```bash
make nhl_totals_predict
```

### Individual Steps

```bash
# Fetch data for current season
make nhl_totals_fetch NHL_SEASON=20242025

# Build features from player stats
make nhl_totals_features

# Train model
make nhl_totals_train

# Generate today's predictions
make nhl_totals_predict
```

## How It Works

### 1. Data Collection (`nhl_fetch_games.py`, `nhl_fetch_player_stats.py`)

Fetches game schedules and player-level boxscores from the new NHL API:
- `https://api-web.nhle.com/v1/schedule/{date}`
- `https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore`

Outputs:
- `data/nhl/raw/games.csv` - All games for the season
- `data/nhl/raw/player_stats.csv` - Player-level stats for each game

### 2. Feature Engineering (`nhl_build_features.py`)

**The key innovation**: Aggregates player stats to team level

For each team-game, creates features like:

**Forwards (aggregated)**:
- Total goals, assists, shots
- Average goals per forward
- Top scorer (max goals)
- Power play goals
- Face-off win %

**Defense (aggregated)**:
- Total blocked shots
- Average plus/minus
- Hits

**Goalie (starting goalie)**:
- Save percentage
- Goals against
- Shots against

**Rolling Windows**:
- Last 5 games average
- Last 10 games average
- Exponential weighted moving average (EWMA)

**Opponent Adjustments**:
- Opponent's goals allowed per game
- Opponent's goalie save %
- Opponent's defensive blocked shots

**Home/Away Splits**:
- Team's home scoring average
- Team's away scoring average

Output: `data/nhl/processed/team_features.csv`

### 3. Model Training (`nhl_train_model.py`)

Trains regression model to predict team total goals.

Model types available:
- `ridge` (default) - Ridge regression, good for correlated features
- `lasso` - Lasso regression, feature selection
- `elastic` - Elastic net, balance of ridge/lasso
- `rf` - Random forest
- `gbm` - Gradient boosting

Uses **time series cross-validation** to preserve temporal order (no data leakage).

Output:
- `data/nhl/models/ridge_team_totals.pkl` - Trained model
- `data/nhl/models/feature_list.txt` - Feature names

### 4. Predictions (`nhl_predict_totals.py`)

Generates predictions for today's games:

1. Fetches today's schedule
2. Loads recent team stats
3. Generates predictions for home team and away team
4. Sums to get total
5. Compares to market odds (if available)
6. Calculates edge

Output: `data/nhl/predictions/today.csv`

## Data Structure

```
data/nhl/
├── raw/
│   ├── games.csv           # All games for season
│   └── player_stats.csv    # Player-level boxscores
├── processed/
│   └── team_features.csv   # Team-level features (training data)
├── models/
│   ├── ridge_team_totals.pkl   # Trained model
│   └── feature_list.txt        # Feature names
└── predictions/
    └── today.csv           # Today's predictions
```

## Key Differences from Old Notebooks

| Old Approach | New Approach |
|-------------|--------------|
| Predict individual player scores | Aggregate player stats, predict team totals |
| Average/max player predictions | One prediction per team |
| No rolling windows | 5-game, 10-game, EWMA |
| No opponent adjustments | Opponent defensive strength |
| Random train/test split | Time series cross-validation |
| Hard-coded credentials | Environment variables |
| Deprecated NHL API | New NHL API (api-web.nhle.com) |
| Manual notebook execution | Automated Makefile pipeline |

## Why This Works Better

### Correct Aggregation

**Old way (WRONG)**:
```python
# Predict each player's goals
player_predictions = [0.1, 0.2, 0.05, 0.15, ...]
team_total = mean(player_predictions)  # or max()
```

**New way (CORRECT)**:
```python
# Aggregate player stats to team features
team_features = {
    'total_forward_goals_l5': 15,
    'avg_shots_per_forward': 2.5,
    'goalie_save_pct': 0.915,
    ...
}
# Predict team total directly
team_total = model.predict(team_features)
```

### More Information from Player-Level Data

By aggregating properly, we capture:
- **Depth scoring** (not just top scorers)
- **Line combinations** (top line vs 4th line production)
- **Special teams** (PP/PK effectiveness)
- **Goalie matchups** (starter vs backup)
- **Defense pairs** (ice time, blocked shots)

All while predicting at the correct level (team totals).

## Example Prediction Output

```
home_team  away_team  home_pred  away_pred  total_pred  market_total  edge
BOS        NYR        3.2        2.8        6.0         6.5           -0.5
TOR        MTL        3.5        2.5        6.0         5.5           +0.5
```

- `home_pred` / `away_pred` - Model's prediction for each team
- `total_pred` - Sum (the over/under number)
- `market_total` - Consensus line from books
- `edge` - Difference (positive = model higher than market)

## Integration with Your Workflow

This follows the same pattern as your NFL pipeline:

1. **Data in `data/nhl/`** (like `data/props/`)
2. **Scripts in `scripts/`** (like `fetch_odds.py`)
3. **Makefile targets** (like `make weekly`)
4. **Environment variables** (like `.env` for API keys)

You can now run:
- `make nhl_totals_all` - Full pipeline
- `make nhl_totals_predict` - Daily predictions

Just like you run `make monday_all` for NFL.

## Next Steps

1. **Test the pipeline** - Run `make nhl_totals_all` to see if it works
2. **Backtest** - Evaluate model performance on historical games
3. **Web interface** - Build NHL page like your NFL props page
4. **Player props** - Apply same methodology to goals, assists, shots
5. **Live updates** - Integrate into daily workflow

## Notes

- NHL season format: `20242025` = 2024-25 season
- API has rate limits - scripts include delays
- First run takes time (fetching full season)
- Subsequent runs are fast (just today's predictions)
- Model retrains when you run `make nhl_totals_train`

## Consensus Edge Strategy 🎯

**The Morning Play** - NHL lines often move throughout the day. Get in early before corrections:

### How It Works

1. **Model predicts**: 5.0 goals
2. **Market consensus**: 5.0 goals (average across all books)
3. **Outlier book**: Has line at 6.0 goals

**Model + Consensus agree** → The true number is ~5.0

**Outlier book is wrong** → Line will likely move from 6.0 → 5.0

**The play**: Bet **Under 6.0** at the outlier book before they correct

### Example Output

```
BOS @ TOR
  Book: DraftKings
  BET: Under 6.0 (-110)
  Consensus: 5.5 | Model: 5.2
  Edge: 0.5 goals from consensus
  Book 6.0 vs Consensus 5.5 (Model: 5.2)
```

This is **structural arbitrage** applied to totals:
- You're not betting against the market
- You're betting on the market correcting an outlier
- Model + consensus give you confidence the outlier is wrong

### When to Run

**Best time**: Early morning when lines first post
**Why**: Books that haven't synced with market yet
**Goal**: Get your bet in before they move the line

```bash
# Run this first thing in the morning
make nhl_totals_daily
```

## Comparison to Your NFL System

| NFL Props | NHL Totals (New) |
|-----------|------------------|
| Player-level predictions | Team-level predictions |
| Family-based parameters | Rolling window features |
| Exponential weighting | ✓ Same approach |
| Opponent adjustments | ✓ Same approach |
| Home/away factors | ✓ Same approach |
| Devigged odds comparison | ✓ Same approach |
| **Structural arbitrage** | ✓ **Consensus edge finder** |

The methodology is consistent - just adapted for team totals instead of player props.
