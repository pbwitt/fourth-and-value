# Defensive Adjustments Implementation Plan

## Current Status
- ✅ L4 (last 4 games) from current season working
- ✅ Family-based modeling working (targets × catch_rate)
- ✅ Exponential weighting (alpha=0.4) working
- ❌ **Defensive adjustments NOT implemented** (function exists but never called)

## What Methods Page Promises
> **Matchup adjustments:**
> - Opponent defensive strength: We compute defensive ratings (0.5-2.0 scale) based on yards allowed per game.
> - Adjustment range: ±15% to player expectations
>   - Weakest defense (0.5) → +15% boost to offense
>   - Average defense (1.0) → No adjustment
>   - Toughest defense (2.0) → -15% penalty to offense

## Implementation Steps

### 1. Calculate Defensive Ratings (NEW FUNCTION)
**File:** `scripts/make_player_prop_params.py`

Create `calculate_defensive_ratings(season: int, week: int) -> pd.DataFrame`:

```python
def calculate_defensive_ratings(season: int, week: int) -> pd.DataFrame:
    """
    Calculate defensive ratings for each team based on yards allowed.

    Returns DataFrame with columns:
        - team: Team abbreviation
        - pass_def_rating: 0.5-2.0 (for receptions, recv_yds, pass_yds, pass_attempts)
        - rush_def_rating: 0.5-2.0 (for rush_yds, rush_attempts)

    Methodology:
        - Use weeks 1 to (week-1) of current season
        - Calculate yards allowed per game
        - Normalize to 0.5-2.0 scale:
            - League worst → 0.5 (easiest matchup, +15% to offense)
            - League average → 1.0 (neutral)
            - League best → 2.0 (toughest matchup, -15% to offense)
    """
    # Load team defense stats from weekly data
    # Group by team, calculate avg yards allowed
    # Normalize to 0.5-2.0 scale
    pass
```

**Data source:** Use nflverse team stats or aggregate from weekly player stats
- Passing yards allowed per game → pass_def_rating
- Rushing yards allowed per game → rush_def_rating

### 2. Add Opponent Lookup to Props
**File:** `scripts/make_player_prop_params.py`

Need to add opponent team to params so we know which defense to apply:

```python
def add_opponent_to_params(params: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent team to params by matching player → team → opponent

    Steps:
        1. Extract player team from props (parse from 'game' column)
        2. Map player → team (handle multiple teams for trades)
        3. Determine opponent from home_team/away_team
        4. Add 'opponent' column to params

    Returns: params with added 'opponent' column
    """
    pass
```

**Example:**
- Saquon Barkley plays for PHI
- Game: "Philadelphia Eagles @ New York Giants"
- Opponent: NYG

### 3. Apply Defensive Adjustments
**File:** `scripts/make_player_prop_params.py` in `build_params()` function

**Current code** (around line 1200):
```python
# Derive market params from latents (ensures coherence)
mu_map, sg_map = {}, {}
for mkt in family_markets:
    mu, sg = derive_market_from_latents(mkt, latents, player_idx)
    mu_map[mkt] = mu
    sg_map[mkt] = sg
```

**NEW code** (after deriving from latents):
```python
# Derive market params from latents
mu_map, sg_map = {}, {}
for mkt in family_markets:
    mu, sg = derive_market_from_latents(mkt, latents, player_idx)
    mu_map[mkt] = mu
    sg_map[mkt] = sg

# ========== Apply Defensive Adjustments ==========
if defensive_ratings is not None:
    for mkt in family_markets:
        # Map market to defense type
        if mkt in ["receptions", "recv_yds"]:
            def_type = "pass_def_rating"
        elif mkt in ["rush_yds", "rush_attempts"]:
            def_type = "rush_def_rating"
        elif mkt in ["pass_yds", "pass_attempts", "pass_completions"]:
            def_type = "pass_def_rating"  # QB facing pass defense
        else:
            continue

        # Apply adjustment per player
        for player in player_idx:
            opponent = opponent_map.get(player)  # Need to create this
            if opponent and opponent in defensive_ratings.index:
                def_rating = defensive_ratings.loc[opponent, def_type]
                mu_map[mkt][player] = apply_defensive_adjustment(
                    mu_map[mkt][player],
                    mkt,
                    def_rating
                )
```

### 4. Update main() to Call Everything
**File:** `scripts/make_player_prop_params.py` in `main()` function

**Add before `build_params()` call:**
```python
# Calculate defensive ratings from current season
defensive_ratings = calculate_defensive_ratings(season, week)
logging.info(f"Calculated defensive ratings for {len(defensive_ratings)} teams")

# Create opponent lookup
opponent_map = create_opponent_map(props)  # New helper function
```

**Pass to build_params:**
```python
params_df = build_params(
    cands,
    logs,
    season,
    week,
    defensive_ratings=defensive_ratings,  # NEW
    opponent_map=opponent_map  # NEW
)
```

### 5. Helper Function: Create Opponent Map
```python
def create_opponent_map(props: pd.DataFrame) -> dict:
    """
    Create player → opponent team mapping from props data.

    Returns:
        dict: {player_name: opponent_team_abbrev}

    Logic:
        - Parse game string to get home/away teams
        - Determine player's team from game
        - Opponent is the other team
    """
    opponent_map = {}
    for _, row in props.iterrows():
        player = row['player']
        game = row.get('game', '')
        home = row.get('home_team', '')
        away = row.get('away_team', '')

        # Determine if player is home or away
        # For now, use simple heuristic or team roster data
        # Map to opponent

    return opponent_map
```

## Testing Plan

### Test 1: Verify Defensive Ratings Calculation
```python
# Expected output:
#     team  pass_def_rating  rush_def_rating
#      NYG             1.85             0.75  (tough pass D, weak rush D)
#      PHI             1.20             1.35
#      ...
```

### Test 2: Verify Saquon Adjustment
**Without defense:**
- Saquon receptions: 3.855 (from 2025 L4)
- Opponent: NYG

**With defense (if NYG pass_def_rating = 1.85):**
- Adjustment: 1.15 - (1.85 - 0.5) * 0.2 = 0.88 (12% penalty)
- Adjusted receptions: 3.855 * 0.88 = 3.39
- This would LOWER the edge (we're projecting fewer receptions)

### Test 3: Full Pipeline
```bash
python3 scripts/make_player_prop_params.py --season 2025 --week 6 \
    --props_csv data/props/latest_all_props.csv \
    --out data/props/params_week6.csv

# Verify params have been adjusted
# Check implied_cr stays the same (defense doesn't affect CR)
# Check mu is adjusted based on opponent
```

## Impact Analysis

**Before defense adjustments:**
- Saquon: 3.855 receptions → P(>2.5) = 96.0%
- Edge: +4,387 bps

**After defense adjustments (estimated):**
- If facing tough defense (1.8 rating): 3.855 * 0.89 = 3.43 receptions
- P(>2.5) = ~88%
- Edge: +3,500 bps (still good, but 887 bps lower)

**This could explain the gap between our model and market!** The market may already be pricing in defensive strength, while we're not.

## Files to Modify
1. `scripts/make_player_prop_params.py` (main implementation)
2. `data/props/params_week6.csv` (regenerate)
3. `data/props/props_with_model_week6.csv` (rebuild edges)
4. `docs/props/index.html` (rebuild page)
5. `docs/blog/early-lines-sharp-advantage.html` (update Saquon numbers)

## Estimated Effort
- Defensive ratings calculation: 30 min
- Opponent lookup logic: 30 min
- Integration into build_params: 30 min
- Testing and debugging: 60 min
- **Total: ~2.5 hours**

## Notes
- Need to handle team abbreviation mismatches (e.g., "Philadelphia Eagles" vs "PHI")
- Need to handle players who changed teams mid-season
- May want to add defensive rating to params CSV for transparency
- Should add defensive adjustment info to methods page details
