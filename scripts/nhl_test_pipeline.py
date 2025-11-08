"""
Test the NHL totals pipeline with a small sample
"""
import sys
import os

print("=" * 70)
print("NHL Totals Pipeline - Quick Test")
print("=" * 70)

# Test 1: Can we import the scripts?
print("\n[1/4] Testing imports...")
try:
    sys.path.insert(0, 'scripts')
    import nhl_fetch_games
    import nhl_fetch_player_stats
    import nhl_build_features
    import nhl_train_model
    print("✓ All scripts import successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Can we fetch today's schedule?
print("\n[2/4] Testing schedule fetch...")
try:
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    games = nhl_fetch_games.get_todays_games()
    print(f"✓ Fetched {len(games)} games for {today}")
    if len(games) > 0:
        print(f"  Example: {games.iloc[0]['away_team']} @ {games.iloc[0]['home_team']}")
except Exception as e:
    print(f"✗ Schedule fetch error: {e}")

# Test 3: Check data directories
print("\n[3/4] Checking data directories...")
required_dirs = [
    'data/nhl/raw',
    'data/nhl/processed',
    'data/nhl/models',
    'data/nhl/predictions'
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"✓ {dir_path} exists")
    else:
        print(f"  Creating {dir_path}...")
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ {dir_path} created")

# Test 4: Check if we have historical data
print("\n[4/4] Checking for existing data...")
data_files = {
    'data/nhl/raw/games.csv': 'Historical games',
    'data/nhl/raw/player_stats.csv': 'Player stats',
    'data/nhl/processed/team_features.csv': 'Team features',
    'data/nhl/models/ridge_team_totals.pkl': 'Trained model'
}

has_data = False
for file_path, description in data_files.items():
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"✓ {description}: {file_path} ({size_mb:.2f} MB)")
        has_data = True
    else:
        print(f"  {description}: Not found (will be created)")

print("\n" + "=" * 70)
print("Pipeline Test Summary")
print("=" * 70)

if has_data:
    print("\n✓ Existing data found - you can run predictions with:")
    print("  make nhl_totals_predict")
else:
    print("\n  No historical data found - run full pipeline with:")
    print("  make nhl_totals_all NHL_SEASON=20242025")

print("\nThis will:")
print("  1. Fetch all games for the season from NHL API")
print("  2. Fetch player stats for each game")
print("  3. Aggregate player stats to team-level features")
print("  4. Train a Ridge regression model")
print("  5. Generate predictions for today's games")

print("\n" + "=" * 70)
print("Ready to go! 🏒")
print("=" * 70)
