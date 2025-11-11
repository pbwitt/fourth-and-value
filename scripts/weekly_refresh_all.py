"""
Complete Weekly Refresh - All Sports
Runs the full pipeline for NFL and NHL
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Load environment variables from .env
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

def run_script(script_path, description, required=True):
    """Run a script and handle errors"""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)

    result = subprocess.run(['python3', script_path], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ ERROR in {description}")
        print(result.stderr)
        if required:
            print("\n⚠️  CRITICAL ERROR - Stopping pipeline")
            sys.exit(1)
        else:
            print("⚠️  Non-critical error - continuing...")
    else:
        print(result.stdout)
        print(f"✓ {description} completed")

    return result.returncode == 0

print("="*80)
print(f"WEEKLY REFRESH - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# NFL PIPELINE
# ============================================================================

print("\n" + "#"*80)
print("# NFL PIPELINE")
print("#"*80)

# 1. Fetch NFL lines
print("\n[NFL 1/7] Fetching NFL lines...")
result = subprocess.run([
    'python3', 'scripts/nfl_fetch_totals_spreads.py',
    '--output', 'data/nfl/lines/lines_latest.csv'
], capture_output=True, text=True, env=os.environ.copy())
if result.returncode == 0:
    print(result.stdout)
    print("✓ NFL Lines Fetch completed")
else:
    print("⚠️  NFL Lines fetch failed (non-critical)")
    print(result.stderr)

# 2. Fetch scores for grading
print("\n[NFL 2/7] Fetching NFL scores...")
result = subprocess.run([
    'python3', 'scripts/nfl_fetch_scores.py',
    '--output', 'data/nfl/scores/scores_latest.csv'
], capture_output=True, text=True, env=os.environ.copy())
if result.returncode == 0:
    print(result.stdout)
    print("✓ NFL Scores Fetch completed")
else:
    print("❌ ERROR in NFL Scores Fetch")
    print(result.stderr)
    sys.exit(1)

# 3. Grade NFL bets
print("\n[NFL 3/7] Grading NFL bets...")
run_script('scripts/grade_bets_nfl.py', 'NFL Bet Grading')

# 4. Train all NFL models
print("\n[NFL 4/7] Training EPA model...")
result = subprocess.run([
    'python3', 'scripts/nfl_build_team_features.py',
    '--pbp', 'data/pbp/pbp_2025_updated.parquet',
    '--output', 'data/nfl/processed/team_features.csv'
], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
    print("✓ NFL EPA Features completed")
else:
    print("❌ ERROR in NFL EPA Features")
    print(result.stderr)
    sys.exit(1)

run_script('scripts/nfl_train_totals_model.py', 'NFL EPA Model Training')

print("\n[NFL 5/7] Training Box Score + Pace models...")
run_script('scripts/nfl_train_boxscore_model.py', 'NFL Box Score Model')
run_script('scripts/nfl_train_pace_model.py', 'NFL Pace Model')

# 5. Generate predictions
print("\n[NFL 6/7] Generating predictions...")
run_script('scripts/nfl_predict_totals.py', 'NFL EPA Predictions')
run_script('scripts/nfl_predict_pace.py', 'NFL Pace Predictions')
run_script('scripts/nfl_simple_calibration.py', 'NFL Calibration')
run_script('scripts/nfl_final_ensemble.py', 'NFL Ensemble Predictions')

# 6. Build NFL pages
print("\n[NFL 7/7] Building NFL pages...")
result = subprocess.run([
    'python3', 'scripts/nfl_build_totals_page.py',
    '--week', '11'
], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
    print("✓ NFL Totals Page completed")
else:
    print("❌ ERROR in NFL Totals Page")
    print(result.stderr)
    sys.exit(1)

# ============================================================================
# NHL PIPELINE
# ============================================================================

print("\n" + "#"*80)
print("# NHL PIPELINE")
print("#"*80)

# 1. Fetch NHL data
print("\n[NHL 1/6] Fetching NHL games...")
result = subprocess.run([
    'python3', 'scripts/nhl/fetch_nhl_schedule.py',
    '--output', 'data/nhl/raw/games.csv'
], capture_output=True, text=True)
print(result.stdout)

print("\n[NHL 2/6] Fetching NHL player stats...")
result = subprocess.run([
    'python3', 'scripts/nhl_fetch_player_stats.py',
    '--games', 'data/nhl/raw/games.csv',
    '--output', 'data/nhl/raw/player_stats.csv'
], capture_output=True, text=True)
print(result.stdout)

print("\n[NHL 2.5/6] Processing NHL stats (skater logs, season stats)...")
result = subprocess.run([
    'python3', 'scripts/nhl/fetch_nhl_stats.py'
], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
    print("✓ NHL Stats Processing completed")
else:
    print("❌ ERROR in NHL Stats Processing")
    print(result.stderr)
    sys.exit(1)

# 2. Train NHL models
print("\n[NHL 3/6] Training NHL models...")
run_script('scripts/nhl/train_scoring_models.py', 'NHL Scoring Models')

# 3. Generate predictions
print("\n[NHL 4/6] Generating NHL predictions...")
result = subprocess.run([
    'python3', 'scripts/nhl/predict_nhl_totals.py',
    '--games', 'data/nhl/raw/games.csv',
    '--stats', 'data/nhl/raw/player_stats.csv',
    '--model', 'data/nhl/models/goals_model_latest.pkl',
    '--output', 'data/nhl/predictions/predictions.csv'
], capture_output=True, text=True)
print(result.stdout)

# 4. Build NHL pages
print("\n[NHL 5/6] Building NHL totals page...")
run_script('scripts/nhl_build_totals_page.py', 'NHL Totals Page')

print("\n[NHL 6/6] Building NHL props page...")
run_script('scripts/nhl/build_nhl_props_page.py', 'NHL Props Page')

# ============================================================================
# PROPS PIPELINE (if needed)
# ============================================================================

print("\n" + "#"*80)
print("# PROPS PIPELINE (Optional)")
print("#"*80)

# NFL Props
print("\n[PROPS] Fetching NFL props...")
result = subprocess.run([
    'python3', 'scripts/fetch_all_player_props.py'
], capture_output=True, text=True)
if result.returncode == 0:
    print(result.stdout)
    print("\n[PROPS] Building props parameters...")
    run_script('scripts/make_player_prop_params.py', 'Props Parameters', required=False)

    print("\n[PROPS] Calculating edges...")
    run_script('scripts/make_props_edges.py', 'Props Edges', required=False)

    print("\n[PROPS] Building props page...")
    run_script('scripts/build_props_site.py', 'Props Page', required=False)
else:
    print("⚠️  Props pipeline skipped (non-critical)")

# ============================================================================
# DEPLOYMENT
# ============================================================================

print("\n" + "#"*80)
print("# DEPLOYMENT")
print("#"*80)

# Check if we should push to GitHub Pages
print("\n[DEPLOY] Checking for changes...")
result = subprocess.run(['git', 'status', '--porcelain', 'docs/'],
                       capture_output=True, text=True, cwd='/Users/pwitt/fourth-and-value')

if result.stdout.strip():
    print("Changes detected in docs/")

    response = input("\n🚀 Push updates to production? (y/n): ")

    if response.lower() == 'y':
        print("\n[DEPLOY] Pushing to GitHub...")

        # Add changes
        subprocess.run(['git', 'add', 'docs/'], cwd='/Users/pwitt/fourth-and-value')

        # Commit
        commit_msg = f"Weekly refresh - {datetime.now().strftime('%Y-%m-%d')}"
        subprocess.run(['git', 'commit', '-m', commit_msg],
                      cwd='/Users/pwitt/fourth-and-value')

        # Push
        result = subprocess.run(['git', 'push'],
                              capture_output=True, text=True,
                              cwd='/Users/pwitt/fourth-and-value')

        if result.returncode == 0:
            print("✓ Deployed to production!")
            print("\n📊 Site will be live at: https://pwitt-nfl.github.io/fourth-and-value/")
        else:
            print("❌ Deployment failed:")
            print(result.stderr)
    else:
        print("⏸️  Deployment skipped")
else:
    print("No changes detected - nothing to deploy")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WEEKLY REFRESH COMPLETE")
print("="*80)

print("\n📋 Summary:")
print("  ✓ NFL: Data fetched, models trained, predictions generated")
print("  ✓ NHL: Data fetched, models trained, predictions generated")
print("  ✓ Props: Parameters calculated, edges identified")
print("  ✓ Pages: All pages rebuilt")

print("\n📂 Key Files:")
print("  NFL Predictions: data/nfl/predictions/week11_final_calibrated.csv")
print("  NHL Predictions: data/nhl/predictions/predictions.csv")
print("  NFL Page: docs/nfl/totals.html")
print("  NHL Page: docs/nhl/totals.html")

print("\n✓ All done!")
