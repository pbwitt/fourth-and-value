#!/bin/bash
# Daily NHL Refresh - Run at 7 AM EST
# Fetches latest data, grades bets, updates predictions, and deploys

set -e  # Exit on error

REPO_DIR="/Users/pwitt/fourth-and-value"
cd "$REPO_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "=========================================="
echo "Daily NHL Refresh - $(date)"
echo "=========================================="

# 1. Fetch latest NHL data
echo ""
echo "[1/7] Fetching NHL games and schedule..."
python3 scripts/nhl_fetch_games.py --season 20252026 --output data/nhl/raw/games.csv

echo ""
echo "[2/7] Fetching NHL player stats..."
python3 scripts/nhl_fetch_player_stats.py --games data/nhl/raw/games.csv --output data/nhl/raw/player_stats.csv

echo ""
echo "[3/7] Processing NHL stats (skater logs, season stats)..."
python3 scripts/nhl/fetch_nhl_stats.py

# 2. Grade pending bets
echo ""
echo "[4/7] Grading pending NHL bets..."
python3 scripts/grade_bets_nhl.py

# 3. Train models and generate predictions
echo ""
echo "[5/7] Training NHL models..."
python3 scripts/nhl/train_scoring_models.py

echo ""
echo "[6/7] Generating NHL predictions..."
python3 scripts/nhl/predict_nhl_totals.py \
    --games data/nhl/raw/games.csv \
    --stats data/nhl/raw/player_stats.csv \
    --model data/nhl/models/goals_model_latest.pkl \
    --output data/nhl/predictions/predictions.csv

# 4. Build pages
echo ""
echo "[7/7] Building NHL pages..."
python3 scripts/nhl_build_totals_page.py
python3 scripts/nhl/build_nhl_props_page.py

# 5. Commit and push to GitHub
echo ""
echo "=========================================="
echo "Checking for changes to deploy..."
echo "=========================================="

# Check if there are any changes in docs/
if git status --porcelain docs/ data/bets/ | grep -q .; then
    echo "✓ Changes detected"

    # Stage changes
    git add docs/
    git add docs/data/bets/bets.csv 2>/dev/null || true

    # Create commit
    COMMIT_MSG="Daily NHL refresh - $(date '+%Y-%m-%d %I:%M %p EST')

- Updated NHL games and player stats
- Graded pending bets
- Refreshed predictions and pages

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

    git commit -m "$COMMIT_MSG"

    # Push to GitHub
    echo ""
    echo "Pushing to production..."
    git push

    echo ""
    echo "✅ Successfully deployed to https://pbwitt.github.io/fourth-and-value/"
else
    echo "ℹ️  No changes detected - nothing to deploy"
fi

echo ""
echo "=========================================="
echo "Daily NHL Refresh Complete - $(date)"
echo "=========================================="
