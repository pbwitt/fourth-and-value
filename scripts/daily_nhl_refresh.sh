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

# 3. Fetch odds and build consensus
TODAY=$(date '+%Y-%m-%d')
echo ""
echo "[5/10] Fetching NHL odds for ${TODAY}..."
python3 scripts/nhl/fetch_nhl_odds.py --date "${TODAY}"

echo ""
echo "[6/10] Computing NHL consensus..."
python3 scripts/nhl/make_nhl_consensus.py --date "${TODAY}"

echo ""
echo "[7/10] Finding consensus edges..."
python3 scripts/nhl_find_consensus_edges.py

# 4. Train models and generate predictions
echo ""
echo "[8/10] Training NHL models..."
python3 scripts/nhl/train_scoring_models.py

echo ""
echo "[9/10] Generating NHL predictions..."
python3 scripts/nhl_predict_totals.py

# 5. Build pages
echo ""
echo "[10/11] Building NHL pages..."
python3 scripts/nhl_build_totals_page.py
python3 scripts/nhl/build_nhl_props_page.py

# 6. QC - Validate data freshness
echo ""
echo "[11/11] Running QC - Validating data freshness..."
if ! python3 scripts/validate_nhl_freshness.py; then
    echo ""
    echo "🚨 CRITICAL: Data freshness validation FAILED"
    echo "🚨 Deployment BLOCKED to prevent publishing stale data"
    echo "🚨 Check the errors above and re-run the pipeline"
    exit 1
fi

# 7. Commit and push to GitHub
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
