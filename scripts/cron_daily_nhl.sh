#!/bin/bash
# Cron wrapper for daily NHL refresh
# Activates conda environment and runs the refresh script

# Set up PATH for conda
export PATH="/Users/pwitt/opt/anaconda3/bin:$PATH"

# Activate conda environment
source /Users/pwitt/opt/anaconda3/etc/profile.d/conda.sh
conda activate nfl2025

# Run the daily refresh
cd /Users/pwitt/fourth-and-value
bash scripts/daily_nhl_refresh.sh
