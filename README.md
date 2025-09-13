# NFL-2025 🏈 — Data Pulls + Player ML Predictions

End-to-end workflow to:
1) pull **NFL raw player/supplemental data** (historical + weekly updates),
2) train **player-level ML models** with recent-form + season-to-date + opponent-defense features,
3) generate **weekly predictions** (QB passing yards, RB rushing yards, WR/TE receiving yards, Fantasy PPR),
4) optionally roll those up later to game-level outcomes.

> Built around `nfl_data_py` (nflverse). Parquet for speed, CSV for portability.

---

## Repo Structure
NFL-2025/
├─ scripts/
│ ├─ pull_nfl_player_data.py # player-level datasets (full history + latest-week mode)
│ ├─ pull_nfl_supplemental_data.py # schedules, PBP (optional), etc.
│ ├─ ml_player_pipeline.py # feature engineering + modeling + per-week predictions
│ ├─ run_all_predictions.sh # (optional) run all positions/targets in one go
├─ data/ # outputs from pull scripts (CSV/Parquet)
│ ├─ weekly_player_stats.csv
│ ├─ schedules.csv
├─ preds/ # saved prediction CSVs
├─ notebooks/ # optional: your experiments
├─ requirements.txt
└─ README.md
