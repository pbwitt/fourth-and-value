# ---- Config ----
PY ?= python3
SEASON ?=
WEEK   ?=
DATE   ?= $(shell date +%Y-%m-%d)

# Require SEASON/WEEK for NFL targets
ifeq ($(strip $(SEASON)),)
  ifeq ($(filter nhl_%,$(MAKECMDGOALS)),)
    $(error SEASON=YYYY required (e.g., SEASON=2025))
  endif
endif
ifeq ($(strip $(WEEK)),)
  ifeq ($(filter nhl_%,$(MAKECMDGOALS)),)
    $(error WEEK=N required (e.g., WEEK=3))
  endif
endif

DOCS_DIR   := docs
PROPS_DIR  := data/props
ODDS_DIR   := data/odds
QC_DIR     := data/qc

PROPS_ALL  := $(PROPS_DIR)/latest_all_props.csv
PARAMS     := $(PROPS_DIR)/params_week$(WEEK).csv
MERGED     := $(PROPS_DIR)/props_with_model_week$(WEEK).csv

PROPS_HTML := $(DOCS_DIR)/props/index.html
TOP_HTML   := $(DOCS_DIR)/props/top.html
INSIGHTS_HTML := $(DOCS_DIR)/props/insights.html
INSIGHTS_JSON := data/ai/insights_week$(WEEK).json
ARB_HTML   := $(DOCS_DIR)/props/arbitrage.html
ODDS_CSV   := $(ODDS_DIR)/latest.csv
FAM_ARB_CSV := data/qc/family_arbitrage.csv
INCOH_CSV  := data/qc/incoherent_books.csv

# ---- Phony targets ----
.PHONY: monday_all monday_all_pub weekly qc publish_pages props_now_pages serve_preview clean_pages clean
.PHONY: nhl_daily nhl_daily_pub nhl_odds nhl_stats nhl_consensus nhl_edges nhl_page
.PHONY: nhl_totals_fetch nhl_totals_features nhl_totals_train nhl_totals_predict nhl_totals_consensus nhl_totals_all

# Main weekly build (no publish)
monday_all: weekly qc
	@echo "[OK] Weekly build complete: SEASON=$(SEASON) WEEK=$(WEEK)"
	@echo "[OK] All QC checks passed ✓"

# Weekly pipeline (consensus removed - now integrated into Props page)
weekly: $(PROPS_HTML) $(TOP_HTML) $(INSIGHTS_HTML) $(ARB_HTML)

# QC checks (run after weekly build)
.PHONY: qc
qc: $(MERGED) $(PARAMS) $(PROPS_ALL)
	@echo "===================================================================="
	@echo "Running QC checks for Week $(WEEK)..."
	@echo "===================================================================="
	$(PY) scripts/weekly_qc_checks.py \
		--season $(SEASON) \
		--week $(WEEK) \
		--props $(PROPS_ALL) \
		--params $(PARAMS) \
		--edges $(MERGED)
	@echo "===================================================================="
	@echo "QC checks complete!"
	@echo "===================================================================="

# ---- Steps ----
# 0) Ensure dirs exist
$(PROPS_DIR) $(DOCS_DIR)/props $(ODDS_DIR) $(QC_DIR) data/ai:
	mkdir -p $@

# 1) Odds (stdout to file, as your script expects)
$(ODDS_CSV): scripts/fetch_odds.py | $(ODDS_DIR)
	$(PY) scripts/fetch_odds.py > $@

# 2) Fetch all player props → latest_all_props.csv
# (Your script typically writes to $(PROPS_ALL) without args.)
$(PROPS_ALL): scripts/fetch_all_player_props.py | $(PROPS_DIR) $(ODDS_CSV)
	$(PY) scripts/fetch_all_player_props.py
	@test -s $(PROPS_ALL) || (echo "[ERR] $(PROPS_ALL) not created"; exit 1)
	@echo "[VALIDATION] Checking props data freshness..."
	@$(PY) scripts/validate_data_freshness.py --props $(PROPS_ALL) || (echo "[ERR] Props data is stale!"; exit 1)

# 3) Build params
$(PARAMS): scripts/make_player_prop_params.py $(PROPS_ALL) | $(PROPS_DIR)
	$(PY) scripts/make_player_prop_params.py \
	  --season $(SEASON) --week $(WEEK) \
	  --out $@

# 4) Compute edges → merged CSV
$(MERGED): scripts/make_props_edges.py $(PARAMS) $(PROPS_ALL) | $(PROPS_DIR)
	$(PY) scripts/make_props_edges.py \
	  --season $(SEASON) --week $(WEEK) \
	  --props_csv $(PROPS_ALL) \
	  --params_csv $(PARAMS) \
	  --out $@

# 4b) QC family coherence → incoherent books and family arbitrage CSVs
$(INCOH_CSV) $(FAM_ARB_CSV): scripts/qc_family_coherence.py $(MERGED) $(PARAMS) | $(QC_DIR)
	$(PY) scripts/qc_family_coherence.py \
	  --props $(MERGED) \
	  --params $(PARAMS) \
	  --out-dir data/qc

# 5) Build pages
$(PROPS_HTML): scripts/build_props_site.py $(MERGED) | $(DOCS_DIR)/props
	$(PY) scripts/build_props_site.py \
	  --merged_csv $(MERGED) \
	  --out $@ \
	  --season $(SEASON) \
	  --week $(WEEK) \
	  --title "Fourth & Value — Player Props (Week $(WEEK))" \
	  --drop_no_scorer

$(TOP_HTML): scripts/build_top_picks.py $(MERGED) | $(DOCS_DIR)/props
	$(PY) scripts/build_top_picks.py \
	  --merged_csv $(MERGED) \
	  --out $@ \
	  --season $(SEASON) \
	  --week $(WEEK) \
	  --title "Top Picks — Week $(WEEK)"

# 5b) Generate AI insights JSON
$(INSIGHTS_JSON): scripts/make_ai_commentary.py $(MERGED) | data/ai
	$(PY) scripts/make_ai_commentary.py \
	  --season $(SEASON) \
	  --week $(WEEK) \
	  --merged_csv $(MERGED) \
	  --out_json $@ \
	  --force

$(INSIGHTS_HTML): scripts/build_insights_page.py $(INSIGHTS_JSON) | $(DOCS_DIR)/props
	$(PY) scripts/build_insights_page.py \
	  --season $(SEASON) \
	  --week $(WEEK) \
	  --title "Fourth & Value — Insights (Week $(WEEK))" \
	  --out $@

$(ARB_HTML): scripts/build_arbitrage_page.py $(INCOH_CSV) $(MERGED) | $(DOCS_DIR)/props
	$(PY) scripts/build_arbitrage_page.py \
	  --incoherent-csv $(INCOH_CSV) \
	  --props-csv $(MERGED) \
	  --out $@


# Pages-only rebuild (when CSV already exists)
props_now_pages: $(PROPS_HTML) $(TOP_HTML) $(INSIGHTS_HTML)
	@echo "[OK] Pages rebuilt from $(MERGED)"

# Local preview
serve_preview:
	cd $(DOCS_DIR) && $(PY) -m http.server 8010

# Publish (guarded)
publish_pages:
	@touch $(DOCS_DIR)/.nojekyll
	@if [ "$(PUBLISH)" = "1" ] && [ "$(CONFIRM)" = "LIVE" ]; then \
	  git add -A; \
	  git commit -m "Publish Week $(WEEK) pages"; \
	  git push; \
	else \
	  echo "Not publishing. Use: make monday_all_pub SEASON=$(SEASON) WEEK=$(WEEK) PUBLISH=1 CONFIRM=LIVE"; \
	fi

# Monday full run + publish (opt-in)
monday_all_pub: monday_all publish_pages

# Cleanup
clean_pages:
	rm -f $(PROPS_HTML) $(TOP_HTML) $(INSIGHTS_HTML)

clean:
	rm -f $(PARAMS) $(MERGED)


# ========================================================================
# NHL Daily Pipeline
# ========================================================================

# NHL directories
NHL_DATA_DIR := data/nhl
NHL_PROC_DIR := $(NHL_DATA_DIR)/processed
NHL_CONS_DIR := $(NHL_DATA_DIR)/consensus
NHL_PROPS_DIR := $(NHL_DATA_DIR)/props
NHL_DOCS_DIR := docs/nhl/props

# NHL outputs
NHL_ODDS_PROPS := $(NHL_PROC_DIR)/odds_props_$(DATE).csv
NHL_ODDS_GAMES := $(NHL_PROC_DIR)/odds_games_$(DATE).csv
NHL_STATS_SKATERS := $(NHL_PROC_DIR)/skater_logs_$(DATE).parquet
NHL_STATS_GOALIES := $(NHL_PROC_DIR)/goalie_logs_$(DATE).parquet
NHL_CONSENSUS_PROPS := $(NHL_CONS_DIR)/consensus_props_$(DATE).csv
NHL_CONSENSUS_GAMES := $(NHL_CONS_DIR)/consensus_games_$(DATE).csv
NHL_MODELS := $(NHL_DATA_DIR)/models/sog_model_latest.pkl
NHL_PROPS_MODEL := $(NHL_PROPS_DIR)/props_with_model_$(DATE).csv
NHL_PAGE := $(NHL_DOCS_DIR)/index.html

# Full daily pipeline
nhl_daily: $(NHL_PAGE)
	@echo "===================================================================="
	@echo "NHL daily build complete for $(DATE)"
	@echo "===================================================================="
	@echo "✓ Odds:      $(NHL_ODDS_PROPS)"
	@echo "✓ Stats:     $(NHL_STATS_SKATERS)"
	@echo "✓ Consensus: $(NHL_CONSENSUS_PROPS)"
	@echo "✓ Edges:     $(NHL_PROPS_MODEL)"
	@echo "✓ Page:      $(NHL_PAGE)"
	@echo "===================================================================="
	@echo "Running QC checks..."
ifeq ($(LIVE),1)
	@$(PY) scripts/nhl/nhl_qc_checks.py --date $(DATE) --warn-only
else
	@$(PY) scripts/nhl/nhl_qc_checks.py --date $(DATE)
endif

# NHL daily build + publish to prod (requires LIVE=1)
nhl_daily_pub: nhl_daily
ifeq ($(LIVE),1)
	@echo "===================================================================="
	@echo "Publishing NHL props page to production..."
	@echo "===================================================================="
	@git add $(NHL_PAGE) $(NHL_PROPS_MODEL) $(NHL_QC_REPORT) || true
	@git commit -m "NHL: Update props page for $(DATE)" || echo "No changes to commit"
	@git push
	@echo "✓ Published to GitHub Pages"
else
	@echo "===================================================================="
	@echo "DRY RUN - NHL page built but not published"
	@echo "===================================================================="
	@echo "To publish to production, run:"
	@echo "  make nhl_daily_pub DATE=$(DATE) LIVE=1"
	@echo "===================================================================="
endif

# Individual steps
nhl_odds: $(NHL_ODDS_PROPS) $(NHL_ODDS_GAMES)

nhl_stats: $(NHL_STATS_SKATERS) $(NHL_STATS_GOALIES)

nhl_consensus: $(NHL_CONSENSUS_PROPS) $(NHL_CONSENSUS_GAMES)

nhl_edges: $(NHL_PROPS_MODEL)

nhl_page: $(NHL_PAGE)

# Fetch NHL odds
$(NHL_ODDS_PROPS) $(NHL_ODDS_GAMES): scripts/nhl/fetch_nhl_odds.py | $(NHL_PROC_DIR)
	$(PY) scripts/nhl/fetch_nhl_odds.py --date $(DATE)

# Fetch NHL stats
$(NHL_STATS_SKATERS) $(NHL_STATS_GOALIES): scripts/nhl/fetch_nhl_stats.py | $(NHL_PROC_DIR)
	$(PY) scripts/nhl/fetch_nhl_stats.py --date $(DATE)

# Train models (uncalibrated)
$(NHL_MODELS): $(NHL_STATS_SKATERS) scripts/nhl/train_sog_model.py scripts/nhl/train_scoring_models.py | $(NHL_DATA_DIR)/models
	$(PY) scripts/nhl/train_sog_model.py --date $(DATE)
	$(PY) scripts/nhl/train_scoring_models.py --date $(DATE)

# Compute consensus (needed for calibration targets)
$(NHL_CONSENSUS_PROPS) $(NHL_CONSENSUS_GAMES): $(NHL_ODDS_PROPS) scripts/nhl/make_nhl_consensus.py | $(NHL_CONS_DIR)
	$(PY) scripts/nhl/make_nhl_consensus.py --date $(DATE)

# Calibrate models (depends on consensus)
$(NHL_DATA_DIR)/models/.calibrated_$(DATE): $(NHL_MODELS) $(NHL_CONSENSUS_PROPS) scripts/nhl/calibrate_nhl_models.py
	$(PY) scripts/nhl/calibrate_nhl_models.py --date $(DATE)
	touch $@

# Compute edges (depends on calibrated models)
$(NHL_PROPS_MODEL): $(NHL_ODDS_PROPS) $(NHL_CONSENSUS_PROPS) $(NHL_DATA_DIR)/models/.calibrated_$(DATE) scripts/nhl/make_nhl_edges.py | $(NHL_PROPS_DIR)
	$(PY) scripts/nhl/make_nhl_edges.py --date $(DATE)

# Build page
$(NHL_PAGE): $(NHL_PROPS_MODEL) scripts/nhl/build_nhl_props_page.py | $(NHL_DOCS_DIR)
	$(PY) scripts/nhl/build_nhl_props_page.py --date $(DATE)

# Ensure NHL directories exist
$(NHL_PROC_DIR) $(NHL_CONS_DIR) $(NHL_PROPS_DIR) $(NHL_DOCS_DIR) $(NHL_DATA_DIR)/models:
	mkdir -p $@

# ========================================================================
# NHL Team Totals Pipeline (NEW)
# ========================================================================

NHL_SEASON ?= 20242025
NHL_GAMES_CSV := data/nhl/raw/games.csv
NHL_PLAYER_STATS_CSV := data/nhl/raw/player_stats.csv
NHL_TEAM_FEATURES_CSV := data/nhl/processed/team_features.csv
NHL_MODEL_PKL := data/nhl/models/ridge_team_totals.pkl
NHL_PREDICTIONS_CSV := data/nhl/predictions/today.csv
NHL_CONSENSUS_EDGES_CSV := data/nhl/consensus/edges.csv
NHL_CONSENSUS_CSV := data/nhl/consensus/consensus.csv
NHL_TOTALS_PAGE := docs/nhl/totals/index.html

# Fetch historical games for the season
nhl_totals_fetch:
	@echo "===================================================================="
	@echo "Fetching NHL games for season $(NHL_SEASON)..."
	@echo "===================================================================="
	$(PY) scripts/nhl_fetch_games.py --season $(NHL_SEASON) --output $(NHL_GAMES_CSV)
	@echo "===================================================================="
	@echo "Fetching player stats for games..."
	@echo "===================================================================="
	$(PY) scripts/nhl_fetch_player_stats.py --games $(NHL_GAMES_CSV) --output $(NHL_PLAYER_STATS_CSV)
	@echo "✓ Data fetched"

# Build team-level features from player stats
nhl_totals_features:
	@echo "===================================================================="
	@echo "Building team-level features..."
	@echo "===================================================================="
	$(PY) scripts/nhl_build_features.py --input $(NHL_PLAYER_STATS_CSV) --output $(NHL_TEAM_FEATURES_CSV)
	@echo "✓ Features built"

# Train model
nhl_totals_train:
	@echo "===================================================================="
	@echo "Training team totals model..."
	@echo "===================================================================="
	$(PY) scripts/nhl_train_model.py --input $(NHL_TEAM_FEATURES_CSV) --model-type ridge --output-dir data/nhl/models
	@echo "✓ Model trained"

# Generate predictions for today
nhl_totals_predict:
	@echo "===================================================================="
	@echo "Generating predictions for today..."
	@echo "===================================================================="
	$(PY) scripts/nhl_predict_totals.py --model $(NHL_MODEL_PKL) --team-features $(NHL_TEAM_FEATURES_CSV) --output $(NHL_PREDICTIONS_CSV)
	@echo "✓ Predictions generated"

# Find consensus edges (books out of sync with market)
nhl_totals_consensus:
	@echo "===================================================================="
	@echo "Finding consensus edges..."
	@echo "===================================================================="
	$(PY) scripts/nhl_find_consensus_edges.py --predictions $(NHL_PREDICTIONS_CSV) --output $(NHL_CONSENSUS_EDGES_CSV)
	@echo "✓ Consensus edges identified"

# Build HTML page
nhl_totals_page:
	@echo "===================================================================="
	@echo "Building NHL totals page..."
	@echo "===================================================================="
	$(PY) scripts/nhl_build_totals_page.py --predictions $(NHL_PREDICTIONS_CSV) --consensus $(NHL_CONSENSUS_CSV) --edges $(NHL_CONSENSUS_EDGES_CSV) --output $(NHL_TOTALS_PAGE)
	@echo "✓ Page built: $(NHL_TOTALS_PAGE)"

# Full pipeline: fetch → features → train → predict → consensus → page
nhl_totals_all: nhl_totals_fetch nhl_totals_features nhl_totals_train nhl_totals_predict nhl_totals_consensus nhl_totals_page
	@echo "===================================================================="
	@echo "✓ NHL Totals Pipeline Complete"
	@echo "===================================================================="
	@echo "Predictions: $(NHL_PREDICTIONS_CSV)"
	@echo "Consensus edges: $(NHL_CONSENSUS_EDGES_CSV)"
	@echo "Page: $(NHL_TOTALS_PAGE)"

# Daily run (just predictions + consensus + page, assumes model already trained)
nhl_totals_daily: nhl_totals_predict nhl_totals_consensus nhl_totals_page
	@echo "===================================================================="
	@echo "✓ Daily NHL totals update complete"
	@echo "===================================================================="
	@echo "View at: $(NHL_TOTALS_PAGE)"

# ========================================================================
# End NHL Team Totals Pipeline
# ========================================================================

# ========================================================================
# End NHL Pipeline
# ========================================================================


# ========================================================================
# NFL Team Totals Pipeline
# ========================================================================

NFL_PBP := data/pbp/pbp_2024_2025.parquet
NFL_TEAM_FEATURES := data/nfl/processed/team_features.csv
NFL_TOTALS_MODEL := data/nfl/models/ridge_totals.pkl
NFL_TOTALS_PREDS := data/nfl/predictions/week_predictions.csv
NFL_TOTALS_CONSENSUS := data/nfl/consensus/consensus.csv
NFL_TOTALS_EDGES := data/nfl/consensus/edges.csv
NFL_TOTALS_PAGE := docs/nfl/totals/index.html

# Phony targets
.PHONY: nfl_totals_fetch nfl_totals_features nfl_totals_train nfl_totals_predict nfl_totals_lines nfl_totals_consensus nfl_totals_page nfl_totals_daily

# Fetch PBP data (one-time or when new season starts)
nfl_totals_fetch:
	@echo "====================================================================="
	@echo "Fetching NFL play-by-play data..."
	@echo "====================================================================="
	$(PY) -c "import nfl_data_py as nfl; pbp = nfl.import_pbp_data([2024, 2025]); pbp.to_parquet('$(NFL_PBP)'); print(f'✓ Fetched {len(pbp)} plays')"
	@echo "✓ PBP data saved to $(NFL_PBP)"

# Build team features from PBP
nfl_totals_features: $(NFL_TEAM_FEATURES)

$(NFL_TEAM_FEATURES): $(NFL_PBP) scripts/nfl_build_team_features.py
	@echo "====================================================================="
	@echo "Building NFL team features..."
	@echo "====================================================================="
	$(PY) scripts/nfl_build_team_features.py --pbp $(NFL_PBP) --output $(NFL_TEAM_FEATURES)

# Train model
nfl_totals_train: $(NFL_TOTALS_MODEL)

$(NFL_TOTALS_MODEL): $(NFL_TEAM_FEATURES) scripts/nfl_train_totals_model.py
	@echo "====================================================================="
	@echo "Training NFL totals model..."
	@echo "====================================================================="
	$(PY) scripts/nfl_train_totals_model.py --input $(NFL_TEAM_FEATURES) --output-dir data/nfl/models

# Generate predictions for specific week
nfl_totals_predict:
	@echo "====================================================================="
	@echo "Generating NFL totals predictions for Week $(WEEK)..."
	@echo "====================================================================="
	$(PY) scripts/nfl_predict_totals.py \
		--model $(NFL_TOTALS_MODEL) \
		--team-features $(NFL_TEAM_FEATURES) \
		--week $(WEEK) \
		--output $(NFL_TOTALS_PREDS)

# Fetch totals and spreads from sportsbooks
nfl_totals_lines:
	@echo "====================================================================="
	@echo "Fetching NFL totals and spreads from sportsbooks..."
	@echo "====================================================================="
	$(PY) scripts/nfl_fetch_totals_spreads.py \
		--output data/nfl/lines/totals_spreads.csv
	@echo "✓ Book lines fetched"

# Calculate consensus across books
nfl_totals_consensus: nfl_totals_lines
	@echo "====================================================================="
	@echo "Calculating consensus totals and spreads..."
	@echo "====================================================================="
	$(PY) scripts/nfl_calculate_totals_consensus.py \
		--lines data/nfl/lines/totals_spreads.csv \
		--output data/nfl/consensus/totals_spreads_consensus.csv
	@echo "✓ Consensus calculated"

# Build HTML page
nfl_totals_page:
	@echo "====================================================================="
	@echo "Building NFL totals page..."
	@echo "====================================================================="
	$(PY) scripts/nfl_build_totals_page.py \
		--predictions $(NFL_TOTALS_PREDS) \
		--consensus data/nfl/consensus/totals_spreads_consensus.csv \
		--edges $(NFL_TOTALS_EDGES) \
		--lines data/nfl/lines/totals_spreads.csv \
		--output $(NFL_TOTALS_PAGE) \
		--week $(WEEK)

# Weekly run: predict + consensus + page
nfl_totals_daily: nfl_totals_predict nfl_totals_consensus nfl_totals_page
	@echo "====================================================================="
	@echo "✓ NFL totals update complete for Week $(WEEK)"
	@echo "====================================================================="
	@echo "View at: $(NFL_TOTALS_PAGE)"

# ========================================================================
# End NFL Team Totals Pipeline
# ========================================================================


.FORCE:
	$(ODDS_CSV): .FORCE scripts/fetch_odds.py | $(ODDS_DIR)
		$(PY) scripts/fetch_odds.py > $@

	$(PROPS_ALL): .FORCE scripts/fetch_all_player_props.py | $(PROPS_DIR) $(ODDS_CSV)
		$(PY) scripts/fetch_all_player_props.py
		@test -s $(PROPS_ALL) || (echo "[ERR] $(PROPS_ALL) not created"; exit 1)
