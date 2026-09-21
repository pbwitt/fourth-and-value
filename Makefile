# ---- Config ----
PY ?= python3
SEASON ?=
WEEK   ?=
DATE   ?= $(shell date +%Y-%m-%d)

# Require SEASON/WEEK for NFL targets
ifeq ($(strip $(SEASON)),)
  ifeq ($(filter nhl_% nba_%,$(MAKECMDGOALS)),)
    $(error SEASON=YYYY required (e.g., SEASON=2025))
  endif
endif
ifeq ($(strip $(WEEK)),)
  ifeq ($(filter nhl_% nba_%,$(MAKECMDGOALS)),)
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
INJURY_REPORT := data/injuries/injuries_week$(WEEK).csv

PROPS_HTML := $(DOCS_DIR)/props/index.html
TOP_HTML   := $(DOCS_DIR)/props/top.html
INSIGHTS_HTML := $(DOCS_DIR)/props/insights.html
INSIGHTS_JSON := docs/data/ai/insights_week$(WEEK).json
ARB_HTML   := $(DOCS_DIR)/props/arbitrage.html
ODDS_CSV   := $(ODDS_DIR)/latest.csv
FAM_ARB_CSV := data/qc/family_arbitrage.csv
INCOH_CSV  := data/qc/incoherent_books.csv

# AI Insights spend API credits. Scheduled site refreshes can leave the last
# generated Insights page in place; opt in with SKIP_AI_INSIGHTS=0.
ifeq ($(SKIP_AI_INSIGHTS),1)
WEEKLY_INSIGHTS :=
else
WEEKLY_INSIGHTS := $(INSIGHTS_HTML)
endif

# ---- Phony targets ----
.PHONY: monday_all monday_all_pub weekly qc publish_pages props_now_pages serve_preview clean_pages clean injuries
.PHONY: nhl_daily nhl_daily_pub nhl_odds nhl_stats nhl_consensus nhl_edges nhl_page
.PHONY: nhl_totals_fetch nhl_totals_features nhl_totals_train nhl_totals_predict nhl_totals_consensus nhl_totals_all

# Main weekly build (no publish)
monday_all: weekly qc
	@echo "[OK] Weekly build complete: SEASON=$(SEASON) WEEK=$(WEEK)"
	@echo "[OK] All QC checks passed ✓"

# Weekly pipeline (consensus removed - now integrated into Props page)
weekly: $(PROPS_HTML) $(TOP_HTML) $(WEEKLY_INSIGHTS) $(ARB_HTML)
	$(PY) scripts/build_site_metadata.py

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
$(PROPS_DIR) $(DOCS_DIR)/props $(ODDS_DIR) $(QC_DIR) data/ai docs/data/ai:
	mkdir -p $@

# 1) Odds (stdout to file, as your script expects)
$(ODDS_CSV): scripts/fetch_odds.py | $(ODDS_DIR)
	$(PY) scripts/fetch_odds.py > $@

# 2) Fetch all player props → latest_all_props.csv
# Scoped to the current SEASON/WEEK to avoid burning odds-API quota
# pulling props for the entire season on every run.
$(PROPS_ALL): scripts/fetch_all_player_props.py | $(PROPS_DIR) $(ODDS_CSV)
	$(PY) scripts/fetch_all_player_props.py --season $(SEASON) --week $(WEEK)
	@test -s $(PROPS_ALL) || (echo "[ERR] $(PROPS_ALL) not created"; exit 1)
	@echo "[VALIDATION] Checking props data freshness..."
	@$(PY) scripts/validate_data_freshness.py --props $(PROPS_ALL) || (echo "[ERR] Props data is stale!"; exit 1)

# 3) Fetch the target week's injury designations before modeling.
injuries: $(INJURY_REPORT)

$(INJURY_REPORT): scripts/fetch_injuries.py scripts/injury_adjustments.py
	$(PY) scripts/fetch_injuries.py --season $(SEASON) --week $(WEEK)

# 4) Build params
$(PARAMS): scripts/make_player_prop_params.py scripts/injury_adjustments.py $(PROPS_ALL) $(INJURY_REPORT) | $(PROPS_DIR)
	$(PY) scripts/make_player_prop_params.py \
	  --season $(SEASON) --week $(WEEK) \
	  --out $@

# 5) Compute edges → merged CSV
$(MERGED): scripts/make_props_edges.py scripts/market_math.py models/nfl_prop_calibration.json $(PARAMS) $(PROPS_ALL) | $(PROPS_DIR)
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
$(PROPS_HTML): scripts/build_props_site.py scripts/site_metadata.py scripts/market_math.py $(MERGED) | $(DOCS_DIR)/props
	$(PY) scripts/build_props_site.py \
	  --merged_csv $(MERGED) \
	  --out $@ \
	  --season $(SEASON) \
	  --week $(WEEK) \
	  --title "Fourth & Value — Player Props (Week $(WEEK))" \
	  --drop_no_scorer

$(TOP_HTML): scripts/build_top_picks.py scripts/build_props_site.py scripts/site_metadata.py scripts/market_math.py $(MERGED) | $(DOCS_DIR)/props
	$(PY) scripts/build_top_picks.py \
	  --merged_csv $(MERGED) \
	  --out $@ \
	  --season $(SEASON) \
	  --week $(WEEK) \
	  --title "Top Picks — Week $(WEEK)"

# 5b) Generate AI insights JSON
# Written directly into docs/ so it's published alongside the HTML pages
# (data/ is gitignored - anything written only there never reaches the site).
$(INSIGHTS_JSON): scripts/make_ai_commentary.py $(MERGED) | docs/data/ai
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
# NHL regular-season publishing (legacy model scripts are research-only).
.PHONY: nhl_daily nhl_pages nhl_test nhl_daily_pub nhl_totals_daily
nhl_daily:
	$(PY) scripts/nhl/refresh.py

nhl_pages:
	$(PY) scripts/nhl/refresh.py --offline

nhl_test:
	$(PY) -m unittest discover -s tests -p 'test_nhl_refresh.py'

nhl_totals_daily: nhl_daily

# Publish through the isolated, tested workflow rather than staging a workspace.
nhl_daily_pub:
	gh workflow run nhl-daily.yml --ref main

# Retired shortcuts must not overwrite the current pages with old model outputs.
nhl_odds nhl_stats nhl_consensus nhl_edges nhl_page nhl_totals_fetch nhl_totals_features nhl_totals_train nhl_totals_predict nhl_totals_consensus nhl_totals_page nhl_totals_all:
	@echo "Retired NHL model shortcut. Use make nhl_daily or nhl_pages; see NHL_README.md."
	@exit 1

# ========================================================================


# ========================================================================
# NFL Team Totals Pipeline
# ========================================================================

NFL_PBP := data/pbp/pbp_2022_2025.parquet
NFL_TEAM_FEATURES := data/nfl/processed/team_features.csv
NFL_TOTALS_MODEL := data/nfl/models/ridge_totals.pkl
NFL_TOTALS_PREDS := data/nfl/predictions/week_predictions.csv
NFL_TOTALS_CONSENSUS := data/nfl/consensus/consensus.csv
NFL_TOTALS_EDGES := data/nfl/consensus/edges.csv
NFL_TOTALS_PAGE := docs/nfl/totals/index.html
NFL_INJURY_SIGNAL := data/nfl/injuries/injury_totals_week$(WEEK).csv
NFL_INJURY_PAGE := docs/nfl/injuries/index.html

# Phony targets
.PHONY: nfl_totals_fetch nfl_totals_features nfl_totals_train nfl_totals_predict nfl_totals_lines nfl_totals_consensus nfl_totals_page nfl_totals_daily injury_totals_signal injury_totals_page

# Fetch PBP data (one-time or when new season starts)
nfl_totals_fetch:
	@echo "====================================================================="
	@echo "Fetching NFL play-by-play data..."
	@echo "====================================================================="
	$(PY) -c "import nfl_data_py as nfl; pbp = nfl.import_pbp_data([2022, 2023, 2024, 2025], downcast=True); pbp.to_parquet('$(NFL_PBP)'); print(f'✓ Fetched {len(pbp)} plays')"
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
		--season $(SEASON) \
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
nfl_totals_page: injury_totals_signal
	@echo "====================================================================="
	@echo "Building NFL totals page..."
	@echo "====================================================================="
	$(PY) scripts/nfl_build_totals_page.py \
		--predictions $(NFL_TOTALS_PREDS) \
		--consensus data/nfl/consensus/totals_spreads_consensus.csv \
		--edges $(NFL_TOTALS_EDGES) \
		--lines data/nfl/lines/totals_spreads.csv \
		--output $(NFL_TOTALS_PAGE) \
		--week $(WEEK) \
		--injury-signal $(NFL_INJURY_SIGNAL)

injury_totals_signal: $(NFL_INJURY_SIGNAL)

$(NFL_INJURY_SIGNAL): scripts/build_injury_totals_signal.py data/injuries/injuries_week$(WEEK).csv $(NFL_TOTALS_PREDS) data/nfl/consensus/totals_spreads_consensus.csv data/nfl/lines/line_movement.csv
	$(PY) scripts/build_injury_totals_signal.py \
		--injuries data/injuries/injuries_week$(WEEK).csv \
		--predictions $(NFL_TOTALS_PREDS) \
		--consensus data/nfl/consensus/totals_spreads_consensus.csv \
		--movement data/nfl/lines/line_movement.csv \
		--lines data/nfl/lines/totals_spreads.csv \
		--output $@

injury_totals_page: $(NFL_INJURY_PAGE)

$(NFL_INJURY_PAGE): scripts/build_injury_totals_page.py $(NFL_INJURY_SIGNAL)
	$(PY) scripts/build_injury_totals_page.py --input $(NFL_INJURY_SIGNAL) --season $(SEASON) --week $(WEEK) --output $@

# Weekly run: predict + consensus + page
nfl_totals_daily: nfl_totals_predict nfl_totals_consensus nfl_totals_page injury_totals_signal injury_totals_page
	@echo "====================================================================="
	@echo "✓ NFL totals update complete for Week $(WEEK)"
	@echo "====================================================================="
	@echo "View at: $(NFL_TOTALS_PAGE)"

# ========================================================================
# End NFL Team Totals Pipeline
# ========================================================================

# NBA market pipeline; no paid commentary generation.
.PHONY: nba_daily nba_pages nba_test
nba_daily:
	$(PY) scripts/nba/pipeline.py

nba_pages:
	$(PY) scripts/nba/pipeline.py --offline

nba_test:
	$(PY) -m unittest discover -s tests -p 'test_nba.py'


.FORCE:
	$(ODDS_CSV): .FORCE scripts/fetch_odds.py | $(ODDS_DIR)
		$(PY) scripts/fetch_odds.py > $@

	$(PROPS_ALL): .FORCE scripts/fetch_all_player_props.py | $(PROPS_DIR) $(ODDS_CSV)
		$(PY) scripts/fetch_all_player_props.py
		@test -s $(PROPS_ALL) || (echo "[ERR] $(PROPS_ALL) not created"; exit 1)
