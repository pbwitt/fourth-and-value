# ---- Config ----
PY ?= python3
SEASON ?=
WEEK   ?=

# Require SEASON/WEEK
ifeq ($(strip $(SEASON)),)
  $(error SEASON=YYYY required (e.g., SEASON=2025))
endif
ifeq ($(strip $(WEEK)),)
  $(error WEEK=N required (e.g., WEEK=3))
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
CONS_HTML  := $(DOCS_DIR)/props/consensus.html
INSIGHTS_HTML := $(DOCS_DIR)/props/insights.html
ARB_HTML   := $(DOCS_DIR)/props/arbitrage.html
ODDS_CSV   := $(ODDS_DIR)/latest.csv
FAM_ARB_CSV := data/qc/family_arbitrage.csv
INCOH_CSV  := data/qc/incoherent_books.csv

# ---- Phony targets ----
.PHONY: monday_all monday_all_pub weekly publish_pages props_now_pages serve_preview clean_pages clean

# Main weekly build (no publish)
monday_all: weekly
	@echo "[OK] Weekly build complete: SEASON=$(SEASON) WEEK=$(WEEK)"

# Weekly pipeline
weekly: $(PROPS_HTML) $(TOP_HTML) $(CONS_HTML) $(INSIGHTS_HTML) $(ARB_HTML)

# ---- Steps ----
# 0) Ensure dirs exist
$(PROPS_DIR) $(DOCS_DIR)/props $(ODDS_DIR) $(QC_DIR):
	mkdir -p $@

# 1) Odds (stdout to file, as your script expects)
$(ODDS_CSV): scripts/fetch_odds.py | $(ODDS_DIR)
	$(PY) scripts/fetch_odds.py > $@

# 2) Fetch all player props → latest_all_props.csv
# (Your script typically writes to $(PROPS_ALL) without args.)
$(PROPS_ALL): scripts/fetch_all_player_props.py | $(PROPS_DIR) $(ODDS_CSV)
	$(PY) scripts/fetch_all_player_props.py
	@test -s $(PROPS_ALL) || (echo "[ERR] $(PROPS_ALL) not created"; exit 1)

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

$(CONS_HTML): scripts/build_consensus_page.py $(MERGED) | $(DOCS_DIR)/props
	$(PY) scripts/build_consensus_page.py \
	  --merged_csv $(MERGED) \
	  --out $@ \
	  --season $(SEASON) \
	  --week $(WEEK)

$(INSIGHTS_HTML): scripts/build_insights_page.py | $(DOCS_DIR)/props
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
props_now_pages: $(PROPS_HTML) $(TOP_HTML) $(CONS_HTML) $(INSIGHTS_HTML)
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
	rm -f $(PROPS_HTML) $(TOP_HTML) $(CONS_HTML) $(INSIGHTS_HTML)

clean:
	rm -f $(PARAMS) $(MERGED)


.FORCE:
	$(ODDS_CSV): .FORCE scripts/fetch_odds.py | $(ODDS_DIR)
		$(PY) scripts/fetch_odds.py > $@

	$(PROPS_ALL): .FORCE scripts/fetch_all_player_props.py | $(PROPS_DIR) $(ODDS_CSV)
		$(PY) scripts/fetch_all_player_props.py
		@test -s $(PROPS_ALL) || (echo "[ERR] $(PROPS_ALL) not created"; exit 1)
