# =========================
# Fourth & Value — Makefile
# Clean, migration-safe pipeline
# =========================

SHELL := /bin/bash

# -------- Config (defaults; can be overridden on the command line) --------
SEASON ?= 2025
WEEK   ?= 2
PY     ?= python3

DOCS_DIR  := docs
PROPS_DIR := $(DOCS_DIR)/props
DATA_DIR  := data

# Load .env for every recipe (handles ODDS_API_KEY, OPENAI_API_KEY, PUBLISH, etc.)
ENV := set -a; [ -f .env ] && . ./.env; set +a;
RUNPY = $(ENV) $(PY)

# Ensure output dirs
$(shell mkdir -p $(PROPS_DIR) $(DATA_DIR)/props >/dev/null 2>&1)

# -------- Phony targets --------
.PHONY: help weekly monday_all monday_all_pub data_only props_now props_now_pages insights site_home publish serve clean

help:
	@echo "Targets:"
	@echo "  make props_now SEASON=YYYY WEEK=N    # fetch props -> params -> edges -> pages (preview)"
	@echo "  make weekly SEASON=YYYY WEEK=N       # data pipeline only (up to merged CSV)"
	@echo "  make data_only SEASON=YYYY WEEK=N    # same as 'weekly' (alias)"
	@echo "  make props_now_pages                  # rebuild pages from existing merged CSV"
	@echo "  make insights SEASON=YYYY WEEK=N      # build AI insights page (needs OPENAI_API_KEY)"
	@echo "  make site_home                         # build homepage (optional)"
	@echo "  make monday_all SEASON=YYYY WEEK=N     # site_home + props_now (preview)"
	@echo "  make monday_all_pub SEASON=YYYY WEEK=N # monday_all then publish (PUBLISH=1 required)"
	@echo "  make publish                            # commit & push docs/ only (guarded)"
	@echo "  make serve                              # local preview server on :8080"
	@echo "  make clean                              # remove derived files"

# -------- Data pipeline primitives --------

# 1) Fetch all player props from The Odds API
$(DATA_DIR)/props/latest_all_props.csv: scripts/fetch_all_player_props.py
	@echo "[fetch_props] writing $@"
	@$(RUNPY) scripts/fetch_all_player_props.py
	@ls -lh $@ || true

# 2) Fetch weekly player stats parquet (auto; migration-proof)
$(DATA_DIR)/weekly_player_stats_$(SEASON).parquet: scripts/fetch_weekly_player_stats.py
	@echo "[weekly_stats] $(SEASON) -> $@"
	@$(RUNPY) scripts/fetch_weekly_player_stats.py --season $(SEASON) --out $@
	@ls -lh $@ || true

# 3) Build per-player params for the week (depends on weekly stats + props)
$(DATA_DIR)/props/params_week$(WEEK).csv: scripts/make_player_prop_params.py $(DATA_DIR)/props/latest_all_props.csv $(DATA_DIR)/weekly_player_stats_$(SEASON).parquet
	@echo "[make_params] -> $@"
	@$(RUNPY) scripts/make_player_prop_params.py --season $(SEASON) --week $(WEEK) --out $@
	@ls -lh $@ || true

# 4) Merge + compute model edges (the canonical merged CSV)
$(DATA_DIR)/props/props_with_model_week$(WEEK).csv: scripts/make_props_edges.py $(DATA_DIR)/props/latest_all_props.csv $(DATA_DIR)/props/params_week$(WEEK).csv
	@echo "[make_edges] -> $@"
	@$(RUNPY) scripts/make_props_edges.py --season $(SEASON) --week $(WEEK) \
		--props_csv $(DATA_DIR)/props/latest_all_props.csv \
		--params_csv $(DATA_DIR)/props/params_week$(WEEK).csv \
		--out $@
	@ls -lh $@ || true

# -------- Pages --------

# Props table
$(PROPS_DIR)/index.html: scripts/build_props_site.py $(DATA_DIR)/props/props_with_model_week$(WEEK).csv
	@echo "[build_props] -> $@"
	@$(RUNPY) scripts/build_props_site.py --season $(SEASON) --week $(WEEK) --out $@
	@touch $(DOCS_DIR)/.nojekyll

# Top Picks (NOTE: builder intentionally does NOT take --week)
$(PROPS_DIR)/top.html: scripts/build_top_picks.py $(DATA_DIR)/props/props_with_model_week$(WEEK).csv
	@echo "[build_top] -> $@"
	@$(RUNPY) scripts/build_top_picks.py --out $@
	@touch $(DOCS_DIR)/.nojekyll

# Consensus page
$(PROPS_DIR)/consensus.html: scripts/build_consensus_page.py $(DATA_DIR)/props/props_with_model_week$(WEEK).csv
	@echo "[build_consensus] -> $@"
	@$(RUNPY) scripts/build_consensus_page.py --out $@
	@touch $(DOCS_DIR)/.nojekyll

# Optional: site homepage (kept simple, no UI surgery)
$(DOCS_DIR)/index.html: scripts/build_edges_site.py
	@echo "[site_home] -> $@"
	@$(RUNPY) scripts/build_edges_site.py --out $@
	@touch $(DOCS_DIR)/.nojekyll

# -------- Composite flows --------

# Data only (through merged CSV)
weekly: $(DATA_DIR)/props/props_with_model_week$(WEEK).csv
data_only: weekly

# Full props flow incl. pages (preview)
props_now: $(PROPS_DIR)/index.html $(PROPS_DIR)/top.html $(PROPS_DIR)/consensus.html
	@echo "[props_now] pages ready under $(PROPS_DIR)/"

# Rebuild pages only
props_now_pages: $(PROPS_DIR)/index.html $(PROPS_DIR)/top.html $(PROPS_DIR)/consensus.html

# Insights page (AI; requires OPENAI_API_KEY in .env)
insights: $(DATA_DIR)/props/props_with_model_week$(WEEK).csv scripts/make_ai_commentary.py scripts/build_insights_page.py
	@echo "[insights] generating commentary & page"
	@$(RUNPY) scripts/make_ai_commentary.py --season $(SEASON) --week $(WEEK) --merged_csv $(DATA_DIR)/props/props_with_model_week$(WEEK).csv
	@$(RUNPY) scripts/build_insights_page.py --season $(SEASON) --week $(WEEK) \
		--title "Fourth & Value — Insights (Week $(WEEK))" \
		--out $(PROPS_DIR)/insights.html
	@touch $(DOCS_DIR)/.nojekyll
	@ls -lh $(PROPS_DIR)/insights.html || true

# Monday all (preview)
monday_all: $(DOCS_DIR)/index.html props_now
	@echo "[monday_all] complete (preview). Set PUBLISH=1 to publish."

# Monday all + publish
monday_all_pub: monday_all publish

# -------- Publish (docs/ only; guarded) --------
publish:
	@if [ "$$PUBLISH" != "1" ]; then \
		echo "[guard] Not publishing: set PUBLISH=1 in .env or on the command line"; \
		exit 1; \
	fi
	@echo "[publish] committing and pushing $(DOCS_DIR)/ only"
	@git add -f $(DOCS_DIR)
	@git commit -m "Publish pages (Season $(SEASON) Week $(WEEK))" || true
	@git push

# -------- Utilities --------
serve:
	@echo "Serving $(DOCS_DIR) at http://127.0.0.1:8080/"
	@$(ENV) $(PY) -m http.server 8080 -b 127.0.0.1 -d $(DOCS_DIR)

clean:
	@echo "[clean] removing derived files"
	@rm -f $(DATA_DIR)/weekly_player_stats_*.parquet
	@rm -f $(DATA_DIR)/props/params_week*.csv
	@rm -f $(DATA_DIR)/props/props_with_model_week*.csv
	@rm -f $(DATA_DIR)/props/latest_all_props.csv
	@echo "[clean] done"


docs/props/index.html: scripts/build_props_site.py data/props/props_with_model_week$(WEEK).csv
	$(ENV) $(PY) scripts/build_props_site.py \
			--merged_csv data/props/props_with_model_week$(WEEK).csv \
			--out $@ \
			--title "Fourth & Value — Props (Week $(WEEK))" \
			--drop_no_scorer

docs/props/top.html: scripts/build_top_picks.py data/props/props_with_model_week$(WEEK).csv
	$(ENV) $(PY) scripts/build_top_picks.py \
			--merged_csv data/props/props_with_model_week$(WEEK).csv \
			--out $@

docs/props/consensus.html: scripts/build_consensus_page.py data/props/props_with_model_week$(WEEK).csv
	$(ENV) $(PY) scripts/build_consensus_page.py \
			--merged_csv data/props/props_with_model_week$(WEEK).csv \
			--out $@
