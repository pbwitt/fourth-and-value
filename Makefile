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

PROPS_ALL  := $(PROPS_DIR)/latest_all_props.csv
PARAMS     := $(PROPS_DIR)/params_week$(WEEK).csv
MERGED     := $(PROPS_DIR)/props_with_model_week$(WEEK).csv

PROPS_HTML := $(DOCS_DIR)/props/index.html
TOP_HTML   := $(DOCS_DIR)/props/top.html
CONS_HTML  := $(DOCS_DIR)/props/consensus.html
ODDS_CSV   := $(ODDS_DIR)/latest.csv

# ---- Phony targets ----
.PHONY: monday_all monday_all_pub weekly publish_pages props_now_pages serve_preview clean_pages clean

# Main weekly build (no publish)
monday_all: weekly
	@echo "[OK] Weekly build complete: SEASON=$(SEASON) WEEK=$(WEEK)"

# Weekly pipeline
weekly: $(PROPS_HTML) $(TOP_HTML) $(CONS_HTML)

# ---- Steps ----
# 0) Ensure dirs exist
$(PROPS_DIR) $(DOCS_DIR)/props $(ODDS_DIR):
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

# 5) Build pages
$(PROPS_HTML): scripts/build_props_site.py $(MERGED) | $(DOCS_DIR)/props
	$(PY) scripts/build_props_site.py \
	  --merged_csv $(MERGED) \
	  --out $@ \
	  --title "Fourth & Value — Player Props (Week $(WEEK))" \
	  --drop_no_scorer

$(TOP_HTML): scripts/build_top_picks.py $(MERGED) | $(DOCS_DIR)/props
	$(PY) scripts/build_top_picks.py \
	  --merged_csv $(MERGED) \
	  --out $@ \
	  --title "Top Picks — Week $(WEEK)"

$(CONS_HTML): scripts/build_consensus_page.py $(MERGED) | $(DOCS_DIR)/props
	$(PY) scripts/build_consensus_page.py \
	  --merged_csv $(MERGED) \
	  --out $@ \
	  --week $(WEEK)


# Pages-only rebuild (when CSV already exists)
props_now_pages: $(PROPS_HTML) $(TOP_HTML) $(CONS_HTML)
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
	rm -f $(PROPS_HTML) $(TOP_HTML) $(CONS_HTML)

clean:
	rm -f $(PARAMS) $(MERGED)


.FORCE:
	$(ODDS_CSV): .FORCE scripts/fetch_odds.py | $(ODDS_DIR)
		$(PY) scripts/fetch_odds.py > $@

	$(PROPS_ALL): .FORCE scripts/fetch_all_player_props.py | $(PROPS_DIR) $(ODDS_CSV)
		$(PY) scripts/fetch_all_player_props.py
		@test -s $(PROPS_ALL) || (echo "[ERR] $(PROPS_ALL) not created"; exit 1)


		# --- Game predictions (baseline) ---


game_preds:
	python3 scripts/make_game_preds.py --season $(SEASON) --week $(WEEK)

game_page:
	python3 scripts/build_game_page.py --season $(SEASON) --week $(WEEK) \
		--csv data/games/model_preds_week$(WEEK).csv \
		--out docs/games/index.html

schedule:
	python3 scripts/fetch_schedule.py --season $(SEASON)

	# ===== Props quick-refresh (canonical) =====
.PHONY: props_now props_now_pages props_now_pub publish_pages

# Load .env for API keys (reuse existing if you already have ENV_LOADER)
ENV_LOADER = set -a; [ -f .env ] && . ./.env; set +a

# Full props refresh: fetch → params → edges → pages
props_now:
	@echo "[props_now] fetch → params → edges → pages (SEASON=$(SEASON) WEEK=$(WEEK))"
	@$(ENV_LOADER); python3 scripts/fetch_all_player_props.py \
		--sport americanfootball_nfl \
		--out data/props/latest_all_props.csv
	@python3 scripts/make_player_prop_params.py --season $(SEASON) --week $(WEEK)
	@python3 scripts/make_props_edges.py --season $(SEASON) --week $(WEEK) \
		--props_csv data/props/latest_all_props.csv \
		--out data/props/props_with_model_week$(WEEK).csv
	@python3 scripts/build_props_site.py       --season $(SEASON) --week $(WEEK)
	@python3 scripts/build_top_picks.py        --season $(SEASON) --week $(WEEK)
	@python3 scripts/build_consensus_page.py   --season $(SEASON) --week $(WEEK)
	@touch docs/.nojekyll
	@echo "[props_now] done."

# Pages-only rebuild (when CSV already exists)
props_now_pages:
	@echo "[pages-only] props/top/consensus (SEASON=$(SEASON) WEEK=$(WEEK))"
	@python3 scripts/build_props_site.py       --season $(SEASON) --week $(WEEK)
	@python3 scripts/build_top_picks.py        --season $(SEASON) --week $(WEEK)
	@python3 scripts/build_consensus_page.py   --season $(SEASON) --week $(WEEK)
	@touch docs/.nojekyll

# Minimal publisher (guards) — use if you don't already have publish_pages
publish_pages:
	@if [ "$(PUBLISH)" = "1" ] && [ "$(CONFIRM)" = "LIVE" ]; then \
		echo "[publish] pushing docs/ to origin..."; \
		git add -A docs && git commit -m "publish pages (SEASON=$(SEASON) WEEK=$(WEEK))" || true; \
		git push; \
	else \
		echo "Not publishing. Use: make $@ PUBLISH=1 CONFIRM=LIVE"; \
	fi

# Convenience wrapper: build props now, then publish pages
props_now_pub:
	$(MAKE) props_now SEASON=$(SEASON) WEEK=$(WEEK)
	$(MAKE) publish_pages PUBLISH=1 CONFIRM=LIVE
# ===========================================
# --- Consensus (data + page) ---
CONS_CSV := data/props/consensus_week$(WEEK).csv

make_consensus:
	python scripts/make_consensus.py \
	  --merged_csv data/props/latest_all_props.csv \
	  --out_csv $(CONS_CSV) \
	  --week $(WEEK)

consensus_page: make_consensus
	python -m scripts.build_consensus_page \
	  --merged_csv data/props/latest_all_props.csv \
	  --out docs/props/consensus.html \
	  --week $(WEEK) \
	  --title "Fourth & Value — Consensus (Week $(WEEK))"


qc_props:
	python -m scripts.qc_props --week $(WEEK)

fetch_stats:
	@mkdir -p data/nflverse
	curl -L -o data/nflverse/stats_player_week_$(SEASON).parquet \
		  "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_$(SEASON).parquet"



			# in your pipeline:
			# make make_player_prop_params  ->  make qc_props  ->  make make_props_edges
qc:
	python qc_datapull.py
				python - <<'PY'
	import json; r=json.load(open("data/qc/run_qc_week4.json"))
	f=r.get("signals",{}).get("fatal",[])
	assert not f, f"FATAL QC issues: {f}"
	print("QC PASS")
	PY
