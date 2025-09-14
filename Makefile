# Variables
PY=python3
SEASON?=2025
WEEK?=1
DOCS_DIR=docs

# Core targets
weekly: odds props_now insights
monday_all: weekly
monday_all_pub: weekly
	@if [ "$(PUBLISH)" = "1" ] && [ "$(CONFIRM)" = "LIVE" ]; then \
		git add -A; \
		git commit -m "Publish Week $(WEEK)"; \
		git push origin main; \
	fi

# Odds fetch
odds:
	$(PY) scripts/fetch_odds.py > data/odds/latest.csv



# Props pipeline
props_now: data/props/props_with_model_week$(WEEK).csv docs/props/index.html docs/props/top.html docs/props/consensus.html
	touch $(DOCS_DIR)/.nojekyll

props_now_pages: docs/props/index.html docs/props/top.html docs/props/consensus.html
	touch $(DOCS_DIR)/.nojekyll

data/props/props_with_model_week$(WEEK).csv: data/props/params_week$(WEEK).csv
	$(PY) scripts/make_props_edges.py --season $(SEASON) --week $(WEEK) \
		--params_csv data/props/params_week$(WEEK).csv \
		--out_csv $@

data/props/params_week$(WEEK).csv:
	$(PY) scripts/make_player_prop_params.py --season $(SEASON) --week $(WEEK) \
		--out_csv $@

data/props/latest_all_props.csv:
	$(PY) scripts/fetch_all_player_props.py --season $(SEASON) --week $(WEEK) \
		--out_csv $@

# Pages
docs/props/index.html: data/props/props_with_model_week$(WEEK).csv
	$(PY) scripts/build_props_site.py \
		--merged_csv data/props/props_with_model_week$(WEEK).csv \
		--out $@

docs/props/top.html: data/props/props_with_model_week$(WEEK).csv
	$(PY) scripts/build_top_picks.py \
		--merged_csv data/props/props_with_model_week$(WEEK).csv \
		--out $@

docs/props/consensus.html: data/props/props_with_model_week$(WEEK).csv
	$(PY) scripts/build_consensus_page.py \
		--merged_csv data/props/props_with_model_week$(WEEK).csv \
		--out $@

# Insights
insights:
	$(PY) scripts/make_ai_commentary.py --season $(SEASON) --week $(WEEK) \
		--merged_csv data/props/props_with_model_week$(WEEK).csv \
		--out_json $(DOCS_DIR)/data/ai/insights_week$(WEEK).json
	$(PY) scripts/build_insights_page.py --season $(SEASON) --week $(WEEK) \
		--title "Fourth & Value — Insights (Week $(WEEK))" \
		--out $(DOCS_DIR)/props/insights.html
	touch $(DOCS_DIR)/.nojekyll


site_home:
	@if [ -f data/edges/edges_week$(WEEK).csv ]; then \
		$(PY) scripts/build_edges_site.py --out $(DOCS_DIR)/home_edges.html; \
	else \
		echo "[site_home] edges missing → keeping static Home"; \
	fi
	touch $(DOCS_DIR)/.nojekyll
