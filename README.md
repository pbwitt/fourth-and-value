# Fourth & Value

A static sports-analysis site published from `docs/`, with Python pipelines for NFL player props and NFL/NHL research. The main NFL board compares sportsbook quotes; model estimates are experimental and show their coverage explicitly.

Read the [NFL site review and traffic plan](reports/NFL_SITE_REVIEW_2026-09-09.md) for the September 2026 changes, validation results and remaining model limitations.

## Preview

```sh
python3 -m http.server 8010 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8010/`. The NFL hub is `/nfl/`, the full props board is `/props/`, and the shortlist is `/props/top.html`.

## Build

Install `requirements.txt` in a Python 3.11 virtual environment. The complete NFL pipeline requires configured data API credentials and explicit season/week:

```sh
make --always-make monday_all PY=.venv/bin/python SEASON=2026 WEEK=1
```

This refreshes inputs and builds/QCs pages; it does not publish. `--always-make` prevents a cached `latest_all_props.csv` from suppressing a new odds pull. Provider quote timestamps are required by the freshness gate. Current-season game results are not expected before Week 1; prior-season career data must be present.

To rebuild the NFL comparison pages from an existing merged CSV without fetching data:

```sh
.venv/bin/python scripts/build_props_site.py --merged_csv /path/to/merged.csv --out docs/props/index.html --season 2026 --week 1
.venv/bin/python scripts/build_top_picks.py --merged_csv /path/to/merged.csv --out docs/props/top.html --season 2026 --week 1
.venv/bin/python scripts/build_site_metadata.py
```

Top Picks requires upcoming games, provider timestamps within 48 hours, player evidence, fitted calibration and a positive model edge. An empty shortlist is expected when those requirements are not met. Never replace an unknown quote timestamp with the page's build time.

## Check

```sh
.venv/bin/python -m unittest discover -s tests -v
node tests/props_interactions.cjs
```

The interaction smoke test uses the checked-in snapshot and a clock anchored to that snapshot. For visual checks on a supported machine, install Playwright and its browser, start the preview, then run `node tests/browser_review.cjs`. `PLAYWRIGHT_MODULE`, `CHROME_PATH` and `BROWSER` can point to an existing installation. Browser verification was blocked on the review machine (macOS 12); see the report.

## Main files

- `scripts/market_math.py`: exact-line de-vig, consensus, push probabilities and expected value.
- `scripts/make_player_prop_params.py`: NFL player parameters and forecast cutoffs.
- `scripts/make_props_edges.py`: model probabilities, fitted calibration, provenance and merged offers.
- `scripts/build_props_site.py`: shared NFL board/shortlist generator.
- `docs/assets/props.js` and `docs/assets/site.css`: filtering, pagination and layout.
- `scripts/build_site_metadata.py`: static search metadata and sitemap.
- `.github/workflows/nfl-weekly.yml`: scheduled NFL refresh.

The NFL totals page is historical research. Its inputs and model evaluation need revalidation before current forecasts resume. Private bet tracking uses Supabase; credentials belong in environment variables, never in public site files.
