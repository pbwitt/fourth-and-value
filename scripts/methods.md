Fourth & Value — Methods (2025 Week 4)

Last updated: Sep 28, 2025 (Week 4). This reflects how the live pipeline actually works today.

1) Data sources

Odds (player props): The Odds API → saved to data/props/latest_all_props.csv.

Weekly player stats: nflverse stats_player_week_2025.parquet (fallback: .csv.gz) → data/nflverse/.

Schedule/timezone: Kickoff times converted to ET for display.

2) Market normalization

We map sportsbook market strings to canonical keys (market_std) used throughout the pipeline:

Normal (Gaussian) markets: rush_yds, recv_yds, pass_yds, receptions, rush_attempts, pass_attempts, pass_completions.

Poisson markets: anytime_td, pass_tds, pass_interceptions.

Common aliases are resolved (e.g., receiving_yards → recv_yds, player_rushing_yards → rush_yds, interceptions → pass_interceptions).

3) Parameter building (μ, σ, λ)

Script: scripts/make_player_prop_params.py → outputs data/props/params_week{W}.csv with per–player×market parameters and metadata.

3.1 Observations window

For each player×market we use the most recent 1–3 games that match the 2025 season to estimate parameters.

The actual Week 4 run shows n_games ≈ 2.57 on average across markets.

3.2 Estimators

Normal markets:

We compute a mean μ and standard deviation σ from the last up-to-3 games for the stat distribution.

We apply light recency weighting and shrinkage to reduce volatility when sample is small, then enforce floors so σ≥σ_min and μ≥0.

Poisson markets:

We estimate a rate λ for counts (TDs, INTs). For anytime_td, λ is the player’s combined rushing+receiving TD scoring rate.

With ≤3 games, we apply shrinkage toward a small prior to avoid extreme implied probabilities.

3.3 Zero-prior handling

When a player lacks usable recent data (“zero-prior”), we currently:

Record them in data/props/zero_prior_week{W}.csv.

Skip parameterization for betting markets (models will not generate probabilities/edges for these rows).

Typical examples: D/ST props, “No Scorer,” recent call-ups, or players without 2025 usage yet.

4) From parameters to probabilities

Script: scripts/make_props_edges.py merges latest_all_props.csv with params_week{W}.csv (keyed on player_key then fallbacks like name_std). For each sportsbook row:

Side detection: We derive side (Over/Under or Yes) from the market/price text.

Model probability (model_prob):

Normal markets: Γiven a posted line L, we compute
𝑃
(
𝑋
>
𝐿
)
P(X>L) for Over or
𝑃
(
𝑋
<
𝐿
)
P(X<L) for Under assuming
𝑋
∼
𝑁
𝑜
𝑟
𝑚
𝑎
𝑙
(
𝜇
,
𝜎
)
X∼Normal(μ,σ).

Poisson markets: For "Yes" (e.g., anytime TD), we compute
𝑃
(
𝑋
≥
1
)
=
1
−
𝑒
−
𝜆
P(X≥1)=1−e
−λ
; for thresholds (e.g., INTs O/U 0.5) we use the Poisson CDF accordingly.

Model line (display): For Normal O/U, we also compute a model line (the 50th percentile / median under the assumed distribution). This is used for comparisons and is shown on site pages where available.

5) Prices, consensus, and edge

Book implied probability: Convert American odds to implied probability per row. (De‑vigging across books is supported in code paths but not always available on every page; see Consensus below.)

Consensus (experimental): A separate build aggregates across books to a consensus line and price by (name_std, market_std, side). When available, the Consensus page shows both book and consensus metrics.

Edge (bps): Our primary ranking metric is the difference between model probability and market implied probability, expressed in basis points (1% = 100 bps). For example, edge_bps = (model_prob − market_prob) * 10,000.

Fair odds: Back out fair odds from model_prob using the inverse of the American odds conversion.

6) Coverage & current limitations (Week 4 reality)

Overall, ~85–86% of props rows had usable parameters.

Normal markets: good coverage but with small samples in early weeks; many μ are near 0 for bench/low-usage players.

Poisson markets: λ is small early-season; probabilities cluster near 0–20% for anytime TD unless the player has clear usage.

Zero-priors: ~120 entities this week (includes D/ST and “No Scorer”). These rows are filtered from model outputs.

Known issues we actively guard against:

Negative μ for yards on noisy samples → clamped to 0 with a log for QC.

σ too small → floor to prevent overconfident edges.

Name joins can miss (ID drift) → normalized keys (player_key → name_std → name_slug) to minimize misses.

7) What you see on the site

Props page: Per-book rows with Game, Player, Market, Side, Line, Book/Price, Model %, Market %, Edge (bps), Fair Odds, Kick (ET). Rows without parameters are hidden to avoid blanks.

Top Picks: Collapsed view (per player×market×side) ranking by signal (edge) and execution (best book/EV).

Consensus (in progress): Adds Market Consensus Line/Impl. % by aggregating books. Current Week 4 build merges consensus by (name_std, market_std, side) and shows model vs market vs consensus where available.

8) Roadmap (near-term)

Improve priors: incorporate career & previous season baselines with role-weighted shrinkage.

De‑vig consistently: compute edges vs de‑vig consensus and best book side-by-side.

Expand modeled markets: longest reception/rush, 1st/last TD (with discrete/zero‑inflated models).

Wire actuals: post‑game grading and calibration plots on-site.

9) QC checks we run weekly

Coverage by market; % with μ/σ or λ.

Distinct model_prob variance by market (catch constant/flat models).

Join health: % matched on player_key, then fallbacks.

Sanity clamps triggered (μ<0, σ<floor) and counts.

Spot-check top edges against usage/news.
