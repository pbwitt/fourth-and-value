# NHL v2: independent forecasts and price research

The seven existing markets and five NHL routes remain compatible. The replacement uses
game-level, lagged hockey features, coherent game scores and player opportunity/count
distributions. It keeps independent forecasts, exact paired market consensus and offer
pricing separate. All betting recommendations remain disabled: four-season predictive
testing does not establish an executable betting edge.

- [v2.3 change: opponent defense in player forecasts](reports/nhl-v2.3/README.md) (current model)
- [v2.2 change: player history aged by games played](reports/nhl-v2.2/README.md)
- [Actual evaluation and limitations](reports/nhl-rebuild/EVALUATION.md) (v2.1)
- [Historical market comparison](reports/nhl-rebuild/HISTORICAL_MARKETS.md)
- [Model card](reports/nhl-rebuild/MODEL_CARD.md) (v2.1; v2.2 changes player-history aging and v2.3 adds the opponent adjustment)
- [Sources and blocked inputs](reports/nhl-rebuild/SOURCES.md)
- [Public contract](reports/nhl-rebuild/CONTRACT.md)
- [Build, operations, analyst review and rollback](reports/nhl-rebuild/RUNBOOK.md)
- [Audit and decision log](reports/nhl-rebuild/PLAN.md)

```bash
make nhl_daily PY=.venv/bin/python   # existing authorized feeds, local output only
make nhl_pages PY=.venv/bin/python   # offline, preserves market snapshot timestamps
make nhl_test PY=.venv/bin/python
.venv/bin/python scripts/validate_nhl_freshness.py
```

Install `requirements-nhl.txt` for the pinned model stack. Restore archived source data
with `scripts/nhl/v2/restore.py`, run `make nhl_evaluate`, and render the report with
`scripts/nhl/v2/report.py`. Both write to the current version's report folder
(`reports/nhl-v2.3`) and refuse to overwrite an earlier version's evidence. Training is separate from daily inference. A bounded historical game-price sample uses the existing authorized odds plan; no
upgrade or restricted xG dataset is used.

Routes: `/nhl/`, `/nhl/props/`, `/nhl/totals/`, `/nhl/top.html`, `/nhl/methods.html`.
Public feed: `docs/nhl/data/latest.json`. Production refreshes start at 07:05 (Morning
Picks Edition) and 16:30 America/New_York (Afternoon Market Refresh); Supabase timers
start both on time and GitHub's own schedules are backups (see `MORNING_SCHEDULER.md`). Main-branch automation archives its
inputs and publishes; branch tests are read-only. Do not dispatch production, merge or
deploy this rebuild without the requested approval.

Historical references remain labeled separately. Missing models remain null, stale data
are hidden, and feed failures do not resurrect old forecasts. Legacy leaking training
scripts/artifacts and saved ledgers are not reused or regraded.

## Cross-market coherence (research ledger)

`scripts/nhl/v2/coherence.py` checks whether one player market agrees with another. Each
market's paired no-vig price implies an expected count. The model's ratios (goals per shot,
assists per point, share of team goals) move that count into a different market, and the
model's count distribution prices the offer. An offer's own market never feeds its estimate.
Team estimates fit regulation goals to the consensus moneyline and main total.

Every source pair is archived and graded separately. Rule `nhl-coherence-1` flags an offer
only when at least two mechanical links (shots and goals; goals or assists and points; team
goals) each price it favorably and together reach 3% EV. The rule was fixed before any
result was graded. Model-free containment checks share `arbitrage.contains`: same-book
contradictions on the Arbitrage page and cross-book floors, where a wider bet is priced
below the fair price of a narrower bet it contains.

- Live refreshes freeze each snapshot to `artifacts/nhl/coherence/<snapshot_id>.json.gz`;
  an existing file is never rewritten. Offline builds only write `docs/nhl/data/coherence.json`.
- `python scripts/nhl/v2/coherence.py backfill` scores archived runs made by the current
  model artifact. Backfills are labeled and graded as a separate cohort.
- `python scripts/nhl/v2/coherence.py grade --cached-history` writes
  `artifacts/nhl/coherence/evaluation.json`. It reports the log loss and Brier score of every
  source against the market and the model on the same outcomes, with game-cluster intervals,
  plus flat-unit results and later-snapshot price movement for flagged offers.

Nothing here feeds Top Picks, Market Watch or recommendations. Using these signals there is a
Top Picks policy change and needs the process-page update in `AGENTS.md`.
