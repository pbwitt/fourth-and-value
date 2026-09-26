# NHL v2: independent forecasts and price research

The seven existing markets and five NHL routes remain compatible. The replacement uses
game-level, lagged hockey features, coherent game scores and player opportunity/count
distributions. It keeps independent forecasts, exact paired market consensus and offer
pricing separate. All betting recommendations remain disabled: four-season predictive
testing does not establish an executable betting edge.

- [Actual evaluation and limitations](reports/nhl-rebuild/EVALUATION.md)
- [Model card](reports/nhl-rebuild/MODEL_CARD.md)
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
`scripts/nhl/v2/report.py`. Training is separate from daily inference. No paid historical
odds endpoint or restricted xG dataset is used.

Routes: `/nhl/`, `/nhl/props/`, `/nhl/totals/`, `/nhl/top.html`, `/nhl/methods.html`.
Public feed: `docs/nhl/data/latest.json`. Production schedules target 10:30 and 16:30
America/New_York, gated for daylight saving time. Main-branch automation archives its
inputs and publishes; branch tests are read-only. Do not dispatch production, merge or
deploy this rebuild without the requested approval.

Historical references remain labeled separately. Missing models remain null, stale data
are hidden, and feed failures do not resurrect old forecasts. Legacy leaking training
scripts/artifacts and saved ledgers are not reused or regraded.
