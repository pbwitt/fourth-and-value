# NHL regular-season setup

The production NHL refresh was rebuilt for 2026–27. See [NHL_README.md](NHL_README.md) for current commands, scheduled updates, data sources and model limitations.

```sh
make nhl_daily PY=.venv/bin/python
make nhl_pages PY=.venv/bin/python
make nhl_test PY=.venv/bin/python
```

The previous totals/model training instructions are retired. Old models included information unavailable before puck drop; their outputs must not be republished as current picks. Historical outputs remain available for research.
