# NHL research paper — FV-2026-02

The manuscript describes the NHL v2.1 system deployed under PR #6, using the numerical
evidence retained at commit `e1f91609524f74a6b99cf86cdebb2328cd5e08d1`. It is a separate
editorial change, approved by the owner for publication under Research. It makes no
peer-review or validated-profitability claim.

- Authored source: `scripts/research/nhl_paper.html`.
- Web output: `docs/research/nhl-forecasting-and-market-pricing.html`.
- Print output: same basename with `.pdf`.
- Numerical supplement: same basename with `.json`, including original report hashes.
- Research-index link and independent stylesheet match the existing research paper's
  typography, numbered sections/equations, citations, and light PDF theme.

Generate the tables and HTML without network access:

```bash
python3 scripts/research/build_nhl_paper.py
```

The source manuscript includes interpreted results as prose. If numerical inputs change,
review and revise that prose, study status, version and date; do not publish new tables
with an old interpretation. The build does not retrain models or request API data.

Generate the PDF and check the site rendering (Node plus Playwright 1.55.1):

```bash
npm install --no-save --package-lock=false playwright@1.55.1
npx playwright install chromium
node scripts/research/render_nhl_paper.cjs
```

Alternatively, set `CHROME_PATH` to an installed Chrome executable and `NODE_PATH` if
Playwright lives outside this checkout. The renderer uses a local HTTP server, checks
the research index and paper at 390/768/1440px, citation anchors, all 11 tables, numerical
supplement, horizontal overflow and JavaScript exceptions. It writes the PDF and saves
browser screenshots/check results to the system temporary directory. Browser versions
can change PDF binary bytes without changing research results.

The source report's reconstruction commands appear in the paper appendix. Those commands
must run in the pinned NHL environment; PDF generation is separate from statistical
reproduction. No model parameters, betting selectors, scheduled workflows or live NHL
predictions are changed by this editorial branch.

Validation: all six route/viewport combinations passed; the five existing site-notice
tests passed; the evidence JSON has no credential-bearing URL markers. PDF text and
selected raster pages are checked before review. See `../nhl-rebuild/DEPLOYMENT.md` for
the completed production launch, including its successful deployment retry.
