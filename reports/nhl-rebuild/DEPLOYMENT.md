# Approved NHL production launch — September 26, 2026

The owner explicitly authorized going live after reviewing PR #6 and its readiness.
This supersedes the pre-approval status recorded in PLAN.md and RUNBOOK.md; their
descriptions of the original review boundary remain historical context.

- PR #6 merged at 22:01:58 UTC: `e1f91609524f74a6b99cf86cdebb2328cd5e08d1`.
- Production refresh [36274974028](https://github.com/pbwitt/fourth-and-value/actions/runs/36274974028)
  passed tests, freshness validation, archival and publishing.
- Published snapshot commit: `7a563e0bd6dfae622921c999dfb550fe656c42da`.
- Pages deployment [36275025406](https://github.com/pbwitt/fourth-and-value/actions/runs/36275025406)
  initially failed on a GitHub OIDC token timeout. Retrying failed jobs succeeded;
  no application or permission changes were required.
- Live browser verification at 22:08 UTC checked all five NHL routes at 390px and
  1440px, HTTP 200, rendered feed status, horizontal overflow and JavaScript errors.
- Public `latest.json`: model `nhl-v2.1`, status `ready`, successful source refresh
  `2026-09-26T22:03:29.939416Z`, 794 quotes, 285 upcoming regular-season fixtures,
  no model error, zero recommendations. Independent forecasts are withheld for
  fixtures beyond the 48-hour inference window.

Launch does not promote forecasts into validated recommendations or enable market
blending. The new research manuscript is a separate editorial review change.
