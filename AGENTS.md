# Fourth & Value maintenance

When changing sports refresh schedules, Top Picks eligibility/ranking, research
stages, spending limits, freshness rules, or review semantics:

1. Update the reader-facing flow in `scripts/research/daily_process.html` and the
   operational details in `ANALYST_RESEARCH.md` as needed. Describe the actual
   execution order and distinguish scheduled start times from publication times.
2. Run `python scripts/build_daily_process.py` and commit the generated
   `docs/research/daily-process.html` with the code change. PR checks run `--check`.
3. Keep Research and Today's Picks linked to this current process page. Historical
   research results must not be relabeled as evidence for a new selection policy.

The source-of-truth selector is `docs/assets/briefing-picks.js`, shared by the
browser and Node research adapter. Preserve exact offered lines/quotes, missing
probabilities, push semantics and the separation from Market Watch. A count of
games submitted to discovery is not a count of games exhaustively researched.

## SEO (standing requirement)

Every public page must meet `SEO_POLICY.md` without being asked: a search title of
65 characters or fewer, a specific 70–160 character description, canonical URL, one H1,
Open Graph/Twitter tags with a branded preview image, Article JSON-LD for articles,
sitemap entry and internal links. Generators must emit compliant tags. Run
`python scripts/seo_check.py --check` before shipping; PR checks enforce it, and
`tests/seo_baseline.json` (known legacy gaps) may only shrink.
