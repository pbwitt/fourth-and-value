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

4. Keep [RESEARCH_SYSTEM.md](RESEARCH_SYSTEM.md) (architecture, schemas, statuses,
   provenance, decision log) consistent with the change. Bump
   `MODEL_VERSION` in `scripts/make_player_prop_params.py` when NFL forecasting logic
   changes; an artifact fitted for another version is reported as incompatible.

The source-of-truth selector is `docs/assets/briefing-picks.js`, shared by the
browser and Node research adapter. Preserve exact offered lines/quotes, missing
probabilities, push semantics and the separation from Market Watch. A count of
games submitted to discovery is not a count of games exhaustively researched.

## SEO (standing requirement)

Every public page must meet `SEO_POLICY.md` without being asked: a search title of
65 characters or fewer, a specific 70–160 character description, canonical URL, one H1,
Open Graph/Twitter tags with a branded preview image, Article JSON-LD for articles,
sitemap entry and internal links. Generators must emit compliant tags. Run
`python scripts/seo_check.py --changed` before shipping; it reports gaps and never
fails a build.
