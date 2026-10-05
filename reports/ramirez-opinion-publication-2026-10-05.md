# GOAT opinion publication — October 5, 2026

Owner approved the complete draft in chat and requested publication plus a few
days in the homepage slider. Byline: GOAT — a Fourth & Value contributor.

- Permanent URL: `/editorial/articles/jose-ramirez-cleveland-playoffs-help.html`.
- Catalog: `config/editorial.json`; clearly labeled Opinion, linked from the
  homepage opinion section, permanent Opinion archive, Blog index and sitemap.
- Feature ends October 8, 2026 at 6 p.m. America/New_York
  (`2026-10-08T22:00:00+00:00`). The article remains published afterward.
- `featured_opinions` requires explicit `featured` and `featured_until`. Opinions
  remain excluded from the market briefing and analysis-only feature list.
- `home_slides` reserves one slot for the latest opted-in opinion while preserving
  the existing featured-blog slot. Subsequent daily articles cannot displace it.
- Browser expiry removes a stale opinion slide on page load even between builds;
  scheduled homepage renders also remove expired promotions.
- Approved prose retained; chat citations converted to numbered source links.
  Calculations documented on page. Metadata identifies GOAT as the author.
- No private contributor queue, approval or automatic writing rules changed.

Validation: editorial unit tests, expiry/slot regression, HTML metadata audit,
JavaScript DOM-harness expiry checks before/after the cutoff. Browser layout
verification was unavailable because the Chromium download failed; existing
article/slider styles and templates are reused.
