# SEO policy

Every public page is optimized for search and for sharing. This is a standing
requirement, not a per-task request: any change that adds or edits a public page
(hand-written, templated or generated) must meet this policy before it ships.

## Every indexed page

1. **Title tag ≤ 65 characters**, including the brand. Lead with what a reader would
   search for. Append ` | Fourth & Value` only when it fits; otherwise drop it. The
   on-page H1 can be longer and more editorial.
2. **Meta description 70–160 characters**, written for that page: what the reader
   gets and why it is useful. No generic filler such as "Sports analysis and research
   from Fourth & Value."
3. **One canonical URL** (`<link rel="canonical">`, absolute, https).
4. **Exactly one `<h1>`.**
5. **Open Graph and Twitter tags**: `og:title`, `og:description`, `og:url`,
   `og:type`, `og:image`, and `twitter:card` = `summary_large_image`.
6. **A branded preview image.** Shared links must show Fourth & Value branding.
   Use a page-specific card when one exists (1200×630 or 1280×720, ≤ 300 KB);
   otherwise the default `https://fourthandvalue.com/assets/social-card.png`.
   Never use a story photo we do not own.
7. **In `docs/sitemap.xml`**, unless deliberately `noindex` (templates, archives,
   redirects).

## Articles and blog posts (in addition)

8. **Article JSON-LD** with `headline`, `description`, `datePublished`,
   `dateModified`, `author` and `publisher` (Organization: Fourth & Value), and
   `image` when the article has its own card. FAQ JSON-LD when the page has an FAQ.
9. **`article:published_time`** and **`article:section`** meta tags.
10. **Internal links**: listed in the Blog index and, for explainers, the Learn hub;
    linked from at least one relevant methods or hub page. Link to related posts.
11. **Readable URLs**: lowercase, hyphenated, descriptive slugs; never change a
    published URL.

## Shareable features

12. Share buttons send the branded image only (no link, which would add a second
    preview card); Copy link is separate. Deep links open the exact view shared.

## Checking

`python scripts/seo_check.py` reports every indexed page's gaps (`--changed` for
pages changed on your branch). It is an audit, not a gate: it never fails a build.
Fix gaps when you touch a page. Generators (`scripts/site_metadata.py`,
`scripts/editorial_seo.py`, `scripts/build_site_metadata.py`, page builders and
templates) must emit compliant tags, so automated pages comply without review.
