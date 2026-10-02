#!/usr/bin/env python3
"""Audit indexed pages against SEO_POLICY.md.

--check            fail when a page changed on this branch has an issue not in the baseline
--update-baseline  record current issues (the baseline should only shrink)
(default)          print a summary of every indexed page
"""
import argparse
from html import unescape
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
SITE = 'https://fourthandvalue.com'
BASELINE = ROOT / 'tests/seo_baseline.json'
GENERIC = 'Sports analysis and research from Fourth & Value'
ARTICLE_PREFIXES = ('blog/', 'editorial/articles/')
ARTICLE_EXCLUDE = ('blog/index.html', 'blog/post-template.html', 'blog/series-template.html')


def meta(html, attr, name):
    m = re.search(rf'<meta\b[^>]*{attr}=["\']{re.escape(name)}["\'][^>]*>', html, re.I)
    if not m:
        return None
    c = re.search(r'content=["\']([^"\']*)["\']', m.group(0), re.I)
    return unescape(c.group(1)) if c else ''


def indexed_pages():
    """Sitemap URLs mapped to their files."""
    pages = {}
    for loc in re.findall(r'<loc>(.*?)</loc>', (DOCS / 'sitemap.xml').read_text()):
        rel = loc[len(SITE):].lstrip('/')
        if not rel or rel.endswith('/'):
            rel += 'index.html'
        elif not rel.endswith('.html'):
            rel += '/index.html'
        pages[rel] = DOCS / rel
    return pages


def audit(rel, html):
    issues = []
    t = re.search(r'<title[^>]*>(.*?)</title>', html, re.S | re.I)
    title = unescape(re.sub(r'<[^>]+>', '', t.group(1))).strip() if t else ''
    if not title:
        issues.append('title:missing')
    elif len(title) > 65:
        issues.append('title:too-long')
    desc = meta(html, 'name', 'description')
    if not desc:
        issues.append('description:missing')
    elif GENERIC in desc:
        issues.append('description:generic')
    elif not 70 <= len(desc) <= 160:
        issues.append('description:length')
    if not re.search(r'<link\b[^>]*rel=["\']canonical["\']', html, re.I):
        issues.append('canonical:missing')
    if len(re.findall(r'<h1[\s>]', html, re.I)) != 1:
        issues.append('h1:count')
    for key in ('og:title', 'og:description', 'og:url', 'og:image'):
        if not meta(html, 'property', key):
            issues.append(f'{key}:missing')
    if meta(html, 'name', 'twitter:card') != 'summary_large_image':
        issues.append('twitter:card')
    if rel.startswith(ARTICLE_PREFIXES) and rel not in ARTICLE_EXCLUDE:
        if not re.search(r'"@type"\s*:\s*"(?:Article|BlogPosting|NewsArticle)"', html):
            issues.append('article:schema')
        if not meta(html, 'property', 'article:published_time'):
            issues.append('article:published_time')
    return issues


def audit_all():
    return {rel: audit(rel, path.read_text()) for rel, path in indexed_pages().items() if path.exists()}


def changed_pages(base):
    try:
        out = subprocess.run(['git', 'diff', '--name-only', f'{base}...HEAD', '--', 'docs'],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return {p[len('docs/'):] for p in out.split() if p.endswith('.html')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--update-baseline', action='store_true')
    ap.add_argument('--base', default='origin/main', help='branch point for --check (default origin/main)')
    ap.add_argument('--all', action='store_true', help='with --check, check every page, not only changed ones')
    a = ap.parse_args()
    results = audit_all()
    if a.update_baseline:
        BASELINE.write_text(json.dumps({k: v for k, v in sorted(results.items()) if v}, indent=1) + '\n')
        print(f'Baseline: {sum(map(len, results.values()))} known issues on {sum(1 for v in results.values() if v)} pages')
        return 0
    if a.check:
        baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        scope = set(results) if a.all else changed_pages(a.base)
        if scope is None:
            scope = set(results)
        new = {rel: [i for i in results[rel] if i not in baseline.get(rel, [])] for rel in scope if rel in results}
        new = {k: v for k, v in new.items() if v}
        for rel, issues in sorted(new.items()):
            print(f'{rel}: {", ".join(issues)}')
        if new:
            print(f'SEO policy: {len(new)} page(s) have new issues. See SEO_POLICY.md.', file=sys.stderr)
            return 1
        print(f'SEO policy: {len(scope & set(results))} changed indexed page(s) checked; no new issues.')
        return 0
    counts = {}
    for issues in results.values():
        for i in issues:
            counts[i] = counts.get(i, 0) + 1
    print(f'{len(results)} indexed pages; {sum(1 for v in results.values() if v)} with issues')
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f'  {k}: {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
