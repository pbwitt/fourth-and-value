#!/usr/bin/env python3
"""Audit indexed pages against SEO_POLICY.md. Report only; never fails a build.

--changed   only pages changed on this branch (vs --base, default origin/main)
(default)   every indexed page
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
    ap.add_argument('--changed', action='store_true')
    ap.add_argument('--base', default='origin/main')
    a = ap.parse_args()
    results = audit_all()
    if a.changed:
        scope = changed_pages(a.base)
        results = {k: v for k, v in results.items() if scope is None or k in scope}
    counts = {}
    for rel, issues in sorted(results.items()):
        if a.changed and issues:
            print(f'{rel}: {", ".join(issues)}')
        for i in issues:
            counts[i] = counts.get(i, 0) + 1
    print(f'{len(results)} indexed pages; {sum(1 for v in results.values() if v)} with issues')
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f'  {k}: {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
