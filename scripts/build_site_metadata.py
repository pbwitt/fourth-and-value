#!/usr/bin/env python3
"""Maintain static discovery metadata; exclude archives and authoring templates."""
import re
from pathlib import Path
from html import escape, unescape
from site_metadata import SITE, DOCS, metadata

DESCRIPTIONS={
    'methods.html':'How Fourth & Value estimates NFL and NHL probabilities, compares sportsbook prices and evaluates model limitations.',
    'blog/index.html':'NFL betting explainers covering sportsbook margins, model uncertainty, line shopping and football analysis.',
    'research/index.html':'Read Fourth & Value research on probability calibration, Brier scores and evaluating sports models.',
    'tracking/index.html':'Track your sports bets and review your results with a free Fourth & Value account.',
}


def main():
    urls=[]
    for path in sorted(DOCS.rglob('*.html')):
        rel=path.relative_to(DOCS).as_posix();html=path.read_text()
        if 'template' in path.name and not re.search(r'name=["\']robots',html):
            html=html.replace('</head>','<meta name="robots" content="noindex,follow"></head>')
        match=re.search(r'<title>(.*?)</title>',html,re.S)
        title=unescape(re.sub('<[^>]+>','',match[1])).strip() if match else 'Fourth & Value'
        desc=DESCRIPTIONS.get(rel, f'{title}. Sports analysis and research from Fourth & Value.')
        if not re.search(r'name=["\']description',html):
            html=html.replace('</head>',f'<meta name="description" content="{escape(desc,quote=True)}"></head>')
        canonical=SITE+'/'+(rel[:-10] if rel.endswith('index.html') else rel)
        if not re.search(r'rel=["\']canonical',html):
            html=html.replace('</head>',f'<link rel="canonical" href="{canonical}"></head>')
        if not re.search(r'property=["\']og:title',html):
            html=html.replace('</head>',f'<meta property="og:title" content="{escape(title,quote=True)}"><meta property="og:description" content="{escape(desc,quote=True)}"><meta property="og:url" content="{canonical}"><meta property="og:type" content="website"><meta name="twitter:card" content="summary"></head>')
        path.write_text(html)
        if not re.search(r'name=["\']robots["\'][^>]*noindex',html) and 'http-equiv="refresh"' not in html:
            urls.append(canonical)
    (DOCS/'sitemap.xml').write_text('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'+''.join(f'  <url><loc>{escape(u)}</loc></url>\n' for u in urls)+'</urlset>\n')
    (DOCS/'robots.txt').write_text('User-agent: *\nAllow: /\n\nSitemap: '+SITE+'/sitemap.xml\n')
    print(f'Metadata checked; {len(urls)} canonical pages in sitemap.')

if __name__=='__main__':main()
