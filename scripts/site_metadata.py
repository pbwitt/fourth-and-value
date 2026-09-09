"""Static, crawlable page metadata and shared NFL context."""
from html import escape
from pathlib import Path
import os

SITE = 'https://fourthandvalue.com'
DOCS = Path(__file__).resolve().parents[1] / 'docs'


def root_relative(out):
    return Path(os.path.relpath(DOCS, Path(out).resolve().parent)).as_posix()


def metadata(out, title, description):
    try:
        path = Path(out).resolve().relative_to(DOCS).as_posix()
    except ValueError:
        path = 'props/index.html'
    url = SITE + '/' + (path[:-10] if path.endswith('index.html') else path)
    return f'''<meta name="description" content="{escape(description, quote=True)}">
<link rel="canonical" href="{url}">
<meta property="og:type" content="website">
<meta property="og:title" content="{escape(title, quote=True)}">
<meta property="og:description" content="{escape(description, quote=True)}">
<meta property="og:url" content="{url}">
<meta name="twitter:card" content="summary">'''


def nfl_links(rel, current=''):
    links = [('NFL overview', 'nfl/'), ('Player props', 'props/'), ('Top picks', 'props/top.html'),
             ('Game totals', 'nfl/totals/'), ('Methods', 'methods.html#edge-nfl')]
    return '<nav class="subnav" aria-label="NFL">' + ''.join(
        f'<a href="{rel}/{href}"' + (' aria-current="page"' if label == current else '') + f'>{label}</a>'
        for label, href in links) + '</nav>'
