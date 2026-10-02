"""Static, crawlable page metadata and shared NFL context."""
from html import escape
from pathlib import Path
import os

SITE = 'https://fourthandvalue.com'
# Default branded preview card (SEO_POLICY.md); pages with their own card override it.
SOCIAL_CARD = SITE + '/assets/social-card.png'
BRAND = ' | Fourth & Value'


def seo_title(title, limit=65):
    """Search title within the policy limit: brand only when it fits, else trim at a word."""
    title = ' '.join(str(title).split())
    if len(title) + len(BRAND) <= limit:
        return title + BRAND
    if len(title) <= limit:
        return title
    cut = title[:limit - 1].rsplit(' ', 1)[0].rstrip(' ,:;–—-')
    return cut + '…'


def seo_description(text, limit=160):
    """Meta description within the policy limit: whole sentences when possible, else whole words."""
    text = ' '.join(str(text).split())
    if len(text) <= limit:
        return text
    sentences, out = text.replace('? ', '?|').replace('. ', '.|').replace('! ', '!|').split('|'), ''
    for s in sentences:
        if len(out) + len(s) + 1 > limit:
            break
        out = (out + ' ' + s).strip()
    if len(out) >= 70:
        return out
    return text[:limit - 1].rsplit(' ', 1)[0].rstrip(' ,:;–—-') + '…'


def social_tags(image=None, alt='Fourth & Value'):
    image = image or SOCIAL_CARD
    image = SITE + image if image.startswith('/') else image
    return (f'<meta property="og:image" content="{escape(image, quote=True)}">'
            f'<meta property="og:image:alt" content="{escape(alt, quote=True)}">'
            '<meta name="twitter:card" content="summary_large_image">'
            f'<meta name="twitter:image" content="{escape(image, quote=True)}">')
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
{social_tags()}'''


def nfl_links(rel, current=''):
    links = [('NFL overview', 'nfl/'), ('Player props', 'props/'), ('Top picks', 'props/top.html'),
             ('Game totals', 'nfl/totals/')]
    return '<nav class="subnav" aria-label="NFL">' + ''.join(
        f'<a href="{rel}/{href}"' + (' aria-current="page"' if label == current else '') + f'>{label}</a>'
        for label, href in links) + '</nav>'
