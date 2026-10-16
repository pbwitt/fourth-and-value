"""Publish the FV-2026-04 pre-registration as a page on the site, so the paper links to it there.

Converts reports/low-scoring-2026/PREREGISTRATION.md (headings, paragraphs, lists, tables, bold,
italics and inline code) into docs/research/<SLUG>.html with the paper's stylesheet. The registered
text is reproduced unchanged.
"""
import html
import json
import re

SLUG = 'low-scoring-games-situational-factors-preregistration'
STUDY = 'low-scoring-games-situational-factors'
SITE = 'https://fourthandvalue.com'
PAGE_URL = f'{SITE}/research/{SLUG}.html'
TITLE = 'Pre-registration: What Makes Games Go Under? | Fourth &amp; Value'
DESCRIPTION = ('The hypotheses, thresholds, seasons and pass rules for FV-2026-04, fixed in writing before any result was '
               'computed, with both dated amendments.')
LIST_ITEM = re.compile(r'(- |(\d+)\. )')


def inline(text):
    t = html.escape(text, quote=False)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    return re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<em>\1</em>', t)


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def markdown(md):
    """Return (h1 text, body HTML) for the subset of Markdown the pre-registration uses."""
    lines, out, h1, i = md.splitlines(), [], '', 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        heading = re.match(r'(#{1,6})\s+(.*)', line)
        if heading:
            level, text = len(heading.group(1)), heading.group(2).strip()
            if level == 1 and not h1:
                h1 = text
            else:
                out.append(f'<h{level} id="{slug(text)}">{inline(text)}</h{level}>')
            i += 1
            continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not all(re.fullmatch(r':?-{3,}:?', c) for c in r)]
            out.append('<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th scope="col">{inline(c)}</th>' for c in head)
                       + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in body)
                       + '</tbody></table></div>')
            continue
        first = LIST_ITEM.match(line)
        if first:
            ordered, items = bool(first.group(2)), []
            while i < len(lines) and (LIST_ITEM.match(lines[i]) or (items and lines[i].startswith('  ') and lines[i].strip())):
                if LIST_ITEM.match(lines[i]):
                    items.append(LIST_ITEM.sub('', lines[i], count=1).strip())
                else:
                    items[-1] += ' ' + lines[i].strip()
                i += 1
            tag = 'ol' if ordered else 'ul'
            start = f' start="{first.group(2)}"' if ordered and first.group(2) != '1' else ''
            out.append(f'<{tag}{start}>' + ''.join(f'<li>{inline(t)}</li>' for t in items) + f'</{tag}>')
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r'(#{1,6}\s|\|)', lines[i]) and not LIST_ITEM.match(lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f'<p>{inline(" ".join(para))}</p>')
    return h1, '\n'.join(out)


def page(md, published):
    h1, body = markdown(md)
    article = {
        '@context': 'https://schema.org', '@type': 'Article', 'headline': 'Pre-registration: What Makes Games Go Under? (FV-2026-04)',
        'description': DESCRIPTION, 'datePublished': published, 'dateModified': published,
        'author': {'@type': 'Organization', 'name': 'Fourth & Value Research Group', 'url': f'{SITE}/research/'},
        'publisher': {'@type': 'Organization', 'name': 'Fourth & Value', 'url': f'{SITE}/',
                      'logo': {'@type': 'ImageObject', 'url': f'{SITE}/assets/logo.svg'}},
        'image': [f'{SITE}/research/{STUDY}-social.png'], 'mainEntityOfPage': PAGE_URL, 'articleSection': 'Research',
        'isPartOf': f'{SITE}/research/{STUDY}.html'}
    image = f'{SITE}/research/{STUDY}-social.png'
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{TITLE}</title>
  <meta name="description" content="{DESCRIPTION}">
  <link rel="canonical" href="{PAGE_URL}">
  <meta property="og:title" content="Pre-registration: What Makes Games Go Under? (FV-2026-04)">
  <meta property="og:description" content="{DESCRIPTION}">
  <meta property="og:url" content="{PAGE_URL}">
  <meta property="og:type" content="article">
  <meta property="og:image" content="{image}">
  <meta property="og:image:width" content="1280">
  <meta property="og:image:height" content="720">
  <meta property="og:image:alt" content="Fourth &amp; Value research: what makes games go under? Situations tested against the closing total.">
  <meta property="article:published_time" content="{published}">
  <meta property="article:modified_time" content="{published}">
  <meta property="article:section" content="Research">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="Pre-registration: What Makes Games Go Under? (FV-2026-04)">
  <meta name="twitter:description" content="{DESCRIPTION}">
  <meta name="twitter:image" content="{image}">
  <script type="application/ld+json">{json.dumps(article, ensure_ascii=False)}</script>
  <link rel="icon" href="/assets/logo.svg">
  <link rel="stylesheet" href="./nhl-forecasting.css">
  <style>
    .prereg h2{{margin-top:2em}} .prereg table{{border-collapse:collapse;width:100%;font-size:.9rem}}
    .prereg th,.prereg td{{text-align:left;vertical-align:top;padding:7px 10px;border-bottom:1px solid var(--rule);font-family:inherit;font-style:normal}}
    .prereg th,.prereg td:first-child,.prereg td:last-child{{white-space:nowrap}}
    .prereg code{{font-family:'JetBrains Mono',monospace;font-size:.88em}}
    .prereg .note{{border:1px solid var(--rule);border-radius:10px;padding:14px 18px;margin:0 0 2em}}
  </style>
</head>
<body>
<div id="nav-root"></div><script src="/nav.js?v=47"></script>
<main class="wrap prereg" id="prereg">
  <div class="paper-head">
    <div class="kicker">Fourth &amp; Value Research · Technical Report FV-2026-04 · Pre-registration</div>
    <h1 class="title">{inline(h1)}</h1>
  </div>
  <p class="note">This is the pre-registration for <a href="./{STUDY}.html">What Makes Games Go Under? Weather, Fatigue, Travel and Officials Against the Closing Total</a>, reproduced word for word: the hypotheses registered on October 15, 2026, and the two amendments, each dated with when it was made relative to the results.</p>
{body}
  <p class="paper-links"><a href="./{STUDY}.html">Read the study</a> · <a href="/research/">All research</a> · <a href="/methods.html">How our models work</a></p>
</main>
</body>
</html>
'''
