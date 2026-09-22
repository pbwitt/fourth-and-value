"""Consistent, visible responsible-use notices for public pages and future builds."""
import argparse
import json
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
NOTICE='''<aside id="fv-betting-notice" class="fv-use-notice" aria-label="Betting information disclaimer"><strong>Analysis, not a guarantee.</strong> Picks and projections are uncertain; odds and player information can change. No outcome or profit is guaranteed. Any wager is your decision and responsibility. <a href="/terms.html#responsible-play">Read our limitations and responsible-play guidance</a>.</aside>'''
FOOTER='''<section id="fv-responsible-footer" class="fv-use-footer" aria-label="Responsible use"><p>Fourth &amp; Value provides sports analysis for informational and entertainment purposes. We do not accept wagers. You are responsible for your wagering decisions and for meeting applicable age and legal requirements. Never wager money you cannot afford to lose.</p><p><a href="/terms.html">Terms &amp; privacy</a> · <a href="/terms.html#responsible-play">Responsible play</a> · U.S. gambling support: call or text <a href="tel:+18006973738">1-800-MY-RESET</a> or <a href="https://www.ncpgambling.org/help-treatment/">find confidential help</a>.</p></section>'''

def contextual(path):
    return path.startswith(('props/','nfl/','nba/','nhl/','mlb/','tracking/','briefing/','research/','editorial/articles/')) or path=='methods.html' or (path.startswith('blog/') and path!='blog/index.html')

def apply(html,path):
    if path.endswith('render.html') or 'template' in path or re.search(r'http-equiv=["\']refresh',html,re.I):return html
    if '</body>' not in html.lower():return html
    # Rebuild only our own marked inserts; preserve authored prose and dates.
    html=re.sub(r'<!-- fv-notices:(?:notice|footer):start -->.*?<!-- fv-notices:(?:notice|footer):end -->','',html,flags=re.S)
    if '/assets/responsible-use.css' not in html:
        html=html.replace('</head>','<link rel="stylesheet" href="/assets/responsible-use.css?v=1"></head>')
    if contextual(path):
        block='<!-- fv-notices:notice:start -->'+NOTICE+'<!-- fv-notices:notice:end -->'
        header=re.search(r'<header\b[^>]*>(?:(?!</header>).)*<h1\b.*?</header>',html,re.S|re.I)
        heading=re.search(r'<h1\b[^>]*>.*?</h1>',html,re.S|re.I)
        match=header or heading
        if match:html=html[:match.end()]+block+html[match.end():]
    return html.replace('</body>','<!-- fv-notices:footer:start -->'+FOOTER+'<!-- fv-notices:footer:end --></body>')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',default='');args=parser.parse_args()
    count=0
    for p in (DOCS/args.scope).rglob('*.html'):
        old=p.read_text();new=apply(old,p.relative_to(DOCS).as_posix())
        if new!=old:p.write_text(new);count+=1
    print(f'Responsible-use notices updated on {count} pages.')

if __name__=='__main__':main()
