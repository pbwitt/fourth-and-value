"""Render Technical Report FV-2026-04 (low-scoring situations) from committed evidence.

Run from the repository root: python3 scripts/research/low_scoring/build_paper.py [--pdf]
Formats reports/low-scoring-2026/results.json into docs/research/<SLUG>.html, its evidence JSON and
social card; --pdf also prints the PDF with the pre-installed Chromium. It does not refit anything
or contact a provider. Every number in the paper is a @@TAG@@ filled from the results file.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(HERE))
from figures import fig_priced, fig_record, pct, record, signed, table, table_sport, win  # noqa: E402
from site_notices import apply  # noqa: E402
from text import FAQ, TEXT_TAGS  # noqa: E402

DEST = ROOT / 'docs/research'
SLUG = 'low-scoring-games-situational-factors'
RESULTS = ROOT / 'reports/low-scoring-2026/results.json'
SPORTS = ['NFL', 'NHL', 'NBA', 'MLB']
CHROME = [p for p in ('/opt/pw-browsers/chromium-1194/chrome-linux/chrome', shutil.which('chromium'), shutil.which('google-chrome')) if p and Path(p).exists()]


def fmt_tags(res):
    f, sports = res['factors'], res['sports']
    tags = {}
    for sport in SPORTS:
        if sport not in sports:
            continue
        m = sports[sport]
        rows = [v for v in f.values() if v['sport'] == sport]
        tags[f'{sport}_FIG'] = fig_record(rows, sport, m['periods'])
        tags[f'{sport}_TABLE'] = table_sport(rows, sport, m['periods'])
        tags[f'{sport}_GAMES'] = f"{m['games']:,}"
        tags[f'{sport}_SEASONS'] = f"{m['periods']['discovery'][0]}–{m['periods']['replication'][1]}"
        tags[f'{sport}_DISC'] = '–'.join(str(x) for x in m['periods']['discovery'])
        tags[f'{sport}_REP'] = '–'.join(str(x) for x in m['periods']['replication'])
        tags[f'{sport}_UNDER'] = pct(m['under_rate'])
        tags[f'{sport}_SD'] = f"{m['residual_sd']:.1f}"
        tags[f'{sport}_MAE'] = f"{m['mae']:.2f}"
        tags[f'{sport}_BIAS'] = signed(m['mean_residual'])
        tags[f'{sport}_N_FACTORS'] = str(len(rows))
        tags[f'{sport}_N_PASS'] = str(sum(r['replicates'] for r in rows))
    for fid, r in f.items():
        key = fid.replace('-', '_')
        side = r.get('tested_side', 'under')
        for label in ('discovery', 'replication', 'full', 'fresh'):
            p = r.get(label) or {}
            if 'under_rate' in p:
                w, lo, hi = win(p, side)
                tags[f'{key}_{label[:3].upper()}_WIN'] = pct(w)
                tags[f'{key}_{label[:3].upper()}_CI'] = f'{pct(lo)}–{pct(hi)}'
                tags[f'{key}_{label[:3].upper()}_REC'] = record(p, side)
                tags[f'{key}_{label[:3].upper()}_N'] = f"{p['n']:,}"
                for k in ('effect', 'priced', 'excess', 'residual'):
                    tags[f'{key}_{label[:3].upper()}_{k.upper()}'] = signed(p.get(k))
        tags[f'{key}_VERDICT'] = r['verdict']
        tags[f'{key}_Q'] = '—' if r.get('q_full') is None else f"{r['q_full']:.2f}"
        if r.get('replication_price_roi'):
            tags[f'{key}_PRICE_ROI'] = signed(100 * r['replication_price_roi']['roi'], 1) + '%'
        if r.get('stability'):
            tags[f'{key}_RHO'] = signed(r['stability']['spearman'], 2)
            tags[f'{key}_RHO_N'] = str(r['stability']['officials'])
    tags['N_TESTS'] = str(sum(1 for v in f.values() if v.get('q_full') is not None))
    tags['N_FACTORS'] = str(len(f))
    tags['N_PASS'] = str(sum(v['replicates'] for v in f.values()))
    tags['N_BET'] = str(sum(v['bettable'] for v in f.values()))
    tags['N_GAMES'] = f"{sum(m['games'] for m in sports.values()):,}"
    tags['PRICED_FIG'] = fig_priced(f, [s for s in SPORTS if s in sports])
    return tags


def render(manuscript, tags):
    for key, value in tags.items():
        manuscript = manuscript.replace('@@' + key + '@@', value)
    for key in ('N_TESTS',):
        manuscript = manuscript.replace('{' + key + '}', tags[key])
    missing = sorted(set(part.split('@@')[0] for part in manuscript.split('@@')[1::2]))
    assert not missing, f'Unfilled manuscript tags: {missing[:20]}'
    return manuscript


def chrome(*args):
    if not CHROME:
        raise SystemExit('No Chromium found for PDF/PNG rendering')
    subprocess.run([CHROME[0], '--headless', '--no-sandbox', '--disable-gpu', '--hide-scrollbars', *args],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)


def social_card(tags):
    card = (HERE / 'social_card.html').read_text()
    for key, value in tags.items():
        card = card.replace('@@' + key + '@@', value)
    out = DEST / f'{SLUG}-social.png'
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / 'card.html'
        page.write_text(card)
        chrome('--window-size=1280,720', f'--screenshot={out}', page.as_uri())
    return out


def pdf(html_path):
    out = DEST / f'{SLUG}.pdf'
    chrome('--no-pdf-header-footer', '--run-all-compositor-stages-before-draw', '--virtual-time-budget=4000',
           f'--print-to-pdf={out}', html_path.as_uri())
    return out


PUBLISHED = '2026-10-16T13:00:00+00:00'
PAGE_URL = f'https://fourthandvalue.com/research/{SLUG}.html'
HEADLINE = 'What Makes Games Go Under? Weather, Fatigue, Travel and Officials Against the Closing Total'


def structured(tags):
    article = {
        '@context': 'https://schema.org', '@type': 'Article', 'headline': HEADLINE, 'description': tags['META_DESCRIPTION'],
        'datePublished': PUBLISHED, 'dateModified': PUBLISHED,
        'author': {'@type': 'Organization', 'name': 'Fourth & Value Research Group', 'url': 'https://fourthandvalue.com/research/'},
        'publisher': {'@type': 'Organization', 'name': 'Fourth & Value', 'url': 'https://fourthandvalue.com/',
                      'logo': {'@type': 'ImageObject', 'url': 'https://fourthandvalue.com/assets/logo.svg'}},
        'image': [f'https://fourthandvalue.com/research/{SLUG}-social.png'], 'mainEntityOfPage': PAGE_URL,
        'articleSection': 'Research', 'keywords': ['over/under', 'game totals', 'closing line', 'weather', 'back-to-back',
                                                   'referees', 'NFL', 'NHL', 'NBA', 'MLB']}
    faq = {'@context': 'https://schema.org', '@type': 'FAQPage', 'mainEntity': [
        {'@type': 'Question', 'name': q, 'acceptedAnswer': {'@type': 'Answer', 'text': a}} for q, a in FAQ(tags)]}
    return json.dumps(article, ensure_ascii=False), json.dumps(faq, ensure_ascii=False)


def main():
    res = json.loads(RESULTS.read_text())
    tags = fmt_tags(res)
    tags.update(TEXT_TAGS(tags))
    tags['JSONLD'], tags['FAQLD'] = structured(tags)
    tags['PUBLISHED'] = PUBLISHED
    shell = (HERE / 'paper.html').read_text()
    for part in ('SYNOPSIS', 'ABSTRACT', 'BODY'):
        shell = shell.replace('@@' + part + '@@', (HERE / f'{part.lower()}.html').read_text().strip())
    manuscript = render(shell, tags)
    DEST.mkdir(parents=True, exist_ok=True)
    evidence = dict(paper='FV-2026-04', paper_version='1.0', results_sha256=hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
                    preregistration='reports/low-scoring-2026/PREREGISTRATION.md', **res)
    (DEST / f'{SLUG}.json').write_text(json.dumps(evidence, indent=1) + '\n')
    page = DEST / f'{SLUG}.html'
    page.write_text(apply(manuscript, f'research/{SLUG}.html'))
    social_card(tags)
    if '--pdf' in sys.argv:
        # Print from a copy whose root-relative links resolve to the local docs folder.
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / page.name
            text = page.read_text().replace('href="/assets/', f'href="{(ROOT / "docs/assets").as_uri()}/').replace(
                'href="./', f'href="{DEST.as_uri()}/').replace('<script src="/nav.js', '<script data-skip="/nav.js')
            local.write_text(text)
            pdf(local)
    print(f'Rendered FV-2026-04: {page.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
