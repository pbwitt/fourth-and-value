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
        tags[f'{sport}_N_PASS'] = 'zero one two three four five six seven eight nine ten'.split()[sum(r['replicates'] for r in rows)]
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
                    if p.get(k) is not None:
                        tags[f'{key}_{label[:3].upper()}_{k.upper()}_ABS'] = f'{abs(p[k]):.2f}' if abs(p[k]) < 1.5 else f'{abs(p[k]):.1f}'
                        tags[f'{key}_{label[:3].upper()}_{k.upper()}_ROUND'] = f'{abs(p[k]):.0f}'
        tags[f'{key}_VERDICT'] = r['verdict']
        tags[f'{key}_Q'] = '—' if r.get('q_full') is None else f"{r['q_full']:.1g}" if r['q_full'] < .01 else f"{r['q_full']:.2f}"
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
    words = 'zero one two three four five six seven eight nine ten eleven twelve'.split()
    tags['N_PASS_WORD'] = words[int(tags['N_PASS'])].capitalize()
    tags['N_BET_WORD_LOWER'] = words[int(tags['N_BET'])]
    tags['PRICED_FIG'] = fig_priced(f, [s for s in SPORTS if s in sports])
    tags.update(extra_tables(res))
    return tags


def rate_cell(x, side='under'):
    if not x or x.get('under_rate') is None:
        return '—'
    w = x['under_rate'] if side == 'under' else 1 - x['under_rate']
    return f"{pct(w)} ({x['n']:,})"


def extra_tables(res):
    f, sports = res['factors'], res['sports']
    t = {}
    src = {'nflverse': 'nflverse closing total', 'sbro': 'archived closing total', 'espn': 'ESPN odds feed (median of books)'}
    rows = []
    situational = {'NFL': 'nflverse weather, roof, rest, referee, starting QB; play-by-play weather text',
                   'NHL': 'NHL schedule and venue time zones, referees, starting goalies',
                   'NBA': 'ESPN schedule and officials (crew chief listed first)',
                   'MLB': 'MLB Stats API game-time weather and wind, plate umpire, day/night, doubleheaders'}
    for sp in SPORTS:
        m = sports[sp]
        lines = '; '.join(f"{src.get(k, k)} {v:,}" for k, v in sorted(m['line_sources'].items(), key=lambda kv: -kv[1]))
        span = f"{m['periods']['discovery'][0]}–{m['periods']['replication'][1]}"
        if sp in ('NHL', 'NBA'):
            span = f"{m['periods']['discovery'][0]}–{str(m['periods']['discovery'][0] + 1)[2:]} to {m['periods']['replication'][1]}–{str(m['periods']['replication'][1] + 1)[2:]}"
        rows.append([sp, span, f"{m['games']:,}", lines, situational[sp]])
    t['DATA_TABLE'] = table('Table 1. Games, closing totals and situational data. Counts are games in the discovery and replication seasons after the pre-registered data-quality filters.',
                            ['League', 'Seasons', 'Games', 'Closing totals (games)', 'Situational data'], rows)
    nfl_s = sports['NFL']['sensitivity']['wind']
    t['WIND_TABLE'] = table('Table 8. NFL wind cut-off sensitivity (post-hoc). Share of outdoor games that went under, and the amount left over against other outdoor games, by minimum wind speed. The pre-registered cut-off is 15 mph.',
                            ['Wind at least', 'Games 2006–15', 'Under 2006–15', 'Left over', 'Games 2016–25', 'Under 2016–25', 'Left over'],
                            [[f"{w['cut']} mph", f"{w['discovery']['n']:,}", pct(w['discovery']['under_rate']), signed(w['discovery']['excess']),
                              f"{w['replication']['n']:,}", pct(w['replication']['under_rate']), signed(w['replication']['excess'])] for w in nfl_s])
    mlb_s = sports['MLB']['sensitivity']
    t['MLB_WIND_TABLE'] = table('Table 9. MLB wind blowing in, cut-off sensitivity (post-hoc). Share of outdoor games that went under, and the runs left over against other outdoor games.',
                                ['Wind in at least', 'Games 2010–17', 'Under 2010–17', 'Left over', 'Games 2018–25', 'Under 2018–25', 'Left over'],
                                [[f"{w['cut']} mph", f"{w['discovery']['n']:,}", pct(w['discovery']['under_rate']), signed(w['discovery']['excess']),
                                  f"{w['replication']['n']:,}", pct(w['replication']['under_rate']), signed(w['replication']['excess'])] for w in mlb_s['wind_in']])
    ov = mlb_s['day_after_night_x_getaway']
    t['OVERLAP_TABLE'] = table('Table 10. MLB day game after a night game, and getaway day, split by overlap (post-hoc). Under rate (games) and runs left over.',
                               ['Games that are…', '2010–17', 'Left over', '2018–25', 'Left over', '2026', 'Left over'],
                               [[name, rate_cell(ov['discovery'][name]), signed(ov['discovery'][name].get('excess')),
                                 rate_cell(ov['replication'][name]), signed(ov['replication'][name].get('excess')),
                                 rate_cell(ov['fresh'][name]), signed(ov['fresh'][name].get('excess'))]
                                for name in ('both', 'day after night only', 'getaway only')])
    rows = []
    for fid in ('NFL-12', 'NHL-11', 'NBA-11', 'MLB-12'):
        r = f[fid]
        st = r.get('stability', {})
        rows.append([f"{r['sport']} {r['name'].lower().replace('low-scoring ', '')}", str(r['officials_rated']), rate_cell(r['discovery']),
                     rate_cell(r['replication']), signed(r['replication'].get('effect')), signed(r['replication'].get('priced')),
                     f"{signed(st.get('spearman'), 2)} ({st.get('officials', 0)})" if st else '—'])
    t['OFFICIALS_TABLE'] = table("Table 6. Officials. The lowest-scoring third of officials, rated on the discovery seasons: their games' under rate in the rating seasons and afterwards, how much less those later games scored, how much lower the line was set, and the rank correlation of every official's tendency between the two periods (officials with 30+ games in both).",
                                 ['League', 'Officials rated', 'Under, rating seasons', 'Under, later seasons', 'Scoring effect later', 'Line moved later', 'Correlation (officials)'], rows)
    rows = []
    for fid, r in f.items():
        fr = r.get('fresh')
        if fr and fr.get('n', 0) >= 10 and (r['replicates'] or fid in ('NFL-9', 'NFL-3', 'NFL-15')):
            rows.append([f"{fid} {r['name']}", r['tested_side'], rate_cell(fr, r['tested_side']), signed(fr.get('excess'))])
    t['FRESH_TABLE'] = table('Table 11. The 2026 seasons, which no rule or threshold saw: NFL Weeks 1–6 and the 2026 MLB season. Win rate on the predicted side (games) and amount left over. The NFL had no game with 15+ mph wind in this window.',
                             ['Situation', 'Side', '2026 win rate', 'Left over'], rows)
    passed = [r for r in f.values() if r['replicates']]
    t['PASS_TABLE'] = table('Table 7. Every situation that met the pre-registered replication rule. Win rate on the predicted side; the break-even rate at −110 is 52.4%.',
                            ['Situation', 'Side', 'Earlier seasons', 'Later seasons', 'Left over (later)', 'q', 'Verdict'],
                            [[f"{r['sport']} · {r['name']}", r['tested_side'], rate_cell(r['discovery'], r['tested_side']),
                              rate_cell(r['replication'], r['tested_side']), signed(r['replication']['excess']),
                              '<0.01' if r['q_full'] < .01 else f"{r['q_full']:.2f}", r['verdict']] for r in sorted(passed, key=lambda r: -(win(r['replication'], r['tested_side'])[0]))])
    return t


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
    for part in ('SYNOPSIS', 'ABSTRACT', 'BODY', 'RESULTS', 'CONCLUSION'):
        shell = shell.replace('@@' + part + '@@', (HERE / f'{part.lower()}.html').read_text().strip('\n'))
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
