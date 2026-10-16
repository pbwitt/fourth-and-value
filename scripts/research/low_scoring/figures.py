"""Inline SVG figures and HTML tables for Technical Report FV-2026-04 (low-scoring situations).

Colors are CSS custom properties set in the paper's stylesheet (dark screen, light print), validated
with the dataviz palette checker: accent #3987e5 / #86b6ef on #101114 (print #2a78d6 / #6da7ec).
"""
import html

SPORT_UNIT = {'NFL': 'points', 'NHL': 'goals', 'NBA': 'points', 'MLB': 'runs'}


def esc(x):
    return html.escape(str(x), quote=True)


def pct(x, digits=1):
    return '—' if x is None else f'{100 * x:.{digits}f}%'


def signed(x, digits=2):
    return '—' if x is None else f'{x:+.{digits}f}'.replace('-', '−')


def win(period, side):
    if not period or period.get('under_rate') is None:
        return None, None, None
    if side == 'under':
        return period['under_rate'], period['under_lo'], period['under_hi']
    return 1 - period['under_rate'], 1 - period['under_hi'], 1 - period['under_lo']


def record(period, side):
    if not period or period.get('under') is None:
        return '—'
    w, l = (period['under'], period['over']) if side == 'under' else (period['over'], period['under'])
    return f"{w:,}–{l:,}"


def fig_record(rows, sport, periods):
    """Dot plot as HTML rows: each factor's win rate on its predicted side, earlier seasons (ring) and
    later seasons (dot) with the later seasons' 95% interval. Marks are positioned in percent, so the
    rows reflow on narrow screens without distorting the dots."""
    rows = [r for r in rows if r.get('replication', {}).get('n')]
    rows.sort(key=lambda r: -(win(r['replication'], r['tested_side'])[0] or 0))
    lo, hi = .30, .70

    def x(v):
        return (min(max(v, lo), hi) - lo) / (hi - lo) * 100
    d0, d1 = periods['discovery']
    r0, r1 = periods['replication']
    rules = (''.join(f'<i class="g" style="left:{x(v):.2f}%"></i>' for v in (.3, .4, .6, .7))
             + f'<i class="b" style="left:{x(.5):.2f}%"></i><i class="be" style="left:{x(110 / 210):.2f}%"></i>')
    out = [f'<div class="dotplot" role="group" aria-label="{sport}: share of bets won on the predicted side, {d0}–{d1} and {r0}–{r1}">']
    for r in rows:
        side = r['tested_side']
        rw, rlo, rhi = win(r['replication'], side)
        dw, _, _ = win(r.get('discovery'), side)
        cls = 'pass' if r['replicates'] else 'fail'
        tip = (f"{r['id']} {r['name']}. Bet the {side}. {r0}–{r1}: {pct(rw)} ({record(r['replication'], side)}), "
               f"95% interval {pct(rlo)}–{pct(rhi)}. "
               + (f"{d0}–{d1}: {pct(dw)} ({record(r.get('discovery'), side)}). " if r['group'] != 'Officials' else 'Officials were rated on the earlier seasons. ')
               + f"Verdict: {r['verdict']}.")
        marks = f'<i class="ci" style="left:{x(rlo):.2f}%;width:{x(rhi) - x(rlo):.2f}%"></i>'
        if dw is not None and r['group'] != 'Officials':
            marks += f'<i class="disc" style="left:{x(dw):.2f}%"></i>'
        marks += f'<i class="rep" style="left:{x(rw):.2f}%"></i>'
        out.append(f'<div class="dp-row {cls}" tabindex="0" data-tip="{esc(tip)}"><span class="dp-label">{esc(r["name"])} <em>→ {side}</em></span>'
                   f'<span class="dp-track" aria-hidden="true">{rules}{marks}</span></div>')
    ticks = ''.join(f'<span style="left:{x(v):.2f}%">{int(v * 100)}%</span>' for v in (.3, .4, .5, .6, .7))
    out.append(f'<div class="dp-row dp-axis" aria-hidden="true"><span class="dp-label"></span><span class="dp-ticks">{ticks}'
               f'<span class="be" style="left:{x(110 / 210):.2f}%">52.4%</span></span></div></div>')
    return ''.join(out)


def fig_priced(results, sports):
    """Small multiples: change in actual scoring (x) against change in the closing line (y)."""
    panels, w, h, pad = [], 360, 300, 46
    for sport in sports:
        rows = [r for r in results.values() if r['sport'] == sport and r.get('full', {}).get('effect') is not None
                and r['group'] != 'Officials' and r['full']['n'] >= 100]
        if not rows:
            continue
        span = max(max(abs(r['full']['effect']), abs(r['full']['priced'])) for r in rows)
        span = max(span * 1.15, .1)
        unit = SPORT_UNIT[sport]

        def sx(v):
            return pad + (v + span) / (2 * span) * (w - pad - 14)

        def sy(v):
            return h - pad - (v + span) / (2 * span) * (h - pad - 18)
        out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{sport}: effect on scoring against the change in the closing total" class="chart-svg">',
               f'<text x="{pad}" y="12" class="panel">{sport} ({unit})</text>',
               f'<line x1="{sx(-span):.1f}" y1="{sy(0):.1f}" x2="{sx(span):.1f}" y2="{sy(0):.1f}" class="grid"/>',
               f'<line x1="{sx(0):.1f}" y1="{sy(-span):.1f}" x2="{sx(0):.1f}" y2="{sy(span):.1f}" class="grid"/>',
               f'<line x1="{sx(-span):.1f}" y1="{sy(-span):.1f}" x2="{sx(span):.1f}" y2="{sy(span):.1f}" class="diag"/>',
               f'<text x="{sx(span) - 4:.1f}" y="{sy(span) + 14:.1f}" text-anchor="end" class="tick">fully priced</text>',
               f'<text x="{(sx(-span) + sx(span)) / 2:.1f}" y="{h - 8}" text-anchor="middle" class="tick">Change in actual scoring</text>',
               f'<text transform="translate(11 {(sy(-span) + sy(span)) / 2:.1f}) rotate(-90)" text-anchor="middle" class="tick">Change in closing total</text>']
        for v in (-span / 1.15, span / 1.15):
            out.append(f'<text x="{sx(v):.1f}" y="{sy(-span) + 14:.1f}" text-anchor="middle" class="tick">{signed(v, 1)}</text>')
        for r in sorted(rows, key=lambda r: r['replicates']):
            f = r['full']
            tip = (f"{r['id']} {r['name']}: games scored {signed(f['effect'])} {unit} vs. other games; "
                   f"the closing total was {signed(f['priced'])} {unit}; left over {signed(f['excess'])}.")
            out.append(f'<g tabindex="0" data-tip="{esc(tip)}"><circle cx="{sx(f["effect"]):.1f}" cy="{sy(f["priced"]):.1f}" r="9" fill="transparent"/>'
                       f'<circle cx="{sx(f["effect"]):.1f}" cy="{sy(f["priced"]):.1f}" r="4.5" class="{"rep" if r["replicates"] else "dot"}"/></g>')
        out.append('</svg>')
        panels.append('<div class="panel-cell">' + ''.join(out) + '</div>')
    return '<div class="small-multiples">' + ''.join(panels) + '</div>'


SHARE_LABEL = 'Research · FV-2026-04'
WORDS = 'none one two three four five six seven eight nine ten eleven twelve'.split()


def share_attrs(**fields):
    """data-share-* attributes read by docs/assets/chart-share.js for the figure's branded share image."""
    return ' '.join(f'data-share-{k}="{esc(v)}"' for k, v in fields.items() if v)


def share_sport(rows, sport, periods, games, span):
    """Headline, takeaway and how-to-read text for one league's dot plot, from the same results as the figure."""
    d0, d1 = periods['discovery']
    r0, r1 = periods['replication']
    passed = sorted((r for r in rows if r['replicates']), key=lambda r: -(win(r['replication'], r['tested_side'])[0] or 0))
    k, bet = len(passed), sum(1 for r in passed if r['bettable'])
    era = 'decade' if r1 - r0 >= 9 else 'period'  # MLB's blocks are eight seasons each
    if not k:
        takeaway = f'None of {len(rows)} situations held up in both {era}s.'
    else:
        items = '; '.join(f"{r['name']} → {r['tested_side']} won {pct(win(r['replication'], r['tested_side'])[0])} "
                          f"({record(r['replication'], r['tested_side'])})" for r in passed)
        verdict = (('and it did.' if bet else 'it fell short.') if k == 1 else
                   f'and all {WORDS[k]} did.' if bet == k else 'none did.' if not bet else f'and {WORDS[bet]} did.')
        takeaway = (f'Of {len(rows)} situations, {WORDS[k]} held up in both {era}s: {items} in {r0}–{r1}. '
                    f"Beating the vig at −110 takes 52.4%{', ' if verdict.startswith('and') else '; '}{verdict}")
    how = (f'Each row is a situation and the side it predicts. The filled dot is the share of bets won on that side in '
           f'{r0}–{r1}, with its 95% interval; the ring is {d0}–{d1}. A situation held up only if it pointed the same way '
           f'in both {era}s and was statistically clear in the later one (one-sided p < 0.05). Officials were rated on '
           f'the earlier {era}, so their rows have no ring.')
    return share_attrs(label=SHARE_LABEL, title=f'What makes {sport} games go under?', subtitle=takeaway, text=how,
                       source=f'Data: {games:,} {sport} games, {span}, with closing totals')


def share_priced(results, n_games):
    wind = results['NFL-1']['full']
    return share_attrs(
        label=SHARE_LABEL, title='What the market already knows',
        subtitle=(f"Most situations sit near the diagonal: big effects came with big line moves. The clearest exception is "
                  f"NFL wind, where scoring fell {abs(wind['effect']):.1f} points and the closing total moved only "
                  f"{abs(wind['priced']):.1f}."),
        text=('Each dot is a situation with at least 100 games, compared with all other games across all seasons. Across: '
              'how much it changed actual scoring. Up: how much it moved the closing total. On the diagonal the line moved '
              'as much as scoring did, so the effect was already priced; the distance off it is what was left over.'),
        source=f'Data: {n_games} NFL, NHL, NBA and MLB games with closing totals')


TABLE_NO = {'NFL': 2, 'NHL': 3, 'NBA': 4, 'MLB': 5}
SHORT = {'Bettable at -110': 'Beat −110', 'Replicates, but not enough to beat the vig': 'Replicated', 'Does not replicate': 'No',
         'Too few games': 'Too few games'}


def table_sport(rows, sport, periods):
    d0, d1 = periods['discovery']
    r0, r1 = periods['replication']
    unit = SPORT_UNIT[sport]
    head = ['Situation', 'Side', f'Games {r0}–{str(r1)[2:]}', f'Effect ({unit})', 'Line moved', 'Left over', f'Won {d0}–{str(d1)[2:]}',
            f'Won {r0}–{str(r1)[2:]}', 'q', 'Passed?']
    body = []
    for r in rows:
        rep, disc = r.get('replication', {}), r.get('discovery', {})
        side = r.get('tested_side', 'under')
        dw = '(rating period)' if r['group'] == 'Officials' else pct(win(disc, side)[0])
        body.append([f"{r['id']} {r['name']}", side, f"{rep.get('n', 0):,}", signed(rep.get('effect')), signed(rep.get('priced')),
                     signed(rep.get('excess')), dw, pct(win(rep, side)[0]),
                     '—' if r.get('q_full') is None else '<0.01' if r['q_full'] < .01 else f"{r['q_full']:.2f}", SHORT.get(r['verdict'], r['verdict'])])
    return table(f'Table {TABLE_NO[sport]}. {sport}: every pre-registered situation. Effect, line movement and the amount left over compare these games with all other games in {r0}–{r1}; "Won" is the share of bets won on the predicted side, pushes excluded; q is the false-discovery-rate value across all {{N_TESTS}} tests. "Beat −110": replicated and won at least 52.4%; "Replicated": met the rule but not the break-even.', head, body, cls='results')


def table(caption, headers, rows, cls=''):
    return (f'<div class="table-wrap"><table class="{cls}"><caption>{esc(caption)}</caption><thead><tr>'
            + ''.join(f'<th scope="col">{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<th scope="row">{esc(c)}</th>' if i == 0 else f'<td>{esc(c)}</td>' for i, c in enumerate(row)) + '</tr>' for row in rows)
            + '</tbody></table></div>')
