"""Build standalone SVG charts and the frozen preview evidence; no API calls."""
import html
import json
from pathlib import Path
import pandas as pd

OUT = Path('docs/blog/week-2-week-3-2026')
LABELS = dict(pass_attempts='Passing attempts', pass_completions='Passing completions',
              pass_tds='Passing touchdowns', pass_yds='Passing yards',
              receptions='Receptions', recv_yds='Receiving yards',
              rush_attempts='Rushing attempts', rush_yds='Rushing yards', interceptions='Interceptions')

def chart(filename, title, subtitle, rows, unit, lo, hi):
    height = 150 + 43 * len(rows)
    x = lambda v: 230 + (v-lo)/(hi-lo)*620
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img"><title>{html.escape(title)}</title>',
             '<rect width="100%" height="100%" fill="#0b1219"/>',
             '<g font-family="Arial,sans-serif" fill="#e7eef9">',
             f'<text x="30" y="38" font-size="25" font-weight="bold">{html.escape(title)}</text>',
             f'<text x="30" y="68" font-size="16" fill="#bac8d6">{html.escape(subtitle)}</text>',
             f'<path d="M{x(0)} 90V{height-35}" stroke="#78909c"/>']
    for i, (label, val, annotation) in enumerate(rows):
        y=105+43*i
        color='#7ce2bd' if val>=0 else '#f4ae91'
        parts.extend([f'<text x="218" y="{y+19}" text-anchor="end" font-size="17">{html.escape(label)}</text>',
                      f'<rect x="{min(x(0),x(val))}" y="{y}" width="{max(abs(x(val)-x(0)),1)}" height="27" fill="{color}" rx="3"/>',
                      f'<text x="865" y="{y+19}" font-size="16">{html.escape(annotation)}</text>'])
    parts.append(f'<text x="30" y="{height-12}" font-size="14" fill="#bac8d6">Fourth &amp; Value · {html.escape(unit)}</text></g></svg>')
    (OUT/filename).write_text(''.join(parts))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    s=json.loads(Path('reports/week2-2026/summary.json').read_text())
    games=pd.read_csv('reports/week2-2026/games.csv').sort_values('closing_error',ascending=False)
    chart('totals.svg','Week 2: 10 unders, 6 overs','Final points minus the recorded closing total · all 16 games',
          [(r.game, -r.closing_error, f'{-r.closing_error:+.1f}') for r in games.itertuples()], 'Points above / below closing total',-40,25)
    chart('props.svg','Where the prop shortlist won and lost','One highest-EV archived offer per player / game / market',
          [(LABELS[k],v['units'],f"{v['units']:+.2f}u") for k,v in sorted(s['ticket_markets'].items(),key=lambda z:-z[1]['units'])],
          '1 unit risked per graded selection · 13 unresolved excluded',-18,28)
    p=pd.read_csv('reports/week3-2026-preview/week_predictions.csv')
    lines=pd.read_csv('reports/week3-2026-preview/totals_spreads.csv')
    p=p.merge(lines.groupby('game').total_over_line.median().rename('market_total'),on='game',validate='one_to_one')
    p['gap']=p.total_pred-p.market_total
    chart('preview.svg','Week 3: disagreement is a research question','Raw baseline minus median market total · September 22, 5:46 AM ET',
          [(r.game,r.gap,f'{r.gap:+.1f}') for r in p.sort_values('gap').itertuples()],
          'Not calibrated betting edges · injury feed empty in this run',-8,8)
    props=pd.read_csv('reports/week3-2026-preview/props_with_model_week3.csv')
    kraft=props[(props.player=='Tucker Kraft') & (props.market_std=='receptions')]
    payload={'as_of':'2026-09-22T09:46:37Z','manifest':json.loads(Path('reports/week3-2026-preview/manifest.json').read_text()),
             'warning':'Automated injury file contains zero rows. Baselines do not incorporate confirmed new injury news.',
             'games':json.loads(p.to_json(orient='records')),'kraft_offers':json.loads(kraft.to_json(orient='records')),
             'totals_offers':json.loads(lines.to_json(orient='records'))}
    (OUT/'preview-data.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
