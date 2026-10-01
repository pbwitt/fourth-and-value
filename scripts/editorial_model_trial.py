#!/usr/bin/env python3
"""Summarize the writer-model trial: Astra (published) versus a shadow model.

Reads reports/editorial-trial/<date>-<slot>.json, written by editorial_writer.py while
writer.shadow is active, and writes report.md plus blind-review.html beside them.
Blind pairs are labelled A/B in a stable random order; the key is at the bottom.
"""
import hashlib
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'reports/editorial-trial'
sys.path.insert(0, str(ROOT / 'scripts'))
import editorial_budget as budget  # noqa: E402

PASSED = {'published', 'passed'}


def load(folder=TRIAL):
    return [json.loads(p.read_text()) for p in sorted(folder.glob('20*.json'))]


def charge(side, write_only=False):
    """Recorded cost of one side; the audit is always Astra."""
    total = budget.cost(side['usage'], side['model']) if side.get('usage') else 0.0
    if not write_only and side.get('review_usage'):
        total += budget.cost(side['review_usage'])
    return total


def summarize(records):
    n = len(records)
    count = lambda side, statuses: sum(r[side].get('status') in statuses for r in records)
    both = sum(r['astra'].get('status') in PASSED and r['shadow'].get('status') in PASSED for r in records)
    sol_written = [r['shadow'] for r in records if r['shadow'].get('usage')]
    astra_written = [r['astra'] for r in records if r['astra'].get('usage')]
    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    words = lambda side: avg([r[side]['words'] for r in records if r[side].get('words')])
    reasons = [(r['date'], r['slot'], r['shadow'].get('status'), r['shadow'].get('audit_reason') or r['shadow'].get('reason', ''))
               for r in records if r['shadow'].get('status') not in PASSED]
    return dict(stories=n, astra_passed=count('astra', PASSED), shadow_passed=count('shadow', PASSED), both_passed=both,
                astra_declined=count('astra', {'skipped'}), shadow_declined=count('shadow', {'declined'}),
                shadow_rejected=count('shadow', {'rejected_by_checks'}), shadow_audit_failed=count('shadow', {'audit_failed'}),
                shadow_errors=count('shadow', {'failed', 'audit_error'}),
                astra_write_cost=avg([charge(s, True) for s in astra_written]), shadow_write_cost=avg([charge(s, True) for s in sol_written]),
                astra_words=words('astra'), shadow_words=words('shadow'), shadow_model=records[0]['shadow']['model'] if records else '',
                reasons=reasons)


def report(s):
    pct = lambda a: f"{a}/{s['stories']}" + (f" ({100 * a / s['stories']:.0f}%)" if s['stories'] else '')
    lines = [f"# Writer trial: gpt-6-astra vs {s['shadow_model']}", '',
             f"Stories with both drafts: **{s['stories']}**", '',
             '| | Astra (published) | ' + s['shadow_model'] + ' (shadow) |', '|---|---|---|',
             f"| Passed checks and Astra audit | {pct(s['astra_passed'])} | {pct(s['shadow_passed'])} |",
             f"| Declined (no publishable angle) | – | {s['shadow_declined']} |",
             f"| Failed automated checks | – | {s['shadow_rejected']} |",
             f"| Failed Astra audit | – | {s['shadow_audit_failed']} |",
             f"| Errors | – | {s['shadow_errors']} |",
             f"| Average words | {s['astra_words']:.0f} | {s['shadow_words']:.0f} |",
             f"| Average writing cost | ${s['astra_write_cost']:.3f} | ${s['shadow_write_cost']:.3f} |", '',
             f"Both drafts passed on {s['both_passed']} of {s['stories']} stories.", '']
    if s['reasons']:
        lines += ['## Shadow drafts that did not pass', ''] + [f"- {d} {slot} · {status}: {why}" for d, slot, status, why in s['reasons']] + ['']
    lines += ['Read the pairs in `blind-review.html` before opening its answer key.', '']
    return '\n'.join(lines)


def article_html(article):
    if not article:
        return '<p class="none">No draft (declined or failed before writing).</p>'
    parts = [f"<h3>{escape(article.get('title', ''))}</h3>", f"<p class='deck'>{escape(article.get('excerpt', ''))}</p>"]
    for section in article.get('sections', []):
        parts.append(f"<h4>{escape(section.get('heading', ''))}</h4>")
        parts += [f'<p>{escape(p)}</p>' for p in section.get('text', '').split('\n\n') if p.strip()]
    return ''.join(parts)


def blind(records):
    pairs, key = [], []
    for i, r in enumerate(r for r in records if r['astra'].get('article') and r['shadow'].get('article')):
        flip = int(hashlib.sha256(f"{r['date']}-{r['slot']}".encode()).hexdigest(), 16) % 2
        a, b = (r['shadow'], r['astra']) if flip else (r['astra'], r['shadow'])
        pairs.append(f"<section><h2>Pair {i + 1} · {escape(r['date'])} · {escape(r.get('game') or r['sport'])}</h2>"
                     f"<div class='pair'><article><p class='label'>Draft A</p>{article_html(a['article'])}</article>"
                     f"<article><p class='label'>Draft B</p>{article_html(b['article'])}</article></div></section>")
        key.append(f"<li>Pair {i + 1}: A = {escape(a['model'])} ({escape(a.get('status', ''))}), "
                   f"B = {escape(b['model'])} ({escape(b.get('status', ''))})</li>")
    style = ('body{font:16px/1.6 Georgia,serif;max-width:1200px;margin:auto;padding:24px;background:#fbfaf7;color:#1d2228}'
             '.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:800px){.pair{grid-template-columns:1fr}}'
             'article{background:#fff;border:1px solid #ddd;border-radius:10px;padding:18px}.label{font:700 12px system-ui;letter-spacing:.1em;'
             'text-transform:uppercase;color:#596'
             '}.deck{color:#555}section{margin:36px 0}details{margin:40px 0;font-family:system-ui}')
    return (f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Writer trial blind review</title><style>{style}</style><h1>Writer trial: blind review</h1>"
            f"<p>Each pair is the same story from the same evidence. Pick the stronger draft before opening the key.</p>"
            + ''.join(pairs) + f"<details><summary>Answer key</summary><ul>{''.join(key)}</ul></details></html>\n")


def main():
    records = load()
    if not records:
        print('No trial records yet.')
        return
    s = summarize(records)
    (TRIAL / 'report.md').write_text(report(s))
    (TRIAL / 'blind-review.html').write_text(blind(records))
    print(report(s))


if __name__ == '__main__':
    main()
