// docs/markets/markets.js — Market Results: what the market expected versus what happened.
// One page for every sport, rendered from docs/markets/data/<sport>.json (scripts/build_market_results.py).
// Readers pick a sport (?sport=); there is no default. It switches in place; /markets/<sport>/ redirect here.
(() => {
  'use strict';
  const root = document.getElementById('markets');
  if (!root) return;
  const sports = [...new Set([...root.querySelectorAll('[data-sport-link]')].map(a => a.dataset.sportLink))];
  const asked = new URLSearchParams(location.search).get('sport');
  let sport = sports.includes(asked) ? asked : root.dataset.sport || null;
  const dataRoot = root.dataset.root || '../data/';
  const C = {ink: '#e7eef9', muted: '#b8c5d6', edge: '#314159', page: '#0f141c', over: '#3987e5',
    under: '#d95926', gray: '#5b6472', mint: '#7ce2bd', band: 'rgba(184,197,214,.12)'};
  const TERMS = {
    ou: {A: 'Overs', B: 'Unders', a: 'over', b: 'under', rate: 'Over rate'},
    side: {A: 'Favorites', B: 'Underdogs', a: 'favorite cover', b: 'underdog cover', rate: 'Favorite cover rate',
      beat: 'covered', missed: 'missed the spread', vs: 'the spread'},
  };
  // A market can rename its sides (a moneyline is won, not covered); vs: null means there is no line to beat.
  const terms = m => ({...TERMS[m.kind], ...(m.terms || {})});
  const $ = sel => root.querySelector(sel);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
  const pct = v => v == null ? '–' : (100 * v).toFixed(1) + '%';
  const signed = (v, d = 1) => {
    const t = Math.abs(v).toFixed(d);
    return (+t === 0 ? '' : v > 0 ? '+' : '−') + t;
  };
  const fmt = v => Number.isInteger(v) ? String(v) : String(+v.toFixed(1));
  const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
  const state = {market: null, win: 'season'};
  let data, rows = {}, latest = 0, lastWidth = 0, resizeTimer, wired = false, firstLoad = true, reveal = false;

  // Sport cards and the toolbar switch both load in place; modified clicks still open the link.
  root.addEventListener('click', e => {
    const a = e.target.closest('[data-sport-link]');
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey) return;
    e.preventDefault();
    // A card sits above the charts; on a phone the stacked cards push them off screen.
    reveal = a.classList.contains('sport');
    if (a.dataset.sportLink !== sport) load(a.dataset.sportLink); else showDashboard();
  });
  if (sport) load(sport);
  else { $('[data-dashboard]').hidden = true; $('[data-pick-prompt]').hidden = false; }

  function load(next) {
    sport = next;
    $('[data-dashboard]').hidden = false;
    $('[data-pick-prompt]').hidden = true;
    // Show only this sport's notes, tables and links; without JavaScript every sport's stay visible.
    root.querySelectorAll('[data-for]').forEach(el => { el.hidden = el.dataset.for !== sport; });
    root.querySelectorAll('[data-sport-link]').forEach(a => {
      if (a.dataset.sportLink === sport) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    fetch(`${dataRoot}${sport}.json`, {cache: 'no-cache'})
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(payload => { if (payload.sport === sport) init(payload); })
      .catch(() => { $('[data-status]').textContent = 'Market data is unavailable right now. Please try again later.'; });
  }

  function init(payload) {
    data = payload;
    rows = {};
    latest = 0;
    for (const m of data.markets) {
      rows[m.key] = (data.rows[m.key] || []).map(a => ({period: a[0], t: a[1], label: a[2], game: a[3], line: a[4],
        p: a[5], actual: a[6], books: a[7], dA: a[8], dB: a[9]}));
      for (const r of rows[m.key]) latest = Math.max(latest, r.t || 0);
    }
    // Deep links on first load: #m=<market>&w=<window>, or a bare #<market>. A sport switch starts fresh.
    const raw = firstLoad ? location.hash.slice(1) : '', hash = new URLSearchParams(raw), known = k => data.markets.some(m => m.key === k);
    if (!data.windows.some(w => w.key === state.win)) state.win = data.windows[0].key;
    if (data.windows.some(w => w.key === hash.get('w'))) state.win = hash.get('w');
    state.market = known(hash.get('m')) ? hash.get('m') : known(raw) ? raw : defaultMarket();
    firstLoad = false;
    windowButtons();
    if (!wired) {
      controls();
      new ResizeObserver(() => {
        const w = root.clientWidth;
        if (Math.abs(w - lastWidth) > 4) { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderAll, 120); }
      }).observe(root);
      wired = true;
    }
    renderAll();
    showDashboard();
  }

  function showDashboard() {
    if (!reveal) return;
    reveal = false;
    const nav = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--nav-h')) || 64;
    const top = $('[data-dashboard]').getBoundingClientRect().top + window.scrollY - nav - 8;
    if (top > window.scrollY + window.innerHeight * 0.5 || top < window.scrollY) window.scrollTo({top, behavior: 'smooth'});
  }

  // The market furthest from what its prices implied is the most useful place to start reading.
  function defaultMarket() {
    let best = null, bestZ = -1;
    for (const m of data.markets) {
      const s = summarize(rows[m.key]);
      if (s.n >= 20 && Math.abs(s.z) > bestZ) { best = m.key; bestZ = Math.abs(s.z); }
    }
    return best || data.markets[0].key;
  }

  function summarize(list) {
    const s = {rows: list.length, a: 0, b: 0, push: 0, E: 0, V: 0, uA: 0, uB: 0, diffs: []};
    for (const r of list) {
      const d = r.actual - r.line;
      s.diffs.push(d);
      if (d === 0) { s.push++; continue; }
      s.E += r.p; s.V += r.p * (1 - r.p);
      if (d > 0) { s.a++; s.uA += r.dA - 1; s.uB -= 1; } else { s.b++; s.uA -= 1; s.uB += r.dB - 1; }
    }
    s.n = s.a + s.b;
    s.rate = s.n ? s.a / s.n : null;
    s.exp = s.n ? s.E / s.n : null;
    s.gap = s.a - s.E;
    s.sd = Math.sqrt(s.V);
    s.z = s.sd ? s.gap / s.sd : 0;
    s.median = quantile([...s.diffs].sort((x, y) => x - y), 0.5);
    return s;
  }

  function quantile(sorted, q) {
    if (!sorted.length) return 0;
    const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }

  function phi(z) {
    const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.3275911 * x);
    const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return z >= 0 ? (1 + erf) / 2 : (1 - erf) / 2;
  }

  function chance(s) {
    const z = Math.abs(s.z);
    if (Math.abs(s.gap) < 0.05) return 'Exactly what the prices implied.';
    if (z < 1) return 'That gap is well within normal chance.';
    const odds = Math.max(2, Math.round(1 / (2 * (1 - phi(z)))));
    return z < 1.645 ? `Chance alone produces a gap that size about 1 time in ${odds}.`
      : `That is outside the usual chance range: chance alone produces a gap that size only about 1 time in ${odds}.`;
  }

  const windowDef = () => data.windows.find(w => w.key === state.win) || data.windows[0];
  function windowPeriods() {
    const w = windowDef();
    return w.last ? data.periods.slice(-w.last) : data.periods;
  }
  function inWindow(list) {
    const w = windowDef();
    if (w.last) { const keep = new Set(windowPeriods().map(p => p.key)); return list.filter(r => keep.has(r.period)); }
    if (w.days) { const cut = latest - w.days * 86400; return list.filter(r => r.t > cut); }
    return list;
  }
  function span(periods) {
    if (!periods.length) return '';
    if (periods.length === 1) return periods[0].label;
    const a = periods[0], b = periods[periods.length - 1];
    return /^Week /.test(a.label) ? `Weeks ${a.key}–${b.key}` : `${a.label} to ${b.label}`;
  }
  // Describe the window by the periods this market actually has results in (game lines start later).
  function when(list) {
    const w = windowDef();
    let ps = windowPeriods();
    if (w.days) return `over the last ${w.days} days`;
    if (list) ps = ps.filter(p => list.some(r => r.period === p.key));
    if (w.key === 'season') return ps.length ? `this season (${span(ps)})` : 'this season';
    return ps.length === 1 ? `in ${span(ps)}` : `over ${span(ps)}`;
  }

  // ---------- page furniture ----------
  function windowButtons() {
    $('[data-windows]').innerHTML = data.windows.map(w => `<button type="button" data-win="${w.key}">${esc(w.label)}</button>`).join('');
  }

  function controls() {
    const box = $('[data-windows]');
    box.addEventListener('click', e => {
      const b = e.target.closest('[data-win]');
      if (b) { state.win = b.dataset.win; renderAll(); }
    });
    $('[data-chips]').addEventListener('click', e => {
      const b = e.target.closest('[data-pick]');
      if (b) select(b.dataset.pick, false);
    });
    $('[data-chart=board]').addEventListener('click', e => {
      const g = e.target.closest('[data-key]');
      if (g) select(g.dataset.key, true);
    });
    $('[data-chart=board]').addEventListener('keydown', e => {
      const g = e.target.closest('[data-key]');
      if (g && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); select(g.dataset.key, true); }
    });
  }

  function select(key, scroll) {
    state.market = key;
    renderAll();
    if (scroll) {
      // Land below the sticky nav and toolbar, whose height changes when the toolbar wraps on phones.
      const bar = $('.toolbar'), stuck = (parseFloat(getComputedStyle(bar).top) || 0) + bar.offsetHeight;
      const top = document.getElementById('detail').getBoundingClientRect().top + window.scrollY - stuck - 12;
      window.scrollTo({top, behavior: 'smooth'});
    }
  }

  function renderAll() {
    lastWidth = root.clientWidth;
    try { history.replaceState(null, '', `${location.pathname}?sport=${sport}#m=${state.market}&w=${state.win}`); } catch (e) { /* embedded frames may refuse */ }
    root.querySelectorAll('[data-win]').forEach(b => b.setAttribute('aria-pressed', b.dataset.win === state.win));
    status();
    board();
    detail();
  }

  function status() {
    const total = Object.values(rows).reduce((n, list) => n + list.length, 0);
    const el = $('[data-status]'), timing = $('[data-timing]');
    if (timing && data.notes.timing) { timing.textContent = data.notes.timing; timing.hidden = false; }
    if (!total) { el.textContent = data.notes.empty || 'No settled lines yet.'; return; }
    const last = data.periods[data.periods.length - 1];
    const games = data.periods.reduce((n, p) => n + (p.games || 0), 0);
    const updated = new Date(data.generated_at).toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
    el.textContent = `Through ${data.through || last.label} · ${games} games · ${total.toLocaleString()} graded lines · Updated ${updated}`;
    const caps = $('[data-captures]');
    if (caps) caps.innerHTML = data.periods.map(p => `<li>${esc(p.label)}: ${esc(p.captured || 'pregame snapshot')}</li>`).join('');
  }

  function board() {
    const items = [];
    let group = null;
    for (const m of data.markets) {
      if (m.group !== group) { group = m.group; items.push({group}); }
      const list = inWindow(rows[m.key]), s = summarize(list), T = terms(m);
      const noun = m.group === 'Game lines' ? 'game' : 'line';
      items.push({key: m.key, label: m.label, sub: plural(s.rows, noun), s, selected: m.key === state.market,
        tip: s.n ? `${m.label} ${when(list)}: ${T.A.toLowerCase()} ${s.a}–${s.b} (${pct(s.rate)}). Prices implied ${pct(s.exp)}, ` +
          `so ${signed(s.gap)} ${T.a}s vs. expected; chance range ±${(1.645 * s.sd).toFixed(1)}. Click for charts.`
          : `${m.label}: no settled lines ${when()}.`});
    }
    dumbbell($('[data-chart=board]'), items, {axis: data.notes.board_axis || 'Share of lines that went over (spreads: share the favorite covered)',
      axisShort: data.notes.board_axis_short || 'Share over (spreads: favorite covered)', buttons: true});
  }

  function detail() {
    const m = data.markets.find(x => x.key === state.market), T = terms(m);
    const list = inWindow(rows[m.key]), s = summarize(list);
    $('[data-chips]').innerHTML = chips();
    $('[data-title]').textContent = m.label;
    const body = $('[data-body]'), empty = $('[data-empty]');
    if (!s.rows) {
      body.hidden = true; empty.hidden = false;
      empty.textContent = rows[m.key].length ? `No settled ${m.label.toLowerCase()} lines ${when()}.`
        : (data.notes.empty || `No settled ${m.label.toLowerCase()} lines yet.`);
      $('[data-headline]').textContent = '';
      return;
    }
    body.hidden = false; empty.hidden = true;
    $('[data-headline]').textContent = headline(m, s, list);
    tiles(m, s);
    tally($('[data-chart=tally]'), list, m);
    periods(m, list);
    histogram($('[data-chart=hist]'), list, m);
    byLine(m, list);
    misses($('[data-chart=misses]'), list, m);
    root.querySelectorAll('[data-term=a]').forEach(el => { el.textContent = T.a; });
    root.querySelectorAll('[data-term=b]').forEach(el => { el.textContent = T.b; });
    root.querySelectorAll('[data-term=A]').forEach(el => { el.textContent = T.A.toLowerCase(); });
  }

  function chips() {
    let html = '', group = null;
    for (const m of data.markets) {
      if (m.group !== group) { group = m.group; html += `<span class="group">${esc(group)}</span>`; }
      html += `<button type="button" data-pick="${m.key}" aria-pressed="${m.key === state.market}">${esc(m.label)}</button>`;
    }
    return html;
  }

  function headline(m, s, list) {
    const T = terms(m);
    if (!s.n) return `Every settled ${m.label.toLowerCase()} line ${when(list)} landed exactly on the number.`;
    const lead = s.a === s.b ? `${T.A} and ${T.B.toLowerCase()} are even at ${s.a}–${s.b}`
      : s.a > s.b ? `${T.A} lead ${s.a}–${s.b}` : `${T.B} lead ${s.b}–${s.a}`;
    return `${lead} ${when(list)}. The books’ prices implied ${s.E.toFixed(1)} ${T.a}s; there were ${s.a}. ${chance(s)}`;
  }

  function tiles(m, s) {
    const T = terms(m), tone = v => v > 0 ? 'pos' : v < 0 ? 'neg' : '';
    const pushes = s.push ? ` · ${s.push} push${s.push === 1 ? '' : 'es'}` : '';
    const median = m.kind === 'side' ? (T.vs ? `Favorite’s median margin vs. ${T.vs}` : `Favorite’s median final margin, in ${m.unit}`)
      : `Median result vs. the line, in ${m.unit}`;
    $('[data-tiles]').innerHTML = [
      [esc(`${s.a}–${s.b}`), `${T.A}–${T.B.toLowerCase()}${pushes}`, ''],
      [pct(s.rate), `${T.rate}; the prices implied ${pct(s.exp)}`, tone(s.gap)],
      [signed(s.median), median, tone(s.median)],
      [`<i>${T.A}</i>${signed(s.uA)}<br><i>${T.B}</i>${signed(s.uB)}`, 'Units from 1 unit on every bet, at the average price', 'pair'],
    ].map(([b, span, cls]) => `<div class="score"><b class="${cls}">${b}</b><span>${esc(span)}</span></div>`).join('');
  }

  // ---------- charts ----------
  const width = el => Math.max(300, Math.floor(el.clientWidth));
  const txt = (x, y, s, o = {}) => `<text x="${x}" y="${y}" fill="${o.fill || C.ink}" font-size="${o.size || 13}"` +
    `${o.anchor ? ` text-anchor="${o.anchor}"` : ''}${o.weight ? ` font-weight="${o.weight}"` : ''}>${esc(s)}</text>`;
  const svg = (w, h, title, body) => `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}">${body}</svg>`;
  const fit = (s, px, size) => { const max = Math.floor(px / (size * 0.53)); return s.length > max ? s.slice(0, max - 1) + '…' : s; };
  const short = name => /^[A-Z][\w'.-]* .+/.test(name) && !/ [-+]?\d/.test(name) ? name.replace(/^(\S)\S* /, '$1. ') : name;
  const color = s => !s.n || Math.abs(s.gap) < 0.05 ? C.gray : s.gap > 0 ? C.over : C.under;

  // One row per item: hollow ring = rate the prices implied, filled dot = what happened.
  function dumbbell(el, items, opt) {
    const W = width(el), narrow = W < 560, rowH = 42, groupH = 28;
    const L = narrow ? 118 : 170, R = narrow ? 58 : 118, x0 = L + 14, x1 = W - R - 12;
    const vals = items.filter(i => i.s && i.s.n).flatMap(i => [i.s.rate, i.s.exp]);
    let half = Math.max(0.15, ...vals.map(v => Math.abs(v - 0.5) + 0.03));
    half = Math.min(0.5, Math.ceil(half * 20) / 20);
    const lo = 0.5 - half, X = v => x0 + (v - lo) / (2 * half) * (x1 - x0);
    const step = narrow || half >= 0.25 ? 0.1 : 0.05;
    const top = 6, inner = items.reduce((h, i) => h + (i.group ? groupH : rowH), 0), H = top + inner + 44;
    let body = '';
    for (let v = Math.ceil(lo / step - 1e-9) * step; vals.length && v <= 0.5 + half + 1e-9; v += step) {
      const x = X(v), mid = Math.abs(v - 0.5) < 1e-9;
      body += `<line x1="${x}" x2="${x}" y1="${top}" y2="${top + inner}" stroke="${mid ? C.muted : C.edge}"${mid ? ' stroke-opacity=".7"' : ''}/>`;
      body += txt(x, top + inner + 17, `${Math.round(v * 100)}%`, {fill: C.muted, size: 12, anchor: 'middle'});
    }
    if (vals.length) body += txt((x0 + x1) / 2, top + inner + 37, narrow && opt.axisShort ? opt.axisShort : opt.axis, {fill: C.muted, size: 12, anchor: 'middle'});
    let y = top;
    for (const it of items) {
      if (it.group) {
        body += txt(4, y + 19, it.group.toUpperCase(), {fill: C.mint, size: 11, weight: 700});
        y += groupH; continue;
      }
      const cy = y + rowH / 2, s = it.s, col = s ? color(s) : C.gray;
      let g = `<rect class="hit" x="0" y="${y + 1}" width="${W}" height="${rowH - 2}" rx="6" fill="${it.selected ? 'rgba(124,226,189,.08)' : 'transparent'}"/>`;
      if (it.selected) g += `<rect x="0" y="${y + 7}" width="3" height="${rowH - 14}" rx="1.5" fill="${C.mint}"/>`;
      g += txt(L, cy - 1, fit(it.label, L - 8, 13.5), {anchor: 'end', size: narrow ? 13 : 13.5, weight: it.selected ? 700 : 500});
      g += txt(L, cy + 14, it.sub, {anchor: 'end', size: 11, fill: C.muted});
      if (s && s.n) {
        const xe = X(s.exp), xa = X(s.rate), strong = Math.abs(s.z) >= 1.645;
        g += `<line x1="${xe}" x2="${xa}" y1="${cy}" y2="${cy}" stroke="${col}" stroke-width="4" stroke-opacity=".5" stroke-linecap="round"/>`;
        g += `<circle cx="${xe}" cy="${cy}" r="6" fill="${C.page}" stroke="${C.muted}" stroke-width="2"/>`;
        g += `<circle cx="${xa}" cy="${cy}" r="7" fill="${col}" stroke="${C.page}" stroke-width="1.5"/>`;
        g += txt(W - R + 6, cy - 1, `${s.a}–${s.b}`, {size: 13, weight: 600});
        g += txt(W - R + 6, cy + 14, narrow ? signed(s.gap) : `${signed(s.gap)} vs. exp.`,
          {size: 11.5, fill: strong ? col : C.muted, weight: strong ? 700 : 400});
      } else {
        g += txt(x0, cy + 4, 'No settled lines yet', {fill: C.muted, size: 12});
      }
      const attrs = it.key && opt.buttons ? ` data-key="${it.key}" role="button" aria-pressed="${!!it.selected}"` : '';
      body += `<g tabindex="0"${attrs} data-tip="${esc(it.tip)}">${g}</g>`;
      y += rowH;
    }
    el.innerHTML = svg(W, H, opt.axis, body);
  }

  // Running count of results above what the prices implied, inside the band chance usually stays in.
  function tally(el, list, m) {
    const T = terms(m), W = width(el), narrow = W < 560, H = narrow ? 240 : 280;
    const pad = {l: 40, r: narrow ? 46 : 58, t: 30, b: 26};
    const seq = list.filter(r => r.actual !== r.line);
    if (seq.length < 2) { el.innerHTML = '<p class="sub">Not enough settled lines in this window to draw a trend.</p>'; return; }
    let cum = 0, v = 0, a = 0, b = 0;
    const pts = seq.map((r, i) => {
      const hit = r.actual > r.line;
      cum += (hit ? 1 : 0) - r.p; v += r.p * (1 - r.p); hit ? a++ : b++;
      return {i, y: cum, band: 1.645 * Math.sqrt(v), period: r.period, a, b};
    });
    const n = pts.length, peak = Math.max(3, ...pts.map(p => Math.max(Math.abs(p.y), p.band)));
    const stepY = niceStep(peak / 2), ymax = Math.ceil(peak * 1.05 / stepY) * stepY;
    const X = i => pad.l + i / (n - 1) * (W - pad.l - pad.r), Y = y => pad.t + (ymax - y) / (2 * ymax) * (H - pad.t - pad.b);
    let body = '';
    for (let t = -ymax; t <= ymax + 1e-9; t += stepY) {
      body += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="${t === 0 ? C.muted : C.edge}"${t === 0 ? ' stroke-opacity=".8"' : ''}/>`;
      body += txt(pad.l - 8, Y(t) + 4, t === 0 ? '0' : `${t > 0 ? '+' : '−'}${Math.abs(+t.toFixed(1))}`, {fill: C.muted, size: 11.5, anchor: 'end'});
    }
    const up = pts.map(p => `${X(p.i)},${Y(p.band)}`), dn = pts.slice().reverse().map(p => `${X(p.i)},${Y(-p.band)}`);
    body += `<polygon points="${up.join(' ')} ${dn.join(' ')}" fill="${C.band}"/>`;
    // Period boundaries and per-period hover targets.
    const groups = [];
    for (const p of pts) {
      const g = groups[groups.length - 1];
      if (!g || g.period !== p.period) groups.push({period: p.period, from: p.i, to: p.i}); else g.to = p.i;
    }
    for (const [k, g] of groups.entries()) {
      const xa = k ? (X(g.from - 1) + X(g.from)) / 2 : pad.l, xb = k < groups.length - 1 ? (X(g.to) + X(g.to + 1)) / 2 : W - pad.r;
      const per = data.periods.find(p => p.key === g.period), last = pts[g.to];
      if (k) body += `<line x1="${xa}" x2="${xa}" y1="${pad.t - 6}" y2="${H - pad.b}" stroke="${C.edge}" stroke-dasharray="3 4"/>`;
      body += txt((xa + xb) / 2, pad.t - 12, per ? (xb - xa > 60 ? per.label : per.short) : '', {fill: C.muted, size: 11.5, anchor: 'middle'});
      const tip = `Through ${per ? per.label : 'this point'}: ${T.A.toLowerCase()} ${last.a}–${last.b}, ` +
        `${Math.abs(last.y).toFixed(1)} ${last.y >= 0 ? 'more' : 'fewer'} ${T.a}s than the prices implied (chance range ±${last.band.toFixed(1)}).`;
      body += `<g tabindex="0" data-tip="${esc(tip)}"><rect class="hit" x="${xa}" y="${pad.t}" width="${Math.max(1, xb - xa)}" height="${H - pad.t - pad.b}" fill="transparent"/></g>`;
    }
    const line = pts.map(p => `${X(p.i).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ');
    const area = `${X(0)},${Y(0)} ${line} ${X(n - 1)},${Y(0)}`, id = `clip-${m.key}`;
    body += `<clipPath id="${id}-a"><rect x="0" y="0" width="${W}" height="${Y(0)}"/></clipPath>` +
      `<clipPath id="${id}-b"><rect x="0" y="${Y(0)}" width="${W}" height="${H}"/></clipPath>`;
    body += `<polygon points="${area}" fill="${C.over}" fill-opacity=".3" clip-path="url(#${id}-a)" pointer-events="none"/>`;
    body += `<polygon points="${area}" fill="${C.under}" fill-opacity=".3" clip-path="url(#${id}-b)" pointer-events="none"/>`;
    body += `<polyline points="${line}" fill="none" stroke="${C.ink}" stroke-width="2" stroke-linejoin="round" pointer-events="none"/>`;
    const end = pts[n - 1], endCol = end.y > 0 ? C.over : end.y < 0 ? C.under : C.gray;
    body += `<circle cx="${X(n - 1)}" cy="${Y(end.y)}" r="5" fill="${endCol}" stroke="${C.page}" stroke-width="1.5" pointer-events="none"/>`;
    body += txt(X(n - 1) + 9, Y(end.y) + 4, signed(end.y), {size: 12.5, weight: 700, fill: endCol});
    body += txt(pad.l + 6, pad.t + 13, `↑ More ${T.a}s than the prices implied`, {fill: C.muted, size: 11});
    body += txt(pad.l + 6, H - pad.b - 7, `↓ More ${T.b}s`, {fill: C.muted, size: 11});
    el.innerHTML = svg(W, H, `${m.label}: running count versus expectation`, body);
  }

  function niceStep(x) {
    const p = Math.pow(10, Math.floor(Math.log10(x))), f = x / p;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
  }

  function periods(m, list) {
    const T = terms(m), ps = windowPeriods().filter(p => list.some(r => r.period === p.key));
    const fig = $('[data-fig=periods]');
    fig.hidden = ps.length < 2;
    if (fig.hidden) return;
    const items = ps.map(p => {
      const s = summarize(list.filter(r => r.period === p.key));
      return {label: p.label, sub: plural(s.rows, m.group === 'Game lines' ? 'game' : 'line'), s,
        tip: s.n ? `${m.label}, ${p.label}: ${T.A.toLowerCase()} ${s.a}–${s.b} (${pct(s.rate)}); prices implied ${pct(s.exp)}.` : `${p.label}: no settled lines.`};
    });
    dumbbell($('[data-chart=periods]'), items, {axis: m.kind === 'side' ? `Share the favorite ${T.beat}` : 'Share that went over'});
  }

  // Distribution of result minus line; colour shows the side that won.
  function histogram(el, list, m) {
    const T = terms(m), W = width(el), narrow = W < 560, H = narrow ? 220 : 250;
    const pad = {l: 36, r: 12, t: 30, b: 46};
    const diffs = list.map(r => r.actual - r.line), nz = diffs.filter(d => d !== 0).sort((x, y) => x - y);
    const pushes = diffs.length - nz.length;
    if (!nz.length) { el.innerHTML = ''; return; }
    let bin = m.bin || 1;
    const edge = Math.max(Math.abs(quantile(nz, 0.03)), Math.abs(quantile(nz, 0.97)), 3 * bin);
    const maxPerSide = narrow ? 7 : 10;
    while (edge / bin > maxPerSide) bin *= 2;
    const k = Math.ceil(edge / bin), counts = new Array(2 * k).fill(0);
    for (const d of nz) counts[Math.max(0, Math.min(2 * k - 1, Math.floor(d / bin) + k))]++;
    const cmax = Math.max(pushes, ...counts), stepY = niceStep(cmax / 3), ymax = Math.ceil(cmax / stepY) * stepY;
    const x0 = pad.l, x1 = W - pad.r, bw = (x1 - x0) / (2 * k), X = v => x0 + (v / bin + k) * bw;
    const Y = c => H - pad.b - c / ymax * (H - pad.t - pad.b);
    let body = '';
    for (let c = 0; c <= ymax; c += stepY) {
      body += `<line x1="${x0}" x2="${x1}" y1="${Y(c)}" y2="${Y(c)}" stroke="${C.edge}"/>`;
      body += txt(x0 - 6, Y(c) + 4, String(c), {fill: C.muted, size: 11, anchor: 'end'});
    }
    const every = Math.ceil(46 / bw);
    for (let i = 0; i <= 2 * k; i++) {
      const v = (i - k) * bin;
      if ((i - k) % every) continue;
      body += txt(X(v), H - pad.b + 16, v === 0 ? '0' : `${v > 0 ? '+' : '−'}${fmt(Math.abs(v))}`, {fill: C.muted, size: 11.5, anchor: 'middle'});
    }
    const unit = m.unit;
    counts.forEach((c, i) => {
      if (!c) return;
      const lo = (i - k) * bin, hi = lo + bin, over = i >= k;
      const edgeBin = i === 0 || i === 2 * k - 1;
      const range = edgeBin ? `${fmt(Math.abs(over ? lo : hi))} or more` : `${fmt(Math.abs(over ? lo : hi))} to ${fmt(Math.abs(over ? hi : lo))}`;
      const what = m.kind === 'side' ? `Favorite ${over ? T.beat : T.missed} by ${range} ${unit}`
        : `Finished ${range} ${unit} ${over ? 'over' : 'under'} the line`;
      const tip = `${what}: ${c} of ${diffs.length} (${pct(c / diffs.length)})`;
      body += `<g tabindex="0" data-tip="${esc(tip)}"><rect class="hit" x="${X(lo)}" y="${pad.t}" width="${bw}" height="${H - pad.t - pad.b}" fill="transparent"/>` +
        `<rect x="${X(lo) + 1.5}" y="${Y(c)}" width="${Math.max(1, bw - 3)}" height="${H - pad.b - Y(c)}" rx="3" fill="${over ? C.over : C.under}"/></g>`;
    });
    if (pushes) {
      body += `<g tabindex="0" data-tip="${esc(`Landed exactly on the line (push): ${pushes}`)}"><rect x="${X(0) - 4}" y="${Y(pushes)}" width="8" height="${H - pad.b - Y(pushes)}" rx="2" fill="${C.muted}"/></g>`;
    }
    body += `<line x1="${X(0)}" x2="${X(0)}" y1="${pad.t - 4}" y2="${H - pad.b}" stroke="${C.muted}"/>`;
    const med = quantile([...diffs].sort((x, y) => x - y), 0.5), xm = Math.max(x0, Math.min(x1, X(med)));
    body += `<line x1="${xm}" x2="${xm}" y1="${pad.t - 10}" y2="${H - pad.b}" stroke="${C.mint}" stroke-dasharray="4 3" pointer-events="none"/>`;
    body += txt(xm + (med >= 0 ? 5 : -5), pad.t - 14, `median ${signed(med)}`, {fill: C.mint, size: 11.5, anchor: med >= 0 ? 'start' : 'end'});
    const axis = m.kind === 'side' ? (T.vs ? `Favorite’s margin minus ${T.vs} (right = favorite ${T.beat})`
      : `Favorite’s final margin in ${unit} (right = favorite ${T.beat})`)
      : `${unit[0].toUpperCase() + unit.slice(1)} minus the line (right = over, left = under)`;
    body += txt((x0 + x1) / 2, H - 8, axis, {fill: C.muted, size: 12, anchor: 'middle'});
    el.innerHTML = svg(W, H, axis, body);
    const sorted = [...diffs].sort((x, y) => x - y), q1 = quantile(sorted, 0.25), q3 = quantile(sorted, 0.75);
    $('[data-cap=hist]').textContent = m.kind === 'side'
      ? (T.vs ? `Half of these favorites finished between ${signed(q1)} and ${signed(q3)} ${unit} of ${T.vs}.`
        : `Half of these favorites had a final margin between ${signed(q1)} and ${signed(q3)} ${unit}.`)
      : `Half of all results landed between ${signed(q1)} and ${signed(q3)} ${unit} of the line. ` +
        `Lines sit near the middle result, not the average: big games pull the average up without changing who wins the bet.`;
  }

  function byLine(m, list) {
    const T = terms(m), fig = $('[data-fig=lines]');
    let buckets = m.buckets;
    if (!buckets) {
      const counts = {};
      for (const r of list) counts[r.line] = (counts[r.line] || 0) + 1;
      buckets = Object.keys(counts).map(Number).sort((a, b) => counts[b] - counts[a]).slice(0, 6).sort((a, b) => a - b)
        .map(l => [l, l + 1e-9, fmt(l)]);
    }
    const items = buckets.map(([lo, hi, label]) => {
      const s = summarize(list.filter(r => (lo == null || r.line >= lo) && (hi == null || r.line < hi)));
      return {label, sub: plural(s.rows, m.group === 'Game lines' ? 'game' : 'line'), s,
        tip: s.n ? `${m.label}, line ${label}: ${T.A.toLowerCase()} ${s.a}–${s.b} (${pct(s.rate)}); prices implied ${pct(s.exp)}${s.n < 10 ? '. Small sample.' : '.'}`
          : `No settled lines at ${label}.`};
    }).filter(i => i.s.rows);
    fig.hidden = items.length < 2;
    if (!fig.hidden) dumbbell($('[data-chart=lines]'), items, {axis: m.kind === 'side' ? `Share the favorite ${T.beat}` : 'Share that went over'});
  }

  // Largest finishes on each side of the line, like the blog's per-player charts.
  function misses(el, list, m) {
    const T = terms(m), d = list.map(r => ({...r, d: r.actual - r.line})).filter(r => r.d !== 0).sort((a, b) => b.d - a.d);
    const n = Math.min(5, Math.floor(d.length / 2));
    if (n < 1) { el.innerHTML = ''; return; }
    const pick = [...d.slice(0, n), null, ...d.slice(-n)];
    const W = width(el), narrow = W < 560, rowH = 30, gapH = 14;
    const L = narrow ? 104 : 150, R = narrow ? 92 : 168, x0 = L + 12, x1 = W - R - 8;
    const maxAbs = Math.max(Math.abs(d[0].d), Math.abs(d[d.length - 1].d));
    const zero = (x0 + x1) / 2, X = v => zero + v / maxAbs * (x1 - x0) / 2;
    const H = pick.reduce((h, r) => h + (r ? rowH : gapH), 0) + 8;
    let body = `<line x1="${zero}" x2="${zero}" y1="0" y2="${H - 4}" stroke="${C.muted}" stroke-opacity=".7"/>`, y = 4;
    for (const r of pick) {
      if (!r) { y += gapH; continue; }
      const cy = y + rowH / 2, over = r.d > 0, per = data.periods.find(p => p.key === r.period);
      const right = m.kind === 'side'
        ? (r.actual > 0 ? `won by ${fmt(r.actual)}` : r.actual < 0 ? `lost by ${fmt(-r.actual)}` : 'tied')
        : `${fmt(r.actual)} on ${fmt(r.line)}`;
      const tip = m.kind === 'side'
        ? `${r.label} (${r.game}, ${per ? per.label : ''}): favorite ${right}` +
          (T.vs ? `; ${over ? 'covered' : 'did not cover'} by ${fmt(Math.abs(r.d))}.` : '.')
        : `${r.label} (${r.game}, ${per ? per.label : ''}): ${fmt(r.actual)} ${m.unit} on a ${fmt(r.line)} line (${signed(r.d)}).`;
      let g = `<rect class="hit" x="0" y="${y}" width="${W}" height="${rowH}" rx="5" fill="transparent"/>`;
      g += txt(L, cy + 4, fit(narrow ? short(r.label) : r.label, L - 6, 13), {anchor: 'end', size: 13});
      g += `<rect x="${Math.min(zero, X(r.d))}" y="${cy - 9}" width="${Math.max(2, Math.abs(X(r.d) - zero))}" height="18" rx="4" fill="${over ? C.over : C.under}"/>`;
      g += txt(W - R + 6, cy + 4, fit(right, R - 8, 12.5), {size: 12.5, fill: C.muted});
      body += `<g tabindex="0" data-tip="${esc(tip)}">${g}</g>`;
      y += rowH;
    }
    el.innerHTML = svg(W, H, `${m.label}: biggest results either side of the line`, body);
  }

  // ---------- tooltip (shared with the blog chart behaviour) ----------
  const tip = document.querySelector('.tip');
  const show = (el, x, y) => {
    tip.textContent = el.dataset.tip; tip.style.opacity = 1;
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(window.innerWidth - r.width - 8, Math.max(8, x + 12)) + 'px';
    tip.style.top = Math.max(8, y - r.height - 12) + 'px';
  };
  const hide = () => { tip.style.opacity = 0; };
  root.addEventListener('mousemove', e => { const el = e.target.closest('[data-tip]'); el ? show(el, e.clientX, e.clientY) : hide(); });
  root.addEventListener('mouseleave', hide);
  root.addEventListener('focusin', e => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (el) { const b = el.getBoundingClientRect(); show(el, b.left + Math.min(b.width / 2, 160), b.top); }
  });
  root.addEventListener('focusout', hide);
  window.addEventListener('scroll', hide, {passive: true});
})();
