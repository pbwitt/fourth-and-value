// docs/assets/chart-share.js (v=2) — share any chart on the site as a branded image that carries its explanation.
// nav.js loads this on every page. The image is drawn on this device with the 2D canvas: SVG charts as images,
// HTML charts box by box. It never uses SVG foreignObject, which Safari will not export. Nothing is uploaded.
// Market Results charts ([data-chart]) keep their own share sheet in markets/markets.js.
//
// Every <figure> holding a chart gets a Share button. The image carries, in order:
//   headline     data-share-title, else the figure's own heading, else the nearest section heading
//   takeaway     data-share-subtitle, else the chart's SVG title, alt text or aria-label
//   explanation  data-share-text, else the paragraph that introduces the figure
//   notes        data-share-notes, else the caption (hover/tap instructions removed)
//   source       data-share-source, plus the page title and its publication date (Eastern Time)
// data-share="off" on a figure or any ancestor opts out; data-share="chart" opts in any other element.
(function () {
  'use strict';

  const clean = s => String(s == null ? '' : s).replace(/\s+/g, ' ').trim();
  const same = (a, b) => !!a && !!b && clean(a).toLowerCase() === clean(b).toLowerCase();
  // Split only where punctuation meets a space, so 17.5 stays whole. No lookbehind: older iOS Safari
  // rejects the whole script when a regex uses it.
  const sentences = s => clean(s).replace(/([.!?]["”’)\]]*)\s+/g, '$1\u0001').split('\u0001').filter(Boolean);
  const HINT = /\b(hover|tap|click|press and hold|mouse over|scroll)\b/i;

  // Captions often end with interaction hints and start with a figure number; neither means anything in an image.
  function stripHints(caption) {
    const text = clean(caption).replace(/^(figure|fig\.?|chart|exhibit)\s*\d+[a-z]?\s*[.:—–-]\s*/i, '');
    return sentences(text).filter(s => !HINT.test(s)).map(clean).join(' ');
  }

  function figureId(caption, index) {
    const m = clean(caption).match(/^(?:figure|fig\.?|chart|exhibit)\s*(\d+[a-z]?)\b/i);
    return m ? `figure-${m[1].toLowerCase()}` : `chart-${index + 1}`;
  }

  const LABELS = [[/^\/research\//, 'Research'], [/^\/blog\//, 'Analysis'], [/^\/editorial\//, 'Editorial'],
    [/^\/recaps\//, 'Recap'], [/^\/markets\//, 'Market Results'], [/^\/learn\//, 'Learn'], [/^\/tools\//, 'Tools'],
    [/^\/prediction-markets\//, 'Prediction Markets'], [/^\/(nfl|nhl|mlb|nba)\//, m => m[1].toUpperCase()]];
  function pageLabel(pathname) {
    for (const [re, label] of LABELS) {
      const m = String(pathname || '').match(re);
      if (m) return typeof label === 'function' ? label(m) : label;
    }
    return '';
  }

  function withUtm(url, source) {
    const [base, hash] = String(url).split('#');
    const tags = `utm_source=${encodeURIComponent(source)}&utm_medium=social&utm_campaign=chart_share`;
    return `${base}${base.includes('?') ? '&' : '?'}${tags}${hash ? '#' + hash : ''}`;
  }

  // The address printed on the image: short enough to read and type.
  function displayUrl(url) {
    const u = String(url).split('#')[0].split('?')[0].replace(/^https?:\/\/(www\.)?/, '').replace(/\/index\.html$/, '/');
    if (u.length <= 72) return u.replace(/\/$/, '');
    const [host, first] = u.split('/');
    return first ? `${host}/${first}` : host;
  }

  // max leaves room for the link on networks that count it (Bluesky 300, Threads 500 characters).
  function socialText(meta, max = 230) {
    const title = clean(meta.title), sub = clean(meta.subtitle);
    let text = sub ? `${title}${/[?!.:]$/.test(title) ? ' ' : ': '}${sub}` : title;
    if (text.length > max) text = text.slice(0, Math.max(1, max - 1)).replace(/\s+\S*$/, '') + '…';
    return text;
  }

  // Publication dates read in Eastern Time. A bare calendar date is shown as written.
  function formatDate(value) {
    const s = clean(value);
    if (!s) return '';
    const opts = {timeZone: 'America/New_York', month: 'long', day: 'numeric', year: 'numeric'};
    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return new Date(s + 'T12:00:00Z').toLocaleDateString('en-US', {...opts, timeZone: 'UTC'});
    const t = Date.parse(s.replace(/(\.\d{3})\d+/, '$1'));
    return isNaN(t) ? '' : new Date(t).toLocaleDateString('en-US', opts);
  }

  function formatCaptured(now) {
    const d = new Date(now);
    const date = d.toLocaleDateString('en-US', {timeZone: 'America/New_York', month: 'short', day: 'numeric', year: 'numeric'});
    const time = d.toLocaleTimeString('en-US', {timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit', timeZoneName: 'short'});
    return `As of ${date}, ${time}`;
  }

  function wrap(measure, text, maxW, maxLines) {
    const lines = [];
    let line = '';
    for (const word of clean(text).split(' ').filter(Boolean)) {
      const next = line ? line + ' ' + word : word;
      if (line && measure(next) > maxW) { lines.push(line); line = word; } else line = next;
    }
    if (line) lines.push(line);
    if (!maxLines || lines.length <= maxLines) return lines;
    const keep = lines.slice(0, maxLines);
    let last = keep[maxLines - 1];
    while (last.includes(' ') && measure(last + '…') > maxW) last = last.replace(/\s+\S+$/, '');
    keep[maxLines - 1] = last + '…';
    return keep;
  }

  const api = {clean, stripHints, figureId, pageLabel, withUtm, displayUrl, socialText, formatDate, formatCaptured, wrap};
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (typeof document === 'undefined' || typeof window === 'undefined' || window.__fvChartShare) return;
  window.__fvChartShare = api;

  // ---------- brand ----------
  const C = {bg: '#0a0d12', bar: '#7ce2bd', mint: '#7ce2bd', green: '#22c55e', ink: '#eef2f8', ink2: '#dfe6f0',
    muted: '#b8c5d6', dim: '#8b93a1', faint: '#6b7380', edge: '#243041', panel: '#0f141c', tile: '#0f1418'};
  const SANS = 'system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
  const SERIF = 'Georgia,"Times New Roman",serif';
  const CARD_W = 1200, PAD = 56, MAX_PIXELS = 16e6;
  const TYPE = {
    title: {font: `700 46px ${SERIF}`, size: 46, lh: 56, color: C.ink, max: 3},
    subtitle: {font: `500 25px ${SANS}`, size: 25, lh: 36, color: C.ink2, max: 6},
    text: {font: `400 21px ${SANS}`, size: 21, lh: 31, color: C.muted, max: 9},
    notes: {font: `400 18px ${SANS}`, size: 18, lh: 27, color: C.dim, max: 7},
    source: {font: `500 17px ${SANS}`, size: 17, lh: 25, color: C.dim, max: 3},
  };
  const ICON = '<svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M18 16a3 3 0 0 0-2.4 1.2l-6.7-3.4a3 3 0 0 0 0-1.6l6.7-3.4A3 3 0 1 0 15 7l-6.7 3.4a3 3 0 1 0 0 3.2L15 17a3 3 0 1 0 3-1Z"/></svg>';
  const SOCIAL = [
    ['x', 'X', (t, u) => `https://twitter.com/intent/tweet?text=${encodeURIComponent(t)}&url=${encodeURIComponent(u)}`],
    ['facebook', 'Facebook', (t, u) => `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(u)}`],
    ['linkedin', 'LinkedIn', (t, u) => `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(u)}`],
    ['reddit', 'Reddit', (t, u) => `https://www.reddit.com/submit?url=${encodeURIComponent(u)}&title=${encodeURIComponent(t)}`],
    ['bluesky', 'Bluesky', (t, u) => `https://bsky.app/intent/compose?text=${encodeURIComponent(t + ' ' + u)}`, 300],
    ['threads', 'Threads', (t, u) => `https://www.threads.net/intent/post?text=${encodeURIComponent(t + ' ' + u)}`, 500],
  ];

  const CSS = `
  /* Pages style every svg and div in a figure (charts at width:100%); the button must not inherit that. */
  .fv-share-row{display:flex!important;justify-content:flex-end;margin:0 0 10px!important;padding:0!important;width:auto!important;grid-column:1/-1}
  .fv-share-btn{display:inline-flex!important;align-items:center;gap:6px;width:auto!important;height:auto!important;min-height:32px;max-width:none!important;padding:4px 12px!important;margin:0!important;border:1px solid #314159;border-radius:999px;background:transparent;color:#b8c5d6;font:500 13px/1.2 ${SANS};white-space:nowrap;cursor:pointer}
  .fv-share-btn svg{display:block!important;flex:none!important;width:15px!important;height:15px!important;max-width:none!important;margin:0!important}
  .fv-share-btn:hover,.fv-share-btn:focus-visible{color:#e7eef9;border-color:#7ce2bd;outline:none}
  [data-fv-share=ready]{scroll-margin-top:84px}
  .fv-share-target{outline:2px solid #7ce2bd;outline-offset:4px}
  .fv-cs-sheet{display:none;position:fixed;inset:0;z-index:10002;background:rgba(0,0,0,.72);overflow-y:auto;padding:24px 16px}
  .fv-cs-sheet.open{display:block}
  .fv-cs-inner{max-width:560px;margin:0 auto;background:#121821;border:1px solid #314159;border-radius:14px;padding:16px;color:#e7eef9;font:15px/1.45 ${SANS}}
  .fv-cs-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 12px}
  .fv-cs-head h2{margin:0;font:600 1.05rem/1.3 ${SANS};color:#e7eef9}
  .fv-cs-x{min-width:40px;min-height:40px;border:1px solid #314159;border-radius:10px;background:transparent;color:#e7eef9;font:500 20px/1 ${SANS};cursor:pointer}
  .fv-cs-preview{position:relative;min-height:140px;display:flex;align-items:center;justify-content:center;border:1px solid #314159;border-radius:8px;background:#0b0e13;overflow:hidden}
  .fv-cs-preview img{display:block;width:100%;height:auto}
  .fv-cs-preview img:not([src]){display:none}
  .fv-cs-status{margin:0;padding:24px 16px;color:#b8c5d6;font-size:.9rem;text-align:center}
  .fv-cs-status:empty{display:none}
  .fv-cs-actions{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 8px}
  .fv-cs-actions>*{flex:1 1 120px;min-height:42px;display:inline-flex;align-items:center;justify-content:center;padding:8px 12px;border:1px solid #314159;border-radius:10px;background:transparent;color:#e7eef9;font:inherit;font-size:.92rem;text-decoration:none;cursor:pointer}
  .fv-cs-actions .primary{background:#7ce2bd;border-color:#7ce2bd;color:#06130e;font-weight:700}
  .fv-cs-actions [hidden]{display:none}
  .fv-cs-actions [aria-disabled=true]{opacity:.45;pointer-events:none}
  .fv-cs-social{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:4px 0 10px}
  .fv-cs-social span{width:100%;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:#7ce2bd;margin-bottom:2px}
  .fv-cs-social button{min-height:36px;padding:6px 12px;border:1px solid #314159;border-radius:999px;background:transparent;color:#e7eef9;font:inherit;font-size:.85rem;cursor:pointer}
  .fv-cs-social button:hover,.fv-cs-social button:focus-visible,.fv-cs-actions>*:focus-visible,.fv-cs-x:focus-visible{border-color:#7ce2bd;outline:none}
  .fv-cs-hint{font-size:.8rem;color:#b8c5d6;text-align:center;margin:0}
  @media print{.fv-share-row,.fv-cs-sheet{display:none!important}}`;

  // ---------- which elements are charts ----------
  function isChart(el) {
    if (el.closest('[data-share="off"]') || el.closest('nav, .fv-nav, .fv-cs-sheet')) return false;
    if (el.querySelector('[data-chart], .share-chart, [data-share-chart]')) return false;
    if (el.dataset.share === 'chart') return true;
    if (el.querySelector('.dotplot, canvas')) return true;
    for (const svg of el.querySelectorAll('svg')) {
      if (svg.closest('button, a, .fv-share-row')) continue;
      const vb = svg.viewBox && svg.viewBox.baseVal;
      const w = (vb && vb.width) || parseFloat(svg.getAttribute('width')) || svg.getBoundingClientRect().width;
      if (w >= 120) return true;
    }
    // Images of our own charts; photographs (JPEG/WebP) are not charts and are never rebranded.
    for (const img of el.querySelectorAll('img')) {
      const src = img.getAttribute('src') || '';
      let sameSite = true;
      try { sameSite = new URL(src, location.href).origin === location.origin; } catch (e) { sameSite = false; }
      if (sameSite && /\.(svg|png)(\?|#|$)/i.test(src)) return true;
    }
    return false;
  }

  let counter = 0;
  function setup(el) {
    el.dataset.fvShare = 'ready';
    if (!el.id) {
      const cap = el.querySelector('figcaption');
      let id = figureId(cap ? cap.textContent : '', counter), n = 2;
      while (document.getElementById(id)) id = `${figureId(cap ? cap.textContent : '', counter)}-${n++}`;
      el.id = id;
    }
    counter++;
    // At the top, like Market Results: on a phone a tall chart's bottom edge is a long scroll away.
    const row = document.createElement('div');
    row.className = 'fv-share-row';
    row.innerHTML = `<button type="button" class="fv-share-btn" aria-haspopup="dialog">${ICON}Share chart</button>`;
    el.insertBefore(row, el.firstChild);
  }

  // Not-yet-charts are checked again on later scans: some pages draw their figures after loading.
  function scan() {
    document.querySelectorAll('figure:not([data-fv-share]), [data-share="chart"]:not([data-fv-share])').forEach(el => {
      if (isChart(el)) setup(el);
    });
  }

  // ---------- what the image says ----------
  function nearestHeading(el) {
    const scope = el.closest('main, article') || document.body;
    let best = null;
    for (const h of scope.querySelectorAll('h2, h3')) {
      if (el.contains(h) || h.closest('.fv-cs-sheet, nav')) continue;
      if (h.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING) best = h; else break;
    }
    return best;
  }

  function describeChart(el) {
    const charts = [...el.querySelectorAll('svg, img')].filter(n => !n.closest('button, a, .fv-share-row'));
    if (charts.length !== 1) {
      const group = el.querySelector('[role=group][aria-label], [role=img][aria-label]');
      return group && charts.length === 0 ? clean(group.getAttribute('aria-label')) : '';
    }
    const c = charts[0];
    const title = c.localName === 'svg' ? c.querySelector(':scope > title') : null;
    const text = clean((title && title.textContent) || c.getAttribute('aria-label') || c.getAttribute('alt'));
    return text.length >= 16 ? text : '';
  }

  function leadParagraph(el) {
    let node = el;
    if (!node.previousElementSibling && node.parentElement && node.parentElement.children.length === 1) node = node.parentElement;
    let p = node.previousElementSibling;
    while (p && (p.matches('script, style, template, .fv-use-notice') || !clean(p.textContent))) p = p.previousElementSibling;
    if (!p || p.localName !== 'p') return '';
    const text = clean(p.textContent);
    return text.length >= 40 && text.length <= 700 ? text : '';
  }

  function pageUrl() {
    const canon = document.querySelector('link[rel="canonical"]');
    if (canon && canon.href) return canon.href.split('#')[0];
    return location.origin + location.pathname;
  }

  function published() {
    const meta = document.querySelector('meta[property="article:published_time"], meta[name="date"], meta[itemprop="datePublished"]');
    const time = document.querySelector('main time[datetime], article time[datetime], header time[datetime]');
    return formatDate(meta ? meta.content : time ? time.getAttribute('datetime') : '') || formatCaptured(Date.now());
  }

  // innerText keeps a <br> inside a heading as a space.
  const textOf = n => n ? clean(n.innerText || n.textContent) : '';

  function describe(el) {
    const d = el.dataset;
    const h1 = textOf(document.querySelector('main h1, article h1, h1'));
    const og = document.querySelector('meta[property="og:title"]');
    const pageTitle = h1 || clean(og && og.content).replace(/\s*[|·–-]\s*Fourth\s*&\s*Value.*$/i, '') || clean(document.title);
    const own = el.querySelector('h2, h3, h4, h5, .chart-title');
    const section = nearestHeading(el);
    let titleFrom = null, title = clean(d.shareTitle);
    if (!title && textOf(own)) { title = textOf(own); titleFrom = own; }
    if (!title) title = textOf(section) || pageTitle || 'Fourth & Value';
    const cap = el.querySelector('figcaption');
    const notes = d.shareNotes != null ? clean(d.shareNotes) : stripHints(cap ? cap.textContent : '');
    let subtitle = d.shareSubtitle != null ? clean(d.shareSubtitle) : describeChart(el);
    let text = d.shareText != null ? clean(d.shareText) : leadParagraph(el);
    if (same(subtitle, title) || same(subtitle, notes)) subtitle = '';
    if (same(text, subtitle) || same(text, notes)) text = '';
    const url = `${pageUrl()}#${el.id}`;
    return {title, subtitle, text, notes, source: clean(d.shareSource), from: same(pageTitle, title) ? '' : pageTitle,
      date: published(), label: clean(d.shareLabel) || pageLabel(location.pathname), url, id: el.id, titleFrom};
  }

  // ---------- capture: lay the chart out at desktop width, then record what to paint ----------
  const escAttr = s => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
  const attrs = n => [...n.attributes].filter(a => !/^on/i.test(a.name)).map(a => ` ${a.name}="${escAttr(a.value)}"`).join('');

  function srcdoc(el, width) {
    const css = [...document.querySelectorAll('link[rel~="stylesheet"][href], style')].filter(n => !n.disabled).map(n =>
      n.localName === 'style' ? `<style${n.media ? ` media="${escAttr(n.media)}"` : ''}>${n.textContent}</style>`
        : `<link rel="stylesheet" href="${escAttr(n.href)}"${n.media && n.media !== 'all' ? ` media="${escAttr(n.media)}"` : ''}>`).join('');
    const chain = [];
    for (let p = el.parentElement; p && p !== document.body && p !== document.documentElement; p = p.parentElement) chain.unshift(p);
    const open = chain.map((p, i) => `<${p.localName}${attrs(p)} data-fvx-wrap${i === chain.length - 1 ? ' data-fvx-host' : ''}>`).join('');
    const close = chain.slice().reverse().map(p => `</${p.localName}>`).join('');
    const fix = `<style>
      html,body{width:auto!important;min-width:0!important;height:auto!important;overflow:visible!important}
      body{margin:0!important;padding:24px!important}
      [data-fvx-wrap]{display:block!important;position:static!important;float:none!important;width:auto!important;min-width:0!important;max-width:none!important;height:auto!important;min-height:0!important;margin:0!important;padding:0!important;border:0!important;transform:none!important;overflow:visible!important;columns:auto!important;contain:none!important}
      [data-fvx-host]{width:${width}px!important}
      [data-fvx-host]>*{margin:0!important}
      .fv-share-row,figcaption,[data-share-skip],.tip,.fv-use-notice{display:none!important}
      *,*::before,*::after{transition:none!important;animation:none!important}
    </style>`;
    const body = chain.length ? `${open}${el.outerHTML}${close}` : `<div data-fvx-host>${el.outerHTML}</div>`;
    return `<!doctype html><html${attrs(document.documentElement)}><head><meta charset="utf-8"><base href="${escAttr(document.baseURI)}">${css}${fix}</head><body${attrs(document.body)}>${body}</body></html>`;
  }

  const timeout = (p, ms) => Promise.race([p, new Promise(r => setTimeout(r, ms))]);

  async function capture(el, meta) {
    const live = el.getBoundingClientRect();
    const width = Math.round(window.innerWidth >= 900 ? Math.min(1000, Math.max(480, live.width)) : 760);
    const frame = document.createElement('iframe');
    frame.setAttribute('sandbox', 'allow-same-origin'); // styles and images load; no script runs inside
    frame.setAttribute('aria-hidden', 'true');
    frame.tabIndex = -1;
    frame.style.cssText = 'position:fixed;left:-30000px;top:0;width:1280px;height:2000px;border:0;opacity:0;pointer-events:none';
    const loaded = new Promise((resolve, reject) => { frame.onload = resolve; setTimeout(() => reject(new Error('Chart layout timed out')), 10000); });
    frame.srcdoc = srcdoc(el, width);
    document.body.appendChild(frame);
    try {
      await loaded;
      const doc = frame.contentDocument, view = frame.contentWindow;
      const fig = doc.querySelector('[data-fvx-host] > *');
      if (!fig) throw new Error('Chart did not lay out');
      fig.getBoundingClientRect();
      await timeout(Promise.all([doc.fonts ? doc.fonts.ready : null,
        ...[...doc.images].map(img => img.decode ? img.decode().catch(() => {}) : null)]), 2000);
      const all = [...el.querySelectorAll('*')], copy = [...fig.querySelectorAll('*')];
      const own = meta.titleFrom ? copy[all.indexOf(meta.titleFrom)] : null;
      const canvases = [...el.querySelectorAll('canvas')];
      return record(fig, view, own, canvases, panelColor(el));
    } finally {
      frame.remove();
    }
  }

  // The colour a transparent figure sits on, so its chart keeps the contrast it was designed for.
  function panelColor(el) {
    for (let p = el; p; p = p.parentElement) {
      const bg = getComputedStyle(p).backgroundColor;
      if (opaque(bg)) return bg;
    }
    return C.panel;
  }

  function opaque(c) {
    if (!c || c === 'transparent') return false;
    let m = c.match(/^rgba\(([^)]+)\)$/);
    if (m) return parseFloat(m[1].split(',')[3]) > 0.005;
    m = c.match(/\/\s*([\d.]+)(%?)\s*\)$/);
    if (m) return parseFloat(m[1]) / (m[2] ? 100 : 1) > 0.005;
    return true;
  }

  const SIDES = ['Top', 'Right', 'Bottom', 'Left'];
  const SVG_PROPS = ['fill', 'fill-opacity', 'fill-rule', 'stroke', 'stroke-width', 'stroke-opacity', 'stroke-dasharray',
    'stroke-dashoffset', 'stroke-linecap', 'stroke-linejoin', 'stroke-miterlimit', 'opacity', 'visibility', 'display',
    'font-family', 'font-size', 'font-weight', 'font-style', 'letter-spacing', 'text-anchor', 'dominant-baseline',
    'alignment-baseline', 'baseline-shift', 'paint-order', 'stop-color', 'stop-opacity', 'marker-start', 'marker-mid',
    'marker-end', 'clip-path', 'mask', 'shape-rendering', 'text-decoration', 'color', 'vector-effect'];
  const localUrl = v => v.replace(/url\((["']?)[^)#"']*#([^)"']+)\1\)/g, 'url(#$2)');

  function svgMarkup(svg, view, r) {
    const copy = svg.cloneNode(true);
    const src = [svg, ...svg.querySelectorAll('*')], dst = [copy, ...copy.querySelectorAll('*')];
    src.forEach((node, i) => {
      const cs = view.getComputedStyle(node);
      let style = '';
      for (const p of SVG_PROPS) {
        const v = cs.getPropertyValue(p);
        if (v) style += `${p}:${localUrl(v)};`;
      }
      if (cs.transform && cs.transform !== 'none' && !node.hasAttribute('transform')) style += `transform:${cs.transform};transform-origin:${cs.transformOrigin};transform-box:${cs.transformBox || 'view-box'};`;
      dst[i].setAttribute('style', i === 0 ? style : (dst[i].getAttribute('style') || '') + ';' + style);
    });
    copy.querySelectorAll('foreignObject, script').forEach(n => n.remove());
    copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    copy.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    if (!copy.getAttribute('viewBox')) copy.setAttribute('viewBox', `0 0 ${r.width} ${r.height}`);
    copy.removeAttribute('width');
    copy.removeAttribute('height');
    return new XMLSerializer().serializeToString(copy); // sized when painted
  }

  function parseShadows(value) {
    if (!value || value === 'none') return [];
    const parts = [];
    let depth = 0, cur = '';
    for (const ch of value) {
      if (ch === '(') depth++;
      if (ch === ')') depth--;
      if (ch === ',' && !depth) { parts.push(cur); cur = ''; } else cur += ch;
    }
    parts.push(cur);
    return parts.map(p => {
      if (/\binset\b/.test(p)) return null;
      const color = (p.match(/(rgba?\([^)]*\)|hsla?\([^)]*\)|[a-z]+\([^)]*\)|#[0-9a-f]{3,8})/i) || [])[0];
      const n = p.replace(color || '', '').trim().split(/\s+/).map(parseFloat).filter(x => !isNaN(x));
      const [x = 0, y = 0, blur = 0, spread = 0] = n;
      return color && opaque(color) && !blur && (spread || x || y) ? {color, x, y, spread} : null;
    }).filter(Boolean);
  }

  function radii(cs, r) {
    return ['TopLeft', 'TopRight', 'BottomRight', 'BottomLeft'].map(c => {
      const v = String(cs[`border${c}Radius`] || '0').split(' ')[0];
      const n = parseFloat(v) || 0;
      return Math.min(v.endsWith('%') ? n / 100 * Math.min(r.width, r.height) : n, r.width / 2, r.height / 2);
    });
  }

  const caseOf = (word, t) => t === 'uppercase' ? word.toUpperCase() : t === 'lowercase' ? word.toLowerCase()
    : t === 'capitalize' ? word.charAt(0).toUpperCase() + word.slice(1) : word;
  const fontOf = cs => clean(`${cs.fontStyle === 'normal' ? '' : cs.fontStyle} ${cs.fontVariantCaps === 'small-caps' ? 'small-caps' : ''} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`);

  function record(root, view, own, canvases, panel) {
    const doc = root.ownerDocument, R = root.getBoundingClientRect(), items = [];
    const copies = [...root.querySelectorAll('canvas')];
    const rootCs = view.getComputedStyle(root);
    let order = 0;
    const at = r => ({x: r.left - R.left, y: r.top - R.top, w: r.width, h: r.height});
    const skip = el => el === own || el.matches('.fv-share-row, figcaption, [data-share-skip], script, style, template, noscript, .tip');
    const range = doc.createRange();

    function text(node, cs, alpha, layer) {
      const s = node.nodeValue;
      if (!/\S/.test(s) || !opaque(cs.color)) return;
      const base = {font: fontOf(cs), color: cs.color, ls: cs.letterSpacing, alpha, layer};
      const re = /\S+/g;
      let m;
      while ((m = re.exec(s))) {
        range.setStart(node, m.index);
        range.setEnd(node, m.index + m[0].length);
        const rects = [...range.getClientRects()].filter(r => r.width > 0);
        if (rects.length === 1) items.push({kind: 'text', ...at(rects[0]), text: caseOf(m[0], cs.textTransform), ...base, order: order++});
        else if (rects.length > 1) {
          for (let i = 0; i < m[0].length; i++) {
            range.setStart(node, m.index + i);
            range.setEnd(node, m.index + i + 1);
            const cr = range.getClientRects()[0];
            if (cr && cr.width > 0) items.push({kind: 'text', ...at(cr), text: caseOf(m[0][i], cs.textTransform), ...base, order: order++});
          }
        }
      }
    }

    function box(el, cs, alpha, layer) {
      let fill = opaque(cs.backgroundColor) ? cs.backgroundColor : '';
      if (!fill && /gradient\(/.test(cs.backgroundImage)) {
        const stop = cs.backgroundImage.match(/(rgba?\([^)]*\))/);
        if (stop && opaque(stop[1])) fill = stop[1];
      }
      const bw = SIDES.map(s => /none|hidden/.test(cs[`border${s}Style`]) ? 0 : parseFloat(cs[`border${s}Width`]) || 0);
      const bc = SIDES.map(s => cs[`border${s}Color`]);
      const bs = SIDES.map(s => cs[`border${s}Style`]);
      const shadows = parseShadows(cs.boxShadow);
      if (!fill && !bw.some((w, i) => w && opaque(bc[i])) && !shadows.length) return;
      const rects = cs.display === 'inline' ? [...el.getClientRects()] : [el.getBoundingClientRect()];
      for (const r of rects) {
        if (r.width < 0.5 || r.height < 0.5) continue;
        items.push({kind: 'box', ...at(r), radius: radii(cs, r), fill, bw, bc, bs, shadows, alpha, layer, order: order++});
      }
    }

    function walk(el, alpha, layer) {
      if (el !== root && skip(el)) return;
      const cs = view.getComputedStyle(el);
      if (cs.display === 'none') return;
      const op = parseFloat(cs.opacity);
      const a = alpha * (isNaN(op) ? 1 : op);
      if (a < 0.01) return;
      let lay = layer;
      if (cs.position !== 'static') {
        const z = parseInt(cs.zIndex, 10);
        lay = isNaN(z) ? Math.max(layer, 2) : z < 0 ? -1 : 2 + Math.min(z, 50);
      }
      const shown = cs.visibility === 'visible';
      const tag = el.localName;
      if (tag === 'svg') {
        const r = el.getBoundingClientRect();
        if (shown && r.width > 0) items.push({kind: 'svg', ...at(r), markup: svgMarkup(el, view, r), alpha: a, layer: lay, order: order++});
        return;
      }
      if (tag === 'img') {
        const r = el.getBoundingClientRect();
        if (shown && r.width > 0 && (el.currentSrc || el.src)) items.push({kind: 'img', ...at(r), src: el.currentSrc || el.src, alpha: a, layer: lay, order: order++});
        return;
      }
      if (tag === 'canvas') {
        const r = el.getBoundingClientRect(), source = canvases[copies.indexOf(el)];
        if (shown && r.width > 0 && source) items.push({kind: 'canvas', ...at(r), source, alpha: a, layer: lay, order: order++});
        return;
      }
      if (shown) box(el, cs, a, lay);
      const textLayer = lay === 0 ? 1 : lay;
      for (const n of el.childNodes) {
        if (n.nodeType === 3) { if (shown) text(n, cs, a, textLayer); } else if (n.nodeType === 1) walk(n, a, lay);
      }
    }

    walk(root, 1, 0);
    if (!items.length) throw new Error('Nothing to draw');
    // Labels may overhang the figure; keep up to 40px of overhang so they are not clipped.
    let x0 = 0, y0 = 0, x1 = R.width, y1 = R.height;
    for (const it of items) {
      x0 = Math.min(x0, it.x); y0 = Math.min(y0, it.y);
      x1 = Math.max(x1, it.x + it.w); y1 = Math.max(y1, it.y + it.h);
    }
    x0 = Math.max(x0, -40); y0 = Math.max(y0, -40); x1 = Math.min(x1, R.width + 40); y1 = Math.min(y1, R.height + 40);
    const ownPanel = opaque(rootCs.backgroundColor) || SIDES.some(s => parseFloat(rootCs[`border${s}Width`]) > 0 && !/none|hidden/.test(rootCs[`border${s}Style`]));
    return {items, x0, y0, w: x1 - x0, h: y1 - y0, ownPanel, panel};
  }

  // ---------- paint ----------
  function roundedPath(ctx, x, y, w, h, r) {
    const [tl, tr, br, bl] = r || [0, 0, 0, 0];
    ctx.beginPath();
    ctx.moveTo(x + tl, y);
    ctx.lineTo(x + w - tr, y);
    if (tr) ctx.arcTo(x + w, y, x + w, y + tr, tr);
    ctx.lineTo(x + w, y + h - br);
    if (br) ctx.arcTo(x + w, y + h, x + w - br, y + h, br);
    ctx.lineTo(x + bl, y + h);
    if (bl) ctx.arcTo(x, y + h, x, y + h - bl, bl);
    ctx.lineTo(x, y + tl);
    if (tl) ctx.arcTo(x, y, x + tl, y, tl);
    ctx.closePath();
  }

  function loadImage(src, crossOrigin) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      if (crossOrigin) img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Image failed: ' + String(src).slice(0, 80)));
      img.src = src;
    });
  }
  const svgData = xml => 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(xml);

  async function itemImage(it, scale) {
    const W = Math.max(1, Math.round(it.w * scale)), H = Math.max(1, Math.round(it.h * scale));
    if (it.kind === 'svg') return loadImage(svgData(it.markup.replace(/^<svg\b/, `<svg width="${W}" height="${H}"`)));
    const url = new URL(it.src, location.href), sameSite = url.origin === location.origin;
    if (sameSite && /\.svg$/i.test(url.pathname)) {
      // Re-size the file's own SVG so it rasterizes sharply at the image's scale in every browser.
      const doc = new DOMParser().parseFromString(await (await fetch(url.href)).text(), 'image/svg+xml');
      const svg = doc.documentElement;
      if (svg.localName !== 'svg') throw new Error('Not an SVG');
      if (!svg.getAttribute('viewBox')) {
        const w = parseFloat(svg.getAttribute('width')) || it.w, h = parseFloat(svg.getAttribute('height')) || it.h;
        svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      }
      svg.setAttribute('width', W);
      svg.setAttribute('height', H);
      svg.querySelectorAll('foreignObject, script').forEach(n => n.remove());
      return loadImage(svgData(new XMLSerializer().serializeToString(svg)));
    }
    return loadImage(url.href, !sameSite);
  }

  const metricCache = new Map();
  function metrics(ctx, font) {
    if (!metricCache.has(font)) {
      ctx.font = font;
      const m = ctx.measureText('Hg'), size = parseFloat((font.match(/([\d.]+)px/) || [])[1]) || 14;
      metricCache.set(font, {
        asc: m.fontBoundingBoxAscent != null ? m.fontBoundingBoxAscent : size * 0.8,
        desc: m.fontBoundingBoxDescent != null ? m.fontBoundingBoxDescent : size * 0.2,
      });
    }
    return metricCache.get(font);
  }

  async function loadFonts(fonts) {
    if (!document.fonts || !document.fonts.load) return;
    await timeout(Promise.all([...new Set(fonts)].map(f => document.fonts.load(f).catch(() => null))), 1500);
  }

  function paintBox(ctx, it) {
    for (const s of it.shadows) {
      ctx.fillStyle = s.color;
      roundedPath(ctx, it.x + s.x - s.spread, it.y + s.y - s.spread, it.w + 2 * s.spread, it.h + 2 * s.spread, it.radius.map(r => r ? r + s.spread : 0));
      ctx.fill();
    }
    if (it.fill) {
      ctx.fillStyle = it.fill;
      roundedPath(ctx, it.x, it.y, it.w, it.h, it.radius);
      ctx.fill();
    }
    const [t, r, b, l] = it.bw;
    if (!(t || r || b || l)) return;
    const uniform = it.bw.every(w => w === t) && it.bc.every(c => c === it.bc[0]) && it.bs.every(s => s === it.bs[0]);
    if (uniform && opaque(it.bc[0])) {
      ctx.save();
      ctx.strokeStyle = it.bc[0];
      ctx.lineWidth = t;
      if (it.bs[0] === 'dashed') ctx.setLineDash([3 * t, 2 * t]);
      if (it.bs[0] === 'dotted') { ctx.setLineDash([t, t]); ctx.lineCap = 'round'; }
      roundedPath(ctx, it.x + t / 2, it.y + t / 2, it.w - t, it.h - t, it.radius.map(x => Math.max(0, x - t / 2)));
      ctx.stroke();
      ctx.restore();
      return;
    }
    const strips = [[it.x, it.y, it.w, t], [it.x + it.w - r, it.y, r, it.h], [it.x, it.y + it.h - b, it.w, b], [it.x, it.y, l, it.h]];
    strips.forEach((s, i) => { if (it.bw[i] && opaque(it.bc[i])) { ctx.fillStyle = it.bc[i]; ctx.fillRect(...s); } });
  }

  async function rasterize(cap, scale) {
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.ceil(cap.w * scale));
    canvas.height = Math.max(1, Math.ceil(cap.h * scale));
    const ctx = canvas.getContext('2d');
    ctx.scale(scale, scale);
    ctx.translate(-cap.x0, -cap.y0);
    const images = await Promise.all(cap.items.map(it => it.kind === 'svg' || it.kind === 'img' ? itemImage(it, scale) : null));
    await loadFonts(cap.items.filter(it => it.kind === 'text').map(it => it.font));
    const order = cap.items.map((it, i) => i).sort((a, b) => (cap.items[a].layer - cap.items[b].layer) || (cap.items[a].order - cap.items[b].order));
    const spacing = 'letterSpacing' in ctx;
    for (const i of order) {
      const it = cap.items[i];
      ctx.globalAlpha = it.alpha;
      if (it.kind === 'box') paintBox(ctx, it);
      else if (it.kind === 'text') {
        const m = metrics(ctx, it.font);
        ctx.font = it.font;
        ctx.fillStyle = it.color;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
        if (spacing) ctx.letterSpacing = it.ls && it.ls !== 'normal' ? it.ls : '0px';
        ctx.fillText(it.text, it.x, it.y + it.h / 2 + (m.asc - m.desc) / 2, Math.max(it.w * 1.08, it.w + 2));
      } else if (it.kind === 'canvas') ctx.drawImage(it.source, it.x, it.y, it.w, it.h);
      else ctx.drawImage(images[i], it.x, it.y, it.w, it.h);
    }
    if (spacing) ctx.letterSpacing = '0px';
    return canvas;
  }

  // ---------- the branded card ----------
  async function drawCard(meta, cap) {
    const measure = document.createElement('canvas').getContext('2d');
    const inner = CARD_W - 2 * PAD;
    const block = (text, spec) => {
      if (!clean(text)) return null;
      measure.font = spec.font;
      return {spec, lines: wrap(t => measure.measureText(t).width, text, inner, spec.max)};
    };
    const sourceLine = [meta.source, meta.from ? `From “${meta.from}”` : '', meta.date].filter(Boolean).join(' · ');
    const top = [block(meta.title, TYPE.title), block(meta.subtitle, TYPE.subtitle), block(meta.text, TYPE.text)].filter(Boolean);
    const bottom = [block(meta.notes, TYPE.notes), block(sourceLine, TYPE.source)].filter(Boolean);
    const blockH = b => b.lines.length * b.spec.lh;
    const inset = cap.ownPanel ? 0 : 28;
    const fit = Math.min(1.7, (inner - 2 * inset) / cap.w, 1150 / cap.h);
    const cw = cap.w * fit, ch = cap.h * fit;
    const HEADER = 128, GAP = 14, FOOTER = 104;
    const H = Math.ceil(HEADER + top.reduce((s, b) => s + blockH(b) + GAP, 0) + 18 + ch + 2 * inset + 26
      + bottom.reduce((s, b) => s + blockH(b) + GAP, 0) + FOOTER);
    const out = Math.min(2, Math.sqrt(MAX_PIXELS / (CARD_W * H)));
    const chart = await rasterize(cap, fit * out);

    const canvas = document.createElement('canvas');
    canvas.width = Math.round(CARD_W * out);
    canvas.height = Math.round(H * out);
    const ctx = canvas.getContext('2d');
    ctx.scale(out, out);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, CARD_W, H);
    ctx.fillStyle = C.bar;
    ctx.fillRect(0, 0, CARD_W, 8);

    // Header: the FV tile and wordmark, and the section label.
    ctx.fillStyle = C.green;
    roundedPath(ctx, PAD, 40, 48, 48, [10, 10, 10, 10]);
    ctx.fill();
    ctx.fillStyle = C.tile;
    ctx.font = `800 24px ${SANS}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('FV', PAD + 24, 65);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = `700 30px ${SANS}`;
    ctx.fillStyle = C.ink;
    ctx.fillText('Fourth & ', PAD + 64, 75);
    ctx.fillStyle = C.green;
    ctx.fillText('Value', PAD + 64 + ctx.measureText('Fourth & ').width, 75);
    const wordEnd = PAD + 64 + ctx.measureText('Fourth & Value').width;
    if (meta.label) {
      ctx.font = `700 15px ${SANS}`;
      const spaced = 'letterSpacing' in ctx;
      if (spaced) ctx.letterSpacing = '2.5px';
      let base = meta.label.toUpperCase(), label = base;
      while (base.length > 4 && CARD_W - PAD - ctx.measureText(label).width < wordEnd + 32) { base = base.slice(0, -1).trim(); label = base + '…'; }
      ctx.fillStyle = C.mint;
      ctx.textAlign = 'right';
      ctx.fillText(label, CARD_W - PAD, 71);
      ctx.textAlign = 'left';
      if (spaced) ctx.letterSpacing = '0px';
    }

    let y = HEADER;
    const draw = b => {
      ctx.font = b.spec.font;
      ctx.fillStyle = b.spec.color;
      for (const line of b.lines) {
        ctx.fillText(line, PAD, y + b.spec.lh / 2 + b.spec.size * 0.35);
        y += b.spec.lh;
      }
      y += GAP;
    };
    top.forEach(draw);
    y += 18;
    if (!cap.ownPanel) {
      ctx.fillStyle = cap.panel;
      ctx.strokeStyle = C.edge;
      ctx.lineWidth = 1;
      roundedPath(ctx, PAD + 0.5, y + 0.5, inner - 1, ch + 2 * inset - 1, [14, 14, 14, 14]);
      ctx.fill();
      ctx.stroke();
    }
    ctx.drawImage(chart, (CARD_W - cw) / 2, y + inset, cw, ch);
    y += ch + 2 * inset + 26;
    bottom.forEach(draw);

    // Footer: where to find it, and the responsible-use line every Fourth & Value share carries.
    const fy = H - FOOTER + 26;
    ctx.fillStyle = C.edge;
    ctx.fillRect(PAD, fy, inner, 1);
    ctx.font = `600 20px ${SANS}`;
    ctx.fillStyle = C.muted;
    const url = displayUrl(meta.url);
    ctx.fillText(url, PAD, fy + 46);
    const urlEnd = PAD + ctx.measureText(url).width;
    ctx.font = `400 16px ${SANS}`;
    ctx.fillStyle = C.dim;
    const tagline = 'Analysis, not a guarantee. Bet responsibly.';
    if (CARD_W - PAD - ctx.measureText(tagline).width > urlEnd + 24) {
      ctx.textAlign = 'right';
      ctx.fillText(tagline, CARD_W - PAD, fy + 46);
      ctx.textAlign = 'left';
    } else ctx.fillText(tagline, PAD, fy + 72);
    return canvas;
  }

  // ---------- share sheet ----------
  let sheet, state = {}, ticket = 0;

  function shareSheet() {
    if (sheet) return sheet;
    sheet = document.createElement('div');
    sheet.className = 'fv-cs-sheet';
    sheet.setAttribute('role', 'dialog');
    sheet.setAttribute('aria-modal', 'true');
    sheet.setAttribute('aria-labelledby', 'fv-cs-heading');
    sheet.innerHTML = '<div class="fv-cs-inner"><div class="fv-cs-head"><h2 id="fv-cs-heading">Share this chart</h2>' +
      '<button type="button" class="fv-cs-x" data-act="close" aria-label="Close">×</button></div>' +
      '<div class="fv-cs-preview"><img alt=""><p class="fv-cs-status" role="status"></p></div>' +
      '<div class="fv-cs-actions"><button type="button" class="primary" data-act="share">Share image</button>' +
      '<a data-act="save" download>Save image</a><button type="button" data-act="copy-image">Copy image</button>' +
      '<button type="button" data-act="copy">Copy link</button></div>' +
      '<div class="fv-cs-social"><span>Post with a link</span>' +
      SOCIAL.map(([id, name]) => `<button type="button" data-social="${id}">${name}</button>`).join('') + '</div>' +
      '<p class="fv-cs-hint">The image carries the chart, its explanation and where it came from. Posting with a link opens the network with a link back to this chart. On a phone you can also press and hold the image.</p></div>';
    document.body.appendChild(sheet);
    sheet.addEventListener('click', onSheetClick);
    sheet.addEventListener('keydown', e => {
      if (e.key === 'Escape') { e.preventDefault(); closeShare(); }
      if (e.key !== 'Tab') return;
      const f = [...sheet.querySelectorAll('button, a[href]')].filter(n => !n.hidden && n.getAttribute('aria-disabled') !== 'true' && n.offsetParent !== null);
      if (!f.length) return;
      if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
    });
    return sheet;
  }

  async function onSheetClick(e) {
    if (e.target === sheet) return closeShare();
    const social = e.target.closest('[data-social]');
    if (social) {
      const [id, , intent, limit] = SOCIAL.find(s => s[0] === social.dataset.social);
      const url = withUtm(state.meta.url, id);
      window.open(intent(socialText(state.meta, limit ? Math.min(230, limit - url.length - 1) : 230), url), '_blank', 'noopener,width=620,height=680');
      return;
    }
    const act = e.target.closest('[data-act]');
    if (!act || act.getAttribute('aria-disabled') === 'true') return;
    const kind = act.dataset.act;
    if (kind === 'close') closeShare();
    if (kind === 'copy') {
      try { await navigator.clipboard.writeText(state.meta.url); act.textContent = 'Link copied'; }
      catch (err) { window.prompt('Copy this link:', state.meta.url); }
    }
    if (kind === 'copy-image') {
      try {
        await navigator.clipboard.write([new ClipboardItem({'image/png': state.blob})]);
        act.textContent = 'Image copied';
      } catch (err) { act.textContent = 'Copy not allowed here'; }
    }
    if (kind === 'share') {
      try {
        // Image only, like the Market Results and Bet Tracker shares: the image already names this page,
        // and a link here would add a second preview in most apps.
        const canFile = state.file && navigator.canShare && navigator.canShare({files: [state.file]});
        await navigator.share(canFile ? {files: [state.file]} : {title: state.meta.title, text: socialText(state.meta), url: state.meta.url});
      } catch (err) { if (err && err.name !== 'AbortError') alert('Sharing isn’t available here. Use Save image or Copy link.'); }
    }
  }

  function setReady(el, ready) {
    for (const kind of ['share', 'save', 'copy-image']) {
      const b = el.querySelector(`[data-act="${kind}"]`);
      b.setAttribute('aria-disabled', ready ? 'false' : 'true');
    }
  }

  async function openShare(fig, trigger) {
    const el = shareSheet(), mine = ++ticket;
    const meta = describe(fig);
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state = {meta, trigger};
    const img = el.querySelector('img'), status = el.querySelector('.fv-cs-status'), save = el.querySelector('[data-act=save]');
    img.removeAttribute('src');
    img.alt = `${meta.title}: chart image to share`;
    save.removeAttribute('href');
    status.textContent = 'Drawing the image…';
    el.querySelector('[data-act=copy]').textContent = 'Copy link';
    el.querySelector('[data-act=copy-image]').textContent = 'Copy image';
    el.querySelector('[data-act=share]').hidden = !navigator.share;
    el.querySelector('[data-act=copy-image]').hidden = !(navigator.clipboard && navigator.clipboard.write && window.ClipboardItem);
    setReady(el, false);
    el.classList.add('open');
    document.body.style.overflow = 'hidden';
    el.querySelector('[data-act=close]').focus();
    try {
      const canvas = await drawCard(meta, await capture(fig, meta));
      const blob = await new Promise((resolve, reject) => canvas.toBlob(b => b ? resolve(b) : reject(new Error('Image export failed')), 'image/png'));
      if (mine !== ticket) return;
      const slug = (location.pathname.split('/').filter(Boolean).pop() || 'home').replace(/\.html$/, '');
      const name = `fourth-and-value-${slug}-${meta.id}.png`.replace(/[^a-z0-9.-]+/gi, '-').toLowerCase();
      state.blob = blob;
      state.file = new File([blob], name, {type: 'image/png'});
      state.objectUrl = URL.createObjectURL(blob);
      img.src = state.objectUrl;
      save.href = state.objectUrl;
      save.download = name;
      status.textContent = '';
      setReady(el, true);
      if (navigator.share) el.querySelector('[data-act=share]').hidden = false;
    } catch (err) {
      if (mine !== ticket) return;
      // The link and posting options still work when this browser cannot draw the image.
      status.textContent = 'This browser couldn’t draw the image. Copy link and the posting options still work.';
      el.querySelector('[data-act=share]').hidden = !navigator.share;
      el.querySelector('[data-act=share]').setAttribute('aria-disabled', 'false');
      if (window.console) console.warn('Chart share image failed:', err);
    }
  }

  function closeShare() {
    if (!sheet) return;
    ticket++;
    sheet.classList.remove('open');
    document.body.style.overflow = '';
    if (state.trigger && state.trigger.focus) state.trigger.focus();
  }

  // ---------- links straight to a chart ----------
  function showTarget() {
    const id = decodeURIComponent(location.hash.slice(1));
    const el = id && document.getElementById(id);
    if (!el || el.dataset.fvShare !== 'ready') return;
    el.scrollIntoView({block: 'start'});
    el.classList.add('fv-share-target');
    setTimeout(() => el.classList.remove('fv-share-target'), 2600);
  }

  api.describe = describe; // read by tests/chart_share_browser.cjs

  function init() {
    if (document.body.dataset.share === 'off') return;
    const style = document.createElement('style');
    style.dataset.fvChartShare = 'true';
    style.textContent = CSS;
    document.head.appendChild(style);
    scan();
    document.addEventListener('click', e => {
      const b = e.target.closest && e.target.closest('.fv-share-btn');
      const fig = b && b.closest('[data-fv-share=ready]');
      if (fig) openShare(fig, b);
    });
    let queued = 0;
    new MutationObserver(() => {
      if (queued) return;
      queued = setTimeout(() => { queued = 0; scan(); }, 300);
    }).observe(document.body, {childList: true, subtree: true});
    window.addEventListener('hashchange', showTarget);
    if (document.readyState === 'complete') showTarget(); else window.addEventListener('load', () => { scan(); showTarget(); }, {once: true});
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {once: true}); else init();
})();
