document.addEventListener('DOMContentLoaded', () => {
  // -------- NAV: force full nav on mobile --------
  const navForce = document.createElement('style');
  navForce.textContent = `
    /* Overpower any "hide on mobile" logic coming from nav.js */
    #nav-root .desktop-only,
    #nav-root .md-only,
    #nav-root .hide-on-mobile,
    #nav-root .consensus-desktop-only,
    #nav-root .hidden,
    #nav-root [style*="display:none"] {
      display: inline-flex !important;
      visibility: visible !important;
      opacity: 1 !important;
    }
  `;
  document.head.appendChild(navForce);

  // Also remove inline display:none that scripts may set
  const unhideNav = () => {
    const nav = document.querySelector('#nav-root');
    if (!nav) return;
    nav.querySelectorAll('[style]').forEach(el => {
      const s = (el.getAttribute('style') || '').toLowerCase();
      if (s.includes('display:none')) el.style.display = 'inline-flex';
      el.removeAttribute('aria-hidden');
    });
    nav.querySelectorAll('.hidden, .md\\:hidden, .sm\\:hidden, .lg\\:hidden').forEach(el => el.classList.remove('hidden','md:hidden','sm:hidden','lg:hidden'));
  };
  // run once after nav.js renders (tiny delay)
  setTimeout(unhideNav, 50);

  // -------- Rotate hint: ensure exists (non-destructive) --------
  if (!document.querySelector('.rotate-hint')) {
    const hint = document.createElement('div');
    hint.className = 'rotate-hint';
    hint.setAttribute('aria-live','polite');
    hint.textContent = 'Tip: turn your phone sideways for the full table.';
    const anchor = document.querySelector('.card, #prop-cards, .table-wrap') || document.body.firstElementChild;
    anchor.parentNode.insertBefore(hint, anchor);
  }

  // -------- Edge (bps) green/red --------
  const edgeColumnIndex = (() => {
    const ths = Array.from(document.querySelectorAll('#tbl thead th'));
    return Math.max(0, ths.findIndex(th => th.textContent.trim().toLowerCase().startsWith('edge')));
  })();
  function colorEdges() {
    const rows = document.querySelectorAll('#tbl tbody tr');
    rows.forEach(tr => {
      const tds = tr.querySelectorAll('td');
      const td = tds[edgeColumnIndex];
      if (!td) return;
      const raw = (td.textContent || '').replace(/[,+%]/g,'').trim();
      const val = Number(raw);
      td.classList.remove('edge-pos','edge-neg');
      if (!Number.isNaN(val)) {
        if (val > 0) td.classList.add('edge-pos');
        else if (val < 0) td.classList.add('edge-neg');
      }
    });
  }
  colorEdges();
  const tbody = document.querySelector('#tbl tbody');
  if (tbody) new MutationObserver(colorEdges).observe(tbody, {childList:true, subtree:true});

  // -------- Cards <-> Table TOGGLE --------
  const btn = document.getElementById('toggle-view');
  const cards = document.getElementById('prop-cards');
  const tableWrap = document.querySelector('.card.table-wrap');
  if (!btn || !cards || !tableWrap) return;

  // Basic card renderer using global DATA (already in the page)
  let cardsBuilt = false;
  function buildCards() {
    if (cardsBuilt || !window.DATA) return;
    const frag = document.createDocumentFragment();
    window.DATA.slice(0, 2400).forEach(r => {
      const card = document.createElement('div');
      card.className = 'prop-card';
      card.innerHTML = `
        <div class="pc-hdr">
          <div class="pc-player">${r.player}</div>
          <div class="pc-game">${r.game}</div>
        </div>
        <div class="pc-body">
          <div><b>${prettyMarket(r.market_std)}</b> <span class="muted">• Line</span> ${r.line_disp ?? '—'}</div>
          <div><span class="muted">Book</span> ${r.bookmaker}</div>
          <div><span class="muted">Mkt%</span> ${r.mkt_pct} <span class="muted">• Model%</span> ${r.model_pct}</div>
          <div class="${edgeClass(r.edge_bps)}"><span class="muted">Edge</span> ${fmtEdge(r.edge_bps)}</div>
          <div class="muted">${r.kick_et}</div>
        </div>
      `;
      frag.appendChild(card);
    });
    cards.appendChild(frag);
    cardsBuilt = true;
  }

  function prettyMarket(m) {
    const map = {
      recv_yds: 'Receiving Yards',
      rush_yds: 'Rushing Yards',
      pass_yds: 'Passing Yards',
      receptions: 'Receptions',
      rush_attempts: 'Rush Attempts',
      pass_attempts: 'Pass Attempts',
      pass_completions: 'Pass Completions'
    };
    return map[m] || (m || '').replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());
  }
  function edgeClass(bps) {
    const n = Number(bps);
    if (isNaN(n)) return '';
    return n > 0 ? 'edge-pos fw' : (n < 0 ? 'edge-neg fw' : '');
  }
  function fmtEdge(bps) {
    const n = Number(bps);
    if (isNaN(n)) return '—';
    return `${Math.round(n)} bps`;
  }

  function showCards() {
    buildCards();
    cards.hidden = false;
    tableWrap.hidden = true;
    btn.textContent = 'Show table view';
  }
  function showTable() {
    cards.hidden = true;
    tableWrap.hidden = false;
    btn.textContent = 'Show card view';
  }

  // Default to TABLE; show the button on small screens
  const smallScreen = () => window.matchMedia('(max-width: 700px)').matches;
  const initToggle = () => {
    btn.hidden = !smallScreen();
    if (!btn.hidden && cards.hidden && tableWrap.hidden) showTable();
  };
  initToggle();
  window.addEventListener('resize', initToggle);

  btn.addEventListener('click', () => {
    if (cards.hidden) showCards();
    else showTable();
  });
});
