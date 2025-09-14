<script>
// docs/props/mobile.js v1 — table helper + mobile cards + hints
document.addEventListener('DOMContentLoaded', () => {
  const isMobile = window.matchMedia('(max-width: 768px)').matches;

  // --- Locate / prepare table ---
  let tbl = document.querySelector('.fv-props') || document.querySelector('#tbl') || document.querySelector('table');
  if (!tbl) return;
  tbl.classList.add('fv-props');

  // Ensure a .table-scroll wrapper exists
  if (!tbl.parentElement.classList.contains('table-scroll')) {
    const wrap = document.createElement('div');
    wrap.className = 'table-scroll';
    tbl.parentElement.insertBefore(wrap, tbl);
    wrap.appendChild(tbl);
  }
  const scrollWrap = tbl.parentElement;

  // --- Swipe hint hide-on-scroll (mobile only) ---
  if (isMobile) {
    const hideHint = () => { if (scrollWrap.scrollLeft > 8) scrollWrap.classList.add('scrolled'); };
    scrollWrap.addEventListener('scroll', hideHint, { passive: true });
    hideHint();
  }

  // --- Find/insert cards container + toggle button ---
  let cardsRoot = document.getElementById('prop-cards');
  if (!cardsRoot) {
    cardsRoot = document.createElement('div');
    cardsRoot.id = 'prop-cards';
    cardsRoot.className = 'prop-cards';
    cardsRoot.hidden = true;
    scrollWrap.parentElement.insertBefore(cardsRoot, scrollWrap);
  }
  let toggleBtn = document.getElementById('toggle-view');
  if (!toggleBtn) {
    toggleBtn = document.createElement('button');
    toggleBtn.id = 'toggle-view';
    toggleBtn.className = 'mobile-only';
    toggleBtn.hidden = true;
    toggleBtn.textContent = 'Show table view';
    cardsRoot.parentElement.insertBefore(toggleBtn, cardsRoot.nextSibling);
  }

  // --- Build cards from header names (robust to column reorders) ---
  const norm = s => (s||'').toLowerCase().replace(/\s+/g,' ').trim();
  const headers = [...tbl.querySelectorAll('thead th')].map(th => norm(th.textContent));
  const idx = name => headers.indexOf(norm(name));
  const col = {
    game: idx('game'),
    player: idx('player'),
    book: idx('book'),
    bet: idx('bet'),
    line: idx('line'),
    mktOdds: idx('mkt odds'),
    fair: idx('fair'),
    mktPct: idx('mkt %'),
    modelPct: idx('model %'),
    edge: idx('edge (bps)'),
    kick: idx('kick (et)'),
  };

  function buildCards() {
    cardsRoot.innerHTML = '';
    const rows = [...tbl.querySelectorAll('tbody tr')];
    rows.forEach(tr => {
      const cells = [...tr.children].map(td => td.textContent.trim());
      const player = cells[col.player] || '';
      const edge   = cells[col.edge]   || '';
      const bet    = cells[col.bet]    || '';
      const line   = cells[col.line]   || '';
      const game   = cells[col.game]   || '';
      const kick   = cells[col.kick]   || '';
      const book   = cells[col.book]   || '';
      const mkt    = cells[col.mktOdds]|| '';
      const fair   = cells[col.fair]   || '';
      const mpct   = cells[col.modelPct]|| '';
      const bpct   = cells[col.mktPct] || '';

      const edgeNum = parseFloat(edge.replace(/[^-0-9.]/g,''));
      const edgeClass = isFinite(edgeNum) ? (edgeNum >= 0 ? 'pos' : 'neg') : '';

      const card = document.createElement('div');
      card.className = 'prop-card';
      card.innerHTML = `
        <div class="prop-top">
          <div class="prop-player">${player}</div>
          <div class="prop-edge ${edgeClass}">${edge}</div>
        </div>
        <div class="prop-mid">
          <span>${game}</span>${kick ? `<span>• ${kick}</span>` : ''}
        </div>
        <div class="prop-bottom">
          <span><strong>${bet}</strong>${line ? ` @ ${line}` : ''}</span>
          ${book ? `<span>• ${book}</span>` : ''}
          ${mkt  ? `<span>• ${mkt}</span>`  : ''}
          ${fair ? `<span>• Fair ${fair}</span>` : ''}
          ${bpct ? `<span>• Mkt ${bpct}</span>` : ''}
          ${mpct ? `<span>• Model ${mpct}</span>` : ''}
        </div>
      `;
      cardsRoot.appendChild(card);
    });
  }

  // --- Mobile behavior: show cards by default, hide table (toggleable) ---
  if (isMobile) {
    buildCards();
    cardsRoot.hidden = false;
    tbl.classList.add('mobile-hidden');
    toggleBtn.hidden = false;
    toggleBtn.textContent = 'Show table view';
    toggleBtn.addEventListener('click', () => {
      const showingTable = tbl.classList.toggle('mobile-hidden') === false;
      toggleBtn.textContent = showingTable ? 'Show cards' : 'Show table view';
      if (!showingTable) { buildCards(); window.scrollTo({ top: cardsRoot.offsetTop - 80, behavior: 'smooth' }); }
    });
  }
});
</script>
