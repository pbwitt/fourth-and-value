document.addEventListener('DOMContentLoaded', () => {
  // --- 3a) Force full nav on mobile (override any "hide on mobile" classes) ---
  const navFix = document.createElement('style');
  navFix.textContent = `
    #nav-root .desktop-only,
    #nav-root .md-only,
    #nav-root .hide-on-mobile,
    #nav-root .consensus-desktop-only { display:inline-flex !important; }
  `;
  document.head.appendChild(navFix);

  // --- 3b) Ensure a rotate-hint exists (non-destructive) ---
  if (!document.querySelector('.rotate-hint')) {
    const hint = document.createElement('div');
    hint.className = 'rotate-hint';
    hint.setAttribute('aria-live','polite');
    hint.textContent = 'Tip: turn your phone sideways for the full table.';
    const anchor = document.querySelector('.card, #prop-cards, .table-wrap') || document.body;
    anchor.parentNode.insertBefore(hint, anchor);
  }

  // --- 3c) Color Edge (bps) cells green/red (bring back the "value" look) ---
  // Edge (bps) is the 10th data column in your header (0-based idx = 9).
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

  // Run once now; if your page re-renders rows later, call colorEdges() again.
  colorEdges();

  // (Optional) Observe tbody for changes and recolor automatically.
  const tbody = document.querySelector('#tbl tbody');
  if (tbody) {
    const mo = new MutationObserver(colorEdges);
    mo.observe(tbody, {childList:true, subtree:true});
  }
});
