// docs/nav.js  — single-source nav with logo (Fourth & Value)
(function () {
  const slot = document.getElementById('nav-root');
  if (!slot || slot.dataset.fvNavMounted) return;

  // Figure out the base URL from THIS script's path
  const script = document.currentScript || Array.from(document.scripts).find(s => (s.src||'').includes('nav.js'));
  const rootHref = new URL(script?.getAttribute('src') || 'nav.js', window.location.href).href.replace(/nav\.js.*$/,'');
  const href = (p) => rootHref + p.replace(/^\//,'');

  // Inject minimal styles once
  if (!document.getElementById('fv-nav-style')) {
    const style = document.createElement('style');
    style.id = 'fv-nav-style';
    style.textContent = `
      .fv-nav { display:flex; align-items:center; justify-content:space-between; gap:12px; width:100%; box-sizing:border-box; padding:12px 16px; position:sticky; top:0; z-index:1000; background:#0b0f19; border-bottom:1px solid rgba(255,255,255,0.08); min-height:72px; }
      .fv-left { display:flex; align-items:center; gap:10px; min-width:0; }
      .fv-logo { display:flex; align-items:center; gap:10px; text-decoration:none; }
      .fv-logo img { height:60px !important; max-height:60px; width:auto; display:block; object-fit:contain; margin-top:3px; }
      .fv-brand {
        font-weight:700; font-size:18px; color:#fff; white-space:nowrap;
        letter-spacing:0.2px;
      }
      .fv-right { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
      .fv-link { color:#cbd5e1; text-decoration:none; padding:8px 10px; border-radius:10px; line-height:1; }}
      .fv-link:hover { color:#fff; background:rgba(255,255,255,0.06); }
      .fv-active { color:#fff; background:rgba(255,255,255,0.12); }
      @media (max-width:640px){
        .fv-brand { display:none; }
      }
    `;
    document.head.appendChild(style);
  }

  // Detect active path for highlighting
  const path = window.location.pathname;
  const isActive = (pats) => pats.some(p => path.endsWith(p) || path.includes(p));

  const links = [
    { href: 'index.html', label: 'Home', active: isActive(['/index.html','/']) },
      { href: 'props/insights.html', label: 'Insights', active: isActive(['/props/insights.html']) },
    { href: 'props/index.html', label: 'Props', active: isActive(['/props/index.html']) },
    { href: 'props/top.html', label: 'Top Picks', active: isActive(['/props/top.html']) },
    { href: 'props/consensus.html', label: 'Consensus', active: isActive(['/props/consensus.html']) },
    { href: 'methods.html', label: 'Methods', active: isActive(['/methods.html']) },
  ];

  // Build left (logo + brand)
  const logoSrc = href('assets/logo.svg'); // <-- place your logo at docs/assets/logo.svg
  const leftHTML = `
    <a class="fv-logo" href="${href('index.html')}" aria-label="Fourth & Value — Home">
      <img src="${logoSrc}" alt="Fourth & Value logo" onerror="this.style.display='none'; this.nextElementSibling.style.display='inline-block'">
      <span class="fv-brand" style="display:none;">Fourth &amp; Value</span>
    </a>
  `;

  // Build right (nav links)
  const rightHTML = links.map(l =>
    `<a class="fv-link ${l.active ? 'fv-active' : ''}" href="${href(l.href)}">${l.label}</a>`
  ).join('');

  slot.innerHTML = `
    <nav class="fv-nav">
      <div class="fv-left">${leftHTML}</div>
      <div class="fv-right">${rightHTML}</div>
    </nav>
  `;
  slot.dataset.fvNavMounted = '1';
})();
