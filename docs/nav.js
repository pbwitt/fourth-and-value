// docs/nav.js
(function () {
  const slot = document.getElementById('nav-root');
  if (!slot) return;

  // figure relative base: pages under /props/ are one level deeper
  const inProps = location.pathname.includes('/props/');
  const base = inProps ? '..' : '.';

  const items = [
    { href: `${base}/index.html`,            label: 'Home' },
    { href: `${base}/props/insights.html`,   label: 'Insights' },
    { href: `${base}/props/index.html`,      label: 'Props' },
    { href: `${base}/props/top.html`,        label: 'Top Picks' },
    { href: `${base}/props/consensus.html`,  label: 'Consensus' },
    { href: `${base}/methods.html`,  label: 'Methods' },
  ];

  const here = location.pathname.replace(/\/+$/, '');

  const nav = document.createElement('nav');
  nav.className = 'fv-nav';
  nav.setAttribute('data-fv-nav', '');

  // Scoped styles so page CSS can't override colors
  nav.innerHTML = `
    <style>
      #nav-root .fv-nav { display:flex; gap:12px; align-items:center; padding:12px 16px; }
      #nav-root .fv-link {
        display:inline-block; padding:6px 12px; border-radius:999px;
        border:1px solid rgba(255,255,255,.12); color:#cfd5e7; text-decoration:none; font-weight:600;
      }
      #nav-root .fv-link:is(:hover,:focus) { color:#fff; border-color:rgba(255,255,255,.32); text-decoration:none; }
      /* Use a nav-specific active class so generic .active on pages won't bleed in */
      #nav-root .fv-link.is-active {
        color:#e6e6ff;
        border-color:#8b5cf6;            /* purple accent */
        background:rgba(139,92,246,.12); /* subtle pill bg */
      }
    </style>
  `;

  items.forEach(it => {
    const a = document.createElement('a');
    a.className = 'fv-link';
    a.href = it.href;
    a.textContent = it.label;

    const targetPath = new URL(it.href, location.origin).pathname.replace(/\/+$/, '');
    if (here.endsWith(targetPath)) a.classList.add('is-active');

    nav.appendChild(a);
  });

  slot.replaceChildren(nav);
})();
