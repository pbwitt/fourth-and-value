(function () {
  const slot = document.getElementById('nav-root');
  if (!slot) return;

  // Compute site base from this script's URL (works local + Pages + custom domain)
  const script = document.currentScript || Array.from(document.scripts).find(s => (s.src||'').includes('nav.js'));
  const base = new URL('.', (script && script.src) || location.href).pathname.replace(/\/$/, '');

  const links = [
    { href: `${base}/index.html`,            label: 'Home' },
    { href: `${base}/props/insights.html`,   label: 'Insights' },
    { href: `${base}/props/index.html`,      label: 'Player Props' },
    { href: `${base}/props/top.html`,        label: 'Top Picks' },
    { href: `${base}/props/consensus.html`,  label: 'Consensus' },
    { href: `${base}/methods.html`,          label: 'Methods' },
  ];

  const cur = location.pathname.replace(/\/+$/, '');
  const nav = document.createElement('nav');
  nav.style.display = 'flex';
  nav.style.gap = '12px';
  nav.style.padding = '12px 0';
  nav.style.flexWrap = 'wrap';

  links.forEach(({ href, label }) => {
    const a = document.createElement('a');
    a.href = href;
    a.textContent = label;
    a.style.color = '#9fb0c3';
    a.style.textDecoration = 'none';
    a.style.padding = '6px 10px';
    a.style.borderRadius = '10px';
    const abs = new URL(href, location.origin).pathname.replace(/\/+$/, '');
    if (abs === cur) a.style.color = '#fff';
    a.onmouseenter = () => (a.style.color = '#fff');
    a.onmouseleave = () => (a.style.color = (abs === cur ? '#fff' : '#9fb0c3'));
    nav.appendChild(a);
  });

  slot.appendChild(nav);
})();
