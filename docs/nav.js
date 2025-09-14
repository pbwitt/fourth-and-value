// docs/nav.js (v=23) — responsive nav; "Consensus" hidden on mobile
(function () {
  // --- Find script & compute base (works locally + GitHub Pages) ---
  const scriptEl =
    document.currentScript ||
    Array.from(document.querySelectorAll('script[src]')).find(s => s.src.includes('nav.js'));
  if (!scriptEl) return;
  const scriptURL = new URL(scriptEl.getAttribute('src'), location.href);
  const base = scriptURL.pathname.replace(/\/nav\.js.*/i, ''); // e.g. "/fourth-and-value"

  const LOGO = `${base}/assets/logo-fv.svg`; // change path if needed

  // --- Styles ---
  const css = `
  :root{
    --nav-h:64px; --nav-bg:#0b1220; --nav-fg:#ffffff; --nav-fg-dim:#cbd5e1; --nav-border:#1f2937;
  }
  @media (max-width:768px){ :root{ --nav-h:56px; } }

  .fv-nav{position:sticky;top:0;z-index:9999;width:100%;background:var(--nav-bg);color:var(--nav-fg);border-bottom:1px solid var(--nav-border);}
  .fv-nav-inner{max-width:1100px;margin:0 auto;padding:0 12px;height:var(--nav-h);display:grid;grid-template-columns:1fr auto;align-items:center;gap:12px;}

  .fv-left{display:flex;align-items:center;gap:12px;min-width:0;}
  .fv-logo{display:flex;align-items:center;gap:10px;min-width:0;text-decoration:none;}
  .fv-logo img{height:60px!important;max-height:60px;width:auto;display:block;object-fit:contain;margin-top:2px;}
  .fv-logo .fv-brand{font-weight:700;letter-spacing:.2px;white-space:nowrap;color:var(--nav-fg);}
  @media (max-width:640px){
    .fv-logo img{height:40px!important;max-height:40px;margin-top:0;}
    .fv-logo .fv-brand{font-size:15px;}
  }

  .fv-links{display:flex;align-items:center;gap:14px;}
  .fv-links a{color:var(--nav-fg-dim);text-decoration:none;padding:8px 10px;border-radius:10px;line-height:1;font-size:15px;}
  .fv-links a:hover,.fv-links a[aria-current="page"]{color:var(--nav-fg);background:rgba(255,255,255,0.06);}

  /* Burger */
  .fv-burger{display:none;align-items:center;justify-content:center;width:40px;height:40px;border-radius:10px;border:1px solid var(--nav-border);background:transparent;color:var(--nav-fg);}
  .fv-burger span{display:block;width:22px;height:2px;background:currentColor;margin:3px 0;transition:transform .2s,opacity .2s;}

  /* Mobile menu */
  @media (max-width:768px){
    .fv-burger{display:flex;}
    .fv-links{
      position:absolute;left:0;right:0;top:var(--nav-h);
      display:none;flex-direction:column;align-items:stretch;gap:4px;
      background:#0f172a;border-bottom:1px solid var(--nav-border);padding:10px 12px 14px;
    }
    .fv-nav.menu-open .fv-links{display:flex;}
    .fv-links a{padding:12px 10px;font-size:16px;}
  }

  /* Burger animation */
  .fv-nav.menu-open .fv-burger span:nth-child(1){transform:translateY(5px) rotate(45deg);}
  .fv-nav.menu-open .fv-burger span:nth-child(2){opacity:0;}
  .fv-nav.menu-open .fv-burger span:nth-child(3){transform:translateY(-5px) rotate(-45deg);}

  /* Utility: hide on mobile */
  @media (max-width:768px){
    .hide-mobile{display:none!important;}
  }
  `;
  const style = document.createElement('style');
  style.setAttribute('data-fv','nav');
  style.textContent = css;
  document.head.appendChild(style);

  // --- Mount point ---
  const slot = document.getElementById('nav-root');
  if (!slot) return;

  // --- Build structure ---
  const nav = document.createElement('nav');
  nav.className = 'fv-nav';
  nav.setAttribute('role','navigation');
  nav.setAttribute('aria-label','Main');

  const inner = document.createElement('div');
  inner.className = 'fv-nav-inner';

  // Left: logo + brand
  const left = document.createElement('div');
  left.className = 'fv-left';

  const logoLink = document.createElement('a');
  logoLink.className = 'fv-logo';
  logoLink.href = `${base}/index.html`;
  logoLink.setAttribute('aria-label','Fourth & Value — Home');

  const img = document.createElement('img');
  img.src = LOGO;
  img.alt = 'Fourth & Value';
  img.onerror = () => img.remove();

  const brand = document.createElement('span');
  brand.className = 'fv-brand';
  brand.textContent = 'Fourth & Value';

  logoLink.appendChild(img);
  logoLink.appendChild(brand);
  left.appendChild(logoLink);

  // Burger
  const burger = document.createElement('button');
  burger.className = 'fv-burger';
  burger.setAttribute('aria-label','Open menu');
  burger.setAttribute('aria-expanded','false');
  burger.setAttribute('aria-controls','fv-menu');
  burger.innerHTML = '<span></span><span></span><span></span>';

  // Links
  const links = document.createElement('div');
  links.className = 'fv-links';
  links.id = 'fv-menu';

  const pages = [
    { href: `${base}/index.html`,        label: 'Home' },
    { href: `${base}/props/index.html`,  label: 'Player Props' },
    { href: `${base}/props/top.html`,    label: 'Top Picks' },
    // Hide Consensus on mobile only:
    { href: `${base}/props/consensus.html`, label: 'Consensus', hideOnMobile: true },
    { href: `${base}/methods.html`,      label: 'Methods' },
  ];

  const here = location.pathname.replace(/\/index\.html$/, '/');

  pages.forEach(p => {
    const a = document.createElement('a');
    a.href = p.href;
    a.textContent = p.label;
    if (p.hideOnMobile) a.classList.add('hide-mobile');

    const normalized = p.href.replace(/\/index\.html$/, '/');
    if (here === new URL(normalized, location.origin).pathname) {
      a.setAttribute('aria-current', 'page');
    }

    a.addEventListener('click', () => {
      nav.classList.remove('menu-open');
      burger.setAttribute('aria-expanded','false');
    });
    links.appendChild(a);
  });

  // Compose
  inner.appendChild(left);
  inner.appendChild(burger);
  inner.appendChild(links);
  nav.appendChild(inner);
  slot.replaceWith(nav);

  // Events
  burger.addEventListener('click', () => {
    const open = nav.classList.toggle('menu-open');
    burger.setAttribute('aria-expanded', String(open));
  });

  // Close menu on escape / outside click (mobile)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && nav.classList.contains('menu-open')) {
      nav.classList.remove('menu-open');
      burger.setAttribute('aria-expanded','false');
    }
  });
  document.addEventListener('click', (e) => {
    if (!nav.contains(e.target) && nav.classList.contains('menu-open')) {
      nav.classList.remove('menu-open');
      burger.setAttribute('aria-expanded','false');
    }
  });
})();
