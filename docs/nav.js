// docs/nav.js (v=31) — Added NHL Team Totals to navigation
(function () {
  // --- Find script & compute base (works locally and deployed) ---
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
    --nav-h:64px; --nav-bg:#0b0b0b; --nav-fg:#ffffff; --nav-fg-dim:#cbd5e1; --nav-border:#27324a;
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
  .fv-brand .fv-accent { color: #22c55e; } /* green-500; tweak if you had a different hex */

  .fv-links{display:flex;align-items:center;gap:14px;}
  .fv-links a{color:var(--nav-fg-dim);text-decoration:none;padding:8px 10px;border-radius:10px;line-height:1;font-size:15px;}
  .fv-links a:hover,.fv-links a[aria-current="page"]{color:var(--nav-fg);background:rgba(255,255,255,0.06);}

  /* Sport dropdowns */
  .fv-sport-dropdown{position:relative;}
  .fv-sport-toggle{cursor:pointer;user-select:none;display:flex;align-items:center;gap:4px;}
  .fv-sport-toggle::after{content:'▼';font-size:10px;opacity:0.7;}
  .fv-sport-menu{
    display:none;position:absolute;top:100%;left:0;margin-top:8px;
    background:#1a1a1f;border:1px solid var(--nav-border);border-radius:8px;
    min-width:160px;padding:6px;box-shadow:0 4px 12px rgba(0,0,0,0.4);
  }
  .fv-sport-dropdown.open .fv-sport-menu{display:block;}
  .fv-sport-menu a{display:block;padding:10px 12px;color:var(--nav-fg-dim);text-decoration:none;border-radius:6px;font-size:14px;}
  .fv-sport-menu a:hover{background:rgba(255,255,255,0.08);color:var(--nav-fg);}
  .fv-sport-menu a[aria-current="page"]{background:rgba(34,197,94,0.15);color:#22c55e;font-weight:600;}

  /* NHL accent color for NHL dropdown items */
  .fv-sport-dropdown.nhl-sport .fv-sport-menu a[aria-current="page"]{background:rgba(79,195,247,0.15);color:#4FC3F7;}

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
    .fv-links > a{padding:12px 10px;font-size:16px;}

    /* Mobile sport dropdowns */
    .fv-sport-dropdown{width:100%;}
    .fv-sport-toggle{padding:12px 10px;font-size:16px;width:100%;justify-content:space-between;}
    .fv-sport-menu{
      position:static;margin:0;border:none;border-radius:0;
      box-shadow:none;background:transparent;padding:0 0 0 20px;
    }
    .fv-sport-menu a{font-size:15px;padding:10px 12px;}
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
  brand.innerHTML = 'Fourth &amp; <span class="fv-accent">Value</span>';

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

  // Navigation structure with sport dropdowns
  const navStructure = [
    { type: 'link', href: `${base}/index.html`, label: 'Home' },
    {
      type: 'dropdown',
      label: 'NFL 🏈',
      className: 'nfl-sport',
      items: [
        { href: `${base}/props/insights.html`, label: 'Insights' },
        { href: `${base}/props/index.html`, label: 'Player Props' },
        { href: `${base}/props/top.html`, label: 'Top Picks' },
        { href: `${base}/props/arbitrage.html`, label: 'Arbitrage' },
        { href: `${base}/nfl/totals/index.html`, label: 'Team Totals' },
      ]
    },
    {
      type: 'dropdown',
      label: 'NHL 🏒',
      className: 'nhl-sport',
      items: [
        { href: `${base}/nhl/props/index.html`, label: 'Props' },
        { href: `${base}/nhl/totals/index.html`, label: 'Team Totals' },
      ]
    },
    { type: 'link', href: `${base}/methods.html`, label: 'Methods' },
    { type: 'link', href: `${base}/blog/`, label: 'Blog' },
  ];

  const here = location.pathname.replace(/\/index\.html$/, '/');

  // Helper to check if current page matches
  function isCurrentPage(href) {
    const normalized = href.replace(/\/index\.html$/, '/');
    return here === new URL(normalized, location.origin).pathname;
  }

  // Helper to check if dropdown contains current page
  function containsCurrentPage(items) {
    return items.some(item => isCurrentPage(item.href));
  }

  // Build navigation items
  navStructure.forEach(item => {
    if (item.type === 'link') {
      const a = document.createElement('a');
      a.href = item.href;
      a.textContent = item.label;

      if (isCurrentPage(item.href)) {
        a.setAttribute('aria-current', 'page');
      }

      a.addEventListener('click', () => {
        nav.classList.remove('menu-open');
        burger.setAttribute('aria-expanded','false');
      });

      links.appendChild(a);

    } else if (item.type === 'dropdown') {
      const dropdown = document.createElement('div');
      dropdown.className = `fv-sport-dropdown ${item.className || ''}`;

      const toggle = document.createElement('div');
      toggle.className = 'fv-sport-toggle';
      toggle.textContent = item.label;
      toggle.setAttribute('role', 'button');
      toggle.setAttribute('aria-expanded', 'false');
      toggle.setAttribute('tabindex', '0');

      // Highlight if any child page is active
      if (containsCurrentPage(item.items)) {
        toggle.style.color = 'var(--nav-fg)';
        toggle.style.background = 'rgba(255,255,255,0.06)';
      }

      const menu = document.createElement('div');
      menu.className = 'fv-sport-menu';
      menu.setAttribute('role', 'menu');

      item.items.forEach(subItem => {
        const a = document.createElement('a');
        a.href = subItem.href;
        a.textContent = subItem.label;
        a.setAttribute('role', 'menuitem');

        if (isCurrentPage(subItem.href)) {
          a.setAttribute('aria-current', 'page');
        }

        a.addEventListener('click', () => {
          nav.classList.remove('menu-open');
          burger.setAttribute('aria-expanded','false');
          dropdown.classList.remove('open');
        });

        menu.appendChild(a);
      });

      // Toggle dropdown on click
      toggle.addEventListener('click', (e) => {
        e.stopPropagation();
        const wasOpen = dropdown.classList.contains('open');

        // Close all other dropdowns
        document.querySelectorAll('.fv-sport-dropdown.open').forEach(d => {
          d.classList.remove('open');
        });

        if (!wasOpen) {
          dropdown.classList.add('open');
          toggle.setAttribute('aria-expanded', 'true');
        } else {
          toggle.setAttribute('aria-expanded', 'false');
        }
      });

      // Keyboard support
      toggle.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggle.click();
        }
      });

      dropdown.appendChild(toggle);
      dropdown.appendChild(menu);
      links.appendChild(dropdown);
    }
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
    if (e.key === 'Escape') {
      if (nav.classList.contains('menu-open')) {
        nav.classList.remove('menu-open');
        burger.setAttribute('aria-expanded','false');
      }
      // Close any open dropdowns
      document.querySelectorAll('.fv-sport-dropdown.open').forEach(d => {
        d.classList.remove('open');
        d.querySelector('.fv-sport-toggle').setAttribute('aria-expanded', 'false');
      });
    }
  });
  document.addEventListener('click', (e) => {
    // Close mobile menu if clicking outside
    if (!nav.contains(e.target) && nav.classList.contains('menu-open')) {
      nav.classList.remove('menu-open');
      burger.setAttribute('aria-expanded','false');
    }

    // Close dropdowns if clicking outside
    if (!e.target.closest('.fv-sport-dropdown')) {
      document.querySelectorAll('.fv-sport-dropdown.open').forEach(d => {
        d.classList.remove('open');
        d.querySelector('.fv-sport-toggle').setAttribute('aria-expanded', 'false');
      });
    }
  });
})();
