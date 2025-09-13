#!/usr/bin/env python3
import re, pathlib

ROOT = pathlib.Path("docs")
PAGES = sorted(p for p in ROOT.rglob("*.html") if p.is_file())

NAV_CSS = """
/* === NAV (injected) === */
.site-header{position:sticky;top:0;z-index:50;background:rgba(20,20,24,.65);
  backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  border-bottom:1px solid var(--border,#2a2a2e)}
.navbar{max-width:1100px;margin:0 auto;padding:10px 16px;display:flex;align-items:center;gap:12px}
.brand{font-weight:800;text-decoration:none;color:var(--text,#e9edf6)}
.nav-links{margin-left:auto;display:flex;flex-wrap:wrap;gap:8px}
.nav-link{display:inline-flex;align-items:center;gap:6px;padding:8px 12px;border-radius:10px;text-decoration:none;color:var(--text,#e9edf6)}
.nav-link:hover{background:var(--btn-hover,rgba(76,117,255,.14))}
.nav-link.active{border:1px solid var(--accent,#4c74ff);background:rgba(76,117,255,.10)}
@media (max-width:720px){.navbar{padding:8px 12px}.nav-links{gap:6px}}
.nav-link:focus-visible{outline:2px solid var(--accent,#4c74ff);outline-offset:2px;border-radius:10px}
""".strip()

AUTO_ACTIVE_JS = """
<!-- auto-highlight current nav tab -->
<script defer>
(function(){
  const here = location.pathname.replace(/\\/+$/,'');
  document.querySelectorAll('.nav-link').forEach(a=>{
    const href = a.getAttribute('href'); if(!href) return;
    const url = new URL(href, location.href);   // works under /user/repo/
    if (url.pathname.replace(/\\/+$/,'') === here) a.classList.add('active');
  });
})();
</script>
""".strip()

def depth_for(p: pathlib.Path) -> int:
  rel = p.relative_to(ROOT)
  return len(rel.parents) - 1  # index.html => 0; props/*.html => 1

def nav_html(depth:int)->str:
  pre = "../"*depth
  tabs = [
    (f"{pre}index.html","Home"),
    (f"{pre}props/index.html","Props"),
    (f"{pre}props/consensus.html","Consensus"),
    (f"{pre}props/top.html","Top Picks"),
    (f"{pre}methods.html","Methods"),
  ]
  links = "\n      ".join(f'<a class="nav-link" href="{h}">{t}</a>' for h,t in tabs)
  return f'''<header class="site-header">
  <nav class="navbar">
    <a class="brand" href="{pre}index.html">Fourth &amp; Value 🏈</a>
    <div class="nav-links">
      {links}
    </div>
  </nav>
</header>'''

def inject_css(html:str)->str:
  if "/* === NAV (injected) === */" in html: return html
  if re.search(r"</style>", html, re.I|re.S):
    return re.sub(r"</style>", "\n"+NAV_CSS+"\n</style>", html, 1, flags=re.I|re.S)
  if re.search(r"</head>", html, re.I|re.S):
    return re.sub(r"</head>", "<style>\n"+NAV_CSS+"\n</style>\n</head>", html, 1, flags=re.I|re.S)
  return "<style>\n"+NAV_CSS+"\n</style>\n"+html

def strip_old_navs_and_crumbs(html:str)->str:
  # 1) drop ALL previous site-header blocks (even multiple)
  html = re.sub(r'(?is)<header[^>]*class=["\']site-header["\'][\s\S]*?</header>\s*', "", html)

  # 2) remove "Back to Home" / "Back to site root" crumbs
  html = re.sub(r'(?is)<(?:div|p)[^>]*>[\s\S]{0,400}?Back to (?:Home|site root)[\s\S]*?</(?:div|p)>\s*', "", html)

  # 3) remove inline row like "HomePropsConsensusTop PicksMethods" (no separators)
  html = re.sub(
    r'(?is)<(?:div|p|nav)[^>]*>(?=[\s\S]{0,1000})(?=[\s\S]*Home)(?=[\s\S]*Props)(?=[\s\S]*Consensus)(?=[\s\S]*Top\s*Picks)(?=[\s\S]*Methods)[\s\S]*?</(?:div|p|nav)>\s*',
    "", html)

  # 4) kill any literal "\1" left by a bad backref
  html = re.sub(r'^\s*\\1\s*\n?', "", html, count=1)
  return html

def insert_nav_after_body(html:str, p:pathlib.Path)->str:
  nav = nav_html(depth_for(p))
  if re.search(r"(<body[^>]*>)", html, re.I):
    return re.sub(r"(<body[^>]*>)", r"\1\n"+nav+"\n", html, 1, flags=re.I)
  return nav + html

def insert_js_before_body_end(html:str)->str:
  if "auto-highlight current nav tab" in html: return html
  if re.search(r"</body>", html, re.I):
    return re.sub(r"</body>", AUTO_ACTIVE_JS+"\n</body>", html, 1, flags=re.I)
  return html + "\n" + AUTO_ACTIVE_JS + "\n"

for p in PAGES:
  html = p.read_text(encoding="utf-8", errors="ignore")
  html = inject_css(html)
  html = strip_old_navs_and_crumbs(html)
  html = insert_nav_after_body(html, p)
  html = insert_js_before_body_end(html)
  p.write_text(html, encoding="utf-8")

print(f"Cleaned & unified nav on {len(PAGES)} files under docs/")
