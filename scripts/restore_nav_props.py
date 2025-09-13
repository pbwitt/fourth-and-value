#!/usr/bin/env python3
import re, pathlib

ROOT = pathlib.Path("docs/props")
TARGETS = [ROOT/"consensus.html", ROOT/"top.html"]

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

AUTO_JS = """
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

NAV_HTML = """
<header class="site-header">
  <nav class="navbar">
    <a class="brand" href="../index.html">Fourth &amp; Value 🏈</a>
    <div class="nav-links">
      <a class="nav-link" href="../index.html">Home</a>
      <a class="nav-link" href="index.html">Props</a>
      <a class="nav-link" href="consensus.html">Consensus</a>
      <a class="nav-link" href="top.html">Top Picks</a>
      <a class="nav-link" href="../methods.html">Methods</a>
    </div>
  </nav>
</header>
""".strip()

def ensure_css(html: str) -> str:
  if "/* === NAV (injected) === */" in html: return html
  # add to end of an existing <style> or before </head>
  if re.search(r"</style>", html, re.I|re.S):
    return re.sub(r"</style>", "\n"+NAV_CSS+"\n</style>", html, 1, flags=re.I|re.S)
  if re.search(r"</head>", html, re.I|re.S):
    return re.sub(r"</head>", "<style>\n"+NAV_CSS+"\n</style>\n</head>", html, 1, flags=re.I|re.S)
  return "<style>\n"+NAV_CSS+"\n</style>\n"+html

def ensure_nav(html: str) -> str:
  # remove any existing site-header first (safety)
  html = re.sub(r'(?is)<header[^>]*class=["\']site-header["\'][\s\S]*?</header>\s*', "", html)
  if re.search(r"<body[^>]*>", html, re.I):
    return re.sub(r"(<body[^>]*>)", r"\1\n"+NAV_HTML+"\n", html, 1, flags=re.I)
  return NAV_HTML + "\n" + html

def ensure_js(html: str) -> str:
  if "auto-highlight current nav tab" in html: return html
  if re.search(r"</body>", html, re.I):
    return re.sub(r"</body>", AUTO_JS+"\n</body>", html, 1, flags=re.I)
  return html + "\n" + AUTO_JS + "\n"

for p in TARGETS:
  if not p.exists():
    print("Skip (missing):", p)
    continue
  html = p.read_text(encoding="utf-8", errors="ignore")
  html = ensure_css(html)
  html = ensure_nav(html)
  html = ensure_js(html)
  # clean any stray \1 or single-quote-only lines
  html = re.sub(r"(?m)^\s*\\1\s*$", "", html)
  html = re.sub(r"(?m)^\s*'\s*$", "", html)
  p.write_text(html, encoding="utf-8")
  print("Restored nav on:", p)
