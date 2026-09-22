// Fallback for newly generated pages; static notices are added during publication.
(function(){
function mount(){
 if(!document.querySelector('link[href*="responsible-use.css"]')){const css=document.createElement('link');css.rel='stylesheet';css.href='/assets/responsible-use.css?v=1';document.head.appendChild(css);}
 const path=location.pathname.replace(/^\//,'');
 const contextual=/^(props|nfl|nba|nhl|mlb|tracking|briefing|research)\//.test(path)||path==='methods.html'||path.startsWith('editorial/articles/')||(path.startsWith('blog/')&&!['blog/','blog/index.html'].includes(path));
 if(contextual&&!document.getElementById('fv-betting-notice')){const h=document.querySelector('h1');if(h)(h.closest('header')||h).insertAdjacentHTML('afterend',NOTICE);}
 if(!document.getElementById('fv-responsible-footer'))document.body.insertAdjacentHTML('beforeend',FOOTER);
}
const NOTICE="<aside id=\"fv-betting-notice\" class=\"fv-use-notice\" aria-label=\"Betting information disclaimer\"><strong>Analysis, not a guarantee.</strong> Picks and projections are uncertain; odds and player information can change. No outcome or profit is guaranteed. Any wager is your decision and responsibility. <a href=\"/terms.html#responsible-play\">Read our limitations and responsible-play guidance</a>.</aside>",FOOTER="<section id=\"fv-responsible-footer\" class=\"fv-use-footer\" aria-label=\"Responsible use\"><p>Fourth &amp; Value provides sports analysis for informational and entertainment purposes. We do not accept wagers. You are responsible for your wagering decisions and for meeting applicable age and legal requirements. Never wager money you cannot afford to lose.</p><p><a href=\"/terms.html\">Terms &amp; privacy</a> \u00b7 <a href=\"/terms.html#responsible-play\">Responsible play</a> \u00b7 U.S. gambling support: call or text <a href=\"tel:+18006973738\">1-800-MY-RESET</a> or <a href=\"https://www.ncpgambling.org/help-treatment/\">find confidential help</a>.</p></section>";
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();
})();
