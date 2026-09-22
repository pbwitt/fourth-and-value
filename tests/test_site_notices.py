from pathlib import Path
import sys
import json
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import site_notices as n

class Notices(unittest.TestCase):
    def test_publication_is_repeatable_and_preserves_authored_content(self):
        html='<html><head></head><body><header><h1>Original pick</h1><p>Original date</p></header><p>Original analysis.</p></body></html>'
        out=n.apply(html,'blog/example.html')
        self.assertEqual(out,n.apply(out,'blog/example.html'))
        self.assertIn('<p>Original analysis.</p>',out)
        self.assertIn('</header><!-- fv-notices:notice:start -->',out)
        self.assertEqual(out.count('id="fv-betting-notice"'),1)
        self.assertIn('tel:+18006973738',out)
    def test_opinion_footer_without_betting_banner(self):
        out=n.apply('<head></head><body><h1>Opinion</h1></body>','editorial/opinion.html')
        self.assertNotIn('id="fv-betting-notice"',out)
        self.assertIn('id="fv-responsible-footer"',out)
    def test_fallback_copy_matches_static_copy(self):
        script=(n.DOCS/'assets/responsible-use.js').read_text()
        self.assertIn(json.dumps(n.NOTICE),script)
        self.assertIn(json.dumps(n.FOOTER),script)
    def test_video_render_and_redirect_not_modified(self):
        html='<head></head><body><h1>Video slide</h1></body>'
        self.assertEqual(n.apply(html,'videos/topic/render.html'),html)
        redirect='<head><meta http-equiv="refresh" content="0;url=/props/"></head><body></body>'
        self.assertEqual(n.apply(redirect,'nfl/props/index.html'),redirect)
    def test_current_public_pages_have_static_notices(self):
        count=0
        for p in n.DOCS.rglob('*.html'):
            rel=p.relative_to(n.DOCS).as_posix();html=p.read_text()
            if 'template' in rel or rel.endswith('render.html') or 'http-equiv="refresh"' in html:continue
            self.assertIn('id="fv-responsible-footer"',html,rel)
            if n.contextual(rel):self.assertIn('id="fv-betting-notice"',html,rel)
            count+=1
        self.assertGreater(count,40)

if __name__=='__main__':unittest.main()
