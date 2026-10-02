"""SEO_POLICY.md: changed pages may not add issues beyond the shrinking baseline."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import seo_check
from site_metadata import seo_description, seo_title


class SeoPolicy(unittest.TestCase):
    def test_no_page_has_issues_beyond_the_baseline(self):
        baseline = json.loads(seo_check.BASELINE.read_text())
        new = {rel: [i for i in issues if i not in baseline.get(rel, [])] for rel, issues in seo_check.audit_all().items()}
        self.assertEqual({k: v for k, v in new.items() if v}, {})

    def test_audit_flags_each_rule(self):
        bad = '<html><head><title>' + 'x' * 70 + '</title></head><body><h1>a</h1><h1>b</h1></body></html>'
        issues = seo_check.audit('blog/new-post.html', bad)
        for rule in ('title:too-long', 'description:missing', 'canonical:missing', 'h1:count',
                     'og:image:missing', 'twitter:card', 'article:schema'):
            self.assertIn(rule, issues)

    def test_title_and_description_fit_the_policy(self):
        self.assertEqual(seo_title('Points'), 'Points | Fourth & Value')
        long = 'Chargers–Bills: the tight-end reset is bigger than the targets everyone is talking about this week'
        self.assertLessEqual(len(seo_title(long)), 65)
        self.assertLessEqual(len(seo_description('word ' * 60)), 160)


if __name__ == '__main__':
    unittest.main()
