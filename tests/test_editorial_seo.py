import json
from pathlib import Path
import re
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_seo as seo

class SeoTests(unittest.TestCase):
    def item(self):
        return {'title':'Team A–Team B: a measured forecast','excerpt':'An original matchup analysis with current market and model context.',
                'url':'/editorial/articles/2026-09-26-0-mlb.html','date':'2026-09-26','published_at':'2026-09-26T12:00:00+00:00','sport':'MLB'}
    def test_metadata_has_canonical_social_and_article_schema(self):
        item=self.item();head=seo.metadata(item)
        schema=json.loads(re.search(r'application/ld\+json">(.*?)</script>',head,re.S).group(1))
        self.assertEqual(schema['datePublished'],item['published_at'])
        self.assertEqual(schema['dateModified'],item['published_at'])
        self.assertEqual(schema['mainEntityOfPage']['@id'],seo.BASE+item['url'])
        self.assertEqual(schema['author']['@type'],'Organization')
        for key in ['og:url','og:title','twitter:card','description','canonical']:self.assertIn(key,head)
        self.assertNotIn('image',schema)
    def test_backfill_is_idempotent_and_does_not_change_body(self):
        page='<html><head><title>Old</title><meta name="description" content="old"><link rel="canonical" href="old"></head><body><h1>Existing article</h1></body></html>'
        first=seo.update_page(page,self.item());second=seo.update_page(first,self.item())
        self.assertEqual(first,second)
        self.assertEqual(first.split('</head>')[1],page.split('</head>')[1])
        self.assertEqual(first.count('rel="canonical"'),1)
    def test_schema_cannot_break_out_of_script(self):
        item=self.item();item['title']='Danger </script><img src=x onerror=alert(1)>'
        head=seo.metadata(item)
        self.assertEqual(head.count('</script>'),1)
        self.assertNotIn('<img',head)

if __name__=='__main__':unittest.main()
