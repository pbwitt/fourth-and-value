"""Thursday allocation uses Eastern kickoff dates and only matchup evidence."""
from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_writer as w
import editorial_sources as sources

class ThursdayPreview(unittest.TestCase):
    now=datetime(2026,9,24,9,7,tzinfo=timezone.utc)
    game={'id':'night','sport':'NFL','commence_time':'2026-09-25T00:15:00Z','game':'Atlanta Falcons @ Green Bay Packers'}

    def test_eastern_thursday_includes_friday_utc(self):
        self.assertEqual(w.thursday_games([self.game],self.now),[self.game])
        self.assertEqual(w.thursday_games([self.game],datetime(2026,9,23,23,tzinfo=timezone.utc)),[])
        self.assertEqual(w.thursday_games([self.game],datetime(2026,9,25,1,tzinfo=timezone.utc)),[])

    def test_priority_even_if_nfl_was_recently_covered(self):
        selected=w.slots([self.game],catalog=[{'sport':'NFL','date':'2026-09-23'}],now=self.now)
        self.assertEqual(selected[0],('NFL','thursday-preview:night'))
        self.assertEqual(len(selected),2)
        self.assertNotEqual(selected[1][0],'NFL')

    def test_multiple_games_one_slot_evening_first(self):
        early=dict(self.game,id='early',commence_time='2026-09-24T17:00:00Z')
        baseball=dict(self.game,id='baseball',sport='MLB')
        self.assertEqual(w.slots([early,baseball,self.game],now=self.now)[0],('NFL','thursday-preview:night,early'))

    def test_focus_excludes_other_games_and_models(self):
        packet={'markets':[dict(self.game,id='total-night'),dict(self.game,id='total-sunday')],
                'model_rows':[{'event_id':'night'},{'event_id':'sunday'}],
                'model_references':[{'game':self.game['game']},{'game':'Buffalo Bills @ Miami Dolphins'}]}
        result=w.focus_preview(packet,'thursday-preview:night')
        self.assertEqual([x['id'] for x in result['markets']],['total-night'])
        self.assertEqual(result['model_rows'],[{'event_id':'night'}])
        self.assertEqual(len(result['model_references']),1)
        self.assertEqual(w.preview_terms(result,'thursday-preview:night'),('falcons','packers'))
        self.assertEqual(len(packet['markets']),2)

    def test_targeted_sources_exclude_unrelated_reporting(self):
        found=[{'title':title,'url':url,'published_timestamp':self.now.isoformat()} for title,url in [
            ('Packers injury report','https://www.espn.com/packers'),
            ('Falcons preview','https://sports.yahoo.com/falcons'),
            ('Unrelated news','https://www.cbssports.com/unrelated')]]
        with patch.object(sources,'candidates',return_value=found),patch.object(sources,'fetch',return_value='body'),patch.object(sources,'text_content',return_value='verified '*110):
            result=sources.collect('NFL',self.now,terms=('falcons','packers'))
        self.assertEqual(len(result),2)
        self.assertTrue(all('unrelated' not in x['url'] for x in result))

    def test_general_research_searches_beyond_three_headlines(self):
        rows=[{'title':'News', 'url':f'https://www.espn.com/{i}',
               'published_timestamp':self.now.isoformat()} for i in range(3)]
        rows += [{'title':'Usable news','url':url,'published_timestamp':self.now.isoformat()}
                 for url in ('https://www.espn.com/usable','https://www.cbssports.com/usable')]
        def candidates(sport,now,limit):return rows[:limit]
        def fetch(url):return 'verified '*110 if url.endswith('usable') else 'too short'
        with patch.object(sources,'candidates',side_effect=candidates),patch.object(sources,'fetch',side_effect=fetch),patch.object(sources,'text_content',side_effect=lambda text:text):
            result=sources.collect('NFL',self.now)
        self.assertEqual(len(result),2)
        self.assertTrue(all(row['url'].endswith('usable') for row in result))

class OfficialReporting(unittest.TestCase):
    now=datetime(2026,9,26,12,tzinfo=timezone.utc)

    def test_official_fallback_rejects_stale_and_undated_articles(self):
        import json
        index='<a href="https://www.nfl.com/news/old">old</a><a href="https://www.nfl.com/news/missing">missing</a><a href="https://www.nfl.com/news/current">current</a>'
        def article(date):
            return '<script type="application/ld+json">'+json.dumps({'@type':'NewsArticle','headline':'Current news','datePublished':date,'articleBody':'evidence '*110})+'</script>'
        pages={'https://www.nfl.com/news':index,'https://www.nfl.com/news/old':article('2026-09-01T12:00:00Z'),
            'https://www.nfl.com/news/missing':article(''),'https://www.nfl.com/news/current':article('2026-09-26T10:00:00Z')}
        with patch.object(sources,'fetch',side_effect=pages.__getitem__):
            result=sources.nfl_reporting(self.now)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['url'],'https://www.nfl.com/news/current')

    def test_official_fallback_restores_two_publishers(self):
        row={'title':'News','url':'https://www.cbssports.com/news','published_timestamp':self.now.isoformat()}
        official=dict(row,url='https://www.nfl.com/news/current',excerpt='evidence '*110)
        with patch.object(sources,'candidates',return_value=[row]),patch.object(sources,'fetch',return_value='evidence '*110),patch.object(sources,'text_content',side_effect=lambda s:s),patch.object(sources,'nfl_reporting',return_value=[official]):
            result=sources.collect('NFL',self.now)
        self.assertEqual(len(result),2)
        self.assertEqual([r['id'] for r in result],['s1','s2'])

if __name__=='__main__':unittest.main()
