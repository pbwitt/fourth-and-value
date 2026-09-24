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

if __name__=='__main__':unittest.main()
