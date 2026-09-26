import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import export_nfl_editorial as export

class ExportTests(unittest.TestCase):
    now=datetime(2026,9,26,12,tzinfo=timezone.utc)
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        path=self.root/'data/nfl/refresh/manifest.json';path.parent.mkdir(parents=True)
        self.manifest={'season':2026,'week':3,'checked_at':'2026-09-26T11:30:00Z','history_through':'2026-09-21'}
        path.write_text(json.dumps(self.manifest))
        self.csv('data/nfl/predictions/week_predictions.csv',{'game':'LAC @ BUF','season':2026,'week':3,'gameday':'2026-09-27','total_pred':48,'home_pred':27,'away_pred':21})
        self.csv('data/nfl/lines/totals_spreads.csv',{'game':'LAC @ BUF','event_id':'correct-game','commence_time':'2026-09-27T17:00:00Z'})
    def csv(self,path,row):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=row);writer.writeheader();writer.writerow(row)
    def test_export_carries_real_forecast_and_provenance_without_fake_ev(self):
        board=export.export(self.root,2026,3,self.now);row=board['rows'][0]
        self.assertEqual(row['event_id'],'correct-game');self.assertEqual(row['model_mean'],48)
        self.assertEqual(row['model_input_through'],'2026-09-21')
        self.assertNotIn('model_ev_pct',row);self.assertFalse(row['is_model_pick'])
    def test_wrong_slate_and_old_prepare_are_rejected(self):
        with self.assertRaises(ValueError):export.export(self.root,2026,4,self.now)
        self.manifest['checked_at']='2026-09-23T12:00:00Z'
        (self.root/'data/nfl/refresh/manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):export.export(self.root,2026,3,self.now)
    def test_missing_event_match_is_unavailable_not_a_current_forecast(self):
        self.csv('data/nfl/lines/totals_spreads.csv',{'game':'NYJ @ NE','event_id':'other','commence_time':'2026-09-27T17:00:00Z'})
        board=export.export(self.root,2026,3,self.now)
        self.assertEqual(board['rows'],[]);self.assertEqual(board['status'],'unavailable')

if __name__=='__main__':unittest.main()
