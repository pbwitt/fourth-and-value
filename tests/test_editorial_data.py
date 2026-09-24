from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import editorial_writer as w

class DataOrdering(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,9,23,12,tzinfo=timezone.utc)
        self.board={'status':'ready','last_success_at':self.now.isoformat(),'model_checked_at':self.now.isoformat(),'model_status':'Independent MLB forecasts available','model_summary':{'history_through':'2026-09-22'}}
        self.briefing={'generated_at':self.now.isoformat()}
    def test_current_mlb_history_required(self):
        self.assertTrue(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
        self.board['model_summary']['history_through']='2026-09-21'
        self.assertFalse(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
    def test_stale_or_failed_refresh_rejected(self):
        self.board['last_success_at']=(self.now-timedelta(minutes=91)).isoformat()
        self.assertFalse(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
        self.board['last_success_at']=self.now.isoformat();self.board['status']='feed_error'
        self.assertFalse(w.data_readiness('MLB',self.board,self.briefing,self.now)['ready'])
    def test_previous_day_briefing_rejected_even_within_six_hours(self):
        now=datetime(2026,9,23,5,tzinfo=timezone.utc)
        self.briefing['generated_at']='2026-09-23T03:00:00Z'
        self.assertFalse(w.data_readiness('NFL',{},self.briefing,now)['ready'])
    def test_compaction_preserves_real_market_and_model(self):
        packet={'markets':[{'id':str(i),'game':'Yankees at Rays','quotes':[{'line':7.5+j/2,'over_price':-110,'under_price':-110,'label':'Book '+str(j),'quoted_at':self.now.isoformat()} for j in range(30)]} for i in range(3)],'model_rows':[{'id':'model','model_mean':5.2,'model_inputs':{'last_five':5}}], 'reporting':[{'title':'Yankees news','excerpt':'facts '*1000,'url':'https://example.org/'+str(i)} for i in range(3)],'methods':'Methods '*300}
        result=w.compact(packet)
        self.assertTrue(result['markets']);self.assertTrue(result['model_rows'])
        self.assertLessEqual(len(result['markets'][0]['quotes']),4)
        self.assertLess(len(json.dumps(result).encode()),11500)
    def test_missing_data_stops_before_research_or_spend(self):
        with TemporaryDirectory() as directory,patch.object(w,'STATE',Path(directory)),patch.object(w,'evidence',return_value={'data_readiness':{'ready':False,'reason':'stale'}}),patch.object(w.reporting,'collect') as collect,patch.object(w,'call_api') as api,patch.object(w.budget,'reserve') as reserve,patch.object(w.ed,'render_home'):
            w.run(self.now)
            api.assert_not_called();reserve.assert_not_called();collect.assert_not_called()
    def test_empty_packet_not_publishable(self):
        with self.assertRaises(ValueError):w.require_data({'data_readiness':{'ready':True},'markets':[],'model_rows':[]})

if __name__=='__main__':unittest.main()
