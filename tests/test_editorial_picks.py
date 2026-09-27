"""The morning briefing's NFL export must preserve the actual priced offer."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_props_site as props
import editorial


class BriefingContractTests(unittest.TestCase):
    def test_nfl_export_matches_embedded_shortlist_without_retiming_quotes(self):
        now=datetime.now(timezone.utc)
        quote=(now-timedelta(minutes=7)).isoformat()
        row=dict(game_id='game',game='A @ B',player='Player',bookmaker='book',book_label='Book',
            market_std='receptions',market_label='Receptions',name='Over',point=2.5,price=110,
            model_status='Calibration fitted',model_prob=.6,edge_bps=100,
            commence_time=(now+timedelta(hours=1)).isoformat(),last_update=quote,kick_et='Today')
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'docs/props/top.html'
            args=SimpleNamespace(merged_csv='unused',out=str(out),season=2026,week=3)
            with patch.object(props,'prepare_records',return_value=[row,dict(row,player='Unsupported',model_status='Legacy estimate')]):
                props.build_page(args,top_only=True)
            feed=json.loads((out.parent/'top-picks.json').read_text())
            packed=json.loads(re.search(r'id="props-data">(.*?)</script>',out.read_text()).group(1))
            decoded=[{f:packed['dictionary'][f][r[i]] if f in packed['dictionary'] else r[i] for i,f in enumerate(packed['fields'])} for r in packed['rows']]
            self.assertEqual(feed['rows'],decoded)
            self.assertEqual(feed['rows'],[row])
            self.assertEqual(feed['rows'][0]['last_update'],quote)

    def test_current_picks_never_appear_as_historical_selections(self):
        now=datetime(2026,9,26,12,tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            docs=Path(td);public=docs/'briefing'
            with patch.object(editorial,'DOCS',docs),patch.object(editorial,'PUBLIC',public):
                editorial.render_briefing({'generated_at':now.isoformat()},now)
            current=(public/'index.html').read_text();archived=(public/'2026-09-26.html').read_text()
            self.assertIn('id="daily-picks"',current)
            self.assertIn('Model prediction</th>',current)
            self.assertIn('Market consensus</th>',current)
            self.assertNotIn('Astra',current)
            self.assertIn('historical-outcome calibration',current)
            self.assertNotIn('id="daily-picks"',archived)
            self.assertNotIn('What changed and what’s next',current)
            self.assertLess(current.index('id="daily-picks-heading"'),current.index('<h2>The price rundown</h2>'))
