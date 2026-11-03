"""Pricing identity, leakage, blank markets and upstream failure regressions."""
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nba.pipeline import compare, add_baselines, refresh, flatten, iso, FeedError

NOW = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)


def quote(book='a', side='Over', line=20.5, market='player_points', price=-110, event='game'):
    from market_math import implied_probability
    return dict(event_id=event, player='Test Player' if market.startswith('player_') else '',
                market=market, line=line, side=side, book=book, price=price,
                book_probability=implied_probability(price), home_team='Home', away_team='Away')


class NBATests(unittest.TestCase):
    def test_exact_line_event_player_and_unique_books(self):
        rows=[quote(),quote(side='Under'),quote(),quote(book='b'),quote(book='b',side='Under'),
              quote(line=21.5),quote(event='other',side='Under')]
        result=compare(rows)
        self.assertEqual(len(result),6)
        self.assertEqual(result[0]['paired_books'],2)
        self.assertEqual(result[0]['fair_probability'],.5)
        self.assertIsNone(result[-1]['fair_probability'])
        self.assertIsNone(result[-2]['fair_probability'])

    def test_spreads_pair_opposite_signs(self):
        paired=compare([quote(side='Home',line=-3.5,market='spreads'),quote(side='Away',line=3.5,market='spreads')])
        self.assertEqual(paired[0]['fair_probability'],.5)
        bad=compare([quote(side='Home',line=-3.5,market='spreads'),quote(side='Away',line=-3.5,market='spreads')])
        self.assertIsNone(bad[0]['fair_probability'])

    def test_conflicting_duplicate_cannot_supply_a_pair(self):
        result=compare([quote(),quote(price=-120),quote(side='Under')])
        self.assertTrue(all(r['fair_probability'] is None for r in result))

    def test_watch_requires_three_other_paired_books(self):
        rows=[quote(book=b,side=s) for b in ['a','b','c'] for s in ['Over','Under']]
        self.assertTrue(all(r['consensus_ev'] is None for r in compare(copy.deepcopy(rows))))
        rows.extend([quote(book='outlier',price=120),quote(book='outlier',side='Under',price=-150)])
        outlier=next(r for r in compare(rows) if r['book']=='outlier' and r['side']=='Over')
        self.assertEqual(outlier['other_book_probability'],.5)
        self.assertAlmostEqual(outlier['consensus_ev'],10)

    def test_baseline_excludes_future_duplicate_and_dnp_games_and_handles_push(self):
        history=[dict(GAME_ID=str(i),PLAYER_ID=1,PLAYER_NAME='Test Player',MIN=30,PTS=20,
                      GAME_DATE=(NOW-timedelta(days=i+1)).date().isoformat()) for i in range(25)]
        history.extend([dict(history[0],PTS=999),dict(history[0],GAME_ID='future',GAME_DATE='2026-10-22',PTS=999),dict(history[0],GAME_ID='dnp',MIN=0,PTS=999)])
        rows=add_baselines([quote(line=20)],{'players':history},NOW)
        self.assertEqual(rows[0]['baseline_games'],25)
        self.assertEqual(rows[0]['baseline_mean'],20)
        self.assertEqual(rows[0]['baseline_push'],1)
        self.assertIsNone(rows[0]['model_probability'])

    def test_positions_by_player_id_then_unique_name(self):
        history=[dict(GAME_ID=str(i),PLAYER_ID=1,PLAYER_NAME='Test Player',MIN=30,PTS=20,
                      GAME_DATE=(NOW-timedelta(days=i+1)).date().isoformat()) for i in range(3)]
        positions=[dict(id=1,name='Test Player',position='F'),dict(id=1,name='Test Player',position='G-F'),
                   dict(id=7,name='Rookie Guard',position='G'),dict(id=8,name='Same Name',position='C'),dict(id=9,name='Same Name',position='F')]
        rows=[quote(line=20),dict(quote(),player='Rookie Guard'),dict(quote(),player='Same Name'),dict(quote(),player='Nobody'),
              quote(market='totals',line=220.5)]
        rows=add_baselines(rows,{'players':history,'positions':positions},NOW)
        self.assertEqual([r.get('player_position') for r in rows],['G-F','G',None,None,None],
                         'history ID first (latest season wins), then a unique name; namesakes stay blank')
        self.assertIsNone(add_baselines([quote()],{'players':history},NOW)[0]['player_position'],'no positions saved yet')

    def test_no_events_is_success_and_does_not_spend_on_props(self):
        class Empty:
            requests=1
            quota_remaining='100'
            def get(self,suffix,**kwargs):
                self.suffix=suffix
                return []
        client=Empty();result=refresh(client,NOW,{})
        self.assertEqual(result['status'],'waiting_for_markets')
        self.assertEqual(client.suffix,'events')

    def test_provider_errors_propagate_not_empty_success(self):
        class Broken:
            def get(self,*args,**kwargs):raise FeedError('HTTP 401')
        with self.assertRaises(FeedError):refresh(Broken(),NOW,{})

    def test_started_and_stale_quotes_are_excluded(self):
        event=dict(id='g',home_team='Home',away_team='Away',commence_time=iso(NOW+timedelta(hours=2)),bookmakers=[dict(key='a',title='A',last_update=iso(NOW-timedelta(hours=25)),markets=[dict(key='totals',outcomes=[dict(name='Over',point=220.5,price=-110)])])])
        self.assertEqual(flatten(event,NOW),[])
        event['bookmakers'][0]['last_update']=iso(NOW)
        self.assertEqual(len(flatten(event,NOW)),1)
        event['commence_time']=iso(NOW-timedelta(minutes=1))
        self.assertEqual(flatten(event,NOW),[])


def context(opp='@ BOS', hits=11):
    return dict(schema_version=1, source='ESPN regular-season box scores', through='2026-10-30', sample_games=30,
                games=[dict(date='2026-10-30', opp=opp, minutes=32, value=24)], opponent=dict(team=opp[-3:]),
                build=dict(steps=[dict(label='Games over the line', value=hits, unit='of 30')]))


def feed():
    rows = []
    for book in 'abc':
        for side, hits in [('Over', 11), ('Under', 19)]:
            rows.append(dict(quote(book=book, side=side), player_context=context(hits=hits)))
    rows.append(dict(quote(event='later'), player_context=context(opp='vs NYK')))  # another opponent
    rows.append(dict(quote(market='totals', line=220.5), player_context=None))      # game line
    return dict(status='ready', rows=rows)


class NBAPackedContextTests(unittest.TestCase):
    """Nov 2, 2026: repeated player histories made latest.json 128.8 MB; GitHub rejects 100 MB."""

    def test_shared_parts_are_stored_once_and_restore_exactly(self):
        from nba.pipeline import pack_player_contexts, unpack_player_contexts
        original = feed()
        packed = pack_player_contexts(copy.deepcopy(original))
        # Two histories (two opponents) and two line parts (Over/Under), not eight copies.
        self.assertEqual(len(packed['player_contexts']), 4)
        self.assertTrue(all('player_context' not in r for r in packed['rows'][:7]))
        self.assertEqual(len({tuple(r['player_context_parts']) for r in packed['rows'][:6]}), 2)
        self.assertNotEqual(packed['rows'][0]['player_context_parts'][0], packed['rows'][6]['player_context_parts'][0])
        self.assertIsNone(packed['rows'][7]['player_context'])
        self.assertNotIn('player_context_parts', packed['rows'][7])
        self.assertEqual(unpack_player_contexts(copy.deepcopy(packed)), original)
        # A failed refresh re-saves the packed feed: nothing is lost or re-keyed.
        self.assertEqual(pack_player_contexts(copy.deepcopy(packed)), packed)
        # A part missing from the map yields no context, never another row's.
        broken = copy.deepcopy(packed)
        del broken['player_contexts'][packed['rows'][0]['player_context_parts'][1]]
        self.assertIsNone(unpack_player_contexts(broken)['rows'][0]['player_context'])
        self.assertNotIn('player_contexts', pack_player_contexts(dict(rows=[dict(quote())])))

    def test_compact_feed_file(self):
        import json, tempfile
        from nba.pipeline import save_json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'latest.json'
            save_json(path, feed(), compact=True)
            text = path.read_text()
            self.assertNotIn('\n  ', text)
            self.assertEqual(json.loads(text), feed())

    @unittest.skipUnless(__import__('shutil').which('node'), 'node is not installed')
    def test_browser_readers_restore_python_packed_rows(self):
        import json, subprocess
        from nba.pipeline import pack_player_contexts
        root = Path(__file__).resolve().parents[1]
        script = ("const fs=require('fs'),d=JSON.parse(fs.readFileSync(0,'utf8'));"
                  "const a=JSON.parse(JSON.stringify(d)),b=JSON.parse(JSON.stringify(d));"
                  "require('./docs/assets/briefing-picks.js').unpackContexts(a);"
                  "require('./docs/tracking/player-stats.js').rowsFrom(b,'NBA');"
                  "process.stdout.write(JSON.stringify([a.rows,b.rows]));")
        packed = json.dumps(pack_player_contexts(feed()))
        result = subprocess.run(['node', '-e', script], input=packed, capture_output=True, text=True,
                                cwd=root, timeout=30, check=True)
        for rows in json.loads(result.stdout):
            self.assertEqual(rows, feed()['rows'])


if __name__=='__main__':unittest.main()
