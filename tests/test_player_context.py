import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from player_context import nhl_context, mlb_context, windows, nba_matchup, distribution, ranked, nba_defense
from mlb.models import State
from nhl.v2.features import History
from nba.pipeline import add_baselines


class PlayerContextTests(unittest.TestCase):
    def test_nhl_observed_form_is_distinct_from_weighted_model_inputs(self):
        history=History()
        for day in range(1,8):
            history.add_player(dict(player_id=1,game_date=f'2026-09-{day:02}',available_at=f'2026-09-{day+1:02}T12:00:00Z',
                shots=day,goals=0,assists=1,points=1,toi=20,position='F'))
        features=history.player_features(1,'F','2026-09-10',datetime(2026,9,10,tzinfo=timezone.utc))
        original=copy.deepcopy(features)
        result=nhl_context(history.players[1],features,0,'opportunity_nb','nhl-v2.1')
        self.assertEqual(result['recent'],[dict(games=5,mean=5,workload=20),dict(games=7,mean=4,workload=20)])
        self.assertAlmostEqual(result['inputs'][1]['value'],features['opportunity_means'][0]/features['projected_toi']*60,places=3)
        self.assertTrue(all(i['used'] for i in result['inputs']))
        self.assertIn('30-day',result['note'])
        self.assertEqual(features,original)
        self.assertEqual(result['through'],'2026-09-07')

    def test_mlb_uses_same_predate_window_and_decimal_innings(self):
        history=State()
        history.pitchers[1]=[dict(date='2026-09-01',outs=17,strikeOuts=0),dict(date='2026-09-02',outs=16,strikeOuts=6),
                             dict(date='2026-09-03',outs=99,strikeOuts=99),dict(date='2024-01-01',outs=99,strikeOuts=99)]
        features=dict(starter_outs5=15,starter_pitches5=85,starter_bf=22,starter_k_rate=.25,opp_k_rate=.20,starter_rest=1)
        result=mlb_context(history,'2026-09-03',{'id':1},'pitcher_strikeouts',features,{'kind':'rolling'})
        self.assertEqual(result['sample_games'],2)
        self.assertEqual(result['recent'],[dict(games=2,mean=3,workload=5.5,pitches=None)])
        self.assertEqual(result['through'],'2026-09-02')
        used=[i['label'] for i in result['inputs'] if i['used']]
        self.assertEqual(used,['Pitcher strikeout rate','Opponent strikeout rate','Batters faced / start'])
        self.assertEqual(result['inputs'][0]['value'],5)
        self.assertEqual(result['inputs'][2]['value'],25)
        self.assertIn('thirds',result['note'])

    def test_selected_model_controls_input_labels(self):
        history=State()
        history.pitchers[1]=[dict(date='2026-09-01',outs=17,strikeOuts=1)]
        f=dict(starter_outs5=15,starter_k_rate=.25)
        result=mlb_context(history,'2026-09-03',{'id':1},'pitcher_outs',f,{'kind':'boosting','features':['starter_k_rate']})
        self.assertEqual([i['label'] for i in result['inputs'] if i['used']],['Pitcher strikeout rate'])
        self.assertIsNone(result['inputs'][1]['value'])
        batter=mlb_context(history,'2026-09-03',{'id':2},'batter_hits',{}, {'kind':'rolling'})
        self.assertEqual(batter['recent'],[])
        self.assertIsNone(batter['through'])

    def test_nba_complete_games_share_stat_and_minutes_sample(self):
        games=[dict(GAME_ID=str(i),PLAYER_ID=1,PLAYER_NAME='Test Player',MIN=30,PTS=20,
                    GAME_DATE=f'2026-08-{i:02}') for i in range(1,26)]
        games.append(dict(games[-1],GAME_ID='missing',GAME_DATE='2026-08-26',PTS=None,MIN=99))
        row=dict(player='Test Player',market='player_points',market_label='Points',line=20,side='Under')
        result=add_baselines([row],{'players':games},datetime(2026,9,1,tzinfo=timezone.utc))[0]
        c=result['player_context']
        self.assertEqual(c['sample_games'],25)
        self.assertTrue(all(w['mean']==20 and w['workload']==30 for w in c['recent']))
        self.assertIsNone(result['model_probability'])
        self.assertEqual(result['baseline_push'],1)

    def test_pitcher_game_log_is_newest_first_with_innings_in_thirds(self):
        history=State()
        team=lambda tid,pitches:dict(id=tid,starter=tid*10,batting=dict(runs=3,plateAppearances=38,strikeOuts=8,hits=7,homeRuns=1,
            baseOnBalls=3,totalBases=11),pitching=dict(runs=3,outs=27,numberOfPitches=140),batters=[dict(id=tid*100,slot=1,plateAppearances=4,hits=2,
            totalBases=5,homeRuns=1,rbi=2)],pitchers=[dict(id=tid*10,outs=17,battersFaced=24,strikeOuts=7,baseOnBalls=2,hits=5,homeRuns=1,runs=2,
            earnedRuns=2,numberOfPitches=pitches,gamesStarted=1)])
        for day,pitches in [('2026-09-20',95),('2026-09-26',101)]:
            history.update(dict(date=day,venue=1,home_score=3,away_score=3,teams=dict(home=team(147,pitches),away=team(111,88))))
        features=dict(starter_outs5=16,starter_pitches5=97)
        result=mlb_context(history,'2026-10-01',{'id':1470},'pitcher_strikeouts',features,{'kind':'rolling'})
        self.assertEqual(result['games'][0],dict(date='2026-09-26',opp='vs BOS',ip='5⅔',pitches=101,k=7,bb=2,er=2))
        self.assertEqual([c[0] for c in result['game_columns']],['date','opp','ip','pitches','k','bb','er'])
        self.assertEqual(result['game_focus'],'k')
        self.assertEqual(result['recent'][0]['pitches'],98)
        batter=mlb_context(history,'2026-10-01',{'id':11100},'batter_total_bases',{},{'kind':'rolling'})
        self.assertEqual(batter['games'][0],dict(date='2026-09-26',opp='@ NYY',pa=4,h=2,tb=5,hr=1,rbi=2))
        self.assertEqual(batter['game_focus'],'tb')
        # Histories saved before opponents were recorded keep the log and drop the column.
        for r in history.pitchers[1470]: r.pop('opponent_id')
        older=mlb_context(history,'2026-10-01',{'id':1470},'pitcher_outs',features,{'kind':'rolling'})
        self.assertNotIn('opp',[c[0] for c in older['game_columns']])
        self.assertEqual(older['game_focus'],'ip')

    def test_nhl_game_log_shows_opponent_and_ice_time(self):
        history=History()
        for day in range(1,8):
            history.add_player(dict(player_id=1,game_id=day,game_date=f'2026-09-{day:02}',available_at=f'2026-09-{day+1:02}T12:00:00Z',
                shots=day,goals=day%2,assists=0,points=day%2,toi=18.5+day/60,position='F',home=day%2==0))
        features=history.player_features(1,'F','2026-09-10',datetime(2026,9,10,tzinfo=timezone.utc))
        result=nhl_context(history.players[1],features,0,'opportunity_nb','nhl-v2.1',lambda r:('vs ' if r['home'] else '@ ')+'CHI')
        self.assertEqual(len(result['games']),5)
        self.assertEqual(result['games'][0],dict(date='2026-09-07',opp='@ CHI',toi='18:37',shots=7,goals=1,assists=0,points=1))
        self.assertEqual(result['game_focus'],'shots')
        bare=nhl_context(history.players[1],features,3,'opportunity_nb','nhl-v2.1')
        self.assertEqual([c[0] for c in bare['game_columns']],['date','toi','shots','goals','assists','points'])

    def test_nba_matchup_labels(self):
        self.assertEqual(nba_matchup('BOS vs. NYK'),'vs NYK')
        self.assertEqual(nba_matchup('BOS @ NYK'),'@ NYK')
        self.assertIsNone(nba_matchup(None))
        games=[dict(GAME_ID=str(i),PLAYER_ID=1,PLAYER_NAME='Test Player',MIN=30+i%2,PTS=20+i,REB=5,AST=5,MATCHUP='LAL @ BOS',
                    GAME_DATE=f'2026-03-{i:02}') for i in range(1,26)]
        row=dict(player='Test Player',market='player_points_rebounds_assists',market_label='Pts + Reb + Ast',line=40.5,side='Over')
        c=add_baselines([row],{'players':games},datetime(2026,4,1,tzinfo=timezone.utc))[0]['player_context']
        self.assertEqual(c['games'][0],dict(date='2026-03-25',opp='@ BOS',minutes=31,value=55))
        self.assertEqual(c['game_columns'][-1],['value','PRA'])

    def test_distribution_trims_tails_into_end_bars(self):
        mass=[.001,.002,.1,.4,.3,.19,.004,.003]
        d=distribution(mass)
        self.assertEqual(d['start'],2)
        self.assertTrue(d['low']) ; self.assertTrue(d['high'])
        self.assertAlmostEqual(sum(d['p']),1,places=3)
        self.assertAlmostEqual(d['p'][0],.103,places=3,msg='the first bar carries the lower tail')
        self.assertIsNone(distribution([0,0]))

    def test_rank_wording(self):
        table={1:dict(v=3),2:dict(v=5),3:dict(v=4)}
        self.assertEqual(ranked(table,2,'v','most'),'Most of 3')
        self.assertEqual(ranked(table,3,'v','most'),'2nd most of 3')
        self.assertIsNone(ranked(table,9,'v','most'))

    def mlb_season(self):
        history=State()
        team=lambda tid,k,runs,pitches:dict(id=tid,starter=tid*10,batting=dict(runs=runs,plateAppearances=38,strikeOuts=k,hits=8,homeRuns=1,
            baseOnBalls=3,totalBases=12),pitching=dict(runs=runs,outs=27,numberOfPitches=140),batters=[dict(id=tid*100,slot=2,plateAppearances=4,
            hits=1,totalBases=2,homeRuns=0,rbi=1,strikeOuts=1,baseOnBalls=0)],pitchers=[dict(id=tid*10,outs=17,battersFaced=24,strikeOuts=7,
            baseOnBalls=2,hits=5,homeRuns=1,runs=2,earnedRuns=2,numberOfPitches=pitches,gamesStarted=1)])
        for d in range(1,13):
            history.update(dict(date=f'2026-09-{d:02}',venue=1,home_score=4,away_score=3,teams=dict(home=team(147,9,4,95+d),away=team(111,6,3,90))))
            history.update(dict(date=f'2026-09-{d:02}',venue=2,home_score=5,away_score=2,teams=dict(home=team(119,11,5,100),away=team(144,5,2,88))))
        return history,dict(date='2026-09-20',game_type='R',venue=1,home_id=147,away_id=111,home_starter=1470,away_starter=1110)

    def test_mlb_explains_the_number_the_odds_and_the_opponent(self):
        from mlb.models import means, pmf
        history,game=self.mlb_season()
        x=history.features(game,'home')
        model=dict(kind='rolling',target='pitcher_strikeouts',alpha=.05,sigma=2,calibrator=None,features=sorted(x))
        mass=pmf(means(model,[{'x':x}]),model)[0]
        c=mlb_context(history,'2026-09-20',{'id':1470,'side':'home'},'pitcher_strikeouts',x,model,mass,game)
        steps=c['build']['steps']
        self.assertEqual([s['label'] for s in steps],['Batters faced per start','Strikeout rate','Opponent adjustment','Simple estimate','Adjusted to past results'])
        self.assertAlmostEqual(steps[0]['value']*steps[1]['value']/100*steps[2]['value'],steps[3]['value'],places=2)
        self.assertAlmostEqual(steps[-1]['value'],float(mass@range(len(mass))),places=3)
        self.assertAlmostEqual(sum(c['distribution']['p']),1,places=3)
        self.assertEqual(len(c['trend']['rows']),10)
        self.assertEqual(c['trend']['rows'][-1][:4],['2026-09-12',7.0,None,'vs BOS'])
        self.assertEqual(c['opponent']['team'],'BOS')
        k,runs=c['opponent']['items']
        self.assertTrue(k['used']) ; self.assertFalse(runs['used'],'the strikeout model reads only the lineup strikeout rate')
        self.assertEqual(k['rank'],'3rd highest of 4')
        self.assertAlmostEqual(c['blend'][0]['own'],288/388,places=3)
        boosted=dict(model,kind='boosted',features=sorted(x),estimator=type('E',(),{'predict':lambda self,m:[6.4]*len(m)})())
        c=mlb_context(history,'2026-09-20',{'id':1470,'side':'home'},'pitcher_strikeouts',x,boosted,mass,game)
        self.assertEqual(c['build']['steps'][-2]['label'],f'Machine-learning model ({len(x)} inputs)')
        self.assertTrue(all(i['used'] for i in c['opponent']['items']))

    def test_explanations_never_break_the_context(self):
        history,game=self.mlb_season()
        x=history.features(game,'home')
        c=mlb_context(history,'2026-09-20',{'id':1470,'side':'home'},'pitcher_strikeouts',x,{'kind':'rolling','target':'pitcher_strikeouts'},None,None)
        self.assertEqual(c['sample_games'],12,'the base context survives a model without distribution settings')
        self.assertNotIn('opponent',c)

    def test_nhl_trend_weights_and_blend_follow_the_model(self):
        history=History()
        for day in range(1,8):
            history.add_player(dict(player_id=1,game_id=day,game_date=f'2026-09-{day:02}',available_at=f'2026-09-{day+1:02}T12:00:00Z',
                shots=day,goals=0,assists=1,points=1,toi=20,position='F'))
        f=history.player_features(1,'F','2026-09-10',datetime(2026,9,10,tzinfo=timezone.utc))
        matchup=dict(team='CHI',label='Opposing defense',items=[])
        c=nhl_context(history.players[1],f,0,'opportunity_nb','nhl-v2.1',None,'2026-09-10',[.2,.3,.3,.2],matchup)
        weights=[r[2] for r in c['trend']['rows']]
        self.assertAlmostEqual(weights[-1],2**(-3/120),places=3,msg='a game three days back keeps 98% weight')
        self.assertLess(weights[0],weights[-1])
        steps=c['build']['steps']
        self.assertAlmostEqual(steps[0]['value']*steps[1]['value']/60,steps[2]['value'],places=3)
        own=sum(2**(-(10-d)/120) for d in range(1,8))
        self.assertAlmostEqual(c['blend'][0]['own'],own/(own+12),places=3)
        self.assertIs(c['opponent'],matchup)
        self.assertIn('Opponent defense and goalie',c['missing'])

    def test_nba_defense_pairs_team_logs_and_aliases(self):
        rows=[]
        for i in range(12):
            rows.append(dict(GAME_ID=str(i),TEAM_NAME='LA Clippers',TEAM_ABBREVIATION='LAC',PTS=110,REB=44,AST=25,FG3M=12,BLK=5,STL=7,TOV=13,GAME_DATE=f'2026-03-{i+1:02}'))
            rows.append(dict(GAME_ID=str(i),TEAM_NAME='Boston Celtics',TEAM_ABBREVIATION='BOS',PTS=100+i,REB=40,AST=22,FG3M=14,BLK=4,STL=8,TOV=12,GAME_DATE=f'2026-03-{i+1:02}'))
        d=nba_defense({'x':rows})
        self.assertEqual(d['table']['losangelesclippers']['PTS'],105.5,'what the Clippers allowed: Boston scored 100 to 111')
        self.assertEqual(d['table']['bostonceltics']['PTS'],110)
        self.assertEqual(d['abbreviations']['losangelesclippers'],'LAC')

    def test_missing_observations_are_not_zero(self):
        self.assertEqual(windows([dict(stat=None,toi=None)],[5,10],'stat','toi'),[dict(games=1,mean=None,workload=None)])


if __name__=='__main__':unittest.main()
