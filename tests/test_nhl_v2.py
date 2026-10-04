import copy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nhl.v2.data import normalize,stamp
from nhl.v2.features import build,History,decision_time
from nhl.v2.models import TeamModel,PlayerModel,count_pmf,outcome,game_outcome
from nhl.v2.pricing import compare,price,decimal,fit_blend
from nhl.v2.grading import settle,select,closing_value,betting_metrics
from nhl.v2.review import validate_review,apply_review
from nhl.v2.inference import annotate,enrich,bundle
from nhl.refresh import PROPS,MARKETS,flatten

NOW=datetime(2026,9,26,14,30,tzinfo=timezone.utc)


def quote(book='a',side='Over',line=6.,market='totals',**extra):
    r=dict(event_id='e',nhl_game_id=1,player='',market=market,line=line,book=book,side=side,
           home_team='Home',away_team='Away',price=-110,quoted_at='2026-09-26T14:29:00Z',
           commence_time='2026-09-26T23:00:00Z',settlement_profile='nhl_full_game_ot_so',settlement_verified=True)
    return dict(r,**extra)


def game(gid,date,home=3,away=2):
    return dict(game_id=gid,game_date=date,available_at=(datetime.fromisoformat(date).replace(tzinfo=timezone.utc)+timedelta(days=1,hours=12)).isoformat(),
      season=20252026,game_type=2,home_id=1,away_id=2,home_team='Home',away_team='Away',home_score=home,away_score=away,
      home_reg_goals=home,away_reg_goals=away,home_shots=30,away_shots=28,home_pp_pct=.2,away_pp_pct=.8,
      home_pk_pct=.8,away_pk_pct=.7,extra_time=False,shootout=False)


class PointInTimeTests(unittest.TestCase):
    def test_target_game_position_cannot_change_rookie_prior(self):
        g=game(1,'2025-10-07')
        p=dict(game_id=1,game_date=g['game_date'],available_at=g['available_at'],player_id=5,player='Test',
               position='D',toi=20,shots=3,goals=0,assists=1,points=1)
        a=build([g],[p])[1][0]
        b=build([g],[dict(p,position='C')])[1][0]
        self.assertEqual(a['base_means'],b['base_means']);self.assertEqual(a['position'],'U')

    def test_current_and_future_outcomes_cannot_change_earlier_features(self):
        games=[game(1,'2025-10-07'),game(2,'2025-10-08'),game(3,'2025-10-09')]
        first=build(games,[])[0]
        changed=copy.deepcopy(games);changed[1]['home_reg_goals']=10;changed[2]['home_reg_goals']=12
        second=build(changed,[])[0]
        for i in range(4):
            a={k:v for k,v in first[i].items() if k not in ['target','final_score']}
            b={k:v for k,v in second[i].items() if k not in ['target','final_score']}
            self.assertEqual(a,b)
        self.assertNotEqual(first[4]['attack'],second[4]['attack'])
        self.assertTrue(all(not r['feature_cutoff'] or stamp(r['feature_cutoff'])<stamp(r['decision_at']) for r in first))

    def appearances(self, days, toi=20, shots=4, start='2025-10-07'):
        h=History()
        for i,day in enumerate(days):
            date=(datetime.fromisoformat(start)+timedelta(days=day)).date().isoformat()
            h.add_player(dict(player_id=9,game_id=i,game_date=date,position='C',toi=toi[i] if isinstance(toi,list) else toi,
                              shots=shots,goals=1,assists=1,points=2,
                              available_at=(datetime.fromisoformat(date).replace(tzinfo=timezone.utc)+timedelta(days=1,hours=12)).isoformat()))
        return h

    def test_offseason_or_injury_gap_does_not_age_player_history(self):
        h=self.appearances(range(0,40,2))
        asof=datetime(2026,10,10,14,30,tzinfo=timezone.utc)
        soon=h.player_features(9,'C','2025-11-20',asof)
        after_summer=h.player_features(9,'C','2026-10-08',asof)
        for k in ['projected_toi','base_means','opportunity_means']:
            self.assertEqual(soon[k],after_summer[k])
        # A 20-minute regular keeps his level across the summer; the 15-minute prior (5 games) still
        # holds about a fifth of the weight once history saturates.
        self.assertGreater(after_summer['projected_toi'],18.5)

    def test_player_weights_halve_per_half_life_in_games_and_sparse_history_shrinks(self):
        from nhl.v2.features import PLAYER_HALF_LIVES
        asof=datetime(2026,10,10,14,30,tzinfo=timezone.utc)
        toi=[12.]*30+[20.]*14
        f=self.appearances(range(0,88,2),toi=toi).player_features(9,'C','2026-01-10',asof)
        w=np.exp2(-np.arange(len(toi))[::-1]/PLAYER_HALF_LIVES['toi'])
        self.assertAlmostEqual(f['projected_toi'],(w@np.array(toi)+5*15)/(w.sum()+5))
        self.assertAlmostEqual(w[-1-PLAYER_HALF_LIVES['toi']],.5)
        one=self.appearances([0],toi=25).player_features(9,'C','2025-10-20',asof)
        self.assertAlmostEqual(one['projected_toi'],(25+5*15)/6)
        with self.assertRaises(ValueError):
            # The 2025-10-07 game becomes available at 12:00 UTC the next day.
            self.appearances([0]).player_features(9,'C','2025-10-08',datetime(2025,10,8,11,0,tzinfo=timezone.utc))

    def test_evaluation_never_writes_into_another_versions_evidence(self):
        from nhl.v2 import EVIDENCE, VERSION, evidence_dir
        root=Path(__file__).resolve().parents[1]
        self.assertEqual(evidence_dir(),root/EVIDENCE[VERSION])
        for version,folder in EVIDENCE.items():
            if version!=VERSION:
                with self.assertRaises(ValueError):evidence_dir(root/folder)
        with tempfile.TemporaryDirectory() as d:self.assertEqual(evidence_dir(d),Path(d))

    def test_future_history_rejected_and_dst_is_explicit(self):
        h=History();h.add_game(game(1,'2026-09-26'))
        with self.assertRaises(ValueError):h.team_features(game(2,'2026-09-26'),NOW)
        self.assertEqual(decision_time('2026-01-10').utcoffset(),timedelta(hours=-5))
        self.assertEqual(decision_time('2026-07-10').utcoffset(),timedelta(hours=-4))
        with self.assertRaises(ValueError):stamp('2026-09-26T10:00:00')

    def test_official_shootout_statistics_gain_only_one_settlement_goal(self):
        base=dict(gameId=1,gameDate='2025-10-07',goalsFor=2,goalsAgainst=2,shotsForPerGame=30,
                  winsInRegulation=0,otLosses=0,wins=1,winsInShootout=1)
        h=dict(base,homeRoad='H',teamId=1,teamFullName='Home')
        a=dict(base,homeRoad='R',teamId=2,teamFullName='Away',wins=0,winsInShootout=0,otLosses=1)
        result,_=normalize([h,a],[],20252026)
        self.assertEqual((result[0]['home_score'],result[0]['away_score']),(3,2))
        self.assertEqual((result[0]['home_reg_goals'],result[0]['away_reg_goals']),(2,2))
        with self.assertRaises(ValueError):normalize([h,h,a],[],20252026)


class ProbabilityTests(unittest.TestCase):
    def test_frozen_artifact_loads_with_pinned_stack_and_matches_manifest(self):
        models,manifest=bundle()
        self.assertEqual(manifest['selection']['team'],'poisson_core')
        self.assertFalse(manifest['selection']['recommendations_enabled'])
        means=models['team'].predict([dict(home=1,attack=3.,defense=3.),dict(home=0,attack=3.,defense=3.)])
        self.assertTrue(np.all(np.isfinite(means)))
        self.assertAlmostEqual(models['team'].joint(*means).sum(),1)

    def test_joint_is_normalized_no_final_tie_and_markets_cohere(self):
        m=TeamModel('rate');m.ot_home=.5
        joint=m.joint(3,3)
        self.assertAlmostEqual(joint.sum(),1)
        self.assertAlmostEqual(np.trace(joint),0)
        self.assertAlmostEqual(game_outcome(joint,'h2h',None)['win'],.5)
        self.assertLessEqual(game_outcome(joint,'spreads',-1.5)['win'],game_outcome(joint,'h2h',None)['win'])
        o=game_outcome(joint,'totals',6);u=game_outcome(joint,'totals',6,side='Under')
        self.assertAlmostEqual(o['win']+u['win']+o['push'],1)
        self.assertAlmostEqual(o['push'],u['push'])

    def test_integer_under_and_zero_do_not_include_push(self):
        pmf=count_pmf(.6,.2)
        a=outcome(pmf,1,'Over');b=outcome(pmf,1,'Under')
        self.assertAlmostEqual(a['win']+b['win']+a['push'],1)
        self.assertEqual(outcome(pmf,0,'Under')['win'],0)
        self.assertAlmostEqual(outcome(pmf,.5)['push'],0)

    def test_player_shared_scoring_mean_and_fast_batch_agree(self):
        rows=[dict(opportunity_means=[3,.4,.6,1],base_means=[2,.3,.4,.7],targets=[4,1,1,2])]*20
        for name in ['rate_poisson','opportunity_poisson','opportunity_nb','opportunity_hurdle']:
            m=PlayerModel(name).fit(rows);p=m.pmfs(rows[0]);batch=m.fast_pmfs(rows[:1]);n=np.arange(48)
            self.assertAlmostEqual(p[1]@n+p[2]@n,p[3]@n,places=6)
            for a,b in zip(p,batch):np.testing.assert_allclose(a,b[0],atol=1e-8)

    def test_push_aware_fair_ev_minimum(self):
        r=price(dict(win=.45,push=.1,loss=.45),110,lower_win=.4)
        self.assertAlmostEqual(r['fair_decimal'],2)
        self.assertAlmostEqual(r['estimated_ev'],.045)
        self.assertAlmostEqual(r['minimum_acceptable_decimal'],2.3)
        s=price(dict(win=.45,push=.1,loss=.45),110,scenarios=[dict(win=.4,push=0.,loss=.6)])
        self.assertAlmostEqual(s['minimum_acceptable_decimal'],2.55)
        with self.assertRaises(ValueError):price(dict(win=.5,push=.1,loss=.5),-110)
        with self.assertRaises(ValueError):decimal(0)


class QuoteTests(unittest.TestCase):
    def test_historical_sample_is_fixed_monthly_and_never_uses_current_time(self):
        from nhl.v2.market_evaluate import dates
        sample=dates()
        self.assertEqual(len(sample),21);self.assertEqual(len(set(sample)),21)
        self.assertTrue(all(day.endswith('-15') for day in sample))
        self.assertEqual(sample[0],'2023-10-15');self.assertEqual(sample[-1],'2026-04-15')

    def test_exact_pair_and_push_correction(self):
        rows=compare([quote(b,s) for b in 'abcd' for s in ['Over','Under']],NOW)
        self.assertTrue(all(r['fair_probability']==.5 and r['other_books']==3 for r in rows))
        self.assertTrue(all(r['consensus_ev'] is None for r in rows))
        altered=[quote(side='Over'),quote(side='Under',line=6.5)]
        self.assertTrue(all(r['fair_probability'] is None for r in compare(altered,NOW)))

    def test_pair_time_window_consensus_future_and_book_exclusion(self):
        rows=[quote(b,s,line=6.5,price=150 if b=='a' and s=='Over' else -110) for b in 'abcd' for s in ['Over','Under']]
        got=compare(rows,NOW);a=next(r for r in got if r['book']=='a' and r['side']=='Over')
        self.assertAlmostEqual(a['other_book_probability'],.5)
        rows[1]['quoted_at']='2026-09-26T14:00:00Z'
        self.assertIsNone(compare(rows,NOW)[0]['fair_probability'])
        rows[0]['quoted_at']='2026-09-26T14:31:00Z'
        self.assertFalse(any(r['book']=='a' and r['side']=='Over' for r in compare(rows,NOW)))

    def test_rules_and_conflicting_quotes_do_not_pair(self):
        rows=[quote(),quote(side='Under',settlement_profile='regulation')]
        self.assertTrue(all(r['fair_probability'] is None for r in compare(rows,NOW)))
        rows=[quote(),quote(price=100),quote(side='Under')]
        got=compare(rows,NOW)
        self.assertEqual(len(got),1);self.assertIsNone(got[0]['fair_probability'])
        rows=[quote(),quote(side='Under',commence_time='2026-09-27T23:00:00Z')]
        self.assertTrue(all(r['fair_probability'] is None for r in compare(rows,NOW)))

    def test_spread_sign_and_missing_models(self):
        paired=compare([quote(market='spreads',side='Home',line=-1.5),quote(market='spreads',side='Away',line=1.5)],NOW)
        self.assertTrue(all(r['fair_probability']==.5 for r in paired))
        self.assertIsNone(fit_blend([]));self.assertEqual(select(paired,NOW),[])


class ReviewAndGradingTests(unittest.TestCase):
    def test_dnp_is_not_zero_and_integer_push_grading(self):
        r=quote(market='player_goals',line=1,player_id=5);g=game(1,'2026-09-26')
        self.assertEqual(settle(r,g),'unresolved_participation')
        self.assertEqual(settle(r,g,participation=False),'void')
        self.assertEqual(settle(r,g,dict(game_id=1,player_id=5,goals=1)),'push')
        with self.assertRaises(ValueError):settle(r,g,dict(game_id=1,player_id=6,goals=1))
        self.assertEqual(settle(quote(line=5),g),'push')
        self.assertEqual(settle(quote(market='spreads',side='Home',line=-1.5),g),'lost')

    def test_clv_does_not_compare_changed_lines_or_poststart_quotes(self):
        r=dict(quote(),book_probability=.5);c=dict(r,quoted_at='2026-09-26T22:45:00Z',other_book_probability=.55)
        self.assertEqual(closing_value(r,c)['status'],'same_line_conditional_on_non_push')
        self.assertEqual(closing_value(r,dict(c,line=6.5))['status'],'line_or_contract_changed')
        self.assertEqual(closing_value(r,dict(c,quoted_at=r['commence_time']))['status'],'no_verified_close')

    def test_analyst_override_preserves_original_and_requires_provenance(self):
        r=dict(quote(),offer_id='o',forecast_id='f',independent_probability=.55,final_probability=.55,push_probability=0)
        review=dict(offer_id='o',forecast_id='f',analyst='Test',source_url='https://example.test/source',
            source_published_at='2026-09-26T13:00:00Z',recorded_at='2026-09-26T14:00:00Z',reason='Goalie announcement',
            kind='goalie',status='reviewed',represented_in=['market'],override_probability=.51,
            adjustment_method='Documented manual scenario',double_counting_check='Independent model does not use starter identity')
        out=apply_review(r,validate_review(review,NOW))
        self.assertEqual(out['final_probability'],.55);self.assertEqual(out['analyst_probability'],.51)
        review['recorded_at']='2026-09-27T00:00:00Z'
        with self.assertRaises(ValueError):validate_review(review,NOW)

    def test_metrics_staking_push_drawdown_and_worse_execution(self):
        rows=[dict(quote(),offer_id=str(i),nhl_game_id=i,result=s,price=100) for i,s in enumerate(['won','lost','push'])]
        m=betting_metrics(rows);w=betting_metrics(rows,.05)
        self.assertEqual(m['turnover'],3);self.assertEqual(m['net_units'],0);self.assertEqual(m['max_drawdown'],1)
        self.assertLess(w['net_units'],m['net_units'])


class TrackRecordTests(unittest.TestCase):
    def test_held_out_bins_for_every_market(self):
        from nhl.v2 import VERSION
        from nhl.v2.track import build, merged
        data=build()
        self.assertEqual(data['model_version'],VERSION)
        self.assertEqual(set(data['markets']),{'player_shots_on_goal','player_goals','player_assists','player_points','h2h','totals','spreads'})
        shots=data['markets']['player_shots_on_goal']
        self.assertEqual(sum(b['n'] for b in shots['calibration_bins']),shots['forecasts'])
        self.assertTrue(all(b['n']>=100 for b in shots['calibration_bins']),'no dot rests on a handful of games')
        self.assertEqual(merged([dict(forecast=.1,observed=.2,count=60),dict(forecast=.3,observed=.2,count=60),dict(forecast=.5,observed=.5,count=10)]),
                         [dict(predicted=.2231,observed=.2231,n=130)])


class SiteContractTests(unittest.TestCase):
    def test_broken_artifact_clears_partial_forecast_but_keeps_quote(self):
        r=dict(quote(),independent_probability=.7,estimated_ev=.1,fair_odds=-150,forecast_id='old')
        state=dict(rows=[r])
        with patch('nhl.v2.inference.bundle',side_effect=AttributeError('incompatible pickle')):
            out=enrich(state,NOW)
        self.assertIsNone(out['rows'][0]['independent_probability']);self.assertIsNone(out['rows'][0]['estimated_ev'])
        self.assertIsNone(out['rows'][0]['forecast_id']);self.assertEqual(out['rows'][0]['price'],-110)
        self.assertIn('AttributeError',out['model_error'])

    def test_live_adapter_models_all_markets_independently_and_fails_on_stale_inputs(self):
        past=[game(1,'2026-09-24'),game(2,'2026-09-25')]
        players=[dict(game_id=g['game_id'],game_date=g['game_date'],available_at=g['available_at'],player_id=5,player='Test',
            position='C',toi=20,shots=3,goals=0,assists=1,points=1) for g in past]
        tr,pr=build(past,players);tm=TeamModel('rate').fit(tr,past);pm=PlayerModel('opportunity_nb').fit(pr)
        models=dict(team=tm,shots=pm,scoring=pm)
        manifest=dict(trained_through='2026-04-16',artifact_sha256='test',validation_status='experimental')
        rows=[quote(market=m,line=2.5 if m=='player_shots_on_goal' else .5,player='Test') for m in PROPS]
        rows += [quote(line=6.5),quote(market='h2h',line=None,side='Home'),quote(market='spreads',line=-1.5,side='Home')]
        for r in rows:r['nhl_game_id']=3
        events=[dict(nhl_game_id=3,home_id=1,away_id=2)]
        rows=compare(rows,NOW)
        out=annotate(copy.deepcopy(rows),past,players,events,models,manifest,NOW,NOW.isoformat())
        self.assertEqual(len(out),7);self.assertTrue(all(r['model_probability'] is not None for r in out))
        context=out[0]['player_context']
        self.assertEqual([g['date'] for g in context['games']],['2026-09-25','2026-09-24'],'recent games, newest first')
        self.assertNotIn('opp',[c[0] for c in context['game_columns']],'no opponent without team abbreviations')
        labeled=[dict(r,home=True,team_abbrev='HOM') for r in players]+[dict(r,player_id=9,player='Other',home=False,team_abbrev='AWY') for r in players]
        context=annotate(copy.deepcopy(rows),past,labeled,events,models,manifest,NOW,NOW.isoformat())[0]['player_context']
        self.assertEqual(context['games'][0]['opp'],'vs AWY')
        self.assertTrue(all(not r['recommendation'] for r in out))
        changed=copy.deepcopy(rows)
        for r in changed:r['price']=250
        again=annotate(changed,past,players,events,models,manifest,NOW,NOW.isoformat())
        self.assertEqual([r['independent_probability'] for r in out],[r['independent_probability'] for r in again])
        with self.assertRaises(ValueError):annotate(rows,past,players,events,models,manifest,NOW,(NOW-timedelta(hours=37)).isoformat())
        ambiguous=players+[dict(players[-1],player_id=6)]
        out=annotate(copy.deepcopy(rows),past,ambiguous,events,models,manifest,NOW,NOW.isoformat())
        self.assertTrue(all(r['model_probability'] is None for r in out[:4]))

    def test_all_seven_markets_and_required_row_keys_preserved(self):
        self.assertEqual(set(MARKETS),{'h2h','totals','spreads','player_shots_on_goal','player_goals','player_assists','player_points'})
        event=dict(id='e',nhl_game_id=1,commence_time='2026-09-26T23:00:00Z',home_team='Home',away_team='Away',
            bookmakers=[dict(key='draftkings',title='DraftKings',markets=[dict(key='totals',last_update='2026-09-26T14:29:00Z',
            outcomes=[dict(name=s,point=6.5,price=-110) for s in ['Over','Under']])])])
        r=compare(flatten(event,NOW),NOW)[0]
        required={'event_id','commence_time','home_team','away_team','game','book','book_label','market','market_label','player','side','line','price','book_probability','quoted_at',
                  'fair_probability','consensus_probability','paired_books','best_price','other_book_probability','other_books','consensus_ev'}
        self.assertLessEqual(required,set(r));self.assertEqual(r['nhl_game_id'],1)

    def test_routes_filters_and_additive_forecast_panel(self):
        from nhl.site import build as site_build
        with tempfile.TemporaryDirectory() as d,patch('nhl.site.ROOT',Path(d)):
            # metadata expects repository-relative docs paths, so stub only the SEO helper.
            with patch('nhl.site.metadata',return_value=''):site_build(dict(season=20262027))
            for route in ['index.html','props/index.html','totals/index.html','top.html','methods.html']:
                page=(Path(d)/'docs/nhl'/route).read_text();self.assertIn('data-nhl-page=',page)
                if route in ['props/index.html','totals/index.html','top.html']:
                    for element in ['search','market','book','game','best','reset','more']:self.assertIn(f'id="{element}"',page)


if __name__=='__main__':unittest.main()
