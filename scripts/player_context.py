"""Reader-facing facts from the exact pregame history and selected model inputs.

This module describes a forecast. It does not build features, price bets or select
picks. Observed averages are kept separate from prior-adjusted model inputs.
"""
import math
import re

# MLB Stats API team ids. Used only to label the opponent in a game log.
MLB_TEAMS={108:'LAA',109:'AZ',110:'BAL',111:'BOS',112:'CHC',113:'CIN',114:'CLE',115:'COL',116:'DET',117:'HOU',
           118:'KC',119:'LAD',120:'WSH',121:'NYM',133:'ATH',134:'PIT',135:'SD',136:'SEA',137:'SF',138:'STL',
           139:'TB',140:'TEX',141:'TOR',142:'MIN',143:'PHI',144:'ATL',145:'CWS',146:'MIA',147:'NYY',158:'MIL'}
GAME_LOG=5
TREND=10


def describe(builder, *args):
    """Optional display context must never change forecast availability."""
    try:
        return builder(*args)
    except (KeyError, TypeError, ValueError, ArithmeticError, AttributeError):
        return None


def metric(label, value, unit='', detail='', used=False):
    return dict(label=label, value=round(float(value), 4) if value is not None and math.isfinite(value) else None,
                unit=unit, detail=detail, used=used)


def average(records, field, divisor=1):
    values=[r[field] for r in records if isinstance(r.get(field), (int, float)) and math.isfinite(r[field])]
    return sum(values)/len(values)/divisor if values else None


def windows(records, sizes, stat, workload, workload_divisor=1, extra=None):
    result=[]
    for size in sizes:
        sample=records[-size:]
        if not sample or any(r['games']==len(sample) for r in result):
            continue
        result.append(dict(games=len(sample), mean=average(sample,stat),
                           workload=average(sample,workload,workload_divisor)))
        if extra:
            result[-1][extra[0]]=average(sample,extra[1])
    return result


def count(value):
    return int(round(value)) if isinstance(value,(int,float)) and math.isfinite(value) else None


def innings(outs):
    """Innings in thirds: 17 outs is 5⅔ (box-score 5.2)."""
    outs=count(outs)
    return None if outs is None else f"{outs//3}{['','⅓','⅔'][outs%3]}"


def clock(minutes):
    seconds=count(minutes*60) if isinstance(minutes,(int,float)) else None
    return None if seconds is None else f'{seconds//60}:{seconds%60:02}'


def versus(home, team):
    return ('vs ' if home else '@ ')+team if team else None


def game_log(records, columns, row):
    """The latest games, newest first. A column no game reports is left out."""
    games=[row(r) for r in reversed(list(records)[-GAME_LOG:])]
    return games,[c for i,c in enumerate(columns) if i==0 or any(g.get(c[0]) is not None for g in games)]


# ---- How the model reads a player: small, display-only explanations. ----
# Each piece is optional. A failure here drops that piece, never the forecast.

def num(value, places=3):
    return round(float(value), places) if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None


def trend(records, date, value, label, note, weight=None, opp=None, workload=None):
    """Up to ten latest games, oldest first: [date, value, model weight, opponent, workload]."""
    return dict(label=label, note=note, rows=[[r[date], num(value(r)), num(weight(r)) if weight else None,
                                               opp(r) if opp else None, num(workload(r),2) if workload else None]
                                              for r in list(records)[-TREND:]])


def distribution(mass, tail=.005):
    """Chances of each count, trimmed to the central 99%; the end bars carry the trimmed tails."""
    p=[max(0.,float(x)) for x in mass]
    total=sum(p)
    if not total>0:
        return None
    p=[x/total for x in p]
    start,cum=0,0.
    while start<len(p)-1 and cum+p[start]<=tail:
        cum+=p[start]; start+=1
    end,cum=len(p)-1,0.
    while end>start and cum+p[end]<=tail:
        cum+=p[end]; end-=1
    shown=p[start:end+1]
    shown[0]+=sum(p[:start]); shown[-1]+=sum(p[end+1:])
    return dict(start=start, p=[round(x,4) for x in shown], low=start>0, high=end<len(p)-1)


def step(label, value, unit='', op=None):
    return dict(label=label, value=num(value,4), unit=unit, op=op)


def ordinal(n):
    return f"{n}{'th' if 10<=n%100<=20 else {1:'st',2:'nd',3:'rd'}.get(n%10,'th')}"


def ranked(table, team, key, word):
    """Rank 1 is the largest value: '4th most of 32'."""
    order=sorted(table, key=lambda t:-table[t][key])
    if team not in table:
        return None
    place=order.index(team)+1
    return f'{word.capitalize() if place==1 else ordinal(place)+" "+word} of {len(order)}'


def opposing(table, team, key, label, unit='', scale=1, word='highest', places=2, used=False):
    league=sum(t[key] for t in table.values())/len(table) if table else None
    value=table[team][key] if team in table else None
    return dict(label=label, value=num(value*scale if value is not None else None,places), unit=unit,
                league=num(league*scale if league is not None else None,places), rank=ranked(table,team,key,word), used=used)


def meetings(rows, date, value, team, home=None):
    """Display only, never a model input: a player's results against tonight's opponent.

    `rows` are his earlier games against that team, oldest first. Every value is kept with its
    venue (True at home, False away, None unknown) so the snapshot can show the overall average,
    the home and away averages, and over/under counts at tonight's line.
    """
    rows=[r for r in rows if num(value(r)) is not None]
    if not rows or not team:
        return None
    venue=lambda r:None if home is None or home(r) is None else bool(home(r))
    return dict(team=team,values=[num(value(r),2) for r in rows],home=[venue(r) for r in rows],since=date(rows[0]),
                note='History only, not a model input. In our backtests, results against one opponent did not predict the next meeting beyond a player’s overall form.')


def blend(label, own, detail=''):
    """The share of an estimate that comes from the player's own games; the rest is the starting average."""
    return dict(label=label, own=num(min(max(own,0),1)), detail=detail)


# How each model version fades a player's older games (features.py of that version).
NHL_HALF_LIVES={'nhl-v2.1':dict(rate=120,toi=30,by='days'),'nhl-v2.2':dict(rate=110,toi=14,by='games'),
                'nhl-v2.3':dict(rate=110,toi=14,by='games'),'nhl-v2.4':dict(rate=110,toi=14,by='games')}
OPPONENT_KINDS=('opportunity_nb_opp','opportunity_nb_opp_player')


def nhl_player_factor(features, j):
    """nhl-v2.4: (factor, actual, forecast) from the player's record against his own forecasts."""
    factor=(features.get('player_factors') or [None,None])[0 if j==0 else 1]
    col=0 if j==0 else 3
    actual,expected=features.get('player_actual'),features.get('player_expected')
    return factor,(actual[col] if actual else None),(expected[col] if expected else None)


def nhl_explain(records, features, j, model_kind, version, opponent, day, pmf, matchup):
    from datetime import date as calendar
    stat=['shots','goals','assists','points'][j]
    label=['SOG','goals','assists','points'][j]
    out={}
    toi=features['projected_toi']
    if model_kind=='rate_poisson':
        out['build']=dict(steps=[step('Weighted average per game',features['base_means'][j],label)],
            note='This version uses the player’s recency-weighted average per game, then turns it into the chance of each count.')
    else:
        mean=features['opportunity_means'][j]
        per60=['Shots on goal','Goals','Assists','Points'][j]+' per 60 minutes'
        steps=[step('Projected ice time',toi,'min'),step(per60,mean/toi*60 if toi else None,'','×')]
        adjusted=(features.get('adjusted_means') or [None]*4)[j]
        if model_kind in OPPONENT_KINDS and adjusted is not None and mean:
            own,actual,expected=nhl_player_factor(features,j) if model_kind=='opportunity_nb_opp_player' else (None,None,None)
            own=own if isinstance(own,(int,float)) and own>0 else None
            steps+=[step('Before the opponent',mean,label,'='),
                    step('Opponent '+('shots' if j==0 else 'goals')+' allowed vs. long-run league average',adjusted/mean/(own or 1),'×','×')]
            if own is not None:
                steps.append(step('His results vs. our past forecasts',own,'×','×'))
            steps.append(step('Expected '+('shots' if j==0 else label),adjusted,label,'='))
            note=('Ice time × production per 60 minutes, both recency-weighted and blended with a position average, then scaled by how many '
                  +('shots' if j==0 else 'goals')+' the opponent allows compared with the league.'
                  +(f" Then adjusted for how he has done against our own forecasts: {num(actual,0):g} {'shots' if j==0 else 'points'} against {num(expected,1):g} forecast in his earlier games, shrunk toward no adjustment."
                    if own is not None and actual is not None and expected is not None else '')
                  +' The model then turns this average into the chance of each count.')
        else:
            steps.append(step('Expected '+('shots' if j==0 else label),mean,label,'='))
            note='Ice time × production per 60 minutes, both recency-weighted and blended with a position average. The model then turns this average into the chance of each count.'
        out['build']=dict(steps=steps,note=note)
    if pmf is not None:
        out['distribution']=distribution(pmf)
    life=NHL_HALF_LIVES.get(version)
    target=calendar.fromisoformat(day) if day else None
    if life and (target or life['by']=='games'):
        if life['by']=='games':
            # Newest appearance = 0: an offseason or injury break does not fade his history.
            ages=lambda half:[2**(-(len(records)-1-i)/half) for i in range(len(records))]
            unit='of his games'
        else:
            ages=lambda half:[2**(-(target-calendar.fromisoformat(r['game_date'])).days/half) for r in records]
            unit='days'
        rate,minutes=ages(life['rate']),ages(life['toi'])
        weight=dict(zip((id(r) for r in records),rate))
        note=f"Faded games count less. A game’s weight halves every {life['rate']} {unit} for production and every {life['toi']} {unit} for ice time."
        out['blend']=[blend('Production rate',sum(rate)/(sum(rate)+12),'Recency-weighted games; the rest is the position average'),
                      blend('Ice time',sum(minutes)/(sum(minutes)+5),'Recency-weighted games; the rest is the position average')]
    else:
        weight,note=None,'Recent games shown without model weights.'
    out['trend']=trend(records,'game_date',lambda r:r.get(stat),label,note,
                       weight=(lambda r:weight[id(r)]) if weight else None,opp=opponent,workload=lambda r:r.get('toi'))
    adjusted_kind=model_kind in OPPONENT_KINDS
    if matchup:
        # The opponent-adjusted model reads shots allowed for shots and goals allowed for scoring.
        out['opponent']=dict(matchup,items=[dict(item,used=adjusted_kind and (i==0)==(j==0)) for i,item in enumerate(matchup.get('items',[]))])
    out['missing']=(['The starting goalie (team defense blends its goalies)','Linemates and power-play role','Injuries and late lineup changes']
                    if adjusted_kind else ['Opponent defense and goalie','Linemates and power-play role','Injuries and late lineup changes'])
    return out


def nhl_context(records, features, market_index, model_kind, version, opponent=None, day=None, pmf=None, matchup=None, meeting=None):
    records=list(records)
    stat=['shots','goals','assists','points'][market_index]
    label=['SOG','goals','assists','points'][market_index]
    opportunity=model_kind!='rate_poisson'
    toi=features['projected_toi']
    inputs=[metric('Projected ice time', toi, 'min', 'Prior-adjusted workload estimate', opportunity),
            metric('Weighted '+label+' / 60', features['opportunity_means'][market_index]/toi*60 if toi else None,
                   '', 'Prior-adjusted production per 60 minutes', opportunity)]
    if not opportunity:
        inputs.append(metric('Weighted '+label+' / game', features['base_means'][market_index], '',
                             'Prior-adjusted per-game mean', True))
    if model_kind in OPPONENT_KINDS:
        key='opp_shots_against' if market_index==0 else 'opp_goals_against'
        inputs.append(metric('Opponent '+('shots' if market_index==0 else 'goals')+' allowed / game', features.get(key), '',
                             'Recency-weighted; the same figure the game-line model uses', True))
    if model_kind=='opportunity_nb_opp_player':
        own,actual,expected=nhl_player_factor(features,market_index)
        if own is not None:
            inputs.append(metric('His results vs. our forecasts', own, '×',
                                 f"{num(actual,0):g} actual vs. {num(expected,1):g} forecast {'shots' if market_index==0 else 'points'} in earlier games; shrunk toward 1"
                                 if actual is not None and expected is not None else 'No earlier forecasts; no adjustment', True))
    # Describe only versions whose weighting contract is known here.
    weighting={
        'nhl-v2.1':'Newer games carry more weight. Ice time has a 30-day half-life; production per minute has a 120-day half-life. Position priors retain weight, including after the offseason.',
        'nhl-v2.2':'Newer appearances carry more weight. Ice time has a 14-appearance half-life; production per minute has a 110-appearance half-life. The offseason does not age player history.',
        'nhl-v2.3':'Newer appearances carry more weight. Ice time has a 14-appearance half-life; production per minute has a 110-appearance half-life. The offseason does not age player history. Expected shots scale with the opponent’s shots allowed, and scoring with its goals allowed, relative to the league.',
        'nhl-v2.4':'Newer appearances carry more weight. Ice time has a 14-appearance half-life; production per minute has a 110-appearance half-life. The offseason does not age player history. Expected shots scale with the opponent’s shots allowed, and scoring with its goals allowed, relative to the league. A player who has consistently beaten (or missed) our earlier forecasts gets a shrunk adjustment toward that record.',
    }.get(version, 'Recency-weighted estimates include position priors. See the methods for this model version.')
    games,columns=game_log(records,[['date','Date'],['opp','Opp'],['toi','TOI'],['shots','SOG'],['goals','G'],
                                    ['assists','A'],['points','P']],
        lambda r:dict(date=r['game_date'],opp=opponent(r) if opponent else None,toi=clock(r.get('toi')),
                      shots=count(r.get('shots')),goals=count(r.get('goals')),assists=count(r.get('assists')),
                      points=count(r.get('points'))))
    return dict(schema_version=1, source='NHL completed-game logs', through=features['last_game'],
                sample_games=len(records), sample_label='appearances', stat_label=label,
                workload_label='Ice time', workload_unit='min',
                recent=windows(records,[5,10,20],stat,'toi'), inputs=inputs,
                games=games, game_columns=columns, game_focus=stat,
                **(describe(nhl_explain,records,features,market_index,model_kind,version,opponent,day,pmf,matchup) or {}),
                versus=meeting,
                note=weighting+' Observed averages below are unweighted. Participation and current role remain unconfirmed.')


_mlb_teams={}


def mlb_league(history, day):
    """Every team's pregame rates on this date, for opponent comparisons and ranks."""
    key=(id(history),day)
    if key not in _mlb_teams:
        _mlb_teams.clear()
        teams={tid:history.team(tid,day) for tid in list(history.teams)}
        _mlb_teams[key]={tid:t for tid,t in teams.items() if t['team_games']>=10}
    return _mlb_teams[key]


def mlb_explain(history, day, player, market, features, model, mass, game, records, used):
    from mlb.models import baseline, means
    pitching=market.startswith('pitcher_')
    x=features
    out={}
    final=float(sum(i*p for i,p in enumerate(mass))) if mass is not None else None
    raw=float(means(model,[{'x':x}])[0])
    simple=baseline(x,market)
    label={'pitcher_strikeouts':'K','pitcher_outs':'outs','batter_hits':'hits','batter_total_bases':'total bases',
           'batter_home_runs':'HR','batter_rbis':'RBI'}[market]
    if market=='pitcher_strikeouts':
        steps=[step('Batters faced per start',x['starter_bf']),step('Strikeout rate',x['starter_k_rate']*100,'%','×'),
               step('Opponent adjustment',(x['opp_k_rate']/.225)**.5,'×','×'),step('Simple estimate',simple,label,'=')]
    elif market=='pitcher_outs':
        steps=[step('Recent outs per start',simple,label)]
    else:
        rate={'batter_hits':'Hits','batter_total_bases':'Total bases','batter_home_runs':'Home runs','batter_rbis':'RBIs'}[market]
        key={'batter_hits':'hit','batter_total_bases':'tb','batter_home_runs':'hr','batter_rbis':'rbi'}[market]
        steps=[step('Expected plate appearances',x['projected_pa'],'PA'),step(rate+' per plate appearance',x['batter_'+key+'_rate'],'','×'),
               step('Simple estimate',simple,label,'=')]
    if model['kind']!='rolling':
        steps.append(step(f"Machine-learning model ({len(model.get('features',[]))} inputs)",raw,label,'→'))
    if final is not None:
        steps.append(step('Adjusted to past results',final,label,'→'))
    out['build']=dict(steps=steps,note=(
        'This market uses the simple estimate directly. ' if model['kind']=='rolling' else
        'The machine-learning model starts from the same inputs and adds the opposing starter, bullpen, team form and ballpark. ')
        +'The model then turns its average into the chance of each outcome, adjusted so past forecasts matched how often things happened.')
    if mass is not None:
        out['distribution']=distribution(mass)
    stat={'pitcher_strikeouts':'strikeOuts','pitcher_outs':'outs'}.get(market) or {'batter_hits':'hits','batter_total_bases':'totalBases',
          'batter_home_runs':'homeRuns','batter_rbis':'rbi'}[market]
    opp=lambda r:versus(r.get('at_home'),MLB_TEAMS.get(r.get('opponent_id')))
    out['trend']=trend(records,'date',lambda r:r.get(stat),label,
        'Every start in the window counts equally: up to 15 for rates and the last 5 for workload.' if pitching else
        'Every game in the window counts equally: up to 60 games for rates.',
        opp=opp,workload=(lambda r:r.get('numberOfPitches')) if pitching else (lambda r:r.get('plateAppearances')))
    if pitching:
        bf=sum(r.get('battersFaced',0) for r in records)
        recent=len(records[-5:])
        out['blend']=[blend('Strikeout rate',bf/(bf+100),f'{int(bf)} batters faced in his last {len(records)} starts'),
                      blend('Outs per start',recent/(recent+2),f'His last {recent} starts')]
    else:
        pa=sum(r.get('plateAppearances',0) for r in records)
        weight=150 if market=='batter_home_runs' else 100
        name={'batter_hits':'Hit rate','batter_total_bases':'Total-bases rate','batter_home_runs':'Home-run rate','batter_rbis':'RBI rate'}[market]
        out['blend']=[blend(name,pa/(pa+weight),f'{int(pa)} plate appearances in his last {len(records)} games')]
    if game:
        side=player.get('side')
        team=game['away_id'] if side=='home' else game['home_id'] if side=='away' else None
        table=mlb_league(history,day)
        if team in table:
            if pitching:
                items=[opposing(table,team,'k_rate','Strikeout rate',"%",100,'highest',1,'opp_k_rate' in used),
                       opposing(table,team,'runs','Runs per game','',1,'most',2,'opp_runs' in used)]
            else:
                items=[dict(label='Starter runs allowed per 9',value=num(x['opp_starter_ra9'],2),unit='',league=4.3,
                            rank=None,used='opp_starter_ra9' in used),
                       opposing(table,team,'bullpen_ra9','Bullpen runs allowed per 9','',1,'most',2,'opp_bullpen_ra9' in used),
                       dict(label='Ballpark scoring',value=num(x['park_factor'],2),unit='×',league=1.0,rank=None,used='park_factor' in used)]
            out['opponent']=dict(team=MLB_TEAMS.get(team),label='Opposing lineup' if pitching else 'Opposing pitching',items=items)
    out['missing']=(['Weather and umpire','Today’s actual batting order (team rates are used)','Announced pitch limits and injuries']
                    if pitching else ['Left- or right-handed matchups','Weather and umpire','Injuries and late lineup changes'])
    return out


def mlb_context(history, day, player, market, features, model, mass=None, game=None):
    pitching=market.startswith('pitcher_')
    records=history.past(history.pitchers if pitching else history.batters,player['id'],day,15 if pitching else 60)
    stat={'pitcher_strikeouts':'strikeOuts','pitcher_outs':'outs','batter_hits':'hits',
          'batter_total_bases':'totalBases','batter_home_runs':'homeRuns','batter_rbis':'rbi'}[market]
    label={'strikeOuts':'K','outs':'outs','hits':'hits','totalBases':'total bases','homeRuns':'HR','rbi':'RBI'}[stat]
    rolling={
        'pitcher_strikeouts':{'starter_bf','starter_k_rate','opp_k_rate'},
        'pitcher_outs':{'starter_outs5'},
        'batter_hits':{'projected_pa','batter_hit_rate'},
        'batter_total_bases':{'projected_pa','batter_tb_rate'},
        'batter_home_runs':{'projected_pa','batter_hr_rate'},
        'batter_rbis':{'projected_pa','batter_rbi_rate'},
    }
    used=rolling[market] if model['kind']=='rolling' else set(model.get('features',[]))
    def item(key, title, unit='', detail='', multiplier=1):
        value=features.get(key)
        return metric(title, value*multiplier if value is not None else None, unit, detail, key in used)
    if pitching:
        inputs=[item('starter_outs5','Recent innings / start','IP','Last 5 starts; adjusted toward a 15.5-out prior',1/3),
                item('starter_pitches5','Recent pitches / start','','Last 5 starts; adjusted toward an 85-pitch prior'),
                item('starter_k_rate','Pitcher strikeout rate','%','Up to 15 starts; prior-adjusted K / batters faced',100),
                item('opp_k_rate','Opponent strikeout rate','%','Up to 40 team games; prior-adjusted K / PA',100),
                item('starter_bf','Batters faced / start','','Up to 15 starts; prior-adjusted'),
                item('starter_rest','Days since last start','days','Capped at 30 days')]
        note='Recent workload uses up to 5 starts; pitcher rates use up to 15, and opponent rates up to 40 games. Each window is limited to the prior 370 days. Model rates include fixed priors; observed averages do not. Innings are shown in thirds (5⅔ = five innings and two outs).'
    else:
        rate_key={'batter_hits':'hit','batter_total_bases':'tb','batter_home_runs':'hr','batter_rbis':'rbi'}[market]
        inputs=[item('lineup_slot','Published batting slot','','Current starting order'),
                item('projected_pa','Expected plate appearances','PA','Opportunity estimate from batting slot, venue and team scoring'),
                item('recent_pa','Recent PA / start','PA','Up to 20 starts; prior-adjusted'),
                item('batter_'+rate_key+'_rate',label+' per PA','','Up to 60 games; prior-adjusted'),
                item('opp_starter_ra9','Opponent starter RA / 9','','Runs allowed, including unearned runs'),
                item('park_factor','Venue scoring factor','×','Prior-adjusted scoring; 1.00 is neutral')]
        note='Batter rates use up to 60 games from the prior 370 days; recent starting-game opportunity uses up to 20 starts. Model rates include fixed priors. Observed averages include all recorded appearances in the sample, including substitute appearances.'
    opp=lambda r:versus(r.get('at_home'),MLB_TEAMS.get(r.get('opponent_id')))
    if pitching:
        games,columns=game_log(records,[['date','Date'],['opp','Opp'],['ip','IP'],['pitches','Pitches'],['k','K'],
                                        ['bb','BB'],['er','ER']],
            lambda r:dict(date=r['date'],opp=opp(r),ip=innings(r.get('outs')),pitches=count(r.get('numberOfPitches')),
                          k=count(r.get('strikeOuts')),bb=count(r.get('baseOnBalls')),er=count(r.get('earnedRuns'))))
    else:
        games,columns=game_log(records,[['date','Date'],['opp','Opp'],['pa','PA'],['h','H'],['tb','TB'],
                                        ['hr','HR'],['rbi','RBI']],
            lambda r:dict(date=r['date'],opp=opp(r),pa=count(r.get('plateAppearances')),h=count(r.get('hits')),
                          tb=count(r.get('totalBases')),hr=count(r.get('homeRuns')),rbi=count(r.get('rbi'))))
    result=dict(schema_version=1, source='MLB completed-game box scores',
                through=records[-1]['date'] if records else None, sample_games=len(records),
                sample_label='starts' if pitching else 'appearances', stat_label=label,
                workload_label='Innings / start' if pitching else 'PA / game', workload_unit='IP' if pitching else 'PA',
                recent=windows(records,[5,10,15 if pitching else 60],stat,'outs' if pitching else 'plateAppearances',3 if pitching else 1,
                               ('pitches','numberOfPitches') if pitching else None),
                inputs=inputs, note=note, games=games, game_columns=columns,
                game_focus={'strikeOuts':'k','outs':'ip','hits':'h','totalBases':'tb','homeRuns':'hr','rbi':'rbi'}[stat])
    result.update(describe(mlb_explain,history,day,player,market,features,model,mass,game,records,used) or {})
    if game:
        # Display only: his completed games against tonight's opponent in the loaded history.
        side=player.get('side')
        team=game['away_id'] if side=='home' else game['home_id'] if side=='away' else None
        past=(history.pitchers if pitching else history.batters).get(player['id'],[])
        result['versus']=describe(meetings,[r for r in past if team is not None and r.get('opponent_id')==team and r['date']<day],
                                  lambda r:r['date'],lambda r:r.get(stat),MLB_TEAMS.get(team),lambda r:r.get('at_home'))
    return result


NBA_ALIASES={'laclippers':'losangelesclippers'}


def nba_team(name):
    key=''.join(c for c in str(name or '').lower() if c.isalnum())
    return NBA_ALIASES.get(key,key)


def nba_defense(team_games):
    """Each team's last 20 games: what its opponents scored, from paired team game logs."""
    by_game={}
    for games in team_games.values():
        for g in games:
            by_game.setdefault(g.get('GAME_ID'),[]).append(g)
    allowed,abbreviations={},{}
    for pair in by_game.values():
        if len(pair)==2:
            for own,other in [(pair[0],pair[1]),(pair[1],pair[0])]:
                allowed.setdefault(nba_team(own.get('TEAM_NAME')),[]).append(dict(other,GAME_DATE=own.get('GAME_DATE')))
                abbreviations[nba_team(own.get('TEAM_NAME'))]=own.get('TEAM_ABBREVIATION')
    table={team:{k:average(sorted(rows,key=lambda g:str(g['GAME_DATE']))[-20:],k) for k in ['PTS','REB','AST','FG3M','BLK','STL','TOV']}
           for team,rows in allowed.items() if len(rows)>=10}
    return dict(table=table,abbreviations=abbreviations)


def nba_explain(row, games, complete, keys, short, defense, wins, pushes, probability):
    out={}
    n=len(complete)
    out['trend']=trend(complete,'date',lambda g:g['value'],short,
        'No model yet: each recent game (up to 30) counts equally in this historical reference.',opp=lambda g:g['opp'],workload=lambda g:g['minutes'])
    out['build']=dict(steps=[step(f'Average of the last {n} games',sum(g['value'] for g in complete)/n,short),
                             step(f"Games {'over' if row['side']=='Over' else 'under'} {row['line']:g}",wins,f'of {n}'),
                             step('Smoothed hit rate',probability*100,'%','→')],
        note='No forecasting model yet. This is how often the player cleared this line, nudged toward 50% because a few dozen games is a small sample. Pushes are excluded.')
    out['distribution']=dict(empirical=[num(g['value'],1) for g in complete])
    team=nba_team(games[-1].get('TEAM_NAME'))
    sides=[nba_team(row.get('home_team')),nba_team(row.get('away_team'))]
    if defense and team in sides:
        rival=sides[1-sides.index(team)]
        table={t:dict(v=sum(d[k] for k in keys)) for t,d in defense['table'].items() if all(d.get(k) is not None for k in keys)}
        if rival in table:
            label=('Points' if keys==['PTS'] else short)+' allowed per game'
            out['opponent']=dict(team=defense['abbreviations'].get(rival),label='Opposing defense, last 20 games',
                                 items=[opposing(table,rival,'v',label,'',1,'most',1)])
    out['missing']=['A forecasting model (past results only)','Opponent, minutes and role changes','Injuries and rest']
    return out


def nba_matchup(matchup):
    """'BOS vs. NYK' is a home game against New York; 'BOS @ NYK' is away."""
    found=re.search(r'(vs\.|@)\s*([A-Z]{2,4})\s*$',str(matchup or ''))
    return versus(found.group(1)!='@',found.group(2)) if found else None
