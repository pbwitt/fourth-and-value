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


def nhl_context(records, features, market_index, model_kind, version, opponent=None):
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
    # Describe only versions whose weighting contract is known here.
    weighting={
        'nhl-v2.1':'Newer games carry more weight. Ice time has a 30-day half-life; production per minute has a 120-day half-life. Position priors retain weight, including after the offseason.',
        'nhl-v2.2':'Newer appearances carry more weight. Ice time has a 14-appearance half-life; production per minute has a 110-appearance half-life. The offseason does not age player history.',
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
                note=weighting+' Observed averages below are unweighted. Participation and current role remain unconfirmed.')


def mlb_context(history, day, player, market, features, model):
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
    return dict(schema_version=1, source='MLB completed-game box scores',
                through=records[-1]['date'] if records else None, sample_games=len(records),
                sample_label='starts' if pitching else 'appearances', stat_label=label,
                workload_label='Innings / start' if pitching else 'PA / game', workload_unit='IP' if pitching else 'PA',
                recent=windows(records,[5,10,15 if pitching else 60],stat,'outs' if pitching else 'plateAppearances',3 if pitching else 1,
                               ('pitches','numberOfPitches') if pitching else None),
                inputs=inputs, note=note, games=games, game_columns=columns,
                game_focus={'strikeOuts':'k','outs':'ip','hits':'h','totalBases':'tb','homeRuns':'hr','rbi':'rbi'}[stat])


def nba_matchup(matchup):
    """'BOS vs. NYK' is a home game against New York; 'BOS @ NYK' is away."""
    found=re.search(r'(vs\.|@)\s*([A-Z]{2,4})\s*$',str(matchup or ''))
    return versus(found.group(1)!='@',found.group(2)) if found else None
