"""Official, dated statistical context for requested MLB postseason stories."""
from datetime import timedelta
import json
from pathlib import Path
import re
import requests
import editorial as ed

API='https://statsapi.mlb.com/api/v1/'


def applies(idea):
    return idea.get('sport')=='MLB' and bool(re.search(r'\b(?:wild\s*cards?|playoffs?|postseason)\b',idea.get('idea',''),re.I))


def get(path,params):
    response=requests.get(API+path,params=params,timeout=30)
    response.raise_for_status()
    return response.json(),response.url


def projected_pairs(standings):
    pairs=[]
    for league in (103,104):
        rows=[r for group in standings.get('records',[]) if group.get('league',{}).get('id')==league for r in group.get('teamRecords',[])]
        winners=sorted([r for r in rows if r.get('divisionRank')=='1'],key=lambda r:int(r['leagueRank']))
        wild=sorted([r for r in rows if r.get('wildCardRank') in ('1','2','3')],key=lambda r:int(r['wildCardRank']))
        if len(winners)!=3 or len(wild)!=3:raise ValueError('Complete MLB playoff standings are unavailable')
        for home,away,seed in ((winners[2],wild[2],3),(wild[0],wild[1],4)):
            pairs.append((league,seed,home,away))
    return pairs


def record(row):
    splits={r['type']:r for r in row.get('records',{}).get('splitRecords',[])}
    if not all(k in splits for k in ('home','away')):raise ValueError('MLB home/away records are unavailable')
    return {'team_id':row['team']['id'],'team':row['team']['name'],'season_record':f"{row['wins']}-{row['losses']}",
        **{k+'_record':f"{splits[k]['wins']}-{splits[k]['losses']}" for k in ('home','away')}}


def completed_games(games,through):
    """One result per completed regular-season game, in chronological order."""
    results={}
    for date in games.get('dates',[]):
        for game in date.get('games',[]):
            day=game.get('officialDate',date.get('date',''))
            sides=game.get('teams',{})
            if not day or day>through or game.get('gameType')!='R' or game.get('status',{}).get('abstractGameState')!='Final':continue
            if not all(sides.get(s,{}).get('team',{}).get('id') for s in ('home','away')):continue
            scores=[sides[s].get('score') for s in ('home','away')]
            if any(type(v) is not int or v<0 for v in scores) or scores[0]==scores[1]:continue
            results[game['gamePk']]=dict(game,officialDate=day)
    return sorted(results.values(),key=lambda g:(g['officialDate'],g.get('gameNumber',1),g.get('gameDate',''),g['gamePk']))


def recent_form(team_id,games,venue=None,limit=10):
    selected=[]
    for game in games:
        sides=game['teams']
        side=next((s for s in ('home','away') if sides[s]['team']['id']==team_id),None)
        if side and (venue is None or venue==side):
            selected.append((game,sides[side]['score'],sides['away' if side=='home' else 'home']['score']))
    selected=selected[-limit:]
    if not selected:raise ValueError('Recent completed MLB results are unavailable')
    wins=sum(scored>allowed for _,scored,allowed in selected)
    scored=sum(s for _,s,_ in selected);allowed=sum(a for _,_,a in selected)
    return {'games':len(selected),'from':selected[0][0]['officialDate'],'through':selected[-1][0]['officialDate'],
        'wins':wins,'losses':len(selected)-wins,'runs_for':scored,'runs_against':allowed,'run_differential':scored-allowed}


def matchup(league,seed,home,away,games,through):
    h,a=record(home),record(away);seen=set();wins={h['team_id']:0,a['team_id']:0};runs=dict(wins)
    for date in games.get('dates',[]):
        for game in date.get('games',[]):
            sides=game.get('teams',{})
            ids={side.get('team',{}).get('id') for side in sides.values()}
            if ids!=set(wins) or game.get('gameType')!='R' or game.get('status',{}).get('abstractGameState')!='Final':continue
            if game.get('officialDate',date.get('date',''))>through or game['gamePk'] in seen:continue
            scores={side['team']['id']:side.get('score') for side in sides.values()}
            if any(not isinstance(v,int) for v in scores.values()) or len(set(scores.values()))!=2:continue
            seen.add(game['gamePk']);winner=max(scores,key=scores.get);wins[winner]+=1
            for team,value in scores.items():runs[team]+=value
    return {'id':f'stats-{league}-{seed}','league':'AL' if league==103 else 'NL','higher_seed':seed,'lower_seed':9-seed,
        'higher_seed_team':h,'lower_seed_team':a,'status':'Projected from dated standings, not a confirmed postseason matchup',
        'head_to_head':{'games':len(seen),'higher_seed_wins':wins[h['team_id']],'lower_seed_wins':wins[a['team_id']],
            'higher_seed_runs':runs[h['team_id']],'lower_seed_runs':runs[a['team_id']]}}


def archive_context(root,series):
    """Inspect only retained editorial evidence; never regenerate historical forecasts."""
    names=[s['team'].lower() for s in (series['higher_seed_team'],series['lower_seed_team'])]
    rows=[];files=0;seen=set()
    for path in sorted((Path(root)/'docs/editorial/evidence').glob('*.json')):
        packet=json.loads(path.read_text())
        if packet.get('sport')!='MLB':continue
        files+=1
        for row in packet.get('model_rows',[])+packet.get('model_references',[]):
            teams=[p.strip().lower() for p in row.get('game','').split('@')]
            if len(teams)!=2 or not all(any(t.endswith(name) for t in teams) for name in names):continue
            identity=(row.get('event_id'),row.get('player'),row.get('market'),row.get('model_version'))
            if identity in seen or not row.get('model_version') or row.get('model_mean') is None:continue
            seen.add(identity)
            rows.append({k:row.get(k) for k in ('game','player','market','model_mean','model_version','model_input_through')})
    return {'scope':'Retained published editorial evidence snapshots; raw workflow artifacts are not indexed here',
        'snapshots_checked':files,'matched_forecasts':rows[:2],
        'status':'Stored matchup estimates found; historical context only' if rows else 'No matching historical forecast in the inspected editorial archive'}


def requested_team(standings,idea):
    text=idea.get('idea','').split('\n\nChanges for the next draft:\n')[0].lower()
    found=[]
    for group in standings.get('records',[]):
        for row in group.get('teamRecords',[]):
            name=row['team']['name'].lower()
            short=' '.join(name.split()[-2:]) if name.endswith('sox') else name.split()[-1]
            if any(re.search(r'(?<![a-z])'+re.escape(term)+r'(?![a-z])',text) for term in (name,short)):
                found.append((group,row))
    if len(found)>1:raise ValueError('A team playoff outlook needs one clearly named team')
    return found[0] if found else None


def playoff_outlook(group,row,standings,games,through,now):
    required=('eliminationNumber','wildCardEliminationNumber','clinched')
    if any(k not in row for k in required):raise ValueError('Official playoff status is unavailable')
    team=record(row);identifier=team['team_id'];season=int(through[:4])
    schedule,url=get('schedule',{'sportId':1,'gameType':'R','teamId':identifier,'season':season,
        'startDate':(now.astimezone(ed.ETZ).date()).isoformat(),'endDate':f'{season}-12-31'})
    remaining={}
    for date in schedule.get('dates',[]):
        for game in date.get('games',[]):
            if game.get('gameType')!='R' or game.get('status',{}).get('abstractGameState')=='Final' or game.get('status',{}).get('detailedState')=='Cancelled':continue
            sides=game['teams'];home=sides['home']['team']['id']==identifier
            remaining[game['gamePk']]={'date':game.get('officialDate',date['date']),'venue':'home' if home else 'away',
                'opponent':sides['away' if home else 'home']['team']['name']}
    league=group['league']['id']
    leaders=[t for t in group['teamRecords'] if t.get('divisionRank')=='1']
    wild=[t for g in standings['records'] if g['league']['id']==league for t in g['teamRecords'] if t.get('wildCardRank')=='3']
    if len(leaders)!=1 or len(wild)!=1:raise ValueError('Complete division and Wild Card standings are unavailable')
    eliminated=row['eliminationNumber']=='E' and row['wildCardEliminationNumber']=='E'
    team.update(id='stats-team-'+str(identifier),league='AL' if league==103 else 'NL',
        wins=row['wins'],losses=row['losses'],games_played=row['gamesPlayed'],
        division_rank=row['divisionRank'],division_games_back=row['divisionGamesBack'],
        wild_card_rank=row.get('wildCardRank'),wild_card_games_back=row['wildCardGamesBack'],
        postseason_status='eliminated' if eliminated else 'clinched' if row['clinched'] else 'not_clinched',
        division_elimination_number=row['eliminationNumber'],wild_card_elimination_number=row['wildCardEliminationNumber'],
        last_10=recent_form(identifier,completed_games(games,through)),
        division_leader=record(leaders[0]),third_wild_card=record(wild[0]),
        remaining_scheduled_games=list(remaining.values()))
    return team,url


def build(now,root,idea=None):
    through=(now.astimezone(ed.ETZ).date()-timedelta(days=1)).isoformat();season=int(through[:4])
    standings,standings_url=get('standings',{'leagueId':'103,104','season':season,'standingsTypes':'regularSeason','date':through})
    games,games_url=get('schedule',{'sportId':1,'gameType':'R','season':season,'startDate':f'{season}-03-01','endDate':through})
    matched=requested_team(standings,idea) if idea else None
    if matched:
        team,schedule_url=playoff_outlook(*matched,standings,games,through,now)
        return {'scope':'mlb_team_playoff_outlook','through':through,'checked_at':now.isoformat(),'series':[team],
            'official_sources':[{'id':'stats-standings','title':'MLB official dated standings and playoff elimination status','url':standings_url,'published_at':through},
                {'id':'stats-games','title':'MLB completed regular-season results','url':games_url,'published_at':through},
                {'id':'stats-remaining','title':'MLB remaining regular-season schedule','url':schedule_url,'published_at':now.astimezone(ed.ETZ).date().isoformat()}],
            'limitations':'Official elimination status answers whether a playoff path remains. E in both elimination fields means eliminated; never describe such a team as still chasing a berth. No simulated playoff probability, futures price, causal explanation or tiebreaker conclusion is supplied. Remaining scheduled games can change. Recent form and venue records are descriptive, not calibrated forecasts or run-line cover rates.'}
    if idea and not re.search(r'\bwild\s*cards?\b|\b(?:playoff|postseason)\s+(?:picture|bracket|overview)\b',idea.get('idea',''),re.I):
        raise ValueError('No unambiguous MLB team resolved for the requested playoff outlook')
    series=[matchup(*pair,games,through) for pair in projected_pairs(standings)]
    completed=completed_games(games,through)
    for row in series:
        row['historical_models']=archive_context(root,row)
        for key,venue in (('higher_seed_team','home'),('lower_seed_team','away')):
            team=row[key]
            team['last_10']=recent_form(team['team_id'],completed)
            team['last_10_'+venue]=recent_form(team['team_id'],completed,venue=venue)
    return {'scope':'mlb_wildcard_overview','through':through,'checked_at':now.isoformat(),'series':series,
        'official_sources':[{'id':'stats-standings','title':'MLB official dated standings and home/away records','url':standings_url,'published_at':through},
            {'id':'stats-games','title':'MLB completed regular-season game results','url':games_url,'published_at':through}],
        'limitations':'Pairings are provisional. Recent form is the last up to 10 completed regular-season games overall and at the projected venue, with dates and sample sizes. These overlapping windows, head-to-head and season records are descriptive, not calibrated or opponent-adjusted forecasts. Scores do not establish historical run-line covers, ATS percentages or returns. No postseason series prices are supplied. Do not substitute current regular-season odds or manufacture missing historical forecasts.'}
