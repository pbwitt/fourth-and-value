"""Variety first, then distinct matchups; selection never makes paid requests."""


def select(state,sports,discover,limit=2):
    """discover(sport, excluded events) returns one qualifying candidate or None.

    Keep paid attempts and explicit private assignments stable. Unpaid ordinary
    slots may be replaced when their evidence no longer qualifies.
    """
    previous=list(state.get('allocation',[]))
    chosen={};used={};decisions=[]
    for index,(sport,angle) in enumerate(previous[:limit]):
        slot=state.get('slots',{}).get(f'{index}-{sport.lower()}',{})
        if angle.startswith('idea:') or slot.get('status') not in (None,'waiting_for_data'):
            event=slot.get('event_id') or (angle.split(':',1)[1] if angle.startswith('matchup:') else None)
            chosen[index]={'sport':sport,'angle':angle,'event_id':event,'preserved':True}
            if event:used.setdefault(sport,set()).add(event)
    open_slots=[i for i in range(limit) if i not in chosen]
    candidates=[]
    represented={c['sport'] for c in chosen.values()}
    # Evaluate every sport before a same-sport fallback is considered.
    for sport in sports:
        if sport in represented:continue
        candidate=discover(sport,used.get(sport,set()))
        decisions.append({'sport':sport,'status':'qualified' if candidate else 'not_qualified'})
        if candidate:candidates.append(dict(candidate,sport=sport,selection='different_sport'))
    for candidate in candidates:
        if not open_slots:break
        index=open_slots.pop(0);chosen[index]=candidate
        used.setdefault(candidate['sport'],set()).add(candidate['event_id'])
    # Once diversity has been exhausted, another event from a qualifying sport
    # is permitted. Unknown legacy event identity cannot safely be duplicated.
    if open_slots:
        for sport in sports:
            existing=[c for c in chosen.values() if c['sport']==sport]
            if not existing or any(not c.get('event_id') for c in existing):continue
            while open_slots:
                candidate=discover(sport,used.get(sport,set()))
                if not candidate:break
                if candidate['event_id'] in used.get(sport,set()):break
                index=open_slots.pop(0);chosen[index]=dict(candidate,sport=sport,selection='same_sport_fallback')
                used.setdefault(sport,set()).add(candidate['event_id'])
            if not open_slots:break
    # Never shift a terminal slot: its index also identifies its budget reservation.
    allocation=[]
    for index in range(max(chosen,default=-1)+1):
        if index not in chosen:
            if index<len(previous):allocation.append(previous[index])
            continue
        candidate=chosen[index];sport=candidate['sport']
        if index<len(previous) and not candidate.get('preserved'):
            old_key=f'{index}-{previous[index][0].lower()}'
            if state.get('slots',{}).get(old_key,{}).get('status')=='waiting_for_data':
                state['slots'].pop(old_key)
        allocation.append([sport,candidate.get('angle') or 'matchup:'+candidate['event_id']])
    state['allocation']=allocation
    state['selection_checks']=decisions
    state['selection_choices']={str(i):{k:v for k,v in c.items() if k in ('sport','event_id','selection')}
        for i,c in chosen.items() if not c.get('preserved')}
    return chosen
