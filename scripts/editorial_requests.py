"""Identify explicit rewrite requests without retaining old draft bodies."""
import hashlib
import editorial as ed


def prepare(state,idea,now):
    requested=ed.stamp(idea['write_now_requested_at']) if idea.get('write_now_requested_at') else None
    previous=ed.stamp(state['request_at']) if state.get('request_at') else ed.stamp(state.get('last_writer_check',{}).get('at')) if state.get('last_writer_check',{}).get('at') else None
    if requested and previous and requested>previous and idea.get('status')=='submitted':
        if any(slot.get('status')=='started' for slot in state.get('slots',{}).values()):
            raise ValueError('Previous writing attempt is uncertain; inspect it before requesting another')
        history=list(state.get('request_history',[]))
        history.append({'request_at':state.get('request_at'),'completed_at':state.get('last_writer_check',{}).get('at'),
            'reservation_key':state.get('reservation_key','requested-'+idea['id']),
            'statuses':{key:slot.get('status') for key,slot in state.get('slots',{}).items()}})
        identity=hashlib.sha256(requested.isoformat().encode()).hexdigest()[:16]
        state={'date':now.astimezone(ed.ETZ).date().isoformat(),'slots':{},'request_history':history,
            'reservation_key':'requested-'+idea['id']+'-'+identity}
    if requested:state['request_at']=requested.isoformat()
    if not state.get('allocation'):state['allocation']=[(idea['sport'],'idea:'+idea['id'])]
    return state
