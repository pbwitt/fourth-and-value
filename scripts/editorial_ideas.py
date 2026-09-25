"""Private idea intake and draft routing. Never print idea text or credentials."""
import os
import re
import requests
import editorial as ed


def configured():
    return bool(os.getenv('SUPABASE_URL') and os.getenv('SUPABASE_SERVICE_ROLE_KEY'))


def request(method,path,**kwargs):
    response=requests.request(method,os.environ['SUPABASE_URL'].rstrip('/')+path,
        headers=ed.api_headers(),timeout=30,**kwargs)
    if not response.ok:raise RuntimeError(f'Private idea request HTTP {response.status_code}')
    return response.json() if response.content else []


def is_editor(user_id):
    user=request('GET','/auth/v1/admin/users/'+user_id)
    return user.get('app_metadata',{}).get('fv_editor') is True


def pending(now):
    if not configured():return []
    rows=request('GET','/rest/v1/editorial_ideas',params={'status':'eq.submitted','select':'*','order':'created_at.asc','limit':'100'})
    result=[]
    for row in rows:
        if row.get('write_now_requested_at'):continue
        if row.get('kind')!='analysis' or row.get('body','').strip():continue
        if row.get('sport') not in ed.CFG['sports']:continue
        if row.get('publish_on') and row['publish_on']>now.astimezone(ed.ETZ).date().isoformat():continue
        if not re.fullmatch(r'[0-9a-f-]{36}',row['id']):continue
        # Readers must first be accepted for research by an editor. This prevents
        # public submissions exhausting the owner's writing allowance.
        owner=is_editor(row['user_id']) and not row.get('requires_review',False)
        if owner or row.get('research_requested_at'):
            result.append(dict(row,owner_idea=owner))
    return sorted(result,key=lambda r:(not r['owner_idea'],r['created_at']))


def get(idea_id):
    if not configured():return None
    rows=request('GET','/rest/v1/editorial_ideas',params={'id':'eq.'+idea_id,'select':'*'})
    if not rows:return None
    row=rows[0]
    row['owner_idea']=is_editor(row['user_id']) and not row.get('requires_review',False)
    return row


def claim(row):
    return bool(request('PATCH','/rest/v1/editorial_ideas',params={
        'id':'eq.'+row['id'],'status':'eq.submitted','updated_at':'eq.'+row['updated_at']},json={'status':'researching','research_error':None}))


def context(row,packet):
    """Only expose verified matched games; treat the owner's angle as a topic, not facts."""
    text=row['idea'].lower()
    matched=[g for g in packet.get('markets',[]) if any(
        re.search(r'(?<![a-z])'+re.escape(team.strip().split()[-1].lower())+r'(?![a-z])',text)
        for team in g.get('game','').split('@') if team.strip())]
    terms=tuple(dict.fromkeys(team.strip().split()[-1].lower() for g in matched for team in g['game'].split('@')))
    if not terms:
        # Names are discovery hints, never evidence. Keep full names together so
        # Jayden Daniels cannot match a headline about Jayden Reed.
        topic=re.sub(r"^(?:please\s+)?(?:highlight|feature|investigate|cover|analyze|analyse|discuss|research)\s+",'',row['idea'],flags=re.I)
        names=re.findall(r"\b[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)*",topic)
        terms=tuple(name.lower() for name in names if name not in {'Use','Then','Does','How','What','Why','The','Is','Can','Look'})
    return matched,terms


def save_draft(row,article,as_of=None):
    body='\n\n'.join(section['heading']+'\n'+section['text'] for section in article['sections'])
    if as_of:body='Market snapshot: '+as_of.astimezone(ed.ETZ).strftime('%B %d, %Y at %I:%M %p ET')+'. Prices may have changed.\n\n'+body
    links='\n'.join(source['url'] for source in article['sources'])
    saved=request('PATCH','/rest/v1/editorial_ideas',params={'id':'eq.'+row['id'],'status':'eq.researching'},
        json={'title':article['title'],'body':body,'byline':'Fourth & Value','sources':links,'status':'review'})
    if not saved:raise RuntimeError('Private draft changed while research was running')


def finish(row):
    # Owner-directed articles publish through the automatic writer. Keep the idea
    # privately archived; never bypass approval on a reader-originated row.
    request('PATCH','/rest/v1/editorial_ideas',params={'id':'eq.'+row['id'],'status':'eq.researching'},json={'status':'archived'})


def waiting(row,message):
    try:
        request('PATCH','/rest/v1/editorial_ideas',params={'id':'eq.'+row['id'],'status':'eq.submitted'},json={'research_error':message})
    except (RuntimeError,requests.RequestException):pass


def fail(row):
    request('PATCH','/rest/v1/editorial_ideas',params={'id':'eq.'+row['id'],'status':'eq.researching'},
        json={'status':'archived','research_error':'Research could not produce a verified article. Archived to prevent repeated charges. Submit a revised idea to try again.'})


def notify(now):
    if not configured():return
    rows=request('GET','/rest/v1/editorial_ideas',params={'or':'(and(requires_review.eq.true,status.eq.submitted,notification_sent_at.is.null),and(status.eq.review,draft_notification_sent_at.is.null))','select':'id,status,notification_sent_at,draft_notification_sent_at','order':'created_at.asc','limit':'100'})
    groups=[('submitted','notification_sent_at'),('review','draft_notification_sent_at')]
    key=os.getenv('RESEND_API_KEY');sender=os.getenv('EDITORIAL_NOTIFY_FROM');recipient=os.getenv('EDITORIAL_NOTIFY_EMAIL')
    if not all([key,sender,recipient]):
        print('Email notifications not configured; reader submissions remain visible in the editorial desk.');return
    for status,column in groups:
        for row in rows:
            if row['status']!=status or row.get(column):continue
            subject='New reader article suggestion' if status=='submitted' else 'Article draft ready for your approval'
            # No private submission text is sent to the email provider. A stable
            # idempotency key protects retries if delivery succeeded but marking failed.
            response=requests.post('https://api.resend.com/emails',headers={'Authorization':'Bearer '+key,
                'Idempotency-Key':f"editorial-{row['id']}-{status}"},json={'from':sender,'to':[recipient],
                'subject':'Fourth & Value: '+subject,'text':subject+'. Open your private editorial desk: https://fourthandvalue.com/editorial/inbox.html'},timeout=30)
            if not response.ok:raise RuntimeError(f'Editorial email HTTP {response.status_code}')
            request('PATCH','/rest/v1/editorial_ideas',params={'id':'eq.'+row['id'],column:'is.null'},json={column:now.isoformat()})
    print('Editorial notification check complete.')


if __name__=='__main__':
    from datetime import datetime,timezone
    try:notify(datetime.now(timezone.utc))
    except (RuntimeError,requests.RequestException):
        raise SystemExit('Editorial notifications unavailable; check schema and email configuration. No private content logged.')
