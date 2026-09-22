"""Grant editorial access to the exact confirmed existing sign-in email.

Run privately with server credentials. Does not print email, ID or secrets.
"""
import argparse
import os
import requests

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--email',required=True);args=ap.parse_args()
    base=os.environ['SUPABASE_URL'].rstrip('/');key=os.environ['SUPABASE_SERVICE_ROLE_KEY']
    headers={'apikey':key,'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    found=[]
    for page in range(1,101):
        r=requests.get(base+'/auth/v1/admin/users',headers=headers,params={'page':page,'per_page':100},timeout=30)
        if not r.ok:raise SystemExit(f'Account lookup HTTP {r.status_code}')
        users=r.json().get('users',[])
        found += [u for u in users if u.get('email','').casefold()==args.email.strip().casefold()]
        if len(users)<100:break
    if len(found)!=1:raise SystemExit('No unique existing account matched. Sign in with the intended email first, then retry.')
    u=found[0];meta=dict(u.get('app_metadata',{}));meta['fv_editor']=True
    r=requests.put(base+'/auth/v1/admin/users/'+u['id'],headers=headers,json={'app_metadata':meta},timeout=30)
    if not r.ok:raise SystemExit(f'Role update HTTP {r.status_code}')
    print('Editorial-owner access granted. Sign out and back in to refresh the session.')

if __name__=='__main__':main()
