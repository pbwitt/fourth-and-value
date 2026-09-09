#!/usr/bin/env python3
"""Post a tweet to X using OAuth 1.0a user context.

Credentials come from .env: X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN and
X_ACCESS_TOKEN_SECRET. The bearer token is app-only and cannot post.

Posting is public and effectively irreversible, so this refuses to run without
--confirm and always prints the exact text and character count first.

  python scripts/post_tweet.py --text "..."            # dry run, shows nothing sent
  python scripts/post_tweet.py --text "..." --confirm  # actually posts
"""
import argparse
import base64
import hashlib
import hmac
import os
import secrets
import sys
import time
import urllib.parse

import requests
from dotenv import load_dotenv

load_dotenv()

TWEET_URL = 'https://api.x.com/2/tweets'
VERIFY_URL = 'https://api.x.com/2/users/me'
LIMIT = 280


def _quote(value):
    return urllib.parse.quote(str(value), safe='~')


def oauth_header(method, url, creds, extra_params=None):
    """OAuth 1.0a HMAC-SHA1. A JSON body is not part of the signature base."""
    params = {
        'oauth_consumer_key': creds['api_key'],
        'oauth_nonce': secrets.token_hex(16),
        'oauth_signature_method': 'HMAC-SHA1',
        'oauth_timestamp': str(int(time.time())),
        'oauth_token': creds['access_token'],
        'oauth_version': '1.0',
    }
    signing_params = dict(params)
    signing_params.update(extra_params or {})

    encoded = '&'.join(f'{_quote(k)}={_quote(signing_params[k])}'
                       for k in sorted(signing_params))
    base = '&'.join([method.upper(), _quote(url), _quote(encoded)])
    key = f"{_quote(creds['api_secret'])}&{_quote(creds['access_token_secret'])}"
    signature = base64.b64encode(
        hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()).decode()

    params['oauth_signature'] = signature
    return 'OAuth ' + ', '.join(f'{_quote(k)}="{_quote(v)}"'
                                for k, v in sorted(params.items()))


def load_credentials():
    creds = {
        'api_key': os.getenv('X_API_KEY'),
        'api_secret': os.getenv('X_API_SECRET'),
        'access_token': os.getenv('X_ACCESS_TOKEN'),
        'access_token_secret': os.getenv('X_ACCESS_TOKEN_SECRET'),
    }
    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise SystemExit(f'Missing credentials in .env: {", ".join(missing)}')
    return creds


def whoami(creds):
    r = requests.get(VERIFY_URL,
                     headers={'Authorization': oauth_header('GET', VERIFY_URL, creds)},
                     timeout=30)
    if r.status_code != 200:
        raise SystemExit(f'Credential check failed [{r.status_code}]: {r.text[:400]}')
    return r.json().get('data', {})


def post(text, creds):
    r = requests.post(TWEET_URL,
                      headers={'Authorization': oauth_header('POST', TWEET_URL, creds),
                               'Content-Type': 'application/json'},
                      json={'text': text}, timeout=30)
    if r.status_code not in (200, 201):
        raise SystemExit(f'Post failed [{r.status_code}]: {r.text[:600]}')
    return r.json().get('data', {})


def main():
    ap = argparse.ArgumentParser(description='Post a tweet to X')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', help='Tweet text')
    group.add_argument('--file', help='File containing the tweet text')
    ap.add_argument('--confirm', action='store_true',
                    help='Actually post. Without this it is a dry run.')
    args = ap.parse_args()

    text = args.text if args.text else open(args.file).read().strip()
    n = len(text)

    print('-' * 60)
    print(text)
    print('-' * 60)
    print(f'{n} characters', '(over the 280 limit)' if n > LIMIT else '(fits)')
    if n > LIMIT:
        raise SystemExit('Refusing to post: over the character limit.')

    creds = load_credentials()
    user = whoami(creds)
    print(f'Authenticated as @{user.get("username")} ({user.get("name")})')

    if not args.confirm:
        print('\nDry run. Nothing was posted. Re-run with --confirm to publish.')
        return

    data = post(text, creds)
    tweet_id = data.get('id')
    print(f'\n✓ Posted: https://x.com/{user.get("username")}/status/{tweet_id}')


if __name__ == '__main__':
    main()
