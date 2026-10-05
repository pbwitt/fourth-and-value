"""Signed Kalshi requests restricted to the mock-funds demo host.

There is deliberately no base-URL setting, live credential fallback, retry,
redirect following, transfer endpoint, or production order route.
"""
import base64
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from .client import PublicClient
from .feed import ROOT

DEMO_ORIGIN = 'https://external-api.demo.kalshi.co'
API_PATH = '/trade-api/v2'


class DemoAPIError(Exception):
    def __init__(self, status=None):
        self.status = status
        super().__init__('Kalshi demo request failed' + (f' (HTTP {status})' if status else ' (transport or response error)'))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def signature(private_key, timestamp_ms, method, path):
    message = (str(timestamp_ms)+method.upper()+path.split('?')[0]).encode()
    if isinstance(private_key, rsa.RSAPrivateKey):
        value = private_key.sign(message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                                      salt_length=hashes.SHA256().digest_size), hashes.SHA256())
    elif isinstance(private_key, ed25519.Ed25519PrivateKey):
        value = private_key.sign(message)
    else:
        raise ValueError('Demo signing requires an RSA or Ed25519 private key')
    return base64.b64encode(value).decode()


def allowed(method, path):
    ident = r'[A-Za-z0-9][A-Za-z0-9_.-]{0,150}'
    if method == 'GET':
        return path in ('/portfolio/balance', '/portfolio/orders', '/portfolio/fills', '/events/fee_changes') or bool(
            re.fullmatch(rf'/(?:portfolio/orders/{ident}|markets/{ident}(?:/orderbook)?|series/{ident})', path))
    if method == 'POST':
        return path == '/portfolio/events/orders'
    if method == 'DELETE':
        return bool(re.fullmatch(rf'/portfolio/events/orders/{ident}', path))
    return False


class DemoClient:
    def __init__(self, api_key_id, private_key, *, opener=None, clock=time.time):
        if not isinstance(api_key_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}', api_key_id):
            raise ValueError('Invalid demo API key ID')
        self._key_id, self._key = api_key_id, private_key
        self._opener, self._clock = opener or build_opener(NoRedirect()), clock
        public = private_key.public_key().public_bytes(serialization.Encoding.DER,
                                                       serialization.PublicFormat.SubjectPublicKeyInfo)
        self.account_fingerprint = hashlib.sha256(api_key_id.encode()+public).hexdigest()

    @classmethod
    def from_environment(cls):
        key_id = os.environ.get('KALSHI_DEMO_API_KEY_ID')
        location = os.environ.get('KALSHI_DEMO_PRIVATE_KEY_PATH')
        if not key_id or not location:
            raise ValueError('Set KALSHI_DEMO_API_KEY_ID and KALSHI_DEMO_PRIVATE_KEY_PATH; live credentials are not used')
        path = Path(location).expanduser().resolve()
        if path == ROOT or ROOT in path.parents:
            raise ValueError('Private keys must be outside this repository')
        if not path.is_file() or path.stat().st_mode & 0o077:
            raise ValueError('Demo private key must be an owner-only file (chmod 600)')
        try:
            key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        except (ValueError, TypeError):
            raise ValueError('Unable to load demo private key') from None
        if not isinstance(key, (rsa.RSAPrivateKey, ed25519.Ed25519PrivateKey)):
            raise ValueError('Unsupported demo private-key type')
        return cls(key_id, key)

    def request(self, method, path, payload=None, **params):
        if not allowed(method, path):
            raise ValueError('Endpoint is outside the supported demo order lifecycle')
        full_path = API_PATH+path
        query = urlencode({k:v for k,v in params.items() if v is not None})
        headers = {'Accept':'application/json', 'Content-Type':'application/json',
                   'User-Agent':'FourthAndValue-Demo/1.0'}
        if path.startswith('/portfolio/'):
            timestamp_ms = str(int(self._clock()*1000))
            headers.update({'KALSHI-ACCESS-KEY':self._key_id, 'KALSHI-ACCESS-TIMESTAMP':timestamp_ms,
                            'KALSHI-ACCESS-SIGNATURE':signature(self._key, timestamp_ms, method, full_path)})
        body = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        request = Request(DEMO_ORIGIN+full_path+('?' + query if query else ''), data=body, headers=headers, method=method)
        try:
            with self._opener.open(request, timeout=15) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ValueError('Response too large')
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError('Response is not an object')
            return result
        except HTTPError as exc:
            raise DemoAPIError(exc.code) from None
        except (URLError, OSError, ValueError, HTTPException):
            raise DemoAPIError() from None

    def get(self, path, **params):
        return self.request('GET', path, **params)

    def pages(self, path, key, **params):
        return PublicClient.pages(self, path, key, **params)

    def create_order(self, payload):
        return self.request('POST', '/portfolio/events/orders', payload)

    def cancel_order(self, order_id, market_ticker):
        return self.request('DELETE', '/portfolio/events/orders/'+order_id, market_ticker=market_ticker)
