"""Bounded public GET-only Kalshi client. No credentials, orders or retries."""
import json
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = 'https://external-api.kalshi.com/trade-api/v2'


def ticker(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Z0-9][A-Z0-9_.-]{0,150}', value):
        raise ValueError('Invalid Kalshi ticker')
    return value


class PublicClient:
    def __init__(self, timeout=15, max_requests=250):
        self.timeout, self.max_requests = timeout, max_requests
        self.requests = 0
        self.last_request = 0

    def get(self, path, **params):
        if path not in ('/markets', '/events/fee_changes') and not re.fullmatch(
                r'/(?:series/[A-Z0-9][A-Z0-9_.-]*|markets/[A-Z0-9][A-Z0-9_.-]*(?:/orderbook)?)', path):
            raise ValueError('Only public market-data paths are supported')
        if self.requests >= self.max_requests:
            raise ValueError('Public request budget exhausted')
        time.sleep(max(0, .12 - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        self.requests += 1
        query = urlencode({k: v for k, v in params.items() if v is not None})
        request = Request(BASE + path + ('?' + query if query else ''),
                          headers={'User-Agent': 'FourthAndValue/1.0 (public market research)', 'Accept': 'application/json'})
        with urlopen(request, timeout=self.timeout) as response:
            body = response.read(8_000_001)
        if len(body) > 8_000_000:
            raise ValueError('Market response too large')
        result = json.loads(body)
        if not isinstance(result, dict):
            raise ValueError('Expected a JSON object')
        return result

    def pages(self, path, key, max_pages=10, **params):
        cursor, seen, items = None, set(), []
        for _ in range(max_pages):
            data = self.get(path, cursor=cursor, **params)
            if not isinstance(data.get(key), list):
                raise ValueError('Missing ' + key)
            items.extend(data[key])
            cursor = data.get('cursor')
            if not cursor:
                return items, True
            if cursor in seen:
                raise ValueError('Repeated pagination cursor')
            seen.add(cursor)
        return items, False
