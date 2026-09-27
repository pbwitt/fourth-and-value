"""Public live injury listings. Observation time is NOT source publication time."""
import hashlib
import re
from urllib.parse import urljoin

from .data import digest, iso


def parse_tables(html, sport):
    # Imported lazily: the collector also calls this module for its direct feed.
    from .evidence import plain, matching_text
    teams = {}
    for table in re.finditer(r'<table\b[^>]*>(.*?)</table>', html, re.S | re.I):
        rows = [re.findall(r'<t[hd]\b[^>]*>(.*?)</t[hd]>', r, re.S | re.I)
                for r in re.findall(r'<tr\b[^>]*>(.*?)</tr>', table[1], re.S | re.I)]
        if not rows or [plain(c) for c in rows[0]] != ['Player','Position','Updated','Injury','Injury Status']:
            continue
        headings = re.findall(r'<h4\b[^>]*>(.*?)</h4>', html[:table.start()], re.S | re.I)
        links = re.findall(r'href=["\']/'+sport.lower()+r'/teams/([^/]+)/([^/]+)/["\']', headings[-1] if headings else '')
        if not links:
            raise ValueError('Injury table missing team identity')
        code, slug = links[-1]
        key = matching_text(slug)
        if key in teams:
            raise ValueError('Duplicate injury team table')
        parsed = []
        for cells in rows[1:]:
            if len(cells) != 5:
                raise ValueError('Incomplete injury table row')
            player = re.search(r'CellPlayerName--long.*?<a\b[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', cells[0], re.S | re.I)
            if not player or not all(plain(c) for c in cells[1:]):
                raise ValueError('Incomplete injury player identity/status')
            player_url = urljoin('https://www.cbssports.com', player[1])
            if not re.fullmatch(r'https://www\.cbssports\.com/'+sport.lower()+r'/players/\d+/[a-z0-9-]+/', player_url):
                raise ValueError('Unexpected injury player identity')
            parsed.append(dict(player=plain(player[2]), player_source_url=player_url,
                team=slug.replace('-', ' ').title(), team_source_id=code,
                position=plain(cells[1]), reported_update=plain(cells[2]),
                injury=plain(cells[3]), status=plain(cells[4])))
        teams[key] = parsed
    if not teams:
        raise ValueError('No recognizable injury tables')
    return teams


def collect(rows, clock, sport, fetch):
    from .evidence import matching_text
    url = f'https://www.cbssports.com/{sport.lower()}/injuries/'
    try:
        html = fetch(url)
        observed = clock()
        teams = parse_tables(html, sport)
    except Exception as error:
        # No cached substitute and no inferred health when the publisher changes/fails.
        return [], dict(status='unavailable', url=url, error=type(error).__name__,
                        candidates={r['candidate_id']: dict(status='unavailable') for r in rows})
    sources, coverage = [], {}
    for row in rows:
        injuries, missing = [], []
        for team in (row.get('home_team'), row.get('away_team')):
            key = matching_text(team or '')
            key = {'oakland athletics':'athletics'}.get(key, key)
            if key not in teams:
                missing.append(team or 'Unknown team')
            else:
                injuries.extend(dict(r, team=team) for r in teams[key])
        # The candidate's own status must survive compacting; next prioritize goalies/pitchers.
        injuries.sort(key=lambda r: (matching_text(r['player']) != matching_text(row.get('player') or ''),
                                     r['position'] not in ('G','SP','RP','P')))
        info = dict(status='partial' if missing else 'available', observed_at=iso(observed),
                    listed_count=len(injuries), missing_teams=missing)
        coverage[row['candidate_id']] = info
        if not injuries:
            continue
        header = ('Live injury listings; publication time unknown; not confirmation of participation. '
                  f'{len(injuries)} rows archived; excerpt may include a subset. ')
        if missing:
            header += 'Team coverage missing: '+', '.join(missing)+'. '
        # Complete fact rows only. No interpolated injury analysis or predicted return date.
        lines = [f"{r['team']}: {r['player']} ({r['position']}); {r['injury']}; {r['status']}; updated {r['reported_update']}." for r in injuries]
        excerpt = header
        for line in lines:
            if len(excerpt)+len(line)+1 > 1400:
                continue
            excerpt += '\n'+line
        source = dict(url=url, title=f"{sport} injury listings — {row.get('game', '')}",
            published_at=None, retrieved_at=iso(observed), candidate_ids=[row['candidate_id']],
            source_kind='live_injury_table', publication_basis='unknown_live_snapshot',
            excerpt=excerpt, injury_rows=injuries, missing_teams=missing,
            content_sha256=hashlib.sha256(html.encode()).hexdigest())
        source['source_id'] = digest(source)[:20]
        sources.append(source)
    return sources, dict(status='available', url=url, observed_at=iso(observed), candidates=coverage)
