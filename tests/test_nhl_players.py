"""Static NHL player prop pages: stable slugs, SEO-compliant output, prices and recent
games, no rewrite without a change, kept pages, fail-closed identity and sitemaps."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nhl import players
import build_site_metadata
import seo_check

NOW = datetime(2026, 10, 4, 15, tzinfo=timezone.utc)
START = '2026-10-04T23:00:00Z'
HOME, AWAY = 'Buffalo Sabres', 'Chicago Blackhawks'


def quote(player, market, line, side, book, price, **extra):
    p = 100 / (price + 100) if price > 0 else -price / (100 - price)
    return dict(dict(event_id='e1', nhl_game_id=2026020030, commence_time=START, game=f'{AWAY} @ {HOME}', home_team=HOME,
                     away_team=AWAY, player=player, market=market, line=line, side=side, book=book, book_label=book.title(),
                     price=price, book_probability=p, quoted_at='2026-10-04T14:50:00Z', fair_probability=None,
                     conditional_probability=None, projected_mean=None, settlement_profile='nhl_player_ot_no_so_participation',
                     settlement_scope='Standard full-game only'), **extra)


def state(rows, status='ready'):
    return dict(status=status, season=20262027, rows=rows, events=[], checked_at='2026-10-04T14:55:00Z')


def history():
    games, records = [], []
    for i in range(12):
        gid = 2025021000 + i
        games.append(dict(game_id=gid, season=20252026, game_type=2, game_date=f'2026-03-{10 + i:02d}', home_id=7, away_id=16,
                          home_team=HOME, away_team=AWAY))
        records.append(dict(game_id=gid, player_id=1, player='Tage Thompson', season=20252026, game_date=f'2026-03-{10 + i:02d}',
                            toi=18.5, home=True, team_abbrev='BUF', team_id=7, shots=i % 5, goals=i % 2, assists=0, points=i % 2,
                            position='C'))
    games.append(dict(game_id=2025030001, season=20252026, game_type=3, game_date='2026-04-30', home_id=7, away_id=16,
                      home_team=HOME, away_team=AWAY))
    records.append(dict(game_id=2025030001, player_id=1, player='Tage Thompson', season=20252026, game_date='2026-04-30',
                        toi=20.0, home=True, team_abbrev='BUF', team_id=7, shots=9, goals=3, assists=0, points=3))
    for pid in (10, 11):   # two different players with one name
        records.append(dict(game_id=2025021000, player_id=pid, player='Sebastian Aho', season=20252026, game_date='2026-03-10',
                            toi=15.0, home=False, team_abbrev='CHI', team_id=16, shots=1, goals=0, assists=0, points=0))
    return games, records


ROWS = [
    quote('Tage Thompson', 'player_shots_on_goal', 2.5, 'Over', 'fanduel', 120, player_id=1, fair_probability=.44,
          conditional_probability=.47, projected_mean=2.9),
    quote('Tage Thompson', 'player_shots_on_goal', 2.5, 'Over', 'draftkings', 105, player_id=1, fair_probability=.46,
          conditional_probability=.47, projected_mean=2.9),
    quote('Tage Thompson', 'player_shots_on_goal', 2.5, 'Under', 'fanduel', -150, player_id=1, fair_probability=.56),
    quote('Tage Thompson', 'player_goals', 1.0, 'Over', 'betrivers', 400, player_id=1),
    quote('Connor Bedard', 'player_points', 0.5, 'Over', 'draftkings', -160),
    quote('Sebastian Aho', 'player_goals', 0.5, 'Over', 'draftkings', 300),
    quote('Old Game', 'player_goals', 0.5, 'Over', 'draftkings', 300, commence_time='2026-10-04T14:00:00Z'),
    quote('Bad </script><b>Name', 'player_points', 0.5, 'Over', 'draftkings', 110),
]


class PlayerPageTests(unittest.TestCase):
    def build(self, rows=ROWS, out=None, now=NOW, status='ready'):
        return players.build(state(rows, status), out=out or self.out, now=now, data=history())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'docs/nhl/players'

    def tearDown(self):
        self.tmp.cleanup()

    def page(self, slug):
        return (self.out / slug / 'index.html').read_text()

    def test_slugs_are_stable_and_namesakes_get_their_id(self):
        self.assertEqual(players.slugify('Tim Stützle'), 'tim-stutzle')
        self.assertEqual(players.slugify('A.J. Greer'), 'a-j-greer')
        registry = {'sebastian-aho': dict(name='Sebastian Aho', player_id=10)}
        self.assertEqual(players.assign(registry, 'Sebastian Aho', 10), 'sebastian-aho')
        self.assertEqual(players.assign(registry, 'Sebastian Aho', 11), 'sebastian-aho-11')
        self.assertEqual(players.assign({'x': dict(name='Tage Thompson', player_id=None)}, 'Tage Thompson', 1), 'x',
                         'a name-only page keeps its URL when the identity arrives')

    def test_pages_index_registry_and_sitemap(self):
        result = self.build()
        self.assertEqual(result['with_props'], 3, 'started games and unresolved namesakes are left out')
        self.assertEqual(sorted(p.name for p in self.out.iterdir() if p.is_dir()), ['bad-script-b-name', 'connor-bedard', 'tage-thompson'])
        for path in self.out.rglob('index.html'):
            rel = 'nhl/players/' + path.relative_to(self.out).as_posix()
            self.assertEqual(seo_check.audit(rel, path.read_text()), [], rel)
        html = self.page('tage-thompson')
        self.assertIn('<link rel="canonical" href="https://fourthandvalue.com/nhl/players/tage-thompson/">', html)
        self.assertIn('Tage Thompson Props Today: Shots, Goals &amp; Points Odds', html)
        self.assertIn('id="fv-betting-notice"', html, 'responsible-use notice')
        self.assertIn('"@type": "BreadcrumbList"', html)
        sitemap = (self.out / 'sitemap.xml').read_text()
        self.assertIn('<loc>https://fourthandvalue.com/nhl/players/</loc><lastmod>2026-10-04</lastmod>', sitemap)
        self.assertEqual(len(re.findall('<loc>', sitemap)), 4)
        registry = json.loads((self.out / 'players.json').read_text())['players']
        self.assertEqual(registry['tage-thompson']['team'], 'BUF')
        self.assertEqual(registry['tage-thompson']['position'], 'C')
        self.assertIn('<p class="eyebrow">NHL player props · Buffalo Sabres · Center</p>', html)
        self.assertIn('<p class="eyebrow">NHL player props</p>', self.page('connor-bedard'), 'no history, no position')
        index = (self.out / 'index.html').read_text()
        self.assertIn('Chicago Blackhawks at Buffalo Sabres', index)
        self.assertIn('<a href="/nhl/players/tage-thompson/">Tage Thompson</a> <span class="muted">G 1, SOG 2.5</span>', index)

    def test_prices_fair_odds_model_and_recent_games(self):
        self.build()
        html = self.page('tage-thompson')
        row = re.search(r'<tr><td>Over 2.5</td>.*?</tr>', html).group(0)
        self.assertIn('+120<span class="sub">Fanduel</span>', row, 'best price and its book')
        self.assertIn('<td class="num">2</td>', row, 'two books at this exact line')
        self.assertIn('+122', row, 'fair odds from the median paired probability (45%)')
        self.assertIn('47%', row, 'experimental model estimate')
        self.assertIn('Over 1 · can push', html)
        self.assertIn('Experimental model projection: 2.9 shots.', html)
        self.assertIn('shots over 2.5 in 4 of 10', html, 'last 10 regular-season games; the playoff game is excluded')
        self.assertIn('vs CHI', html)
        self.assertIn('<td>2025–26</td><td class="num">12</td>', html)
        self.assertIn('<a href="/nhl/players/connor-bedard/">Connor Bedard</a>', html, 'also in this game')
        offers = json.loads(re.search(r'<script type="application/json" id="offers">(.*?)</script>', html).group(1))
        self.assertEqual(len(offers), len(re.findall(r'data-offer="\d+"', html)))
        self.assertEqual([offers[0]['sport'], offers[0]['market_label'], offers[0]['book'], offers[0]['price']],
                         ['NHL', 'Shots on goal', 'fanduel', 120], 'offers follow the table: shots first, best price')
        self.assertEqual([offers[-1]['market_label'], offers[-1]['book'], offers[-1]['price']], ['Goals', 'betrivers', 400])

    def test_untrusted_names_are_escaped(self):
        self.build()
        html = self.page('bad-script-b-name')
        self.assertNotIn('<b>Name', re.sub(r'<script\b.*?</script>', '', html, flags=re.S), 'no markup outside script data')
        self.assertIn('Bad &lt;/script&gt;&lt;b&gt;Name props', html)
        data = re.search(r'id="offers">(.*?)</script>', html, re.S).group(1)
        self.assertIn('<\\/script>', data, 'cannot close the JSON block early')
        self.assertEqual(json.loads(data)[0]['player'], 'Bad </script><b>Name')

    def test_no_rewrite_without_change_and_pages_kept_without_props(self):
        self.build()
        self.assertEqual(self.build()['written'], 0, 'an unchanged snapshot rewrites nothing')
        later = NOW + timedelta(days=1)
        self.build(rows=[r for r in ROWS if r['player'] != 'Tage Thompson'], now=later - timedelta(hours=9))
        html = self.page('tage-thompson')
        self.assertIn('No props posted right now', html)
        self.assertIn('Last 10 games', html)
        self.assertIn('nhl/players/tage-thompson/', (self.out / 'sitemap.xml').read_text(), 'the URL stays indexed')
        self.assertIn('Tage Thompson', (self.out / 'index.html').read_text(), 'still in the A–Z list')

    def test_failed_feed_changes_nothing(self):
        result = self.build(status='feed_error')
        self.assertIn('feed status', result['skipped'])
        self.assertFalse(self.out.exists())


class SitemapTests(unittest.TestCase):
    def test_section_sitemaps_are_listed_and_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            (docs / 'nhl/players/a').mkdir(parents=True)
            (docs / 'sitemap.xml').write_text('<urlset><url><loc>https://fourthandvalue.com/about.html</loc></url></urlset>')
            (docs / 'nhl/players/sitemap.xml').write_text('<urlset><url><loc>https://fourthandvalue.com/nhl/players/a/</loc></url></urlset>')
            self.assertEqual(build_site_metadata.section_sitemaps(docs), ['nhl/players/sitemap.xml'])
            self.assertEqual(build_site_metadata.robots(docs), 'User-agent: *\nAllow: /\n\nSitemap: https://fourthandvalue.com/sitemap.xml\n'
                             'Sitemap: https://fourthandvalue.com/nhl/players/sitemap.xml\n')
            self.assertEqual(sorted(seo_check.indexed_pages(docs)), ['about.html', 'nhl/players/a/index.html'])


if __name__ == '__main__':
    unittest.main()
