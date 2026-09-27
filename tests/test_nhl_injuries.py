"""Shared injury evidence: exact teams, timestamps, parsing failures and request survival."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from nhl.v2 import astra, evidence, injuries
from nhl.v2.data import iso
import analyst_review

NOW = datetime(2026,9,27,14,tzinfo=timezone.utc)


def table(sport='NHL', slug='washington-capitals', player='Example Goalie', position='G', status='Day-to-day'):
    return f'''<h4><a href="/{sport.lower()}/teams/TEST/{slug}/">Team city alone</a></h4>
    <table><tr><th>Player</th><th>Position</th><th>Updated</th><th>Injury</th><th>Injury Status</th></tr>
    <tr><td><span class="CellPlayerName--short">E. Goalie</span><span class="CellPlayerName--long"><a href="/{sport.lower()}/players/123/example-goalie/">{player}</a></span></td><td>{position}</td><td>Sat, Sep 26</td><td>Lower Body</td><td>{status}</td></tr></table>'''


def row(sport='NHL'):
    home,away=('Washington Capitals','Pittsburgh Penguins') if sport=='NHL' else ('New York Yankees','Boston Red Sox')
    return dict(candidate_id='c1',player='Example Goalie',home_team=home,away_team=away,game=f'{away} @ {home}')


class InjuryTests(unittest.TestCase):
    def test_exact_team_slug_and_full_player_identity_for_both_sports(self):
        for sport,slug in [('NHL','washington-capitals'),('MLB','new-york-yankees')]:
            html=table(sport,slug)
            parsed=injuries.parse_tables(html,sport)
            r=parsed[slug.replace('-',' ')][0]
            self.assertEqual(r['player'],'Example Goalie')
            self.assertIn('/players/123/',r['player_source_url'])
            self.assertEqual(r['reported_update'],'Sat, Sep 26')
            self.assertNotIn('published_at',r)  # No fabricated year or publication time.
            sources,status=injuries.collect([row(sport)],lambda:NOW,sport,lambda u:html)
            self.assertEqual(status['candidates']['c1']['status'],'partial')
            self.assertEqual(sources[0]['missing_teams'],[row(sport)['away_team']])
            self.assertIsNone(sources[0]['published_at'])
            self.assertTrue(evidence.usable(sources[0],row(sport),NOW))
            self.assertFalse(evidence.usable(sources[0],row(sport),NOW+timedelta(minutes=91)))
            self.assertFalse(evidence.usable(sources[0],row(sport),NOW-timedelta(seconds=1)))
            bad=dict(sources[0],published_at=iso(NOW))
            self.assertFalse(evidence.usable(bad,row(sport),NOW))

    def test_missing_team_or_source_does_not_mean_healthy(self):
        for html in ('unavailable',table(slug='new-york-rangers')):
            sources,status=injuries.collect([row()],lambda:NOW,'NHL',lambda u:html)
            self.assertEqual(sources,[])
            self.assertIn(status['candidates']['c1']['status'],('unavailable','partial'))
        with self.assertRaises(ValueError):
            injuries.parse_tables(table().replace('<td>Lower Body</td>',''),'NHL')
        with self.assertRaises(ValueError):
            injuries.parse_tables(table()+table(),'NHL')

    def test_subject_first_then_goalies_full_rows_archived_and_excerpt_bounded(self):
        html=table(player='Other Player',position='D')
        # Two teams preserve full identifiers even if the city text is ambiguous.
        html+=table(slug='pittsburgh-penguins',player='Example Goalie')
        sources,status=injuries.collect([row()],lambda:NOW,'NHL',lambda u:html)
        self.assertEqual(status['candidates']['c1']['status'],'available')
        self.assertEqual(len(sources[0]['injury_rows']),2)
        self.assertEqual(sources[0]['injury_rows'][0]['player'],'Example Goalie')
        self.assertLessEqual(len(sources[0]['excerpt']),1400)
        self.assertIn('subset',sources[0]['excerpt'])

    def test_shared_collector_reserves_table_even_without_recent_headlines(self):
        for sport,slug in [('NHL','washington-capitals'),('MLB','new-york-yankees')]:
            def fetch(url):
                return table(sport,slug) if url.endswith('/injuries/') else '<rss></rss>'
            with patch.object(evidence,'fetch',side_effect=fetch):
                sources,status=evidence.collect([row(sport)],lambda:NOW,sport=sport)
            self.assertEqual(len(sources),1)
            self.assertEqual(sources[0]['source_kind'],'live_injury_table')
            board=dict(candidates=[row(sport)],generated_at=iso(NOW))
            evidence.attach_context(board,status)
            self.assertEqual(board['candidates'][0]['injury_context']['status'],'partial')
            config=json.loads(analyst_review.CONFIG.read_text())
            request=astra.payload(board,sources,NOW,config)
            packet=json.loads(request['input'])
            self.assertIn('Example Goalie',packet['sources'][0]['excerpt'])
            self.assertIsNone(packet['sources'][0]['published_at'])
            self.assertNotIn('injury_rows',packet['sources'][0])  # Full listing archived, bounded excerpt sent.
            self.assertLess(astra.bounds(request,config),1)

    def test_official_updated_table_survives_sparse_jsonld_and_time_guards(self):
        article={'@type':'NewsArticle','headline':'Texans at Colts injury report',
            'datePublished':iso(NOW-timedelta(days=4)), 'dateModified':iso(NOW-timedelta(days=2)),
            'articleBody':'Houston Texans. Indianapolis Colts. DNP means did not practice.'}
        html='<script type="application/ld+json">'+json.dumps(article)+'</script><h2>Houston Texans</h2>'
        html+='<table><tr><th>POSITION</th><th>PLAYER</th><th>INJURY</th><th>FRI</th><th>GAME STATUS</th></tr>'
        html+=''.join(f'<tr><td>WR</td><td>Example Receiver {i}</td><td>Hamstring</td><td>DNP</td><td>OUT</td></tr>' for i in range(5))+'</table>'
        r=dict(candidate_id='nfl',home_team='Indianapolis Colts',away_team='Houston Texans',player='C.J. Stroud')
        url='https://www.houstontexans.com/news/texans-colts-injury-report'
        def fetch(u):
            if u==url:return html
            if u=='https://www.houstontexans.com/news/':return '<a href="'+url+'">Injuries</a>'
            return '<rss></rss>'
        with patch.object(evidence,'fetch',side_effect=fetch):
            sources,status=evidence.collect([r],lambda:NOW,sport='NFL')
        s=sources[0]
        self.assertIn('Example Receiver 0',s['excerpt'])
        self.assertIn('game status: OUT; practice FRI: DNP',s['excerpt'])
        self.assertEqual(s['published_at'],article['datePublished'])
        self.assertEqual(s['updated_at'],article['dateModified'])
        self.assertTrue(evidence.usable(s,r,NOW))
        for change in [dict(updated_at=iso(NOW+timedelta(seconds=1))),dict(updated_at=iso(NOW-timedelta(days=5))),
                       dict(published_at=iso(NOW-timedelta(days=8))),dict(updated_at=None)]:
            self.assertFalse(evidence.usable(dict(s,**change),r,NOW))

    def test_full_name_initials_and_noninjury_articles(self):
        r=dict(player='C.J. Stroud')
        for text in ('CJ Stroud','c-j-stroud','C.J. Stroud'):
            self.assertTrue(evidence.matches(r,text))
        self.assertFalse(evidence.matches(r,'Stroud'))
        for sport in ('MLB','NHL'):
            body=f'{sport} synthetic lineup update with ordinary article content.'
            html='<script type="application/ld+json">'+json.dumps({'@type':'NewsArticle','articleBody':body})+'</script>'
            self.assertEqual(evidence.article_text(html),body)


if __name__=='__main__':unittest.main()
