"""Historical MLB/NHL recaps preserve offered contracts and evidence boundaries."""
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import sport_weekly_review as review


def quote(**changes):
    row = dict(mlb_game_id=1, nhl_game_id=1, game='Away @ Home', home_team='Home', away_team='Away',
               market='totals', market_label='Game total', player='', side='Over', line=8.5,
               price=-110, book='a', quoted_at='2026-09-30T17:00:00Z',
               commence_time='2026-09-30T20:00:00Z', model_probability=.6,
               model_push_probability=0, model_mean=9)
    return dict(row, **changes)


def run(at, rows, **changes):
    snap = dict(checked_at=at, rows=rows, events=[dict(mlb_game_id=1, nhl_game_id=1,
                                                   commence_time='2026-09-30T20:00:00Z')])
    snap.update(changes)
    return Path('/tmp/' + at.replace(':', '-') + '.json.gz'), {'snapshot': snap}, snap


def card_row(**changes):
    row = quote(sport='MLB', reviewed_candidate=True, qualitative_review=dict(
        status='research_support', reviewed_at='2026-09-30T16:55:00Z', countercase='',
        open_checks=[], evidence=[], assessment=dict(verdict='consider', reason='',
        model_case='', price_case='', context_case='', blocking_checks=[])))
    row.update(changes)
    return row


def nba_feed():
    event = dict(id='401000001', competitions=[dict(id='401000001', date='2026-09-30T20:00:00Z',
        status=dict(type=dict(completed=True, state='post')),
        competitors=[dict(homeAway='home', score='115', team=dict(id='1', displayName='Home')),
                     dict(homeAway='away', score='108', team=dict(id='2', displayName='Away'))])])
    box = dict(header=event, boxscore=dict(players=[dict(team=dict(id='1', displayName='Home'),
        statistics=[dict(labels=['MIN', 'PTS'], athletes=[dict(athlete=dict(id='99', displayName='Player'),
            stats=['35', '25'], didNotPlay=False)])])]))
    return lambda url: deepcopy(dict(events=[event]) if '/scoreboard?' in url else box)


class SportWeeklyReviewTests(unittest.TestCase):
    games = {'1': dict(home_score=5, away_score=3, home_team='Home', away_team='Away')}

    def test_pregame_uses_last_safe_snapshot_and_both_start_times(self):
        first = run('2026-09-30T18:00:00Z', [quote(price=-120)])
        last = run('2026-09-30T19:00:00Z', [quote(price=120, commence_time='2026-09-30T20:00:47Z')])
        late = run('2026-09-30T20:00:01Z', [quote(price=140, commence_time='2026-09-30T20:00:47Z')])
        result = review.pregame([first, last, late], 'mlb')['1']
        self.assertEqual(result['rows'][0]['price'], 120)
        self.assertEqual(result['rows'][0]['commence_time'], '2026-09-30T20:00:47Z')
        self.assertEqual(result['rows'][0]['pregame_cutoff'], '2026-09-30T20:00:00+00:00')
        self.assertEqual(result['captured_at'], '2026-09-30T19:00:00+00:00')

    def test_post_start_quote_ingestion_or_forecast_never_enters_review(self):
        for field in ['quoted_at', 'ingested_at', 'forecast_at', 'decision_at']:
            for value in ['2026-09-30T20:00:00Z', 'invalid']:
                with self.subTest(field=field, value=value):
                    self.assertEqual(review.pregame([run('2026-09-30T19:59:00Z',
                        [quote(**{field: value})])], 'mlb'), {})
        for field in ['captured_at', 'model_prediction_at', 'model_checked_at']:
            with self.subTest(field=field):
                self.assertEqual(review.pregame([run('2026-09-30T19:59:00Z', [quote()],
                    **{field: '2026-09-30T20:00:00Z'})], 'mlb'), {})
        # A response received seconds after the request began is still pregame.
        good = run('2026-09-30T19:59:00Z', [quote(quoted_at='2026-09-30T19:59:05Z',
                   ingested_at='2026-09-30T19:59:06Z')])
        self.assertEqual(len(review.pregame([good], 'mlb')), 1)

    def test_replayed_snapshots_and_invalid_prices_are_not_pregame_evidence(self):
        self.assertEqual(review.pregame([run('2026-09-30T19:00:00Z', [quote()], local_replay=True)], 'nhl'), {})
        for price in [None, True, 0, 99, float('nan')]:
            self.assertEqual(review.pregame([run('2026-09-30T19:00:00Z', [quote(price=price)])], 'mlb'), {})

    def test_representative_uses_most_offered_paired_line_and_exact_best_price(self):
        rows = []
        for book, line, price in [('a', 8.5, -110), ('b', 8.5, 120), ('c', 7.5, 150)]:
            rows.extend([quote(book=book, line=line, price=price), quote(book=book, line=line, side='Under', price=-130)])
        # A lone high-price over and a duplicate book do not change the main line.
        rows.extend([quote(book='d', line=7.5, price=200), deepcopy(rows[4])])
        representative, = review.representatives(rows)
        self.assertEqual((representative['line'], representative['price'], representative['book']), (8.5, 120, 'b'))
        self.assertEqual(representative['paired_books'], 2)
        implied_under = 1 / review.decimal(-130)
        expected = sum((1 / review.decimal(p)) / (1 / review.decimal(p) + implied_under)
                       for p in [-110, 120]) / 2
        self.assertAlmostEqual(representative['market_probability'], expected)

    def test_tied_main_lines_never_create_an_unoffered_median_contract(self):
        rows = [quote(book=b, line=line, side=side) for b, line in [('a', 8), ('b', 9)] for side in ['Over', 'Under']]
        representative, = review.representatives(rows)
        self.assertEqual(representative['line'], 8)
        self.assertEqual(representative['reference_line'], 8)

    def test_pairs_require_same_book_opposite_spread_and_settlement(self):
        home = quote(market='spreads', side='Home', line=-1.5)
        away = quote(market='spreads', side='Away', line=1.5)
        self.assertEqual(len(review.representatives([home, away])), 1)
        for changes in [dict(book='other'), dict(line=-1.5), dict(price=None),
                        dict(settlement_verified=False), dict(settlement_profile='regulation_only')]:
            with self.subTest(changes=changes):
                self.assertEqual(review.representatives([home, dict(away, **changes)]), [])

    def test_missing_best_quote_forecast_stays_missing(self):
        rows = [quote(book=b, side=s, price=p, model_probability=prob)
                for b, p, prob in [('a', -110, .65), ('b', 120, None)] for s in ['Over', 'Under']]
        representative, = review.representatives(rows)
        self.assertIsNone(representative['model_probability'])
        self.assertIsNone(review.metrics([dict(representative, outcome='win')], 'mlb')['probability'])

    def test_exact_published_price_line_and_side_determine_flat_unit_return(self):
        win = review.grade(quote(side='Under', line=8.5, price=135), 'mlb', self.games, {})
        loss = review.grade(quote(side='Over', line=8.5, price=-150), 'mlb', self.games, {})
        push = review.grade(quote(line=8, price=-120), 'mlb', self.games, {})
        self.assertEqual((win['outcome'], win['line'], win['price']), ('win', 8.5, 135))
        self.assertAlmostEqual(win['units'], 1.35)
        self.assertEqual((loss['outcome'], loss['units']), ('loss', -1))
        self.assertEqual((push['outcome'], push['units']), ('push', 0))
        spread = review.grade(quote(market='spreads', side='Home', line=-1.5, price=-125), 'mlb', self.games, {})
        self.assertAlmostEqual(spread['units'], .8)
        moneyline = review.grade(quote(market='h2h', side='Away', line=None), 'mlb', self.games, {})
        self.assertEqual(moneyline['outcome'], 'loss')

    def test_unresolved_results_participation_and_stats_never_become_winning_unders(self):
        row = quote(market='batter_hits', player='Player', model_player_id=9, side='Under', line=.5)
        self.assertEqual(review.grade(row, 'mlb', {}, {})['grading_status'], 'official_result_unavailable')
        for players in [{}, {('1', '9'): dict(player='Player', batting=dict(hits=0, plateAppearances=0))},
                        {('1', '9'): dict(player='Player', batting=dict(plateAppearances=4))}]:
            result = review.grade(row, 'mlb', self.games, players)
            self.assertIsNone(result['outcome'])
            self.assertIsNone(result['units'])
        self.assertIsNone(review.grade(quote(), 'mlb', {'1': {'home_score': 5}}, {})['outcome'])

    def test_official_name_fallback_only_when_no_identifier_and_unique(self):
        row = quote(market='player_goals', player='José Example', side='Under', line=.5)
        players = {('1', '9'): dict(player='Jose Example', goals=0)}
        self.assertEqual(review.grade(row, 'nhl', self.games, players)['outcome'], 'win')
        self.assertIsNone(review.grade(dict(row, player_id=10), 'nhl', self.games, players)['outcome'])
        players[('1', '10')] = dict(player='Jose Example', goals=0)
        self.assertIsNone(review.grade(row, 'nhl', self.games, players)['outcome'])

    def test_pitcher_requires_start_and_total_bases_uses_official_components(self):
        row = quote(market='pitcher_outs', model_player_id=9, side='Under', line=15.5)
        relief = {('1', '9'): dict(pitching=dict(gamesStarted=0, outs=3))}
        self.assertEqual(review.grade(row, 'mlb', self.games, relief)['grading_status'], 'starting_pitcher_unconfirmed')
        starter = {('1', '9'): dict(pitching=dict(gamesStarted=1, outs=15))}
        self.assertEqual(review.grade(row, 'mlb', self.games, starter)['outcome'], 'win')
        batter = {('1', '9'): dict(batting=dict(plateAppearances=4, hits=3, doubles=1, triples=1, homeRuns=1))}
        result = review.grade(quote(market='batter_total_bases', model_player_id=9, line=8.5), 'mlb', self.games, batter)
        self.assertEqual((result['actual'], result['outcome']), (9, 'win'))

    def test_unverified_nhl_settlement_is_unresolved(self):
        for changes in [dict(settlement_verified=False), dict(settlement_profile='unverified:a:totals'),
                        dict(settlement_profile='regulation_only')]:
            result = review.grade(quote(**changes), 'nhl', self.games, {})
            self.assertEqual(result['grading_status'], 'settlement_unconfirmed')
            self.assertIsNone(result['outcome'])

    def test_brier_uses_matched_non_push_probabilities_and_keeps_missing_push_unknown(self):
        base = quote(line=8, model_probability=.54, model_push_probability=.1, market_probability=.5)
        win = dict(base, outcome='win', actual=9, model_mean=8.5)
        loss = dict(base, mlb_game_id=2, outcome='loss', actual=7, model_mean=8.5)
        rows = [win, loss, dict(base, outcome='push'), dict(base, outcome=None),
                dict(win, model_probability=None), dict(win, model_push_probability=None),
                dict(win, market_probability=2)]
        quality = review.metrics(rows, 'mlb')['probability']
        self.assertEqual((quality['n'], quality['games']), (2, 2))
        self.assertAlmostEqual(quality['model_brier'], (.4**2 + .6**2) / 2)
        self.assertAlmostEqual(quality['market_brier'], .25)
        self.assertEqual(len(quality['difference_ci95']), 2)
        self.assertIsNone(review.conditional(dict(base, model_push_probability=None)))
        self.assertAlmostEqual(review.conditional(dict(base, line=8.5, model_push_probability=None)), .54)
        self.assertIsNone(review.conditional(dict(base, model_probability=.95)))

    def test_roi_counts_push_stake_and_excludes_unresolved(self):
        rows = [dict(outcome='win', units=1.2), dict(outcome='loss', units=-1),
                dict(outcome='push', units=0), dict(outcome=None, units=None)]
        result = review.summary(rows)
        self.assertEqual((result['selected'], result['graded'], result['unresolved']), (4, 3, 1))
        self.assertAlmostEqual(result['units'], .2)
        self.assertAlmostEqual(result['roi'], .2 / 3)

    def test_official_mlb_cache_reproduces_grades_without_network(self):
        game = dict(gamePk=1, status=dict(abstractGameState='Final'), teams={
            'home': dict(score=5, team=dict(name='Home')), 'away': dict(score=3, team=dict(name='Away'))})
        pending = dict(game, gamePk=2, status=dict(abstractGameState='Live'))
        box = dict(teams=dict(home=dict(players={'ID9': dict(person=dict(id=9, fullName='Player'),
                   stats=dict(batting=dict(plateAppearances=4, hits=0)))})))
        fetch = Mock(side_effect=[dict(dates=[dict(games=[game, pending])]), box])
        with TemporaryDirectory() as td:
            live = review.mlb_results(date(2026, 9, 28), date(2026, 10, 4), {'1', '2'}, td, fetch)
            offline = review.mlb_results(date(2026, 9, 28), date(2026, 10, 4), {'1', '2'}, td,
                                         Mock(side_effect=AssertionError('network')), offline=True)
            self.assertEqual(live, offline)
            self.assertEqual(set(live[0]), {'1'})
            self.assertEqual(live[1][('1', '9')]['batting']['hits'], 0)
            self.assertEqual(fetch.call_count, 2)

    def test_missing_boxscore_keeps_team_result_but_no_player_zeroes(self):
        game = dict(gamePk=1, status=dict(abstractGameState='Final'), teams={
            'home': dict(score=5, team=dict(name='Home')), 'away': dict(score=3, team=dict(name='Away'))})
        fetch = Mock(side_effect=[dict(dates=[dict(games=[game])]), requests.HTTPError('missing')])
        with TemporaryDirectory() as td:
            games, players = review.mlb_results(date(2026, 9, 28), date(2026, 10, 4), {'1'}, td, fetch)
            self.assertEqual(games['1']['home_score'], 5)
            self.assertEqual(players, {})

    def test_final_morning_edition_keeps_historical_policy_without_reranking(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / 'docs/assets').mkdir(parents=True)
            shutil.copy(review.ROOT / 'docs/assets/briefing-picks.js', root / 'docs/assets/briefing-picks.js')
            cards = root / 'docs/briefing/cards'; cards.mkdir(parents=True)
            base = dict(schema_version=1, kind='morning', decision_date='2026-09-30',
                        published_at='2026-09-30T17:00:00Z', edition_id='first', policy_version='old-policy',
                        rows=[card_row(price=150)])
            editions = [base, dict(base, edition_id='last', published_at='2026-09-30T13:30:00-04:00',
                                   policy_version='saved-policy', rows=[card_row(price=-125, model_probability=None)]),
                        dict(base, edition_id='test', kind='test', published_at='2026-09-30T18:00:00Z'),
                        dict(base, edition_id='bad-schema', schema_version=2, published_at='2026-09-30T19:00:00Z')]
            for i, edition in enumerate(editions):
                (cards / f'{i}.json').write_text(json.dumps(edition))
            result = review.published_picks('mlb', date(2026, 9, 28), date(2026, 10, 4), root)
            row, = result['rows']
            self.assertEqual((row['edition_id'], row['policy_version'], row['price']), ('last', 'saved-policy', -125))
            self.assertIsNone(row['model_probability'])
            # A valid empty final edition represents a no-pick day, not the earlier tickets.
            (cards / 'empty.json').write_text(json.dumps(dict(base, edition_id='empty', rows=[],
                                                      published_at='2026-09-30T19:30:00Z')))
            self.assertEqual(review.published_picks('mlb', date(2026, 9, 28), date(2026, 10, 4), root)['rows'], [])

    def test_nhl_card_only_review_loads_season_and_reports_separate_coverage(self):
        with TemporaryDirectory() as td:
            root = Path(td); (root / 'docs/blog').mkdir(parents=True)
            (root / 'docs/blog/index.html').write_text('<!-- editorial-managed:end -->')
            (root / 'docs/sitemap.xml').write_text('<urlset></urlset>')
            cards = dict(rows=[quote(policy_version='old-policy', edition_id='one', published_at='2026-09-30T17:00:00Z'),
                               quote(policy_version='new-policy', edition_id='two', published_at='2026-09-30T17:30:00Z', side='Under')],
                         editions=[])
            with patch.object(review, 'load_runs', return_value=[]), patch.object(review, 'published_picks', return_value=cards), \
                    patch.object(review, 'nhl_results', return_value=(self.games, {})) as official:
                result = review.review('nhl', date(2026, 10, 4), root)
            self.assertEqual(official.call_args.args[1], {2026})
            self.assertEqual((result['games_with_quotes'], result['published_pick_games']), (0, 1))
            self.assertEqual(set(result['policies']), {'old-policy', 'new-policy'})
            article = (root / 'docs/blog/nhl-recap-2026-10-04.html').read_text()
            self.assertIn('Separately, 2 published morning picks cover 1 games', article)
            self.assertNotIn('evidence begins None', article)
            public = json.loads((root / 'docs/blog/nhl-recap-2026-10-04.json').read_text())
            self.assertEqual(len(public['tickets']), 2)
            self.assertEqual(public['board'], [])

    def test_nba_review_resolves_provider_ids_without_changing_board_or_ticket_contracts(self):
        with TemporaryDirectory() as td:
            root = Path(td); (root / 'docs/blog').mkdir(parents=True)
            (root / 'docs/blog/index.html').write_text('<!-- editorial-managed:end -->')
            (root / 'docs/sitemap.xml').write_text('<urlset></urlset>')
            over = quote(event_id='provider-hash', market='player_points', player='Player',
                         model_player_id=999999, line=25.5, price=-110, model_probability=None)
            under = dict(over, side='Under')
            archived = run('2026-09-30T19:00:00Z', [over, under], events=[dict(
                id='provider-hash', commence_time='2026-09-30T20:00:00Z')])
            ticket = dict(over, line=24.5, price=140, policy_version='saved-policy',
                          edition_id='card-one', published_at='2026-09-30T19:05:00Z')
            cards = dict(rows=[ticket], editions=[])
            with patch.object(review, 'load_runs', return_value=[archived]), \
                    patch.object(review, 'published_picks', return_value=cards):
                result = review.review('nba', date(2026, 10, 4), root, fetch=nba_feed())
            public = json.loads((root / 'docs/blog/nba-recap-2026-10-04.json').read_text())
            board, = public['board']; picked, = public['tickets']
            self.assertEqual(result['status'], 'published')
            self.assertEqual((result['games_with_quotes'], result['games_with_official_results']), (1, 1))
            self.assertEqual((board['event_id'], board['nba_game_id']), ('provider-hash', '401000001'))
            self.assertEqual((board['model_player_id'], board['espn_player_id']), (999999, '99'))
            self.assertEqual((board['line'], board['outcome'], board['units']), (25.5, 'loss', -1))
            self.assertEqual((picked['line'], picked['price'], picked['outcome']), (24.5, 140, 'win'))
            self.assertAlmostEqual(picked['units'], 1.4)
            self.assertEqual(result['policies']['saved-policy']['total']['wins'], 1)
            self.assertIsNone(result['probability'])

    def test_nba_official_tipoff_rechecks_model_capture_and_published_card_timing(self):
        from nba_weekly_results import fetch_results
        with TemporaryDirectory() as td:
            row = quote(event_id='provider-hash', commence_time='2026-09-30T20:10:00Z', line=222.5)
            for field in ['model_checked_at', 'captured_at']:
                with self.subTest(field=field):
                    archived = run('2026-09-30T19:59:00Z', [row], events=[],
                                   **{field: '2026-09-30T20:00:00Z'})
                    pregame = review.pregame([archived], 'nba')['provider-hash']['rows']
                    self.assertEqual(pregame[0]['evidence_at'], '2026-09-30T20:00:00+00:00')
                    games, players, resolved = fetch_results(date(2026, 9, 30), date(2026, 9, 30),
                        pregame, td, nba_feed())
                    result = review.grade(resolved[0], 'nba', games, players)
                    self.assertIsNone(result['outcome'])
                    self.assertEqual(result['grading_status'], 'evidence_at_or_after_official_tipoff')
            ticket = dict(row, published_at='2026-09-30T20:00:00Z')
            games, players, resolved = fetch_results(date(2026, 9, 30), date(2026, 9, 30),
                [ticket], td, nba_feed())
            self.assertEqual(review.grade(resolved[0], 'nba', games, players)['grading_status'],
                             'evidence_at_or_after_official_tipoff')
            # The exact same offered contract remains gradable with wholly pregame evidence.
            resolved[0]['published_at'] = '2026-09-30T19:59:59Z'
            self.assertEqual(review.grade(resolved[0], 'nba', games, players)['outcome'], 'win')
            # The consensus and main-line choice must also come wholly from pregame quotes.
            # An early Over cannot launder a late Under into its market probability.
            late_under = dict(row, side='Under', quoted_at='2026-09-30T20:00:01Z')
            archived = run('2026-09-30T19:59:00Z', [row, late_under], events=[])
            quotes = review.pregame([archived], 'nba')['provider-hash']['rows']
            representative = review.representatives(quotes)
            games, players, resolved = fetch_results(date(2026, 9, 30), date(2026, 9, 30),
                representative, td, nba_feed())
            self.assertEqual(review.grade(resolved[0], 'nba', games, players)['grading_status'],
                             'evidence_at_or_after_official_tipoff')

    def test_mlb_card_only_review_publishes_without_claiming_full_board_coverage(self):
        with TemporaryDirectory() as td:
            root = Path(td); (root / 'docs/blog').mkdir(parents=True)
            (root / 'docs/blog/index.html').write_text('<!-- editorial-managed:end -->')
            (root / 'docs/sitemap.xml').write_text('<urlset></urlset>')
            ticket = quote(side='Under', price=135, policy_version='historical-policy',
                           edition_id='old-card', published_at='2026-09-30T17:00:00Z')
            game = dict(gamePk=1, status=dict(abstractGameState='Final'), teams={
                'home': dict(score=5, team=dict(name='Home')), 'away': dict(score=3, team=dict(name='Away'))})
            fetch = Mock(side_effect=[dict(dates=[dict(games=[game])]), {}])
            with patch.object(review, 'load_runs', return_value=[]), patch.object(review, 'published_picks',
                    return_value=dict(rows=[ticket], editions=[])):
                result = review.review('mlb', date(2026, 10, 4), root, fetch=fetch)
            self.assertEqual(result['status'], 'published')
            self.assertEqual((result['games_with_quotes'], result['games_with_official_results']), (0, 0))
            self.assertEqual((result['published_pick_games'], result['published_pick_games_with_results']), (1, 1))
            self.assertEqual((result['board'], result['probability'], result['first_archived_game']), ({}, None, None))
            self.assertAlmostEqual(result['tickets']['units'], 1.35)
            self.assertEqual(result['tickets']['unresolved'], 0)
            self.assertTrue((root / 'docs/blog/mlb-recap-2026-10-04.html').exists())


if __name__ == '__main__':
    unittest.main()
