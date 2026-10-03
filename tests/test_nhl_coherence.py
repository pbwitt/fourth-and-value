from pathlib import Path
import gzip
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from nhl.v2.arbitrage import contains, find_incoherent
from nhl.v2.coherence import (FLAG_EV, archive, bounds, coherence, evaluate, fit_regulation, game_probabilities,
                              implied_mean, outcome, public, render, report)
from nhl.v2.pricing import compare

PARAMS = dict(version='test-model', means_fields=['opportunity_means']*4, alphas=[.08, .025, .025, .025], ot_home=.52,
              artifact_sha256='x')
# Shots, goals, assists, points: a 10% shooter whose points are 43% goals.
MODEL = [3.0, .3, .4, .7]
GAME = dict(event_id='e1', commence_time='2026-10-10T23:00:00Z', home_team='Carolina Hurricanes',
            away_team='Florida Panthers', nhl_game_id=7)
EVENTS = [dict(nhl_game_id=7, home_id=12, away_id=13)]


def american(p):
    dec = 1/p
    return round(100*(dec-1)) if dec >= 2 else round(-100/(dec-1))


def pair(book, market, line, over, player='A Skater', at='2026-10-10T18:00:00Z', margin=.04, profile='nhl_player_ot_no_so_participation'):
    """Two-sided quotes whose de-vigged Over probability is `over`."""
    base = dict(GAME, book=book, market=market, line=line, player=player, quoted_at=at,
                settlement_profile=profile, settlement_verified=True)
    sides = ['Over', 'Under'] if market not in ('h2h',) else [GAME['home_team'], GAME['away_team']]
    return [dict(base, side=sides[0], price=american(over*(1+margin))),
            dict(base, side=sides[1], price=american((1-over)*(1+margin)))]


def over(mean, alpha, line):
    p = outcome(mean, alpha, line, 'Over')
    return p['win']/(1-p['push'])


def board(extra=()):
    """Shots imply 3.3 shots; goals imply 0.30 goals; same player, two books."""
    raw = [*pair('draftkings', 'player_shots_on_goal', 2.5, over(3.3, .08, 2.5)),
           *pair('fanduel', 'player_shots_on_goal', 2.5, over(3.3, .08, 2.5)),
           *pair('fanduel', 'player_goals', 0.5, over(.30, .025, .5)),
           *pair('draftkings', 'player_goals', 0.5, over(.30, .025, .5)),
           *extra]
    rows = compare(raw)
    for r in rows:
        if r['market'].startswith('player_'):
            r.update(player_id=99, player_team_id=12, model_inputs=dict(opportunity_means=MODEL),
                     estimated_ev=-.01, conditional_probability=.4)
    return rows


class CoherenceMathTests(unittest.TestCase):
    def test_implied_mean_inverts_half_and_whole_lines(self):
        for alpha in (0, .08):
            for line in (.5, 2.5, 3.0):
                self.assertAlmostEqual(implied_mean(over(2.7, alpha, line), line, alpha), 2.7, places=4)
        self.assertIsNone(implied_mean(1.0, 2.5, .08))

    def test_team_fit_reproduces_moneyline_and_total(self):
        home, away = fit_regulation(.62, .48, 6.5, .52)
        win, over_p = game_probabilities(home, away, 6.5, .52)
        self.assertAlmostEqual(win, .62, places=4)
        self.assertAlmostEqual(over_p, .48, places=4)
        self.assertGreater(home, away)


class CoherenceSignalTests(unittest.TestCase):
    def test_goal_offer_uses_shots_market_through_model_ratio_never_its_own_price(self):
        entries = coherence(dict(rows=board(), events=EVENTS), PARAMS)
        goal = next(e for e in entries if e['market'] == 'player_goals' and e['side'] == 'Over')
        self.assertEqual(set(goal['sources']), {'shots'})
        # Whole-number American odds round the quoted probabilities slightly.
        self.assertAlmostEqual(goal['sources']['shots']['mean'], 3.3*.3/3.0, places=2)
        self.assertAlmostEqual(goal['sources']['shots']['model_ratio'], .1)
        self.assertAlmostEqual(goal['own_market_mean'], .30, places=2)
        shots = next(e for e in entries if e['market'] == 'player_shots_on_goal' and e['side'] == 'Over')
        self.assertEqual(set(shots['sources']), {'goals'})
        self.assertAlmostEqual(shots['sources']['goals']['mean'], .30*3.0/.3, places=2)

    def test_one_link_never_flags_two_agreeing_links_can(self):
        # The shots market implies a 0.33-goal mean but the goal price assumes 0.30: one link only.
        rows = board()
        entries = coherence(dict(rows=rows, events=EVENTS), PARAMS)
        goal = next(e for e in entries if e['market'] == 'player_goals' and e['side'] == 'Over' and e['best_price'])
        self.assertGreater(goal['combined']['ev'], 0)
        self.assertFalse(goal['flagged'])
        # A points market implying more scoring adds a second agreeing link.
        rows = board(pair('draftkings', 'player_points', .5, over(.85, .025, .5)))
        entries = coherence(dict(rows=rows, events=EVENTS), PARAMS)
        goal = next(e for e in entries if e['market'] == 'player_goals' and e['side'] == 'Over' and e['best_price'])
        self.assertEqual(goal['combined']['sources'], ['shots', 'points'])
        self.assertEqual(goal['flagged'], goal['combined']['ev'] >= FLAG_EV)
        self.assertTrue(goal['flagged'])

    def test_team_top_down_scales_model_mean_by_market_team_goals(self):
        rows = board([*pair('fanduel', 'h2h', None, .62, player='', profile='nhl_full_game_ot_so'),
                      *pair('fanduel', 'totals', 6.5, .48, player='', profile='nhl_full_game_ot_so')])
        for r in rows:
            if r['market'] in ('h2h', 'totals'):
                r.update(projected_home_reg_goals=3.0, projected_away_reg_goals=2.8)
        goal = next(e for e in coherence(dict(rows=rows, events=EVENTS), PARAMS)
                    if e['market'] == 'player_goals' and e['side'] == 'Over')
        home, _ = fit_regulation(.62, .48, 6.5, .52)
        self.assertAlmostEqual(goal['sources']['team']['mean'], .3*home/3.0, places=2)
        # A player not on either team gets no team estimate.
        for r in rows:
            r['player_team_id'] = 55
        goal = next(e for e in coherence(dict(rows=rows, events=EVENTS), PARAMS)
                    if e['market'] == 'player_goals' and e['side'] == 'Over')
        self.assertNotIn('team', goal['sources'])

    def test_stale_or_unverified_sources_are_excluded(self):
        late = board()
        for r in late:
            if r['market'] == 'player_shots_on_goal':
                r.update(paired_at_min='2026-10-10T19:00:00+00:00', paired_at_max='2026-10-10T19:00:00+00:00')
        entries = coherence(dict(rows=late, events=EVENTS), PARAMS)
        self.assertFalse(any(e['market'] == 'player_goals' for e in entries))
        rows = board()
        for r in rows:
            if r['market'] == 'player_goals':
                r['settlement_verified'] = False
        self.assertFalse(any(e['market'] == 'player_goals' for e in coherence(dict(rows=rows, events=EVENTS), PARAMS)))

    def test_missing_model_or_parameters_produce_no_estimates(self):
        rows = board()
        for r in rows:
            r['model_inputs'] = None
        self.assertEqual(coherence(dict(rows=rows, events=EVENTS), PARAMS), [])
        data = report(dict(rows=board(), events=EVENTS, status='ready', snapshot_id='s'), None)
        self.assertEqual(data['status'], 'model_distribution_unavailable')
        self.assertIn('Model distribution unavailable', render(data))


class ContainmentTests(unittest.TestCase):
    def row(self, market, side, line, **extra):
        return {**GAME, 'player': 'A Skater', 'book': 'draftkings', 'settlement_profile': 'p',
                'market': market, 'side': side, 'line': line, **extra}

    def test_containment_relations(self):
        r = self.row
        self.assertTrue(contains(r('player_shots_on_goal', 'Over', 1.5), r('player_goals', 'Over', 1.5)))
        self.assertTrue(contains(r('player_points', 'Over', .5), r('player_goals', 'Over', 1.5)))
        self.assertTrue(contains(r('player_shots_on_goal', 'Over', 1.5), r('player_shots_on_goal', 'Over', 2.5)))
        self.assertTrue(contains(r('player_goals', 'Under', .5), r('player_points', 'Under', .5)))
        self.assertFalse(contains(r('player_goals', 'Over', .5), r('player_points', 'Over', .5)))
        self.assertFalse(contains(r('player_points', 'Over', 1.5), r('player_goals', 'Over', .5)))
        self.assertFalse(contains(r('player_shots_on_goal', 'Over', 2.0), r('player_shots_on_goal', 'Over', 3.0)))
        self.assertFalse(contains(r('player_shots_on_goal', 'Over', 1.5), r('player_assists', 'Over', 1.5)))
        team = dict(player='', side='Carolina Hurricanes')
        self.assertTrue(contains(r('h2h', **team, line=None), r('spreads', **team, line=-1.5)))
        self.assertTrue(contains(r('spreads', **team, line=1.5), r('h2h', **team, line=None)))
        self.assertFalse(contains(r('h2h', **team, line=None), r('spreads', **team, line=1.5)))

    def test_same_book_ladder_contradiction(self):
        rows = [self.row('player_shots_on_goal', 'Over', 1.5, price=-110, quoted_at='2026-10-10T18:00:00Z', game='g'),
                self.row('player_shots_on_goal', 'Over', 2.5, price=-120, quoted_at='2026-10-10T18:00:00Z', game='g')]
        found = find_incoherent(rows)
        self.assertEqual([(f['narrow']['line'], f['wide']['line']) for f in found], [(2.5, 1.5)])

    def test_cross_book_floor_from_narrower_fair_price(self):
        rows = compare([*pair('fanduel', 'player_goals', .5, .40),
                        dict(GAME, book='draftkings', market='player_points', line=.5, side='Over', player='A Skater',
                             price=160, quoted_at='2026-10-10T18:00:00Z',
                             settlement_profile='nhl_player_ot_no_so_participation', settlement_verified=True)])
        found = bounds(rows)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0]['market'], found[0]['narrow']['market']), ('player_points', 'player_goals'))
        self.assertAlmostEqual(found[0]['bound_probability'], .40, places=2)
        self.assertAlmostEqual(found[0]['bound_ev'], found[0]['bound_probability']*2.6-1)
        rows[-1]['price'] = 140
        self.assertEqual(bounds(rows), [])


class ArchiveAndGradingTests(unittest.TestCase):
    def state(self, snapshot='s1'):
        rows = board(pair('draftkings', 'player_points', .5, over(.85, .025, .5)))
        return dict(rows=rows, events=EVENTS, status='ready', snapshot_id=snapshot,
                    checked_at='2026-10-10T18:05:00Z', decision_session='afternoon')

    def test_archive_is_written_once_and_public_feed_omits_entries(self):
        data = report(self.state(), PARAMS)
        self.assertNotIn('entries', public(data))
        json.dumps(public(data), allow_nan=False)
        with tempfile.TemporaryDirectory() as d:
            path = archive(data, d, computed_at='2026-10-10T18:06:00Z')
            first = path.read_bytes()
            archive(dict(data, offers_checked=-1), d)
            self.assertEqual(path.read_bytes(), first)
            with gzip.open(path, 'rt') as f:
                saved = json.load(f)
            self.assertEqual((saved['rule_version'], saved['backfilled']), ('nhl-coherence-1', False))
            self.assertIsNone(archive(dict(data, feed_status='feed_error', snapshot_id='s2'), d))

    def test_grading_scores_sources_and_separates_backfills(self):
        data = report(self.state(), PARAMS)
        records = [dict(data, backfilled=False), dict(report(self.state('s0'), PARAMS), backfilled=True)]
        games = [dict(game_id=7, home_score=3, away_score=2)]
        players = [dict(game_id=7, player_id=99, shots=4, goals=1, assists=0, points=1)]
        result = evaluate(records, games, players)
        prospective = result['cohorts']['prospective test-model']
        self.assertIn('combined:player_goals', prospective['probability_quality'])
        self.assertIn('shots:player_goals', prospective['probability_quality'])
        # One graded ticket per flagged outcome, however many books or snapshots flagged it.
        outcomes = {(e['market'], e['line'], e['side']) for e in data['entries'] if e['flagged']}
        self.assertGreater(len(outcomes), 0)
        self.assertEqual(prospective['flagged']['count'], len(outcomes))
        self.assertTrue(all(r['result'] in ('won', 'lost') for r in prospective['flagged_rows']))
        self.assertIn('backfilled test-model', result['cohorts'])
        # Unknown participation is unresolved, never a loss.
        unresolved = evaluate(records, games, [])['cohorts']['prospective test-model']
        self.assertEqual(unresolved['probability_quality'], {})
        self.assertTrue(all(r['result'] == 'unresolved_participation' for r in unresolved['flagged_rows']))

    def test_later_movement_needs_a_later_snapshot(self):
        first = report(self.state('s1'), PARAMS)
        games = [dict(game_id=7, home_score=3, away_score=2)]
        players = [dict(game_id=7, player_id=99, shots=4, goals=1, assists=0, points=1)]
        rows = evaluate([dict(first, backfilled=False)], games, players)['cohorts']['prospective test-model']['flagged_rows']
        self.assertTrue(rows and all(r['later_consensus_move'] is None for r in rows))
        later = report(dict(self.state('s2'), checked_at='2026-10-10T20:05:00Z'), PARAMS)
        for e in later['entries']:
            e['consensus_probability'] += .02
        rows = evaluate([dict(first, backfilled=False), dict(later, backfilled=False)], games, players)
        moves = [r['later_consensus_move'] for r in rows['cohorts']['prospective test-model']['flagged_rows']]
        self.assertTrue(moves and all(abs(m-.02) < 1e-9 for m in moves))

    def test_each_model_version_is_its_own_cohort(self):
        games = [dict(game_id=7, home_score=3, away_score=2)]
        players = [dict(game_id=7, player_id=99, shots=4, goals=1, assists=0, points=1)]
        old = report(self.state('s1'), dict(PARAMS, version='old-model'))
        new = report(self.state('s2'), PARAMS)
        cohorts = evaluate([dict(old, backfilled=True), dict(new, backfilled=False)], games, players)['cohorts']
        self.assertEqual(set(cohorts), {'backfilled old-model', 'prospective test-model'})

    def test_snapshot_parameters_come_only_from_the_artifact_that_made_it(self):
        from nhl.v2.coherence import params_for
        retired = json.loads((Path(__file__).resolve().parents[1]/'models/nhl/v2/retired-distributions.json').read_text())
        sha, v21 = next(iter(retired.items()))
        self.assertEqual(params_for(dict(model_manifest=dict(artifact_sha256=sha)))['version'], 'nhl-v2.1')
        self.assertEqual(params_for(dict(model_distribution=PARAMS, model_manifest=dict(artifact_sha256=sha))), PARAMS)
        self.assertIsNone(params_for(dict(model_manifest=dict(artifact_sha256='unknown'))))
        self.assertIsNone(params_for(dict()))

    def test_site_build_survives_a_coherence_failure(self):
        from unittest.mock import patch
        from nhl.site import build as site_build
        with tempfile.TemporaryDirectory() as d, patch('nhl.site.ROOT', Path(d)), patch('nhl.site.metadata', return_value=''), \
                patch('nhl.v2.coherence.report', side_effect=RuntimeError('boom')):
            site_build(dict(season=20262027, rows=[], status='ready'), archive=True)
            self.assertIn('Cross-market checks are unavailable', (Path(d)/'docs/nhl/arbitrage.html').read_text())
            self.assertEqual(json.loads((Path(d)/'docs/nhl/data/coherence.json').read_text())['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
