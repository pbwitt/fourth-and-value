"""Freeze three decision versions over one eligible opportunity universe, before outcomes.

Version 1, baseline: the numerical-model policy alone (same screen, ranking, idea
consolidation and card limit as the morning card, with research ignored).
Version 2, research_filtered: the published card policy (research gates applied).
Version 3, adjusted: predictions and decisions after supported input adjustments.
Only adjustments produced by numerical code from supported facts count (currently the
NHL shots deployment pilot, shadow mode). Every other row records
`none_supported` and keeps its version 2 decision.

Each ledger records selections AND passes with reasons, the exact offer, frozen
probabilities, research state and fact IDs, rules and code fingerprints. Ledgers are
immutable (content-addressed; an existing file with other content is a conflict) and
are never modified by settlement or grading, which write separate files.
Hypothetical decisions here are never wagers; Bet Tracker records actual stakes.
"""
from copy import deepcopy
import hashlib
import json
import math

from nhl.v2.data import ROOT, digest, iso, stamp
from nhl.analyst import immutable

SCHEMA = 'decision-ledger-1'
LEDGERS = ROOT/'artifacts/research/decisions'
VERSIONS = ('baseline', 'research_filtered', 'adjusted')
CARD_LIMIT = 10


def fingerprint(*paths):
    return hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in paths)).hexdigest()[:20]


def decimal(price):
    return 1+price/100 if price > 0 else 1+100/abs(price)


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def probabilities(r):
    """Win probability conditional on no push, push mass, paired market and break-even.

    Same semantics as comparison() in docs/assets/briefing-picks.js. Missing values stay
    None; a book probability is never substituted for a missing model.
    """
    sport = r['sport']
    if r.get('model_withheld'):
        win_cond, push = None, None
    elif sport == 'NFL':
        win_cond, push = r.get('model_prob'), r.get('push_prob')
    else:
        win = r.get('model_probability') if sport == 'MLB' else r.get('final_probability')
        push = r.get('model_push_probability') if sport == 'MLB' else r.get('push_probability')
        win_cond = win/(1-push) if finite(win) and finite(push) and 0 <= push < 1 and 0 <= win <= 1-push else None
    market = r.get('consensus_prob') if sport == 'NFL' else r.get('other_book_probability') if sport == 'MLB' else r.get('market_probability')
    books = r.get('book_count') if sport == 'NFL' else r.get('other_books')
    return dict(model_win_conditional=win_cond if finite(win_cond) else None, push=push if finite(push) else None,
                market_conditional=market if finite(market) and finite(books) and books > 0 else None,
                market_books=books if finite(books) else None,
                market_basis='paired_consensus_includes_offer' if sport == 'NFL' else 'other_books_conditional_on_nonpush',
                raw_probability=(r.get('forecast_health') or {}).get('raw_probability'),
                break_even=1/decimal(r['price']))


def expected_value(win_cond, push, price):
    """EV per unit = P(win)·(d−1) − P(loss), with P(win) unconditional; None if unknown."""
    if not finite(win_cond) or not finite(push):
        return None
    win = win_cond*(1-push)
    return win*(decimal(price)-1)-(1-win-push)


def outcome_key(r):
    return json.dumps([r['sport'], str(r['game_id']), r.get('player') or '', r.get('market_std') or r.get('market'),
                       str(r['side']).lower(), r.get('line'), r.get('settlement_profile') or ''])


def idea_keys(r):
    base = [r['sport'], str(r['game_id']), r.get('player') or '', r.get('market_std') or r.get('market'),
            str(r['side']).lower(), r.get('settlement_profile') or '']
    keys = [json.dumps(base)]
    if r['sport'] == 'MLB' and r.get('line') == .5 and r.get('market') in ('batter_hits', 'batter_total_bases'):
        keys.append(json.dumps(base[:3]+['batter_any_hit']+base[4:]))
    return keys


def _limit(ordered, eligible, value):
    """Card-style consolidation: one idea per subject/market/side, up to CARD_LIMIT."""
    seen, chosen = set(), []
    for r in ordered:
        if not eligible(r):
            continue
        keys = idea_keys(r)
        if any(k in seen for k in keys):
            continue
        seen.update(keys)
        chosen.append(r['_entry'])
        if len(chosen) >= CARD_LIMIT:
            break
    return chosen


def baseline_value(r):
    return r.get('card_rank_score', r.get('_card_value'))


def freeze(selection, card, now, *, edition_id, kind='morning', pilot=None, ledgers=LEDGERS, prompt_version=None):
    """Write and return the immutable ledger for one published edition.

    `selection` is the Node selector output (selected rows carry research_state and
    _card_value from scripts/analyst_shortlist.cjs); `card` is the published row list.
    `pilot` maps entry outcome keys to shadow adjustment records from the NHL pilot.
    """
    pilot = pilot or {}
    universe = [deepcopy(r) for r in selection['selected']]
    entries = []
    for r in universe:
        if not r.get('commence_time') or stamp(r['commence_time']) <= now:
            raise ValueError('Ledger entries must be frozen before the event starts')
        p = probabilities(r)
        entry = dict(entry_id=digest([outcome_key(r), r['book'], r['price'], r['quoted_at'], r.get('forecast_at')])[:24],
            outcome_key=outcome_key(r), sport=r['sport'], game_id=str(r['game_id']), game=r.get('game'),
            home_team=r.get('home_team'), away_team=r.get('away_team'), commence_time=r['commence_time'],
            player=r.get('player') or None, market=r.get('market_std') or r.get('market'), market_label=r.get('market_label'),
            side=r['side'], line=r.get('line'), book=r['book'], price=r['price'], quoted_at=r['quoted_at'],
            forecast_at=r.get('forecast_at'), model_version=r.get('model_version') or r.get('model_status'),
            player_id=r.get('player_id'), event_id=r.get('event_id'), nhl_game_id=r.get('nhl_game_id'),
            mlb_game_id=r.get('mlb_game_id'), settlement_profile=r.get('settlement_profile'),
            probabilities=p, expected_value=expected_value(p['model_win_conditional'], p['push'], r['price']),
            exposure_group=r.get('exposure_group'), discovery_origin=r.get('discovery_origin'),
            research=dict(state=r.get('research_state'), verdict=((r.get('qualitative_review') or {}).get('assessment') or {}).get('verdict'),
                          reviewed_at=(r.get('qualitative_review') or {}).get('reviewed_at'),
                          request_id=(r.get('qualitative_review') or {}).get('request_id'),
                          prompt_version=(r.get('qualitative_review') or {}).get('prompt_version'),
                          fact_ids=[f['fact_id'] for f in (r.get('qualitative_review') or {}).get('facts', [])],
                          failure=(r.get('research_failure') or {}).get('category')),
            decisions={})
        r['_entry'] = entry
        entries.append(entry)

    # Version 1: model and price alone, identical ordering units to the card.
    model_rows = [r for r in universe if r['_entry']['probabilities']['model_win_conditional'] is not None
                  and r.get('_card_value') is not None]
    ordered = sorted(model_rows, key=lambda r: (-(r['_card_value']), r['commence_time'], r['_entry']['outcome_key']))
    base = {e['entry_id'] for e in _limit(ordered, lambda r: True, None)}
    # Version 2: the published card is the research-filtered decision.
    card_ids = {digest([outcome_key(r), r['book'], r['price'], r['quoted_at'], r.get('forecast_at')])[:24] for r in card}
    for r in universe:
        e = r['_entry']
        if e['probabilities']['model_win_conditional'] is None or r.get('_card_value') is None:
            e['decisions']['baseline'] = dict(decision='not_eligible', reason='no_eligible_numerical_model')
        else:
            e['decisions']['baseline'] = dict(decision='select' if e['entry_id'] in base else 'pass',
                                              reason='model_rank_within_card_limit' if e['entry_id'] in base else 'below_card_limit_or_duplicate_idea')
        gate = (r.get('research_state') or {}).get('gate', 'not_reviewed')
        if e['entry_id'] in card_ids:
            e['decisions']['research_filtered'] = dict(decision='select', reason=gate)
        else:
            reason = gate if gate not in ('model_case_only', 'verified_context') else 'below_card_limit_or_duplicate_idea'
            if r.get('human_decision') == 'pass':
                reason = 'analyst_pass'
            e['decisions']['research_filtered'] = dict(decision='pass', reason=reason)
        # Version 3: supported numerical adjustments only, from code, never from the LLM.
        adj = pilot.get(e['outcome_key'])
        decision = dict(e['decisions']['research_filtered'], adjustment='none_supported',
                        model_win_conditional=e['probabilities']['model_win_conditional'])
        if adj and adj.get('status') == 'supported_adjustment':
            p_adj = adj['adjusted_win_conditional']
            ev = expected_value(p_adj, adj.get('push'), e['price'])
            decision.update(adjustment='nhl_deployment_pilot', model_win_conditional=p_adj, expected_value=ev,
                            pilot_record_id=adj['record_id'])
            if decision['decision'] == 'select' and (ev is None or ev < adj['ev_threshold']):
                decision.update(decision='pass', reason='adjusted_ev_below_threshold')
        elif adj:
            decision.update(adjustment=adj.get('status'), pilot_record_id=adj.get('record_id'),
                            scenario_range=adj.get('scenario_range'))
        e['decisions']['adjusted'] = decision
    selector = fingerprint('docs/assets/briefing-picks.js', 'scripts/analyst_shortlist.cjs')
    ledger = dict(schema=SCHEMA, edition_id=edition_id, kind=kind, decision_date=card_date(now),
        frozen_at=iso(now), universe_count=len(entries),
        rules=dict(card_limit=CARD_LIMIT, research_gates_selectable=['model_case_only', 'verified_context'],
                   nfl_screen='3% EV under min(raw, calibrated) probability', mlb_screen='validation gate, 2 paired books, 3% EV',
                   nhl_screen='2% EV and adverse-scenario minimum price', review_max_age_hours=3,
                   baseline='same screen, ordering and consolidation as the card; research ignored',
                   adjusted='research_filtered decisions; supported numerical adjustments re-evaluate eligibility only'),
        code=dict(selector_sha=selector, ledger_sha=fingerprint('scripts/decision_ledger.py'), prompt_version=prompt_version),
        counts={v: dict(select=sum(e['decisions'][v]['decision'] == 'select' for e in entries),
                        passed=sum(e['decisions'][v]['decision'] == 'pass' for e in entries),
                        not_eligible=sum(e['decisions'][v]['decision'] == 'not_eligible' for e in entries)) for v in VERSIONS},
        entries=entries,
        evaluation_status='prospective_shadow_only',
        note='Hypothetical decisions at recorded prices; not wagers. Settlement writes separate grade files.')
    ledger['entries_sha256'] = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    ledger['ledger_id'] = digest([edition_id, ledger['entries_sha256']])[:24]
    path = ledgers/f"{ledger['decision_date']}-{edition_id}.json"
    immutable(path, ledger)
    return ledger


def card_date(now):
    from zoneinfo import ZoneInfo
    return now.astimezone(ZoneInfo('America/New_York')).date().isoformat()


def verify(ledger):
    """Integrity check used by grading; raises if entries changed after freezing."""
    digest_now = hashlib.sha256(json.dumps(ledger['entries'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest_now != ledger['entries_sha256']:
        raise ValueError('Decision ledger was modified after freezing')
    if any(stamp(e['commence_time']) <= stamp(ledger['frozen_at']) for e in ledger['entries']):
        raise ValueError('Ledger contains an entry frozen after its start')
    return True
