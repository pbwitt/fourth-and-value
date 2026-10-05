"""Decision-relevant research facts as structured, point-in-time records.

A fact is built only from an evidence item that already passed local validation:
its source was retrieved before the review, matches the event and player/team, and
its excerpt is a verbatim contiguous quotation. The language model classifies the
fact; it never supplies a probability or an effect size. Numerical use of a fact is
decided by code (see scripts/nhl/v2/deployment_pilot.py) and only for facts that
are not already represented in model features.
"""
from datetime import timedelta
from urllib.parse import urlsplit

from nhl.v2.data import digest, iso, stamp

SCHEMA_VERSION = 'research-fact-1'
ASSUMPTIONS = ['participation', 'role', 'efficiency', 'workload_limit', 'lineup_slot', 'ice_time',
               'power_play', 'goalie', 'bullpen', 'manager_usage', 'weather', 'settlement', 'other']
MATERIALITY = ['consequential', 'minor']
VERIFICATION = ['official', 'original_report', 'secondary_report', 'conflicting', 'opinion']
EFFECTS = ['quantified', 'scenario', 'unresolved', 'none']
APPLIES = ['this_game', 'general_context']
# Verification levels that can establish an adverse fact. Opinion and conflicting
# reports can raise a question but cannot by themselves establish a fact.
ESTABLISHING = ('official', 'original_report', 'secondary_report')
OFFICIAL_HOSTS = ('nfl.com', 'mlb.com', 'nhl.com')


def schema_fields():
    """Additional evidence-item fields for prompt sports-research-7 (all required, enums)."""
    def enum(values):
        return dict(type='string', enum=values)
    return dict(assumption=enum(ASSUMPTIONS), materiality=enum(MATERIALITY), verification=enum(VERIFICATION),
                effect=enum(EFFECTS), applies_to=enum(APPLIES))


INSTRUCTIONS = '''
Classify each evidence item as a fact. assumption: forecast assumption affected. materiality:
consequential only if, when true, the selection's rationale fails or materially weakens; else minor.
verification: official (league/team report), original_report, secondary_report, conflicting
(sources disagree) or opinion (another outlet's pick). effect: quantified only when the excerpt
states the quantity (minutes, pitch limit, snap/route share, lineup slot); scenario if it implies a
direction without a number; unresolved if the fact is unsettled; else none. applies_to: this_game
or general_context. A consequential unresolved or conflicting fact requires wait. A verified
consequential concern that defeats the rationale requires pass. Never estimate any effect size.
'''


def _host(url):
    return (urlsplit(url).hostname or '').removeprefix('www.')


def overlap_guard(represented_in):
    """How a fact may be used numerically, given where it may already be reflected."""
    return {'model_features': 'excluded_already_in_model', 'hockey_features': 'excluded_already_in_model',
            'both': 'excluded_already_in_model', 'market_prices': 'market_may_reflect',
            'neither': 'not_represented', 'unknown': 'overlap_unknown'}.get(represented_in, 'overlap_unknown')


def build_facts(review, row, sources, decision_at, *, recorded_at=None):
    """Return research-fact-1 records for one validated review.

    `sources` maps source_id to the retrieved source (with excerpt). Legacy reviews
    without classification fields yield facts marked not_classified; they are never
    eligible for numerical use.
    """
    facts = []
    asof = stamp(decision_at)
    for item in review.get('evidence', []):
        source = sources.get(item['source_id'])
        if not source:
            continue
        retrieved = stamp(source['retrieved_at'])
        published = stamp(source['published_at']) if source.get('published_at') else None
        if retrieved > asof or (published and published > asof):
            # Never attach information that postdates the decision.
            raise ValueError('Fact source postdates the decision')
        classified = all(k in item for k in ('assumption', 'materiality', 'verification', 'effect', 'applies_to'))
        verification = item.get('verification', 'not_classified')
        basis = 'reviewer_classification' if classified else 'not_classified'
        if source.get('source_kind') == 'professional_opinion' and verification != 'opinion':
            verification, basis = 'opinion', 'source_kind_override'
        if verification == 'official' and _host(source['url']) not in OFFICIAL_HOSTS and \
                source.get('source_kind') not in ('official_injury_report',) and not _team_site(source['url']):
            # Only league/team publishers are official; keep the reported fact, downgrade the label.
            verification, basis = 'secondary_report', 'publisher_override'
        effective = stamp(source.get('updated_at') or source['published_at']) if source.get('published_at') else retrieved
        fact = dict(schema_version=SCHEMA_VERSION,
            candidate_id=row['candidate_id'], offer_id=row.get('offer_id'), forecast_id=row.get('forecast_id'),
            sport=row.get('sport'),
            event=dict(game=row.get('game'), game_id=str(row.get('game_id') or row.get('nhl_game_id') or ''),
                       commence_time=row.get('commence_time'), player=row.get('player') or None,
                       home_team=row.get('home_team'), away_team=row.get('away_team'),
                       market=row.get('market_std') or row.get('market'), side=row.get('side'), line=row.get('line')),
            source=dict(source_id=source['source_id'], url=source['url'], title=source.get('title'),
                        publisher=_host(source['url']), source_kind=source.get('source_kind', 'reporting'),
                        published_at=source.get('published_at'), updated_at=source.get('updated_at'),
                        retrieved_at=source['retrieved_at'], publication_basis=source.get('publication_basis'),
                        content_sha256=source.get('content_sha256')),
            excerpt=item['excerpt'], interpretation=item['interpretation'], kind=item.get('kind'),
            direction=item['direction'], represented_in=item.get('represented_in', 'unknown'),
            assumption=item.get('assumption', 'not_classified'), materiality=item.get('materiality', 'not_classified'),
            verification=verification, classification_basis=basis,
            effect=item.get('effect', 'not_classified'), applies_to=item.get('applies_to', 'not_classified'),
            usage=overlap_guard(item.get('represented_in')),
            freshness=dict(decision_at=iso(asof), source_age_hours=round((asof-effective).total_seconds()/3600, 2),
                           event_starts_after_decision=bool(row.get('commence_time')) and stamp(row['commence_time']) > asof),
            probability_adjustment=None, recorded_at=recorded_at or iso(asof))
        fact['fact_id'] = digest([fact['candidate_id'], fact['source'], fact['excerpt'], fact['direction']])[:20]
        facts.append(fact)
    # Same assumption, opposite directions: retain both and mark the conflict.
    for f in facts:
        f['conflicts_with'] = [g['fact_id'] for g in facts if g is not f and g['assumption'] == f['assumption']
                               and {f['direction'], g['direction']} == {'supports', 'concern'}]
        if f['conflicts_with'] and f['verification'] not in ('opinion',):
            f['verification_note'] = 'conflicting_evidence_in_review'
    return facts


def _team_site(url):
    from nhl.v2.evidence import NFL_TEAM_SITES
    host = _host(url)
    return any(host == h or host.endswith('.'+h) for h in NFL_TEAM_SITES.values())


def is_fresh(source, decision_at, max_age=timedelta(hours=72)):
    """Article freshness under the existing evidence policy; live tables use their own window."""
    effective = source.get('updated_at') or source.get('published_at')
    if not effective:
        return stamp(decision_at)-stamp(source['retrieved_at']) <= timedelta(minutes=90)
    return timedelta(0) <= stamp(decision_at)-stamp(effective) <= max_age
