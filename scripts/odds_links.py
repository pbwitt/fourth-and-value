"""Sportsbook links from The Odds API (requested with includeLinks=true).

Each bookmaker, market and outcome in an odds response can carry the book's own link:
an outcome's adds that bet to the book's bet slip; a market's or bookmaker's opens the
game. Some books' links hold a {state} placeholder for their state-specific sites, so
links are stored exactly as returned and the state is filled in per reader when a link
is shown (scripts/morning_email.py). Requesting links does not change a request's
credit cost, which depends only on markets and regions (scripts/odds_budget.py).
"""
import re

LEVELS = ('bet', 'market', 'event')
STATE = re.compile(r'\{state\}', re.I)


def bet_link(book, market, outcome):
    """The most specific link for this outcome and how far it goes: ('https://…', 'bet'),
    'market' or 'event', or (None, None) when the book sent none."""
    for level, source in zip(LEVELS, (outcome, market, book)):
        link = source.get('link') if isinstance(source, dict) else None
        if isinstance(link, str) and link.startswith('https://'):
            return link, level
    return None, None


def link_fields(book, market, outcome):
    """Row fields for a stored offer; empty when there is no link, so feeds stay small."""
    link, level = bet_link(book, market, outcome)
    return dict(bet_link=link, bet_link_level=level) if link else {}


def for_state(link, state):
    """The link for a reader in `state` (for example 'md'), or None when it needs a state
    we do not have or still holds a placeholder we cannot fill."""
    if not isinstance(link, str) or not link.startswith('https://'):
        return None
    if STATE.search(link):
        if not (isinstance(state, str) and re.fullmatch(r'[A-Za-z]{2}', state)):
            return None
        link = STATE.sub(state.lower(), link)
    return None if '{' in link or '}' in link else link


def is_state_specific(link):
    return isinstance(link, str) and bool(STATE.search(link))
