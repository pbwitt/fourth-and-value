"""Eastern Time presentation policy; source timestamps stay unchanged."""
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo('America/New_York')
TIMEZONE_INSTRUCTIONS = '''
Prose times: convert to America/New_York (EST/EDT), including date rollover;
never relabel UTC or print ISO clocks. Keep source fields/excerpts intact.
Audit conversions against evidence.'''
# Review packets have a tighter byte ceiling than article-writing requests.
REVIEW_TIMEZONE_INSTRUCTIONS = '\nProse dates/times: America/New_York (EST/EDT).'


def eastern_label(value):
    """'Wed, Oct 14, 2026, 7:08 AM EDT': the Eastern weekday, date and time, ready to quote."""
    t = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError('A source time zone is required for Eastern conversion')
    t = t.astimezone(EASTERN)
    return f"{t:%a}, {t:%b} {t.day}, {t.year}, {t.hour % 12 or 12}:{t:%M} {'AM' if t.hour < 12 else 'PM'} {t:%Z}"


def eastern_time(value, fmt='%b %d, %Y at %I:%M %p %Z'):
    """Format an aware datetime or ISO timestamp without changing its instant."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('A source time zone is required for Eastern conversion')
    return value.astimezone(EASTERN).strftime(fmt)
