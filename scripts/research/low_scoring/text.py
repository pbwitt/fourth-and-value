"""Text fragments for FV-2026-04 that depend on the results: description, FAQ and links."""
import html

PREREG_URL = 'https://github.com/pbwitt/fourth-and-value/blob/main/reports/low-scoring-2026/PREREGISTRATION.md'


def FAQ(t):
    return [
        ('Does wind make NFL games go under?',
         f"Yes, by more than the betting line expects. In outdoor games with wind of 15 mph or more, scoring fell about {t['NFL_1_REP_EFFECT_ROUND']} "
         f"points compared with other outdoor games, while closing totals fell only about {t['NFL_1_REP_PRICED_ROUND']}. Unders won "
         f"{t['NFL_1_DIS_WIN']} of these games from 2006 to 2015 and {t['NFL_1_REP_WIN']} from 2016 to 2025."),
        ('Do back-to-backs lead to low-scoring NHL or NBA games?',
         'Not in a way the betting line misses. NHL back-to-backs did not change scoring at all, and NBA fatigue spots such as back-to-backs, '
         'three games in four nights and long road trips won 48% to 51% for the under from 2016 to 2025.'),
        ('Do referees or umpires affect over/unders?',
         'Not in a way you can predict from their history. Officials whose games went under most often in one decade saw about 50% unders '
         'in the next, in all four leagues.'),
        ('What is the closing total?',
         'The final over/under a sportsbook offers before a game starts. It reflects everything the betting market has learned, '
         'which makes it the toughest benchmark to beat.'),
        ('Which situations beat the closing total?',
         f"Four of {t['N_FACTORS']} met our pre-registered rule and the 52.4% break-even at −110: NFL wind of 15+ mph (the clearest), "
         f"MLB day games after night games and MLB getaway days (narrowly, and not in 2026), and NBA overs in the last two weeks of the "
         f"regular season (only in the later decade)."),
    ]


def TEXT_TAGS(tags):
    faq = FAQ(tags)
    return dict(
        PREREG_URL=PREREG_URL,
        FAQ_HTML='\n'.join(f'    <h3>{html.escape(q)}</h3>\n    <p>{html.escape(a)}</p>' for q, a in faq),
        META_DESCRIPTION=(f"We tested {tags['N_FACTORS']} weather, rest, travel and referee situations in {tags['N_GAMES']} NFL, NHL, NBA "
                          f"and MLB games against the closing total. Only NFL wind clearly won."),
    )
