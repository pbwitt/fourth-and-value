"""Text fragments for FV-2026-04 that depend on the results: description, FAQ and links."""

PREREG_URL = 'https://github.com/pbwitt/fourth-and-value/blob/main/reports/low-scoring-2026/PREREGISTRATION.md'


def TEXT_TAGS(tags):
    return dict(
        PREREG_URL=PREREG_URL,
        META_DESCRIPTION=(f"We tested {tags['N_FACTORS']} weather, rest, travel and referee situations in {tags['N_GAMES']} NFL, NHL, "
                          f"NBA and MLB games against the closing total. Few beat the line."),
    )


def FAQ(tags):
    return []
