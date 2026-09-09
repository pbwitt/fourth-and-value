#!/usr/bin/env python3
"""The NFL shortlist uses the same pricing, filters and rendering as the full board."""
from build_props_site import main

if __name__ == '__main__':
    main(top_only=True)
