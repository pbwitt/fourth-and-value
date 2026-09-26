# NHL evidence archive

Daily inference writes content-addressed gzip objects and append-only run manifests here.
They contain public NHL statistics, authorized-feed responses without credentials, and
the exact forecast snapshot. They are committed with the daily public refresh after this
branch is approved and merged. Git history retains corrected/replaced forecasts.

The training source archive is `training-sources.tar.gz`; it contains the 44 raw source
pages actually used by the four-season evaluation plus the season manifests. Original
publication time is unknown; ingestion times are recorded. Data availability in a past
simulation is reconstructed, not certified by a retrieval timestamp from 2026.

Daily source pages are deduplicated. Actions artifacts additionally retain diagnostics
for 90 days; those are not the sole durable archive. Monitor Git growth monthly. If this
archive approaches 500 MB, migrate the existing objects to an approved durable store
and retain content hashes and immutable references before changing retention. Do not
purchase storage or silently delete evidence as part of inference.

All archive content stays outside `docs/`; it is not a new public site route.
