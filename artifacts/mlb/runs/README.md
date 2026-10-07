# MLB pregame archive

One gzip JSON file per MLB refresh (`<UTC stamp>.json.gz`), written by
`scripts/mlb/archive.py`: the pregame quotes (per book) with the forecast attached
to each, and the scheduled games with probable pitchers. Weekly recaps grade each
game from the last snapshot saved before first pitch. Files are never overwritten.
