# YouTube publication package

Status: owner requested creation September 29, 2026. Narrated vertical master runtime: 1:46, 1080×1920, approximately 25MB. No sportsbook is named anywhere in the narration or on-screen text. Owner uploads to YouTube.

MP4: `docs/videos/synthetic-hold/synthetic-hold.mp4`

Thumbnail: `docs/videos/synthetic-hold/poster.png`

Scene spec: `content/videos/02-synthetic-hold.json` (voice: `scripts/generate_basics_voice.py --spec content/videos/02-synthetic-hold.json --output-dir docs/videos/synthetic-hold`; render: `VIDEO_SLUG=synthetic-hold node scripts/render_basics_video.cjs`).

## Recommended title

Synthetic Hold: The True Price of Any Betting Market | Betting Basics

Alternatives:
- Stop Paying 4.5% to Bet: Synthetic Hold Explained
- Same Bet, a Third Cheaper: How Synthetic Hold Works

## Description

Paste exactly as written. The chapter timestamps must stay in the description body starting at 00:00, or YouTube will not generate chapters.

---
Every sportsbook charges a fee to bet. At −110 on both sides, that's a 4.55% hold. But if you can shop more than one book, the price you really pay is set by the best number on each side. That's the synthetic hold.

We build a synthetic market step by step, turn it into fair odds, look at a real NFL board where the synthetic market was cheaper than every single book, and explain why a negative hold is rarely free money.

00:00 The price of a bet
00:11 One book at a time
00:19 Build the synthetic market
00:31 Do the math
00:45 Fair odds
00:56 A real board
01:09 Below zero: arbitrage
01:23 Why it matters on every bet
01:36 Price the market, not the book

Full explainer: https://fourthandvalue.com/blog/synthetic-hold.html
Synthetic Hold Calculator: https://fourthandvalue.com/tools/synthetic-hold.html
Episode 01, why we devig: https://fourthandvalue.com/videos/

Analysis, not a guarantee. No outcome or profit is guaranteed. Fourth & Value does not accept wagers. Must be of legal age where you live. Gambling problem? Call or text 1-800-MY-RESET.
---

## Tags

synthetic hold, sports betting math, line shopping, vig, hold, fair odds, devig, arbitrage betting, NFL betting, betting basics

## Notes

- Sportsbook names are deliberately absent from all narration and on-screen text (YouTube gambling-policy safety). The real-board scene labels books "Book 1–4"; the article keeps the full named price table.
- The Bills/Dolphins spread is hypothetical and is labeled that way on screen. The Eagles–Bears figures come from the September 28, 2026 briefing snapshot.
- The negative-hold scene is framed as a price observation with explicit caveats (stale quotes, limits), not a profit claim.
