# NHL season preview — production notes

The owner requested an NHL preview video and asked whether it should include betting markets. The default treatment combines hockey storylines with market analysis, in the established Fourth & Value dark navy/mint landscape format and Cedar narration.

## Editorial treatment

Twelve scenes, 1,053 spoken words, approximately 6:20 of generated narration and transitions. The canonical script, on-screen cards and per-scene sources are in `nhl-season-preview-2026.json`. The companion `-youtube.md` contains the title, description, chapters and pinned comment.

The script starts with hockey and deployment, explains the model's limited evidence, then uses two explicitly hypothetical examples to explain line settlement and price. It is not a full 32-team ranking or a slate of current betting recommendations. No live odds acquisition was required.

Existing site material:

- [Preseason case study](https://fourthandvalue.com/editorial/articles/2026-09-22-3-nhl.html): a September 22 assessment, not a full season preview. Its old model-withholding language predates the replacement launch and is not repeated as the current system status.
- [NHL methods paper](https://fourthandvalue.com/research/nhl-forecasting-and-market-pricing.html): source of the model evaluation summary and limitations.
- [Current daily process](https://fourthandvalue.com/research/daily-process.html): operational reference, without a promised posting time or pick quota.

## Betting discussion and YouTube

[YouTube's policy](https://support.google.com/youtube/answer/9229611?hl=en) prohibits facilitating access to uncertified gambling sites and promising guaranteed gambling returns. Some gambling content can be age-restricted. Educational framing is not a blanket exemption. This package discusses forecasts and markets without provider promotion or signup links; the owner should assess the final upload's settings. Omitting bookmaker names is an editorial choice, not a claim that all bookmaker mentions are categorically prohibited.

## Production and validation

Generated with the repository's existing narration and landscape-rendering tools. This task does not change scheduled sports research or its budget. The narration helper now preserves the supplied production status instead of assigning every video the old September 22 approval date.

- All scene sources map to the source manifest; the script is under the speech endpoint's per-scene input limit.
- Example arithmetic: 55% at −110 yields +$5.00 per $100; at −135, −$4.26. A half-point line has no push in the example.
- Each WAV has a valid duration and non-silent PCM signal. This is a signal check, not a word-level listening review.
- All 12 scene layouts were checked in Chrome for text escaping safe bounds; representative frames and the thumbnail were visually inspected.
- Phrase captions are approximate timing, not forced alignment.
- The full export is decoded and sampled after rendering; final results are recorded in the render manifest.

## Reproduction

Use the existing Python environment with `requests` and `python-dotenv`, Node with Playwright, and Chrome. Keep the existing OpenAI key in the environment; never add it to a committed command or file. The speech API is a usage-billed production dependency; cached scene audio prevents repeat requests for unchanged narration. No paid advertisement or subscription was purchased.

```bash
python scripts/narrate_editorial_video.py content/videos/nhl-season-preview-2026.json --approved
VIDEO_SLUG=nhl-season-preview-2026 node scripts/render_editorial_video.cjs
```

Set `NODE_PATH` if Playwright is installed outside this checkout. `CHROME_PATH` can identify the local Chrome executable. Approval for this video's creation is the owner's request; the command flag does not authorize a YouTube upload. The [official speech guide](https://developers.openai.com/api/docs/guides/text-to-speech) documents Cedar and the voice-disclosure requirement used in the description.

The audio and video are dated production snapshots. Regenerating with a model alias may produce a different voice rendering; the cached WAVs and hashes preserve this edition. The site Videos page features the master. The owner handles the YouTube upload.
