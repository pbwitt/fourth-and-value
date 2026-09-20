# AI game insights

The Insights page is generated from the same filtered, calibrated prop rows
used by the board. Each game prompt includes only evidence-backed rows with a
positive edge, a model probability, a market probability, and at least three
books. It also includes the exact side, line, price, book count, price-shopping
data, and any target-week injury designations available in the normalized
injury snapshot.

The workflow sets `OPENAI_INSIGHTS_MODEL=gpt-6-astra`. The generator reads
that variable (or accepts `--model`) and uses the OpenAI Responses API. The
model must be enabled for the project associated with `OPENAI_API_KEY`; if the
requested model is unavailable or the account has no quota, the generator
falls back to deterministic, data-backed prose for every game. It never
invents a pick from an empty evidence set.

The fallback is intentionally specific: it names the player, side, market,
line, price, model probability, market probability and book count. This keeps
the page useful during API outages and makes it clear what a later model
summary should be grounded in.

OpenAI’s official API quickstart shows the Responses API pattern used here:
<https://platform.openai.com/docs/quickstart/make-your-first-api-request>.

Insights generation is opt in. The normal scheduled NFL refresh skips the AI
target and leaves the last published Insights page unchanged. To refresh it,
run the **NFL Weekly Update** workflow manually in GitHub Actions and enable
**Generate GPT-6 game insights**. That checkbox is the credit-spend permission;
leaving it off still refreshes the odds, models, and other site pages.
