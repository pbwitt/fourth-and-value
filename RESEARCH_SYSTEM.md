# Research, reliability and model-validation system

Status: in progress (October 5, 2026). This file is the central developer reference
for the research pipeline. It is completed at the end of this change.

## Work plan (audit of 66e7b0a, rechecked against HEAD = 66e7b0a)

1. Verify each audit finding against saved artifacts; diagnose actual causes.
2. Separate research concepts (eligibility, evidence verification, direction,
   material blockers, completion) in data and labels; make failures actionable.
3. Repair NFL validation (cutoff reconstruction, separated calibration/evaluation,
   provenance, compatibility guard); strengthen MLB gates.
4. Structured research facts; NHL shots deployment pilot in shadow mode.
5. Three-version decision ledger with frozen records and shadow grading.
6. Documentation, process page, methods pages.
7. Tests, end-to-end fixture runs, rendered output check.

## Finding-to-change checklist

| # | Finding | Verified cause at HEAD | Change | State |
|---|---------|------------------------|--------|-------|
| 1 | Oct 5 8:41 card: five ideas, empty evidence, `needs_information`, “Consider” | Confirmed in `docs/briefing/cards/2026-10-05-466e6f14….json` | pending | open |
| 2 | 20/38 reviewed, failed batches, NHL refresh failure, ~$1.89 left | pending | pending | open |
| 3 | NFL calibration not refitted/validated after cutoff fix | pending | pending | open |
| 4 | MLB postseason pitcher-outs passes despite calibration error | pending | pending | open |
| 5 | NHL lines/PP/goalie incompletely represented | pending | pending | open |
| 6 | Extend existing research/decision tracking | pending | pending | open |
