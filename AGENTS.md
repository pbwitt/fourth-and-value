# AGENTS.md

## QA Review Instructions for Consensus Page

### 1. Build & Preview
- Run: `make monday_all SEASON=2025 WEEK=<X>`
- Confirm `docs/props/consensus.html` is rebuilt with correct title.
- Start server: `cd docs && python -m http.server 8010`

### 2. Overview Tab
- Columns: Player | Market | Market cons. line | Market cons. % | Mkt Odds | Fair | Books
- Confirm:
  - No “Edge” column present.
  - Market cons. % matches expected from consensus CSV.
  - “Fair” odds align with consensus %.
  - Switching Book filter updates “Mkt Odds”.
  - With “All books,” Mkt Odds shows compact multi-book summary.

### 3. Value Tab
- Columns: Kick | Game | Player | Market | Side | Line | Market cons. % | Model % | Edge (bps) | Best book bet
- Confirm:
  - Model % populates and matches props_with_model CSV.
  - Market cons. % matches consensus line for same side/line.
  - Edge (bps) = (Model% - Market%)*10,000.
  - Best book bet column has both odds and book name.

### 4. Filters
- Test all 4 top filters (Game, Player, Market, Book).
- Confirm filtering applies consistently to both Overview and Value.
- For Player input, confirm partial substring search works.

### 5. Navigation
- Nav bar present and consistent across pages.
- “Consensus” tab is highlighted.

---

### Step 3 — Commit it
```bash
git add AGENTS.md
git commit -m "Add AGENTS.md with QA review instructions for Consensus page"
git push origin main
```
