/* Edit existing ledger rows without creating another ticket. */
(function (global) {
  'use strict';
  const ticketFields = ['league', 'game_date', 'team_home', 'team_away', 'player', 'market_type', 'side', 'line'];
  const fields = [...ticketFields, 'book', 'odds', 'stake_dollars', 'status', 'actual_result', 'payout'];
  const snapshotFields = [...fields, 'graded_timestamp', 'model_prob', 'edge_bps'];
  const numeric = new Set(['line', 'odds', 'stake_dollars', 'actual_result', 'payout', 'model_prob', 'edge_bps']);
  const blank = value => value == null || String(value).trim() === '';
  const same = (key, a, b) => blank(a) && blank(b) || (numeric.has(key) && !blank(a) && !blank(b)
    ? Number(a) === Number(b) : (a ?? null) === (b ?? null));
  const changed = (original, row, keys) => keys.some(key => !same(key, original[key], row[key]));
  const ticketChanged = (original, row) => changed(original, row, ticketFields);

  function amount(value, label, positive = false) {
    const n = Number(value);
    if (blank(value) || !Number.isFinite(n) || (positive ? n <= 0 : n < 0) ||
        !Number.isSafeInteger(Math.round(n * 100)) || Math.abs(n * 100 - Math.round(n * 100)) > 1e-6) {
      throw Error(`Enter ${positive ? 'a positive' : 'a nonnegative'} ${label} in dollars and cents.`);
    }
    return n;
  }

  function payout(stake, odds, status) {
    if (status === 'pending') return null;
    if (status === 'lost') return 0;
    if (status === 'push') return Number(stake);
    return Math.round(Number(stake) * (odds > 0 ? 1 + Number(odds) / 100 : 1 + 100 / Math.abs(odds)) * 100) / 100;
  }

  function buildUpdate(original, values, now = new Date().toISOString()) {
    const row = {};
    for (const key of fields) {
      const value = Object.hasOwn(values, key) ? values[key] : original[key];
      row[key] = blank(value) ? null : numeric.has(key) ? Number(value) : String(value).trim();
    }
    if (!row.league) throw Error('Choose a league.');
    if (row.game_date && (!/^\d{4}-\d{2}-\d{2}$/.test(row.game_date) ||
        !Number.isFinite(Date.parse(row.game_date)) || new Date(row.game_date).toISOString().slice(0, 10) !== row.game_date)) {
      throw Error('Enter a valid game date.');
    }
    if (!Number.isInteger(row.odds) || Math.abs(row.odds) < 100) throw Error('Enter valid American odds, such as -110 or +150.');
    row.stake_dollars = amount(row.stake_dollars, 'stake', true);
    if (row.line !== null && !Number.isFinite(row.line)) throw Error('Enter a valid line or leave it blank.');
    if (row.actual_result !== null && !Number.isFinite(row.actual_result)) throw Error('Enter a valid result or leave it blank.');
    if (!['pending', 'won', 'lost', 'push'].includes(row.status)) throw Error('Choose a result.');

    // A different selection must not retain the previous selection's result or model.
    if (ticketChanged(original, row)) {
      Object.assign(row, { status: 'pending', actual_result: null, payout: null, graded_timestamp: null,
        model_prob: null, edge_bps: null });
      if (Object.hasOwn(original, 'player_team') && changed(original, row, ['league', 'game_date', 'team_home', 'team_away', 'player'])) row.player_team = null;
    } else {
      if (!same('odds', original.odds, row.odds)) row.edge_bps = null;
      if (row.status === 'pending') Object.assign(row, { actual_result: null, payout: null, graded_timestamp: null });
      else {
        const recalculate = changed(original, row, ['stake_dollars', 'odds', 'status']);
        // Preserve an existing custom return on an unrelated edit. An explicit
        // return allows sportsbook rounding, boosts and cash-outs to be recorded.
        const explicit = Object.hasOwn(values, 'payout') && !blank(values.payout);
        row.payout = row.status === 'won' && (explicit || !recalculate && row.payout != null)
          ? amount(row.payout, 'payout') : amount(payout(row.stake_dollars, row.odds, row.status), 'payout');
        if (changed(original, row, ['status', 'actual_result'])) row.graded_timestamp = now;
      }
    }
    return row;
  }

  async function save(original, values) {
    const user = await global.getCurrentUser();
    if (!user || user.id !== original.user_id) return { ok: false, error: 'Sign in to the account that owns this bet before saving.' };
    let row;
    try { row = buildUpdate(original, values); }
    catch (error) { return { ok: false, error: error.message }; }
    let query = global.supabaseClient.from('bets').update(row).eq('id', original.id).eq('user_id', user.id);
    // A grade, edit in another tab, or deletion since opening the form must not
    // be silently overwritten. The database checks the snapshot atomically.
    for (const key of [...snapshotFields, ...(Object.hasOwn(original, 'player_team') ? ['player_team'] : [])]) {
      query = original[key] == null ? query.is(key, null) : query.eq(key, original[key]);
    }
    const { data, error } = await query.select('*').maybeSingle();
    if (error) return { ok: false, error: 'Changes could not be saved. Your entries are still here; try again.' };
    if (!data) return { ok: false, conflict: true, error: 'This bet changed or was deleted since you opened it. Close this form and reopen the bet to review its latest details.' };
    return { ok: true, bet: data };
  }

  const api = { buildUpdate, payout, ticketChanged, save };
  if (typeof module === 'object' && module.exports) { module.exports = api; return; }

  function mount({ onSaved, onConflict }) {
    const dialog = document.createElement('dialog');
    dialog.id = 'editBetDialog';
    dialog.setAttribute('aria-labelledby', 'eb-title');
    dialog.innerHTML = `<h2 id="eb-title">Edit bet</h2>
      <p id="eb-description" class="mb-intro"></p>
      <form id="eb-form" novalidate>
        <fieldset id="eb-fields">
          <div class="form-grid">
            <div class="form-group"><label for="eb-stake_dollars">Stake ($)</label><input id="eb-stake_dollars" type="number" step="0.01" min="0.01" inputmode="decimal" autofocus></div>
            <div class="form-group"><label for="eb-odds">Odds (American)</label><input id="eb-odds" type="number" step="1" inputmode="numeric"></div>
            <div class="form-group"><label for="eb-book">Sportsbook</label><input id="eb-book" list="mb-books"></div>
            <div class="form-group"><label for="eb-status">Result</label><select id="eb-status"><option value="pending">Pending</option><option value="won">Won</option><option value="lost">Lost</option><option value="push">Push</option></select></div>
            <div class="form-group"><label for="eb-payout">Payout ($, including stake)</label><input id="eb-payout" type="number" step="0.01" min="0" inputmode="decimal"></div>
            <div class="form-group"><label for="eb-actual_result">Final stat / score (optional)</label><input id="eb-actual_result" type="number" step="any" inputmode="decimal"></div>
          </div>
          <p class="mb-intro">Changing the stake or odds recalculates the payout. For a win, you can enter the exact amount returned by your sportsbook.</p>
          <details id="eb-details"><summary>Game and pick details</summary><div class="form-grid">
            <div class="form-group"><label for="eb-league">League</label><select id="eb-league"></select></div>
            <div class="form-group"><label for="eb-game_date">Game date</label><input id="eb-game_date" type="date"></div>
            <div class="form-group"><label for="eb-team_away">Away team</label><input id="eb-team_away"></div>
            <div class="form-group"><label for="eb-team_home">Home team</label><input id="eb-team_home"></div>
            <div class="form-group"><label for="eb-player">Player (if applicable)</label><input id="eb-player"></div>
            <div class="form-group"><label for="eb-market_type">Market</label><select id="eb-market_type"></select></div>
            <div class="form-group"><label for="eb-side">Pick (over, under, team, yes/no)</label><input id="eb-side"></div>
            <div class="form-group"><label for="eb-line">Line (if applicable)</label><input id="eb-line" type="number" step="any" inputmode="decimal"></div>
          </div><p class="mb-intro">Changing the game or pick details returns the bet to pending and clears its previous result. Unsupported markets and games over 14 days old need a manual result.</p></details>
          <p id="eb-reset" class="mb-check warn" hidden>This changed pick will be saved as pending. You can reopen it to record a result.</p>
          <div class="edit-actions"><button class="btn" id="eb-save" type="submit">Save changes</button><button class="filter-btn" id="eb-cancel" type="button">Cancel</button></div>
        </fieldset>
        <p id="eb-feedback" class="mb-check" role="status" aria-live="polite"></p>
      </form>`;
    document.body.append(dialog);
    const $ = key => document.getElementById('eb-' + key);
    const values = () => Object.fromEntries(fields.map(key => [key, $(key).value]));
    const option = (value, label) => Object.assign(document.createElement('option'), { value, textContent: label });
    let original, saving = false, reset = false, opener, settlementDraft;

    function markets(keep) {
      const league = $('league').value;
      const choices = global.FVManualBet.marketOptions(league).map(m => [global.FVOfferTracker.leagues[league].markets[m.key], m.label]);
      if (!choices.some(([key]) => key === keep)) choices.unshift([keep || '', keep || 'No market recorded']);
      $('market_type').replaceChildren(...choices.map(([key, label]) => option(key, label)));
      $('market_type').value = keep || '';
    }
    function sync(recalculate = false) {
      const shouldReset = ticketChanged(original, values());
      if (shouldReset !== reset) {
        if (shouldReset) settlementDraft = values();
        reset = shouldReset;
        for (const key of ['status', 'actual_result', 'payout']) $(key).value = reset ? (key === 'status' ? 'pending' : '') : settlementDraft[key] ?? '';
        recalculate = !reset && changed(settlementDraft, values(), ['stake_dollars', 'odds']);
      }
      const status = $('status').value;
      if (recalculate || status !== 'won') {
        const n = payout($('stake_dollars').value, $('odds').value, status);
        $('payout').value = Number.isFinite(n) ? n.toFixed(2) : '';
      }
      if (status === 'pending') $('actual_result').value = '';
      $('status').disabled = reset;
      $('payout').disabled = reset || status !== 'won';
      $('actual_result').disabled = reset || status === 'pending';
      $('reset').hidden = !reset;
    }
    $('form').addEventListener('input', event => {
      $('feedback').textContent = '';
      sync(['eb-stake_dollars', 'eb-odds', 'eb-status'].includes(event.target.id));
    });
    $('league').addEventListener('change', () => { markets($('market_type').value); sync(); });
    $('cancel').addEventListener('click', () => { if (!saving) dialog.close(); });
    dialog.addEventListener('cancel', event => { if (saving) event.preventDefault(); });
    dialog.addEventListener('close', () => {
      document.body.classList.remove('editing-bet');
      if (opener?.isConnected) opener.focus();
    });
    $('form').addEventListener('submit', async event => {
      event.preventDefault();
      if (saving) return;
      const invalid = [...$('form').elements].find(input => input.validity?.badInput);
      if (invalid) { $('feedback').textContent = 'Finish entering a valid number before saving.'; invalid.focus(); return; }
      const draft = values();
      try { buildUpdate(original, draft); }
      catch (error) { $('feedback').textContent = error.message; return; }
      saving = true; $('fields').disabled = true; $('feedback').textContent = 'Saving…';
      try {
        const result = await save(original, draft);
        if (result.ok) { onSaved(result.bet); dialog.close(); }
        else { $('feedback').textContent = result.error; if (result.conflict) onConflict?.(); }
      } catch {
        $('feedback').textContent = 'The save could not be confirmed. Close this form and reload your bets to check before trying again.';
      } finally { saving = false; $('fields').disabled = false; }
    });
    return {
      open(bet, button) {
        if (saving) return;
        original = { ...bet }; opener = button; reset = false;
        $('form').reset(); $('details').open = false; $('feedback').textContent = '';
        const leagues = [...new Set([bet.league, 'NFL', 'NHL', 'MLB', 'NBA'].filter(Boolean))];
        $('league').replaceChildren(...leagues.map(key => option(key, key)));
        $('league').value = bet.league;
        markets(bet.market_type);
        for (const key of fields) $(key).value = bet[key] ?? '';
        $('description').textContent = [bet.game_date, bet.league, [bet.team_away, bet.team_home].filter(Boolean).join(' @ '), bet.player,
          [bet.side, bet.line, $('market_type').selectedOptions[0]?.textContent].filter(value => !blank(value)).join(' ')].filter(Boolean).join(' · ');
        sync();
        document.body.classList.add('editing-bet'); dialog.showModal();
        $('stake_dollars').focus(); $('stake_dollars').select();
      }
    };
  }
  global.FVEditBet = { ...api, mount };
})(typeof window === 'undefined' ? globalThis : window);
