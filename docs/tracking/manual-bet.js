/* Guided manual entry for Bet Tracker. Shared by the browser and Node tests
   (tests/manual_bet.cjs).

   A bet is built from the same schedule feed the grader reads (league -> date
   -> game) and the same ledger market names the offer boards save
   (docs/assets/offer-tracker.js), so a hand-entered bet is stored exactly like
   a tracked offer. Before saving, `gradeCheck` runs the grader's own matching
   (docs/tracking/live-stats.js) and says whether the bet will settle on its
   own. */
(function (global) {
  'use strict';
  const node = typeof module === 'object' && module.exports;
  const offers = node ? require('../assets/offer-tracker.js') : global.FVOfferTracker;
  const live = node ? require('./live-stats.js') : global.FVLiveStats;
  const feeds = node ? require('./live-feeds.js') : global.FVLiveFeeds;

  // scripts/grade_bets.cjs: settles from 4 hours after the start, looks back 14 days.
  const SETTLE_AFTER_HOURS = 4, LOOKBACK_DAYS = 14;

  const YES_NO = new Set(['anytime_td', 'first_td', 'last_td']);
  const TOTAL = { NHL: 'Total goals', MLB: 'Total runs', NFL: 'Total points', NBA: 'Total points' };
  const LABELS = {
    h2h: 'Moneyline', spreads: 'Spread',
    player_shots_on_goal: 'Shots on goal', batter_rbis: 'RBIs', batter_hits_runs_rbis: 'Hits + runs + RBIs',
    pitcher_outs: 'Pitcher outs recorded', player_threes: '3-pointers made',
    player_points_rebounds_assists: 'Points + rebounds + assists', player_points_rebounds: 'Points + rebounds',
    player_points_assists: 'Points + assists', player_rebounds_assists: 'Rebounds + assists', player_blocks_steals: 'Blocks + steals',
    anytime_td: 'Anytime TD', first_td: 'First TD', last_td: 'Last TD', pass_yds: 'Passing yards', pass_tds: 'Passing TDs',
    pass_completions: 'Completions', pass_attempts: 'Pass attempts', interceptions: 'Interceptions thrown',
    rush_yds: 'Rushing yards', rush_attempts: 'Rush attempts', recv_yds: 'Receiving yards',
    rush_reception_yds: 'Rushing + receiving yards', pass_rush_yds: 'Passing + rushing yards', rush_tds: 'Rushing TDs',
    rec_tds: 'Receiving TDs', reception_longest: 'Longest reception', rush_longest: 'Longest rush',
  };
  const pretty = key => key.replace(/^(player|batter)_/, '').replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase());

  function kindOf(market) {
    if (market === 'h2h') return 'moneyline';
    if (market === 'spreads') return 'spread';
    if (market === 'totals') return 'total';
    return YES_NO.has(market) ? 'yesno' : 'prop';
  }

  // Every market the offer boards can save for a league, game markets first.
  function marketOptions(league) {
    const markets = offers.leagues[league]?.markets || {};
    return Object.keys(markets).map(key => {
      const kind = kindOf(key);
      return { key, kind, group: ['moneyline', 'spread', 'total'].includes(kind) ? 'Game' : 'Player',
        label: kind === 'total' ? TOTAL[league] : LABELS[key] || pretty(key),
        manual: offers.manualGrade({ sport: league, market: key }) };
    }).sort((a, b) => (a.group === 'Game' ? 0 : 1) - (b.group === 'Game' ? 0 : 1));
  }

  // The choices for the "Pick" field once a market is chosen.
  function sideOptions(kind, game) {
    if (kind === 'moneyline' || kind === 'spread') {
      return game ? [['away', teamCode(game.away)], ['home', teamCode(game.home)]] : [];
    }
    return kind === 'yesno' ? [['Yes', 'Yes'], ['No', 'No']] : [['over', 'Over'], ['under', 'Under']];
  }
  const teamCode = t => t.abbrev || t.short || t.name;
  const needsLine = kind => kind !== 'moneyline' && kind !== 'yesno';
  const needsPlayer = kind => kind === 'prop' || kind === 'yesno';

  // Form values -> the ledger row saveTrackedBet() stores. Throws a reader-facing message.
  // `box` is the game's box score when loaded; it supplies the player's team.
  function buildTicket({ league, date, game, market, side, line, player, book, odds, stake, box }) {
    if (!game) throw Error('Choose the game.');
    const kind = kindOf(market);
    if (!offers.leagues[league]?.markets[market]) throw Error('Choose a market.');
    player = String(player || '').trim();
    if (needsPlayer(kind) && !player) throw Error('Enter the player.');
    book = String(book || '').trim();
    if (!book) throw Error('Enter the sportsbook.');
    const lineText = String(line ?? '').trim();
    if (needsLine(kind) && (lineText === '' || !Number.isFinite(Number(lineText)))) throw Error('Enter the line, such as 8.5 or -1.5.');
    const pick = kind === 'moneyline' || kind === 'spread'
      ? (side === 'home' || side === 'away' ? teamCode(game[side]) : '') : side;
    if (!pick) throw Error('Choose your pick.');
    const home = teamCode(game.home), away = teamCode(game.away);
    const row = { sport: league, game: `${away} @ ${home}`, home_team: home, away_team: away,
      commence_time: game.start || `${date}T17:00:00Z`, book, market, side: pick,
      line: needsLine(kind) ? Number(lineText) : null, player: needsPlayer(kind) ? player : null };
    // The grader looks the game up on the schedule date the reader picked, so
    // keep that date rather than re-deriving it from the start time.
    const ticket = { ...offers.ticketData(row, odds, stake), game_date: date };
    const found = row.player && box?.game?.id === game.id ? live.findPlayer(row.player, box.players) : null;
    if (found && game[found.side]) ticket.player_team = teamCode(game[found.side]);
    return ticket;
  }

  // Would the grader settle this bet on its own? Same matching as scripts/grade_bets.cjs.
  // `bet` needs league, market_type, game_date, teams and player; `games` is that
  // league's schedule for the date and `box` the game's box score when loaded.
  // (NFL's hand-graded markets keep their offer names, so market_type finds them.)
  function gradeCheck(bet, game, games, box, now = Date.now()) {
    const warn = text => ({ ok: false, text });
    if (offers.manualGrade({ sport: bet.league, market: bet.market_type }) || !live.marketSpec(bet.league, bet.market_type)) {
      return warn('This market isn’t graded automatically, so the bet will stay pending.');
    }
    if (bet.game_date < live.day(now - LOOKBACK_DAYS * 864e5)) {
      return warn(`Games more than ${LOOKBACK_DAYS} days old aren’t graded automatically, so the bet will stay pending.`);
    }
    const matched = live.findGame(bet, games);
    if (!matched) return warn('The grader can’t match this game on the schedule, so the bet will stay pending.');
    if (matched.id !== game.id) {
      return warn('These teams play more than once on this date and the grader may settle this bet against the other game.');
    }
    if (game.state === 'off') return warn('This game was postponed or suspended, so the bet will stay pending.');
    if (bet.player && box && box.game?.id === game.id) {
      const found = live.findPlayer(bet.player, box.players);
      if (!found && game.state === 'final') return warn(`“${bet.player}” isn’t in this game’s box score, so the bet won’t grade. Pick the name from the list.`);
      if (!found) return { ok: true, text: `“${bet.player}” isn’t in the box score yet. It grades once the name matches the final box score.` };
    } else if (bet.player) {
      return { ok: true, text: 'Grades automatically once the game is final, if the player name matches the box score.' };
    }
    return { ok: true, text: `Grades automatically about ${SETTLE_AFTER_HOURS} hours after the game starts, once it’s final.` };
  }

  const api = { marketOptions, sideOptions, buildTicket, gradeCheck, kindOf, needsLine, needsPlayer, teamCode };
  if (node) { module.exports = api; return; }

  // ---- Browser form -----------------------------------------------------------
  // Element ids are prefixed "mb-" in docs/tracking/index.html. fetchJSON and
  // proxy are the page's live-stats loaders; onSaved reloads the bet list.
  function mount({ fetchJSON, proxy, onSaved }) {
    const $ = id => document.getElementById('mb-' + id);
    const run = plan => Promise.all(plan.requests.map(r => r.proxy ? proxy(r.proxy) : fetchJSON(r.url))).then(plan.build);
    const boards = new Map(), boxes = new Map();
    const cached = (map, key, load) => {
      if (!map.has(key)) map.set(key, load().catch(error => { map.delete(key); throw error; }));
      return map.get(key);
    };
    let games = [], box = null, draftId = null, saving = false, loadToken = 0;

    const game = () => games.find(g => g.id === $('game').value) || null;
    const kind = () => kindOf($('market').value);
    const time = iso => Number.isFinite(Date.parse(iso))
      ? new Date(iso).toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' }) + ' ET' : '';
    function gameLabel(g) {
      const matchup = `${teamCode(g.away)} @ ${teamCode(g.home)}`;
      if (g.state === 'final') return `${matchup} · ${g.detail || 'Final'} ${g.away.score}–${g.home.score}`;
      if (g.state === 'live') return `${matchup} · ${g.detail || 'Live'}`;
      if (g.state === 'off') return `${matchup} · ${g.detail || 'Postponed'}`;
      return `${matchup} · ${time(g.start)}`;
    }
    const option = (value, label) => Object.assign(document.createElement('option'), { value, textContent: label });

    function fillMarkets() {
      const select = $('market'), keep = select.value, groups = {};
      select.replaceChildren();
      for (const m of marketOptions($('league').value)) {
        if (!groups[m.group]) select.append(groups[m.group] = Object.assign(document.createElement('optgroup'), { label: m.group === 'Game' ? 'Game' : 'Player props' }));
        groups[m.group].append(option(m.key, m.label + (m.manual ? ' (not auto-graded)' : '')));
      }
      if ([...select.options].some(o => o.value === keep)) select.value = keep;
    }

    function fillSides() {
      const select = $('side'), keep = select.value;
      select.replaceChildren(...sideOptions(kind(), game()).map(([value, label]) => option(value, label)));
      if ([...select.options].some(o => o.value === keep)) select.value = keep;
      $('line-group').hidden = !needsLine(kind());
      $('player-group').hidden = !needsPlayer(kind());
      $('line-label').textContent = kind() === 'spread' ? 'Line (your team, e.g. -1.5)' : 'Line';
    }

    async function loadGames() {
      const token = ++loadToken, league = $('league').value, date = $('date').value;
      games = []; box = null;
      $('game').replaceChildren(option('', date ? 'Loading games…' : 'Choose a date first'));
      $('game').disabled = true;
      fillSides(); update();
      if (!date) return;
      try {
        const list = await cached(boards, `${league}|${date}`, () => run(feeds.scoreboardPlan(league, date)));
        if (token !== loadToken) return;
        games = [...list].sort((a, b) => String(a.start).localeCompare(String(b.start)));
        $('game').replaceChildren(option('', games.length ? 'Choose a game' : `No ${league} games on this date`),
          ...games.map(g => option(g.id, gameLabel(g))));
        $('game').disabled = !games.length;
      } catch {
        if (token !== loadToken) return;
        $('game').replaceChildren(option('', 'Schedule unavailable — try again'));
      }
      fillSides(); update();
    }

    // Player names come from the box score once the game has started.
    async function loadPlayers() {
      const g = game();
      box = null; $('players').replaceChildren();
      if (!g || !needsPlayer(kind()) || !['live', 'final'].includes(g.state)) { update(); return; }
      const token = loadToken, id = g.id;
      try {
        const loaded = await cached(boxes, `${g.league}:${id}`, () => run(feeds.boxPlan(g.league, id)));
        if (token !== loadToken || game()?.id !== id) return;
        box = loaded;
        const names = [...new Set(loaded.players.filter(p => p.played).map(p => p.name))].sort();
        $('players').replaceChildren(...names.map(name => option(name, '')));
      } catch { /* free text still works; the check just can't confirm the name */ }
      update();
    }

    // Live preview of whether the grader will settle the bet.
    function update() {
      draftId = null;
      const check = $('check');
      const g = game();
      if (!g) { check.textContent = ''; check.className = 'mb-check'; return; }
      const v = values(), player = needsPlayer(kind()) ? v.player.trim() : '';
      const result = gradeCheck({ league: v.league, market_type: offers.leagues[v.league].markets[v.market], game_date: v.date,
        team_home: teamCode(g.home), team_away: teamCode(g.away), player: player || null }, g, games, box);
      check.textContent = (result.ok ? '✓ ' : '⚠ ') + result.text;
      check.className = 'mb-check ' + (result.ok ? 'ok' : 'warn');
    }

    const values = () => ({ league: $('league').value, date: $('date').value, game: game(), market: $('market').value,
      side: $('side').value, line: $('line').value, player: $('player').value, book: $('book').value,
      odds: $('odds').value, stake: $('stake').value, box });

    async function save(event) {
      event.preventDefault();
      if (saving) return;
      let ticket;
      try { ticket = buildTicket(values()); }
      catch (error) { $('feedback').textContent = error.message; return; }
      // One ID per unchanged form, so a retry after a lost response can't save twice.
      draftId = draftId || crypto.randomUUID();
      saving = true; $('save').disabled = true; $('feedback').textContent = 'Saving…';
      try {
        const result = await global.saveTrackedBet({ ...ticket, id: draftId });
        if (result.ok) {
          draftId = null;
          for (const id of ['player', 'line', 'odds', 'stake']) $(id).value = '';
          $('feedback').textContent = 'Saved.';
          update();
          onSaved?.();
        } else $('feedback').textContent = result.error;
      } catch {
        $('feedback').textContent = 'The save could not be confirmed. Check your bets below before retrying.';
      } finally { saving = false; $('save').disabled = false; }
    }

    $('date').value = live.day(Date.now());
    fillMarkets();
    $('league').addEventListener('change', () => { fillMarkets(); loadGames(); });
    $('date').addEventListener('change', loadGames);
    $('game').addEventListener('change', () => { fillSides(); loadPlayers(); });
    $('market').addEventListener('change', () => { fillSides(); loadPlayers(); });
    for (const id of ['side', 'line', 'player', 'book', 'odds', 'stake']) $(id).addEventListener('input', () => { $('feedback').textContent = ''; update(); });
    $('form').addEventListener('submit', save);
    loadGames();
  }

  global.FVManualBet = { ...api, mount };
})(typeof window === 'undefined' ? globalThis : window);
