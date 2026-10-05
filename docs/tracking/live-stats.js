/* Live progress for pending Bet Tracker bets. Shared by the browser and Node
   tests (tests/live_stats.cjs).

   A bet is matched to a game by its date and teams, to a player by name, and
   to a stat by its market. The result is a display-only view: it never
   changes a bet's status, which the official graders still settle. */
(function (global) {
  'use strict';
  const feeds = global.FVLiveFeeds || (typeof require === 'function' ? require('./live-feeds.js') : null);

  const day = t => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(t));
  const SUFFIX = /\b(jr|sr|ii|iii|iv|v)\b/g;

  function norm(s) {
    return String(s ?? '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase()
      .replace(/\(.*?\)/g, ' ').replace(/[.'’]/g, '').replace(/[-_]/g, ' ')
      .replace(/[^a-z0-9 ]/g, ' ').replace(SUFFIX, ' ').replace(/\s+/g, ' ').trim();
  }
  const squash = s => norm(s).replace(/ /g, '');

  // ---- Matching ------------------------------------------------------------

  // Common abbreviations that differ from the feed's own (older manual
  // entries were typed as 2-3 letter codes).
  const ALIASES = {
    NHL: { tb: 'tbl', nj: 'njd', la: 'lak', sj: 'sjs', mon: 'mtl', clb: 'cbj', was: 'wsh', veg: 'vgk', lv: 'vgk' },
    MLB: { was: 'wsh', chw: 'cws', kcr: 'kc', sdp: 'sd', sfg: 'sf', tbr: 'tb', az: 'ari', oak: 'ath', cha: 'cws', chn: 'chc' },
    NFL: { was: 'wsh', jac: 'jax', arz: 'ari', la: 'lar', gnb: 'gb', kan: 'kc', nwe: 'ne', nor: 'no', sfo: 'sf', tam: 'tb', lvr: 'lv', oak: 'lv' },
    NBA: { gs: 'gsw', ny: 'nyk', no: 'nop', sa: 'sas', pho: 'phx', bkn: 'bkn', brk: 'bkn', cho: 'cha', uta: 'utah', wsh: 'wsh', was: 'wsh' },
  };

  function teamMatches(betTeam, t, league) {
    let b = squash(betTeam);
    if (!b || !t) return false;
    b = ALIASES[league]?.[b] || b;
    const name = squash(t.name), short = squash(t.short), abbrev = squash(t.abbrev);
    return b === abbrev || b === short || b === name ||
      (b.length >= 3 && name.endsWith(b)) || (short.length >= 3 && b.endsWith(short));
  }

  // The short label a feed gives a team: "NYR", else its short or full name.
  const teamCode = t => (t && (t.abbrev || t.short || t.name)) || null;

  const STATE_RANK = { live: 0, pre: 1, final: 2, off: 3 };

  function findGame(bet, games) {
    const home = bet.team_home, away = bet.team_away, m = (x, t) => teamMatches(x, t, bet.league);
    let best = [], bestScore = 0;
    for (const g of games || []) {
      let score = 0;
      if (home && away) {
        if (m(home, g.home) && m(away, g.away)) score = 3;
        else if (m(home, g.away) && m(away, g.home)) score = 2;
      } else if (home || away) {
        const t = home || away;
        if (m(t, g.home) || m(t, g.away)) score = 1;
      }
      if (score > bestScore) { best = [g]; bestScore = score; }
      else if (score && score === bestScore) best.push(g);
    }
    // Doubleheaders: prefer the game in progress, then the next one to start.
    return best.sort((a, b) => STATE_RANK[a.state] - STATE_RANK[b.state] || String(a.start).localeCompare(String(b.start)))[0] || null;
  }

  function nameParts(name) {
    // NHL: "J. Staal". Initials that are the first name ("J.T. Miller", "J. T. Miller") are a full name.
    const initialForm = /^([a-z])\.\s+([^.\s]{2,}.*)$/i.exec(String(name).trim());
    if (initialForm) return { full: null, initial: initialForm[1].toLowerCase(), last: squash(initialForm[2]) };
    const tokens = norm(name).split(' ').filter(Boolean);
    return { full: tokens.join(''), initial: (tokens[0] || '')[0] || '', last: tokens.slice(1).join('') };
  }

  // Exact full name first, then first initial + last name. Ambiguity is no match.
  function findPlayer(name, players) {
    const want = nameParts(name);
    if (!want.full) return null;
    const exact = (players || []).filter(p => nameParts(p.name).full === want.full);
    if (exact.length) return exact.length === 1 ? exact[0] : null;
    const close = (players || []).filter(p => {
      const got = nameParts(p.name);
      return got.last && got.last === want.last && got.initial === want.initial;
    });
    return close.length === 1 ? close[0] : null;
  }

  // ---- Markets -------------------------------------------------------------
  // Alias -> stats summed for the bet. `moves` marks stats that can go down
  // during a game (yardage), so an over is never "locked" before the final.

  const M = (stats, label, extra) => ({ stats: [].concat(stats), label, ...extra });
  const YDS = { moves: true };
  const MARKETS = {
    NHL: {
      goals: M('goals', 'G'), goal_scorer_anytime: M('goals', 'G'), anytime_goal: M('goals', 'G'),
      assists: M('assists', 'A'), points: M('points', 'PTS'),
      sog: M('sog', 'SOG'), shots: M('sog', 'SOG'), shots_on_goal: M('sog', 'SOG'),
      hits: M('hits', 'Hits'), blocks: M('blocks', 'BLK'), blocked_shots: M('blocks', 'BLK'),
      saves: M('saves', 'Saves'), goalie_saves: M('saves', 'Saves'), pp_goals: M('pp_goals', 'PPG'),
    },
    MLB: {
      hits: M('hits', 'H'), any_hit: M('hits', 'H'), total_bases: M('total_bases', 'TB'),
      home_runs: M('home_runs', 'HR'), rbis: M('rbis', 'RBI'), rbi: M('rbis', 'RBI'),
      runs: M('runs', 'R'), runs_scored: M('runs', 'R'), walks: M('walks', 'BB'),
      stolen_bases: M('stolen_bases', 'SB'), singles: M('singles', '1B'), doubles: M('doubles', '2B'),
      triples: M('triples', '3B'), strikeouts: M('batter_strikeouts', 'K'),
      hits_runs_rbis: M(['hits', 'runs', 'rbis'], 'H+R+RBI'),
      pitcher_strikeouts: M('pitcher_strikeouts', 'K'), pitcher_outs: M('pitcher_outs', 'Outs'),
      pitcher_hits_allowed: M('hits_allowed', 'H allowed'), pitcher_earned_runs: M('earned_runs', 'ER'),
      pitcher_walks: M('walks_allowed', 'BB allowed'),
    },
    NFL: {
      pass_yds: M('pass_yds', 'Pass yds', YDS), passing_yds: M('pass_yds', 'Pass yds', YDS),
      pass_yards: M('pass_yds', 'Pass yds', YDS), passing_yards: M('pass_yds', 'Pass yds', YDS),
      pass_tds: M('pass_tds', 'Pass TD'), pass_td: M('pass_tds', 'Pass TD'), passing_tds: M('pass_tds', 'Pass TD'),
      pass_completions: M('pass_completions', 'Comp'), completions: M('pass_completions', 'Comp'),
      pass_attempts: M('pass_attempts', 'Att'), attempts: M('pass_attempts', 'Att'),
      pass_interceptions: M('pass_interceptions', 'INT'), interceptions: M('pass_interceptions', 'INT'),
      rush_yds: M('rush_yds', 'Rush yds', YDS), rushing_yds: M('rush_yds', 'Rush yds', YDS),
      rush_yards: M('rush_yds', 'Rush yds', YDS), rushing_yards: M('rush_yds', 'Rush yds', YDS),
      rush_attempts: M('rush_attempts', 'Rush att'), rush_att: M('rush_attempts', 'Rush att'), carries: M('rush_attempts', 'Rush att'),
      receptions: M('receptions', 'Rec'), rec: M('receptions', 'Rec'),
      recv_yds: M('recv_yds', 'Rec yds', YDS), rec_yds: M('recv_yds', 'Rec yds', YDS),
      receiving_yds: M('recv_yds', 'Rec yds', YDS), reception_yds: M('recv_yds', 'Rec yds', YDS),
      receiving_yards: M('recv_yds', 'Rec yds', YDS),
      rush_reception_yds: M(['rush_yds', 'recv_yds'], 'Rush+rec yds', YDS), rush_rec_yds: M(['rush_yds', 'recv_yds'], 'Rush+rec yds', YDS),
      pass_rush_yds: M(['pass_yds', 'rush_yds'], 'Pass+rush yds', YDS),
      anytime_td: M(['rush_tds', 'rec_tds'], 'TD'), tds: M(['rush_tds', 'rec_tds'], 'TD'),
      rush_tds: M('rush_tds', 'Rush TD'), rec_tds: M('rec_tds', 'Rec TD'),
      kicking_points: M('kicking_points', 'Kick pts'), field_goals: M('fg_made', 'FG'),
      tackles: M('tackles', 'Tkl'), tackles_assists: M('tackles', 'Tkl'), sacks: M('sacks', 'Sacks'),
    },
    NBA: {
      points: M('points', 'PTS'), pts: M('points', 'PTS'), rebounds: M('rebounds', 'REB'), reb: M('rebounds', 'REB'),
      assists: M('assists', 'AST'), ast: M('assists', 'AST'), threes: M('threes', '3PM'), threes_made: M('threes', '3PM'),
      steals: M('steals', 'STL'), blocks: M('blocks', 'BLK'), turnovers: M('turnovers', 'TO'),
      points_rebounds_assists: M(['points', 'rebounds', 'assists'], 'PRA'), pra: M(['points', 'rebounds', 'assists'], 'PRA'),
      points_rebounds: M(['points', 'rebounds'], 'P+R'), points_assists: M(['points', 'assists'], 'P+A'),
      rebounds_assists: M(['rebounds', 'assists'], 'R+A'), blocks_steals: M(['blocks', 'steals'], 'STL+BLK'),
      steals_blocks: M(['blocks', 'steals'], 'STL+BLK'),
    },
  };
  const GAME_MARKETS = { h2h: 'moneyline', moneyline: 'moneyline', ml: 'moneyline', spreads: 'spread', spread: 'spread',
    totals: 'total', total: 'total', team_total: 'score', team_totals: 'score' };

  function marketSpec(league, market) {
    const key = norm(market).replace(/ /g, '_');
    // Tickets saved before 2026-10-05 recorded NHL game totals as "team_total" (the
    // NHL grader's original name for them), so there it means both teams combined.
    // New tickets record "totals".
    if (league === 'NHL' && key === 'team_total') return { game: 'total' };
    if (GAME_MARKETS[key]) return { game: GAME_MARKETS[key] };
    const table = MARKETS[league] || {};
    return table[key] || table[key.replace(/^player_/, '')] || table[key.replace(/^(batter|player)_/, '')] || null;
  }

  // ---- Verdicts ------------------------------------------------------------

  const fmt = n => Number.isInteger(n) ? String(n) : n.toFixed(1);

  // Over/under on a running number. `settled` only once the game is final.
  function overUnder(value, line, side, { final, moves }) {
    const over = side === 'over';
    const clear = Number.isInteger(line) ? line + 1 : Math.ceil(line);   // first value that wins an over
    if (final) {
      const tone = value === line ? 'push' : (value > line) === over ? 'won' : 'lost';
      return { tone, label: tone === 'push' ? 'Push' : tone === 'won' ? 'Won' : 'Lost', note: 'Final · awaiting official grade' };
    }
    if (over) {
      if (value >= clear) return moves ? { tone: 'ahead', label: 'Over the line' } : { tone: 'won', label: 'Hit' };
      const need = clear - value;
      return { tone: 'alive', label: `Needs ${fmt(need)}` };
    }
    if (value > line) return moves ? { tone: 'behind', label: 'Over the line' } : { tone: 'lost', label: 'Dead' };
    const spare = (Number.isInteger(line) ? line - 1 : Math.floor(line)) - value;
    return { tone: 'alive', label: spare > 0 ? `${fmt(spare)} to spare` : 'Holding' };
  }

  function sideOf(bet, game) {
    if (teamMatches(bet.side, game.home, bet.league)) return 'home';
    if (teamMatches(bet.side, game.away, bet.league)) return 'away';
    return null;
  }

  function gameVerdict(kind, bet, game) {
    const final = game.state === 'final', h = game.home.score ?? 0, a = game.away.score ?? 0;
    if (kind === 'total') {
      const side = String(bet.side || '').toLowerCase(), line = Number(bet.line);
      if (!['over', 'under'].includes(side) || !Number.isFinite(line)) return { value: h + a };
      const clear = Number.isInteger(line) ? line + 1 : Math.ceil(line);
      return { value: h + a, line, side, progress: Math.min(1, (h + a) / (side === 'over' ? clear : Math.max(line, 0.5))),
        ...overUnder(h + a, line, side, { final }) };
    }
    if (kind === 'score') return {};
    const mine = sideOf(bet, game);
    if (!mine) return {};
    const margin = mine === 'home' ? h - a : a - h;
    const adj = kind === 'spread' && Number.isFinite(Number(bet.line)) ? margin + Number(bet.line) : margin;
    if (final) {
      const tone = adj === 0 ? 'push' : adj > 0 ? 'won' : 'lost';
      return { margin, tone, label: tone === 'push' ? 'Push' : tone === 'won' ? 'Won' : 'Lost', note: 'Final · awaiting official grade' };
    }
    const lead = kind === 'spread' ? ['Covering', 'Not covering', 'On the number'] : ['Leading', 'Trailing', 'Tied'];
    const verdict = adj > 0 ? { tone: 'ahead', label: lead[0] } : adj < 0 ? { tone: 'behind', label: lead[1] } : { tone: 'alive', label: lead[2] };
    return { margin, ...verdict };
  }

  function timeLabel(iso) {
    const t = Date.parse(iso);
    return Number.isFinite(t) ? new Date(t).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' }) : '';
  }

  // One bet + its game (+ box when fetched) -> everything the UI draws.
  function evaluate(bet, game, box) {
    if (!game) return { status: 'nogame', label: 'Game not found on the schedule' };
    const g = box?.game && box.game.id === game.id ? box.game : game;
    const base = { status: g.state, game: g, gameKey: `${g.league}:${g.id}` };
    if (g.state === 'off') return { ...base, tone: 'off', label: g.detail || 'Postponed' };
    if (g.state === 'pre') return { ...base, tone: 'pre', label: `Starts ${timeLabel(g.start)}`.trim() };
    const spec = marketSpec(bet.league, bet.market_type);
    const final = g.state === 'final';

    if (spec?.game || !bet.player) return { ...base, kind: 'game', ...gameVerdict(spec?.game || 'score', bet, g) };
    if (!spec) return { ...base, kind: 'unsupported', label: 'Live stat not supported for this market' };
    if (!box) return { ...base, kind: 'player', label: 'Loading box score…' };
    const player = findPlayer(bet.player, box.players);
    if (!player) return { ...base, kind: 'player', unit: spec.label, tone: 'missing',
      label: final ? 'No stats found for this player' : 'Not in the box score yet' };

    const value = spec.stats.reduce((sum, key) => sum + (player.stats[key] ?? 0), 0);
    let side = String(bet.side || '').toLowerCase(), line = bet.line == null || bet.line === '' ? null : Number(bet.line);
    if (side === 'yes') side = 'over';
    if (side === 'no') side = 'under';
    if (line == null && ['over', 'under'].includes(side)) line = 0.5;   // anytime / yes-no props
    const view = { ...base, kind: 'player', value, unit: spec.label, playerName: player.name, playerTeam: teamCode(g[player.side]) };
    if (!['over', 'under'].includes(side) || !Number.isFinite(line)) return view;
    const verdict = overUnder(value, line, side, { final, moves: !!spec.moves });
    const clear = Number.isInteger(line) ? line + 1 : Math.ceil(line);
    const progress = Math.max(0, Math.min(1, value / (side === 'over' ? clear : Math.max(line, 0.5))));
    let pace = null;
    if (!final && g.elapsed != null && g.elapsed >= 0.15 && g.elapsed < 1 && side === 'over' && verdict.tone === 'alive') {
      pace = Math.round(value / g.elapsed * (spec.moves ? 1 : 10)) / (spec.moves ? 1 : 10);
    }
    return { ...view, line, side, progress, pace, ...verdict };
  }

  // ---- Polling -------------------------------------------------------------

  const LIVE_MS = 60e3, IDLE_MS = 5 * 60e3;

  function liveCandidates(bets, now = Date.now()) {
    const today = day(now), yesterday = day(now - 864e5);
    return (bets || []).filter(b => (b.status || 'pending') === 'pending' && feeds.LEAGUES.includes(b.league) &&
      (b.game_date === today || b.game_date === yesterday));
  }

  // fetchJSON(url) -> parsed JSON; proxy(body) -> parsed JSON via the edge
  // function. onUpdate(views: Map<betId, view>, meta) redraws the page.
  function createLiveTracker({ fetchJSON, proxy, onUpdate, now = () => Date.now(), setTimer = setTimeout, clearTimer = clearTimeout }) {
    let bets = [], timer = null, running = false, paused = false, finals = new Map(), lastViews = new Map();

    const request = r => r.proxy ? proxy(r.proxy) : fetchJSON(r.url);
    const run = plan => Promise.all(plan.requests.map(request)).then(plan.build);

    async function refresh() {
      const candidates = liveCandidates(bets, now());
      const views = new Map();
      if (!candidates.length) { lastViews = views; onUpdate(views, { live: 0, pending: 0, errors: 0, at: now() }); return 0; }
      let errors = 0;
      const dayKeys = [...new Set(candidates.map(b => `${b.league}|${b.game_date}`))];
      const boards = new Map(await Promise.all(dayKeys.map(async k => {
        const [league, date] = k.split('|');
        try { return [k, await run(feeds.scoreboardPlan(league, date))]; } catch { errors++; return [k, null]; }
      })));
      const gameFor = new Map(), needBox = new Map();
      for (const bet of candidates) {
        const games = boards.get(`${bet.league}|${bet.game_date}`);
        if (!games) { views.set(bet.id, lastViews.get(bet.id) || { status: 'error', label: 'Live stats unavailable' }); continue; }
        const game = findGame(bet, games);
        gameFor.set(bet.id, game);
        if (game && ['live', 'final'].includes(game.state) && bet.player) needBox.set(`${game.league}:${game.id}`, game);
      }
      const boxes = new Map(await Promise.all([...needBox].map(async ([key, game]) => {
        if (finals.has(key)) return [key, finals.get(key)];
        try {
          const box = await run(feeds.boxPlan(game.league, game.id));
          if (box.game.state === 'final') finals.set(key, box);
          return [key, box];
        } catch { errors++; return [key, null]; }
      })));
      for (const bet of candidates) {
        if (!gameFor.has(bet.id)) continue;
        const game = gameFor.get(bet.id), key = game && `${game.league}:${game.id}`;
        const box = key ? boxes.get(key) : null;
        if (key && needBox.has(key) && !box) { views.set(bet.id, lastViews.get(bet.id) || { status: 'error', label: 'Live stats unavailable' }); continue; }
        views.set(bet.id, evaluate(bet, game, box));
      }
      lastViews = views;
      const live = [...views.values()].filter(v => v.status === 'live').length;
      onUpdate(views, { live, pending: candidates.length, errors, at: now() });
      return live;
    }

    function schedule(ms) { clearTimer(timer); timer = running && !paused ? setTimer(tick, ms) : null; }
    async function tick() {
      let live = 0;
      try { live = await refresh(); } catch { /* keep the loop alive; next tick retries */ }
      if (!liveCandidates(bets, now()).length) { running = false; return; }
      schedule(live ? LIVE_MS : IDLE_MS);
    }

    return {
      setBets(next) {
        bets = next || [];
        const had = running;
        running = liveCandidates(bets, now()).length > 0;
        if (running && !paused) { clearTimer(timer); tick(); }
        else if (!running) { clearTimer(timer); if (had) onUpdate(new Map(), { live: 0, pending: 0, errors: 0, at: now() }); }
      },
      pause() { paused = true; clearTimer(timer); timer = null; },
      resume() { if (!paused) return; paused = false; if (running) tick(); },
      refresh,
    };
  }

  const api = { norm, teamMatches, teamCode, findGame, findPlayer, marketSpec, evaluate, liveCandidates, createLiveTracker, day };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else global.FVLiveStats = api;
})(typeof window === 'undefined' ? globalThis : window);
