/* League feed adapters for Bet Tracker live stats. Shared by the browser and
   Node tests. Each league's public feed becomes one small shape:

   game:   {id, league, start, state: pre|live|final|off, detail, elapsed,
           home:{name, abbrev, short, score}, away:{...}}
   box:    {game, players:[{name, side: home|away, played, stats:{canonical: n}}]}
//
   `elapsed` is the share of regulation played (0-1), or null where a clock
   fraction means little (MLB). Stats are live and unofficial; grading still
   uses the official post-game pipeline.

   MLB and ESPN (NFL, NBA) allow browser requests, so viewers fetch them
   directly. The NHL feed sends no CORS header and goes through the
   live-stats edge function, which returns the upstream JSON unchanged. */
(function (global) {
'use strict';

const LEAGUES = ['NHL', 'MLB', 'NFL', 'NBA'];

const num = v => { const n = Number(v); return Number.isFinite(n) ? n : 0; };
const ordinal = n => n + ({ 1: 'st', 2: 'nd', 3: 'rd' }[n] || 'th');
const team = (name, abbrev, short, score) =>
  ({ name: name || short || abbrev || '', abbrev: abbrev || '', short: short || '', score: score == null || score === '' ? null : num(score) });

// ---- NHL: api-web.nhle.com (via proxy) --------------------------------------------

function nhlTeam(t = {}) {
  const common = t.commonName?.default || t.name?.default || '';
  const place = t.placeName?.default || '';
  return team(place && common ? `${place} ${common}` : common, t.abbrev, common, t.score);
}

function nhlGame(g) {
  const s = g.gameState, sched = g.gameScheduleState;
  const state = sched && sched !== 'OK' ? 'off'
    : ['LIVE', 'CRIT'].includes(s) ? 'live'
    : ['FINAL', 'OFF'].includes(s) ? 'final' : 'pre';
  const pd = g.periodDescriptor || {}, n = num(pd.number || g.period), clock = g.clock || {};
  const type = pd.periodType || 'REG';
  let detail = '', elapsed = null;
  if (state === 'off') detail = sched === 'PPD' ? 'Postponed' : 'Suspended';
  else if (state === 'final') {
    const last = g.gameOutcome?.lastPeriodType;
    detail = last && last !== 'REG' ? `Final/${last}` : 'Final';
    elapsed = 1;
  } else if (state === 'live') {
    const label = type === 'REG' ? ordinal(n) : type;
    detail = clock.inIntermission ? `${label} Int` : type === 'SO' ? 'SO' : `${label} ${clock.timeRemaining || ''}`.trim();
    elapsed = type === 'REG' ? Math.min(1, ((Math.min(n, 3) - 1) * 1200 + (1200 - num(clock.secondsRemaining))) / 3600) : 1;
  }
  return { id: String(g.id), league: 'NHL', start: g.startTimeUTC || null, state, detail, elapsed,
    home: nhlTeam(g.homeTeam), away: nhlTeam(g.awayTeam) };
}

function nhlPlayers(box) {
  const out = [];
  for (const side of ['home', 'away']) {
    const t = box.playerByGameStats?.[side + 'Team'] || {};
    for (const p of [...(t.forwards || []), ...(t.defense || [])]) {
      out.push({ name: p.name?.default || '', side, played: true, stats: {
        goals: num(p.goals), assists: num(p.assists), points: num(p.points), sog: num(p.sog),
        hits: num(p.hits), blocks: num(p.blockedShots), pim: num(p.pim), pp_goals: num(p.powerPlayGoals) } });
    }
    for (const p of t.goalies || []) {
      out.push({ name: p.name?.default || '', side, played: true, stats: {
        saves: num(p.saves), shots_against: num(p.shotsAgainst), goals_against: num(p.goalsAgainst) } });
    }
  }
  return out;
}

// ---- MLB: statsapi.mlb.com -----------------------------------------------

const MLB = 'https://statsapi.mlb.com/api/v1';

function mlbTeam(side = {}) {
  const t = side.team || {};
  return team(t.name, t.abbreviation, t.teamName || t.clubName, side.score);
}

function mlbGame(g) {
  const st = g.status || {}, ls = g.linescore || {};
  const coded = st.codedGameState;
  const state = ['D', 'C'].includes(coded) || /postponed|cancel|suspend/i.test(st.detailedState || '') ? 'off'
    : st.abstractGameState === 'Live' ? 'live'
    : st.abstractGameState === 'Final' ? 'final' : 'pre';
  let detail = '';
  if (state === 'off') detail = st.detailedState || 'Postponed';
  else if (state === 'final') detail = ls.currentInning && ls.currentInning !== (g.scheduledInnings || 9) ? `Final/${ls.currentInning}` : 'Final';
  else if (state === 'live') {
    if (/delay/i.test(st.detailedState || '')) detail = st.detailedState;
    else if (ls.currentInningOrdinal) detail = `${(ls.inningState || '').replace('Middle', 'Mid')} ${ls.currentInningOrdinal}`.trim();
    else detail = 'Live';
  }
  return { id: String(g.gamePk), league: 'MLB', start: g.gameDate || null, state, detail,
    elapsed: state === 'final' ? 1 : null, home: mlbTeam(g.teams?.home), away: mlbTeam(g.teams?.away) };
}

function mlbPlayers(box) {
  const out = [];
  for (const side of ['home', 'away']) {
    const t = box.teams?.[side] || {}, appeared = new Set([...(t.batters || []), ...(t.pitchers || [])]);
    for (const p of Object.values(t.players || {})) {
      const id = p.person?.id, b = p.stats?.batting || {}, pi = p.stats?.pitching || {};
      const played = appeared.has(id);
      const stats = {};
      if (played && (t.batters || []).includes(id)) {
        const hits = num(b.hits);
        Object.assign(stats, { hits, total_bases: num(b.totalBases), home_runs: num(b.homeRuns), rbis: num(b.rbi),
          runs: num(b.runs), walks: num(b.baseOnBalls), stolen_bases: num(b.stolenBases), doubles: num(b.doubles),
          triples: num(b.triples), singles: hits - num(b.doubles) - num(b.triples) - num(b.homeRuns),
          batter_strikeouts: num(b.strikeOuts) });
      }
      if (played && (t.pitchers || []).includes(id)) {
        Object.assign(stats, { pitcher_strikeouts: num(pi.strikeOuts), pitcher_outs: num(pi.outs),
          hits_allowed: num(pi.hits), earned_runs: num(pi.earnedRuns), walks_allowed: num(pi.baseOnBalls),
          pitches: num(pi.numberOfPitches) });
      }
      out.push({ name: p.person?.fullName || '', side, played, stats });
    }
  }
  return out;
}

// ---- NFL / NBA: ESPN site API --------------------------------------------

const ESPN = { NFL: 'football/nfl', NBA: 'basketball/nba' };
const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/';
const PERIOD = { NFL: [4, 900], NBA: [4, 720] };

// "category.key" -> canonical stat. Combined keys ("completions/passingAttempts",
// "threePointFieldGoalsMade-threePointFieldGoalsAttempted") are split first.
const ESPN_STATS = {
  'passing.completions': 'pass_completions', 'passing.passingAttempts': 'pass_attempts',
  'passing.passingYards': 'pass_yds', 'passing.passingTouchdowns': 'pass_tds', 'passing.interceptions': 'pass_interceptions',
  'rushing.rushingAttempts': 'rush_attempts', 'rushing.rushingYards': 'rush_yds', 'rushing.rushingTouchdowns': 'rush_tds',
  'rushing.longRushing': 'rush_long',
  'receiving.receptions': 'receptions', 'receiving.receivingYards': 'recv_yds', 'receiving.receivingTouchdowns': 'rec_tds',
  'receiving.receivingTargets': 'targets', 'receiving.longReception': 'rec_long',
  'kicking.fieldGoalsMade': 'fg_made', 'kicking.totalKickingPoints': 'kicking_points',
  'defensive.totalTackles': 'tackles', 'defensive.sacks': 'sacks', 'interceptions.interceptions': 'def_interceptions',
  '.points': 'points', '.rebounds': 'rebounds', '.assists': 'assists', '.steals': 'steals', '.blocks': 'blocks',
  '.turnovers': 'turnovers', '.threePointFieldGoalsMade': 'threes',
};

function espnTeam(c = {}) {
  const t = c.team || {};
  return team(t.displayName, t.abbreviation, t.shortDisplayName || t.name, c.score);
}

function espnGame(league, id, comp = {}, fallbackStatus) {
  const st = comp.status || fallbackStatus || {}, type = st.type || {};
  const state = /POSTPONED|CANCELED|SUSPENDED/.test(type.name || '') ? 'off'
    : type.state === 'in' ? 'live' : type.state === 'post' ? 'final' : 'pre';
  const [periods, secs] = PERIOD[league];
  let elapsed = state === 'final' ? 1 : null;
  if (state === 'live' && st.period) {
    const p = num(st.period);
    elapsed = p > periods ? 1 : Math.min(1, ((p - 1) * secs + (secs - num(st.clock))) / (periods * secs));
  }
  const sides = Object.fromEntries((comp.competitors || []).map(c => [c.homeAway, c]));
  return { id: String(id), league, start: comp.date || null, state,
    detail: state === 'pre' ? '' : type.shortDetail || type.detail || '', elapsed,
    home: espnTeam(sides.home), away: espnTeam(sides.away) };
}

function espnPlayers(summary) {
  const comps = summary.header?.competitions?.[0]?.competitors || [];
  const sideOf = Object.fromEntries(comps.map(c => [String(c.team?.id), c.homeAway]));
  const byName = new Map();
  for (const t of summary.boxscore?.players || []) {
    const side = sideOf[String(t.team?.id)] || '';
    for (const cat of t.statistics || []) {
      const keys = cat.keys || [];
      for (const a of cat.athletes || []) {
        const name = a.athlete?.displayName || '';
        if (!byName.has(name + side)) byName.set(name + side, { name, side, played: false, stats: {} });
        const p = byName.get(name + side);
        if (a.didNotPlay || !(a.stats || []).length) continue;
        p.played = true;
        keys.forEach((key, i) => {
          // Only combined keys split their value, so "-2" rushing yards stays -2.
          const subkeys = key.split(/[/-]/), raw = String(a.stats[i] ?? '');
          const values = subkeys.length > 1 ? raw.split(/[/-]/) : [raw];
          subkeys.forEach((sub, j) => {
            const canonical = ESPN_STATS[`${cat.name || ''}.${sub}`];
            if (canonical && subkeys.length === values.length) p.stats[canonical] = num(values[j]);
          });
        });
      }
    }
  }
  return [...byName.values()];
}

// ---- Requests --------------------------------------------------------------

const compact = date => date.replaceAll('-', '');

// Each plan lists the requests to make and how to assemble their JSON. A
// request is either a direct URL or a body for the live-stats proxy.
function scoreboardPlan(league, date) {
  if (league === 'NHL') return { requests: [{ proxy: { league, date } }], build: ([d]) => (d.games || []).map(nhlGame) };
  if (league === 'MLB') return { requests: [{ url: `${MLB}/schedule?sportId=1&date=${date}&hydrate=linescore,team` }],
    build: ([d]) => (d.dates || []).flatMap(x => x.games || []).map(mlbGame) };
  return { requests: [{ url: `${ESPN_BASE}${ESPN[league]}/scoreboard?dates=${compact(date)}` }],
    build: ([d]) => (d.events || []).map(e => espnGame(league, e.id, { ...e.competitions?.[0], date: e.date })) };
}

function boxPlan(league, id) {
  if (league === 'NHL') return { requests: [{ proxy: { league, game: id } }],
    build: ([d]) => ({ game: nhlGame(d), players: nhlPlayers(d) }) };
  if (league === 'MLB') return { requests: [{ url: `${MLB}/schedule?sportId=1&gamePk=${id}&hydrate=linescore,team` }, { url: `${MLB}/game/${id}/boxscore` }],
    build: ([s, b]) => {
      const g = (s.dates || []).flatMap(x => x.games || [])[0];
      if (!g) throw new Error('Game not found');
      return { game: mlbGame(g), players: mlbPlayers(b) };
    } };
  return { requests: [{ url: `${ESPN_BASE}${ESPN[league]}/summary?event=${id}` }],
    build: ([d]) => {
      const comp = d.header?.competitions?.[0];
      if (!comp) throw new Error('Game not found');
      return { game: espnGame(league, id, comp), players: espnPlayers(d) };
    } };
}

const api = { LEAGUES, scoreboardPlan, boxPlan };
if (typeof module === 'object' && module.exports) module.exports = api;
else global.FVLiveFeeds = api;
})(typeof window === 'undefined' ? globalThis : window);
