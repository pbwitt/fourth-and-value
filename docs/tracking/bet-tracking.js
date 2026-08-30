// Shared Supabase client + auth helpers for the bet tracker.
//
// This replaces the old github-api.js, which worked by having each visitor
// paste a personal GitHub access token into localStorage and commit every
// bet straight into this site's own source repo. That only ever worked for
// a single person (you) - there's no way to hand out repo-write access to
// the public. This version uses a real per-user account (Supabase Auth) and
// a real per-user database table (Postgres, via Supabase), so any signed-in
// visitor can track their own bets without ever touching this codebase.
//
// SUPABASE_ANON_KEY below is a PUBLIC key - it's safe to ship in this file.
// It can only do what the database's row-level security policies allow
// (see supabase/schema.sql): a signed-in user can read/write their own
// rows, nothing else. The much more powerful service_role key must never
// appear here - it only ever lives in GitHub Actions secrets, used by the
// grading scripts.

const SUPABASE_URL = 'https://fzjonxpzsrbdhbujbhsn.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZ6am9ueHB6c3JiZGhidWpiaHNuIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODgwMzU2ODMsImV4cCI6MjEwMzYxMTY4M30.U640KpqB4uHiF0Q-b0lcAq0bpVv5OKfOKXUKbNg30nQ';

const supabaseClient = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function getCurrentUser() {
  const { data: { session } } = await supabaseClient.auth.getSession();
  return session?.user ?? null;
}

async function signInWithEmail(email) {
  const { error } = await supabaseClient.auth.signInWithOtp({
    email,
    options: { emailRedirectTo: window.location.href },
  });
  return { ok: !error, error };
}

async function signOut() {
  await supabaseClient.auth.signOut();
}

/**
 * One-click "track this bet" call used from the props/totals pages
 * (docs/nfl/totals, docs/nhl/props, docs/nhl/totals). Same function name
 * and shape as before (window.autoTrackBet(betData)) so those pages needed
 * no changes beyond loading this file instead of the old one.
 */
async function autoTrackBet(betData) {
  const user = await getCurrentUser();

  if (!user) {
    const goSignIn = confirm(
      'In order to track your bets, you need to create a free account - it only takes an email, no password required.\n\n' +
      'We will never sell or share your email or personal information with any third party.\n\n' +
      'Click OK to create your free account now.'
    );
    if (goSignIn) window.location.href = '/tracking/';
    return false;
  }

  const row = {
    user_id: user.id,
    league: betData.league,
    game_date: betData.game_date || null,
    team_home: betData.team_home || null,
    team_away: betData.team_away || null,
    player: betData.player || null,
    market_type: betData.market_type || null,
    side: betData.side || null,
    line: betData.line !== '' && betData.line != null ? Number(betData.line) : null,
    book: betData.book || null,
    odds: betData.odds !== '' && betData.odds != null ? Number(betData.odds) : null,
    stake_dollars: Number(betData.stake_dollars),
    status: 'pending',
    model_prob: betData.model_prob || null,
    edge_bps: betData.edge_bps || null,
  };

  const { error } = await supabaseClient.from('bets').insert(row);

  if (error) {
    console.error('Error tracking bet:', error);
    alert(`Error tracking bet: ${error.message}`);
    return false;
  }

  alert(`Bet tracked!\n\n${betData.player || 'Team Total'} ${betData.market_type} ${betData.side} ${betData.line}\nStake: $${betData.stake_dollars}\n\nView at: https://fourthandvalue.com/tracking/`);
  return true;
}

window.supabaseClient = supabaseClient;
window.getCurrentUser = getCurrentUser;
window.signInWithEmail = signInWithEmail;
window.signOut = signOut;
window.autoTrackBet = autoTrackBet;
