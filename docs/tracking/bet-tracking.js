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

// The email allowance belongs to the whole site, not to a reader's submissions.
function signInErrorMessage(error) {
  const detail = error?.message || '';
  if (/for security purposes.*after.*seconds/i.test(detail)) {
    return 'A sign-in email was requested recently. Please wait before requesting another link, and check your inbox for the previous email.';
  }
  if (error?.code === 'over_email_send_rate_limit' || /email rate limit exceeded/i.test(detail)) {
    return 'We can’t send a sign-in email right now because the site’s hourly email limit has been reached. This allowance is shared by everyone using the site. Please try again in about an hour. If you’re already signed in, you can keep using your account.';
  }
  if (error?.code === 'over_request_rate_limit' || error?.status === 429) {
    return 'Too many sign-in attempts were made recently. Please wait a few minutes before trying again.';
  }
  if (error?.code === 'email_address_not_authorized') {
    return 'The site’s email service is not set up to send a sign-in link to this address.';
  }
  return detail || 'We couldn’t request a sign-in email. Please try again later.';
}

async function signOut() {
  await supabaseClient.auth.signOut();
}

// Structured result for the briefing dialog; the legacy alert-based wrapper
// below retains the existing props/totals call contract.
async function saveTrackedBet(betData) {
  const user = await getCurrentUser();
  if (!user) {
    return {ok:false, needsSignIn:true, error:'Sign in to Bet Tracker, then return here to save this bet.'};
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
    model_prob: betData.model_prob ?? null,
    edge_bps: betData.edge_bps ?? null,
  };
  if (!row.league || !Number.isFinite(row.stake_dollars) || row.stake_dollars <= 0 ||
      !Number.isInteger(row.odds) || Math.abs(row.odds) < 100 ||
      (row.line !== null && !Number.isFinite(row.line))) {
    return {ok:false, error:'Enter a positive stake, valid American odds and a valid line.'};
  }
  // A caller may retain this ID across an uncertain network response. Retrying
  // cannot insert a second ticket or overwrite one already saved.
  if (betData.id !== undefined) {
    if (!/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(betData.id)) {
      return {ok:false, error:'Invalid tracking reference. Reopen the bet to try again.'};
    }
    row.id=betData.id;
  }

  const { error } = await supabaseClient.from('bets').insert(row);
  if (error?.code === '23505' && row.id) {
    const existing=await supabaseClient.from('bets').select('*').eq('id',row.id).eq('user_id',user.id).maybeSingle();
    const numeric=new Set(['line','odds','stake_dollars','model_prob','edge_bps']);
    const same=!existing.error && existing.data && Object.keys(row).filter(k=>k!=='status').every(k=>
      numeric.has(k) && row[k]!==null ? existing.data[k]!=null&&Number(existing.data[k])===row[k] : (existing.data[k]??null)===row[k]);
    if (same) return {ok:true,id:row.id};
    return {ok:false,error:'This ticket may already be saved with different details. Check Bet Tracker before logging it again.'};
  }
  if (error) return {ok:false,error:'The bet could not be saved. Check Bet Tracker before retrying if the connection was interrupted.'};
  return {ok:true,id:row.id};
}

async function autoTrackBet(betData) {
  const result=await saveTrackedBet(betData);
  if (result.needsSignIn) {
    const goSignIn = confirm(
      'In order to track your bets, you need to create a free account - it only takes an email, no password required.\n\n' +
      'We will never sell or share your email or personal information with any third party.\n\n' +
      'Click OK to create your free account now.'
    );
    if (goSignIn) window.location.href = '/tracking/';
    return false;
  }
  if (!result.ok) {
    alert(result.error);
    return false;
  }

  alert(`Bet tracked!\n\n${betData.player || 'Team Total'} ${betData.market_type} ${betData.side} ${betData.line}\nStake: $${betData.stake_dollars}\n\nView at: https://fourthandvalue.com/tracking/`);
  return true;
}

window.supabaseClient = supabaseClient;
window.getCurrentUser = getCurrentUser;
window.signInWithEmail = signInWithEmail;
window.signInErrorMessage = signInErrorMessage;
window.signOut = signOut;
window.autoTrackBet = autoTrackBet;
window.saveTrackedBet = saveTrackedBet;

function betTrackerSummary(bets) {
  const settled=bets.filter(b=>['won','lost','push'].includes(b.status));
  const staked=settled.reduce((sum,b)=>sum+Number(b.stake_dollars||0),0);
  const returned=settled.reduce((sum,b)=>sum+Number(b.payout||0),0);
  return {totalBets:bets.length,totalStaked:bets.reduce((sum,b)=>sum+Number(b.stake_dollars||0),0),
    totalReturned:returned,profitLoss:returned-staked,roi:staked>0?(returned-staked)/staked*100:0,
    winRate:settled.length?100*settled.filter(b=>b.status==='won').length/settled.length:0};
}
window.betTrackerSummary=betTrackerSummary;
