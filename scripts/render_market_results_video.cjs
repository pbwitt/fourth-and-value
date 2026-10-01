// Render the Market Results walkthrough: the live pages in a phone frame with captions.
// Frames are captured one at a time so motion is smooth regardless of machine speed,
// then encoded with ffmpeg. When docs/videos/market-results/timeline.json exists
// (from scripts/narrate_editorial_video.py), each caption beat holds until its
// narration ends and the narration is mixed in. Usage:
//   NODE_PATH=$(npm root -g) node scripts/render_market_results_video.cjs
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { execFileSync } = require('node:child_process');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '../docs');
const out = path.join(root, 'videos', 'market-results');
const frames = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'fv-frames-'));
const FPS = 30;
// The video quotes Weeks 1-3 numbers, so it always renders from that frozen snapshot.
const SNAPSHOT = path.join(out, 'nfl-weeks-1-3.json');
const timelinePath = path.join(out, 'timeline.json');
const narration = fs.existsSync(timelinePath) ? JSON.parse(fs.readFileSync(timelinePath, 'utf8')).scenes : null;
const types = {'.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png'};
const server = http.createServer((req, res) => {
  let file = path.resolve(root, '.' + decodeURIComponent(req.url.split('?')[0]));
  if (!file.startsWith(root)) { res.writeHead(403); return res.end(); }
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.setHeader('Content-Type', types[path.extname(file)] || 'application/octet-stream');
    res.end(data);
  });
});

// [eyebrow, headline, subline] for each beat of the walkthrough.
const BEATS = [
  ['New on Fourth & Value', 'Are overs or unders hitting?', 'Market Results grades every prop and game line we track against the books’ own prices.'],
  ['No models. No picks.', 'Just the market and what happened.', 'What the prices implied, next to the results.'],
  ['Every market at a glance', 'Ring: what the prices implied. Dot: what happened.', 'Blue means more overs than expected. Orange means more unders.'],
  ['Tap any market', 'Each one gets the same charts.', 'Here: NFL rushing yards, Weeks 1–3.'],
  ['Is it more than chance?', 'The shaded band is where luck usually keeps the count.', 'Rushing unders lead 134–115. That gap is common by chance.'],
  ['Week by week', 'Week 2 was an under week.', 'Rushing overs went 33–57, outside the usual range. Week 3 snapped back.'],
  ['Season or lately', 'Switch to the last week or two.', 'See what the market is doing right now.'],
  ['Every result vs. the line', 'How far results landed, and the biggest misses.', 'Lines sit near the middle result, not the average.'],
  ['When we price it', 'Prices come from our pregame snapshot.', 'Not closing lines. Every page says when.'],
  ['fourthandvalue.com/markets', 'NFL now. NHL this season.', 'Free to read. No sign-up.'],
];

(async () => {
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || undefined});
  let n = 0;
  try {
    const page = await browser.newPage({viewport: {width: 1080, height: 1920}, deviceScaleFactor: 1});
    await page.route('**/markets/data/nfl.json', route => route.fulfill({contentType: 'application/json', body: fs.readFileSync(SNAPSHOT)}));
    if (narration && narration.length !== BEATS.length) throw new Error(`Narration has ${narration.length} scenes; the video has ${BEATS.length} beats`);
    const starts = [];
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(`${base}/videos/market-results/render.html`);
    await page.waitForFunction(() => window.videoReady);
    const d = (fn, ...args) => page.evaluate(([f, a]) => window.director[f](...a), [fn, args]);
    const shot = async () => { await page.screenshot({path: path.join(frames, `${String(n++).padStart(5, '0')}.jpg`), type: 'jpeg', quality: 92}); };
    const hold = async seconds => {
      const first = path.join(frames, `${String(n).padStart(5, '0')}.jpg`);
      await shot();
      for (let i = 1; i < Math.round(seconds * FPS); i++) fs.copyFileSync(first, path.join(frames, `${String(n++).padStart(5, '0')}.jpg`));
    };
    const ease = t => t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
    const scrollTo = async (y, seconds = 0.9) => {
      const from = await d('scrollY'), steps = Math.round(seconds * FPS);
      for (let i = 1; i <= steps; i++) { await d('setScroll', from + (y - from) * ease(i / steps)); await shot(); }
    };
    // Hold the current frame until the previous beat's narration has finished.
    const finishBeat = async i => {
      if (!narration || i < 0) return;
      const need = Math.ceil((narration[i].speech_duration + 0.45) * FPS) - (n - starts[i]);
      if (need > 0) await hold(need / FPS);
    };
    const beat = async (i, fade = true) => {
      await finishBeat(i - 1);
      await d('progress', i / (BEATS.length - 1));
      if (fade) for (let f = 6; f >= 0; f--) { await d('opacity', f / 6); await shot(); }
      await d('caption', ...BEATS[i]);
      starts[i] = n;
      if (fade) for (let f = 1; f <= 6; f++) { await d('opacity', f / 6); await shot(); }
    };
    const tap = async (sel, after) => {
      const p = await d('point', sel), steps = 14;
      for (let i = 0; i <= steps; i++) {
        await d('tap', p, i / steps);
        if (i === Math.round(steps / 2)) await after();
        await shot();
      }
      await d('tap', p, 1);
    };
    const followPageScroll = async () => { const t = await d('pending'); if (t != null) await scrollTo(t, 1.0); };
    const NAV = 112; // sticky nav + toolbar height inside the 430px-wide page

    await d('load', `${base}/markets/`);
    await d('caption', ...BEATS[0]); await d('progress', 0); starts[0] = n;
    await hold(3.6);
    await beat(1); await hold(2.8);
    await tap('a.sport[href="nfl/"]', async () => {});
    await d('load', `${base}/markets/?sport=nfl#m=rush_yds&w=season`);
    await beat(2, false);
    await scrollTo(await d('targetY', 'figure:has([data-chart=board])', NAV));
    await hold(4.6);
    await beat(3);
    await tap('[data-chart=board] [data-key=rush_yds]', async () => d('click', '[data-chart=board] [data-key=rush_yds]'));
    await followPageScroll();
    await hold(2.8);
    await beat(4);
    await scrollTo(await d('targetY', 'figure:has([data-chart=tally])', NAV + 8));
    await hold(4.6);
    await beat(5);
    await scrollTo(await d('targetY', 'figure[data-fig=periods]', NAV + 10));
    await hold(4.6);
    await beat(6);
    await tap('[data-win=last2]', async () => d('click', '[data-win=last2]'));
    await hold(3.0);
    await beat(7);
    await scrollTo(await d('targetY', '[data-chart=hist]', NAV + 60), 1.1);
    await hold(2.6);
    await scrollTo(await d('targetY', '[data-chart=misses]', NAV + 60), 1.1);
    await hold(2.6);
    await beat(8);
    await scrollTo(0, 1.4);
    await scrollTo(await d('targetY', '[data-timing]', NAV + 40), 0.8);
    await hold(3.6);
    await beat(9);
    await d('load', `${base}/markets/`);
    await hold(3.6);
    await finishBeat(BEATS.length - 1);
    await hold(0.6);
    await page.screenshot({path: path.join(out, 'poster.png')});
    fs.writeFileSync(path.join(frames, 'starts.json'), JSON.stringify(starts));
    if (errors.length) throw new Error(errors.join('\n'));
  } finally { await browser.close(); server.close(); }

  const mp4 = path.join(out, 'market-results.mp4');
  const video = ['-framerate', String(FPS), '-i', path.join(frames, '%05d.jpg')];
  const encode = ['-c:v', 'libx264', '-preset', 'slow', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'];
  if (narration) {
    // Each scene's narration starts just after its caption appears.
    const starts = JSON.parse(fs.readFileSync(path.join(frames, 'starts.json'), 'utf8'));
    const inputs = narration.flatMap(s => ['-i', path.join(out, s.audio)]);
    const delays = narration.map((s, i) => `[${i + 1}:a]adelay=${Math.round((starts[i] / FPS + 0.15) * 1000)}:all=1[a${i}]`);
    const mix = `${delays.join(';')};${narration.map((s, i) => `[a${i}]`).join('')}amix=inputs=${narration.length}:normalize=0:dropout_transition=0[aout]`;
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...video, ...inputs, '-filter_complex', mix,
      '-map', '0:v', '-map', '[aout]', ...encode, '-c:a', 'aac', '-b:a', '160k', '-t', (n / FPS).toFixed(3), mp4]);
  } else {
    execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...video, ...encode, mp4]);
  }
  const info = {width: 1080, height: 1920, duration: +(n / FPS).toFixed(2), bytes: fs.statSync(mp4).size, narrated: !!narration};
  fs.writeFileSync(path.join(out, 'render-info.json'), JSON.stringify(info, null, 2) + '\n');
  fs.rmSync(frames, {recursive: true, force: true});
  console.log('Rendered', info);
})().catch(e => { console.error(e); process.exitCode = 1; server.close(); });
