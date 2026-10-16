// Text rules for the branded chart share image (docs/assets/chart-share.js).
const assert = require('node:assert/strict');
const s = require('../docs/assets/chart-share.js');

// Captions lose their figure number and interaction hints, and never split at a decimal point.
assert.equal(s.stripHints('Figure 1. NFL: share of bets won. Blue rows met the rule. Hover or tap a row for the full record.'),
  'NFL: share of bets won. Blue rows met the rule.');
assert.equal(s.stripHints('Books: the no-vig chance across books quoting 17.5 and 18.5; “exactly 18” is left over. Tap any row on the live page.'),
  'Books: the no-vig chance across books quoting 17.5 and 18.5; “exactly 18” is left over.');
assert.equal(s.stripHints('Rushing yards by week.'), 'Rushing yards by week.');
assert.equal(s.stripHints(''), '');

assert.equal(s.figureId('Figure 3. NBA: share of bets', 0), 'figure-3');
assert.equal(s.figureId('Fig. 12b: totals', 4), 'figure-12b');
assert.equal(s.figureId('One highest-EV offer per player', 4), 'chart-5');

assert.equal(s.pageLabel('/research/low-scoring-games-situational-factors.html'), 'Research');
assert.equal(s.pageLabel('/blog/chris-sale-pitcher-outs-2026.html'), 'Analysis');
assert.equal(s.pageLabel('/nhl/totals/'), 'NHL');
assert.equal(s.pageLabel('/'), '');

const url = 'https://fourthandvalue.com/research/low-scoring-games-situational-factors.html#figure-1';
assert.equal(s.withUtm(url, 'x'),
  'https://fourthandvalue.com/research/low-scoring-games-situational-factors.html?utm_source=x&utm_medium=social&utm_campaign=chart_share#figure-1');
assert.equal(s.withUtm('https://fourthandvalue.com/markets/?sport=nfl', 'reddit'),
  'https://fourthandvalue.com/markets/?sport=nfl&utm_source=reddit&utm_medium=social&utm_campaign=chart_share');
assert.equal(s.displayUrl(url), 'fourthandvalue.com/research/low-scoring-games-situational-factors.html');
assert.equal(s.displayUrl('https://fourthandvalue.com/blog/index.html?x=1'), 'fourthandvalue.com/blog');
assert.equal(s.displayUrl('https://fourthandvalue.com/research/' + 'a'.repeat(80) + '.html'), 'fourthandvalue.com/research');

// Social text: headline plus takeaway, trimmed at a word so the link still fits.
const meta = {title: 'What makes NFL games go under?', subtitle: 'Of 18 situations, one held up in both decades.'};
assert.equal(s.socialText(meta), 'What makes NFL games go under? Of 18 situations, one held up in both decades.');
assert.equal(s.socialText({title: 'Every market at a glance', subtitle: 'Most sit near the line.'}), 'Every market at a glance: Most sit near the line.');
const short = s.socialText({title: 'Headline', subtitle: 'word '.repeat(80)}, 60);
assert.ok(short.length <= 60 && short.endsWith('…'), short);

// Dates read in Eastern Time; a bare calendar date is never shifted to the previous day.
assert.equal(s.formatDate('2026-09-20'), 'September 20, 2026');
assert.equal(s.formatDate('2026-10-16T02:30:00+00:00'), 'October 15, 2026');
assert.equal(s.formatDate('2026-10-14T11:07:59.832907+00:00'), 'October 14, 2026');
assert.equal(s.formatDate('not a date'), '');
assert.match(s.formatCaptured(Date.parse('2026-10-16T16:05:00Z')), /^As of Oct 16, 2026, 12:05 PM EDT$/);
assert.match(s.formatCaptured(Date.parse('2026-12-01T16:05:00Z')), /^As of Dec 1, 2026, 11:05 AM EST$/);

// Wrapping keeps every word and marks a cut with an ellipsis.
const measure = t => t.length * 10;
assert.deepEqual(s.wrap(measure, 'one two three four five', 100), ['one two', 'three four', 'five']);
assert.deepEqual(s.wrap(measure, 'one two three four five', 100, 2), ['one two', 'three…']);

console.log('PASS: chart share captions, links, social text, Eastern dates and wrapping.');
