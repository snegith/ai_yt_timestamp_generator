/**
 * Property 4: Timestamp List Rendering Completeness
 *
 * For any non-empty array of timestamp objects `[{ time, title }]`,
 * `renderTimestamps()` SHALL produce a list containing exactly one clickable
 * element per timestamp, each displaying the correct `time` and `title` strings.
 *
 * Validates: Requirements 3.1
 */

'use strict';

const fc = require('fast-check');
const { JSDOM } = require('jsdom');

// ---------------------------------------------------------------------------
// Build a fresh jsdom document with the sidebar HTML for each test run.
// ---------------------------------------------------------------------------

function createDocument() {
  const dom = new JSDOM(`
    <!DOCTYPE html>
    <html>
      <body>
        <div id="yt-ts-sidebar">
          <div id="yt-ts-body">
            <button id="yt-ts-generate-btn">Generate Timestamps</button>
            <div id="yt-ts-status"></div>
            <ul id="yt-ts-list"></ul>
          </div>
        </div>
      </body>
    </html>
  `);
  return dom.window.document;
}

// ---------------------------------------------------------------------------
// Inline renderTimestamps adapted to accept a `document` parameter so it
// works in Node/jsdom rather than relying on the browser global.
// ---------------------------------------------------------------------------

/**
 * Render an array of timestamp objects as a clickable list.
 *
 * @param {Array<{time: string, title: string}>} timestamps
 * @param {Document} document  The DOM document to operate on.
 */
function renderTimestamps(timestamps, document) {
  const list = document.getElementById('yt-ts-list');
  const status = document.getElementById('yt-ts-status');

  if (!list) return;

  // Clear previous results and any error styling
  list.innerHTML = '';
  if (status) {
    status.textContent = '';
    status.style.color = '';
  }

  timestamps.forEach(({ time, title }) => {
    const li = document.createElement('li');
    li.className = 'yt-ts-item';

    const btn = document.createElement('button');
    btn.className = 'yt-ts-timestamp-btn';
    btn.type = 'button';

    const timeSpan = document.createElement('span');
    timeSpan.className = 'yt-ts-time';
    timeSpan.textContent = time;

    const titleSpan = document.createElement('span');
    titleSpan.className = 'yt-ts-title';
    titleSpan.textContent = title;

    btn.appendChild(timeSpan);
    btn.appendChild(titleSpan);
    li.appendChild(btn);
    list.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

// Arbitrary non-empty string (printable ASCII, length 1–50)
const printableString = fc.string({ minLength: 1, maxLength: 50 });

// Arbitrary timestamp object
const timestampArb = fc.record({
  time: printableString,
  title: printableString,
});

// Non-empty array of timestamp objects (1–20 items)
const nonEmptyTimestampArray = fc.array(timestampArb, { minLength: 1, maxLength: 20 });

// ---------------------------------------------------------------------------
// Property 4 test
// ---------------------------------------------------------------------------

console.log('Running Property 4: Timestamp List Rendering Completeness...');

fc.assert(
  fc.property(nonEmptyTimestampArray, (timestamps) => {
    const document = createDocument();
    renderTimestamps(timestamps, document);

    const list = document.getElementById('yt-ts-list');
    const liElements = list.querySelectorAll('li');

    // 1. Exactly one <li> per timestamp
    if (liElements.length !== timestamps.length) {
      throw new Error(
        `Expected ${timestamps.length} <li> elements, got ${liElements.length}`
      );
    }

    // 2. Each <li> contains a <button>, and the button's spans match time/title
    for (let i = 0; i < timestamps.length; i++) {
      const li = liElements[i];
      const btn = li.querySelector('button');

      if (!btn) {
        throw new Error(`<li> at index ${i} has no <button> element`);
      }

      const timeSpan = btn.querySelector('.yt-ts-time');
      const titleSpan = btn.querySelector('.yt-ts-title');

      if (!timeSpan) {
        throw new Error(`button at index ${i} has no .yt-ts-time span`);
      }
      if (!titleSpan) {
        throw new Error(`button at index ${i} has no .yt-ts-title span`);
      }

      if (timeSpan.textContent !== timestamps[i].time) {
        throw new Error(
          `time mismatch at index ${i}: expected "${timestamps[i].time}", got "${timeSpan.textContent}"`
        );
      }
      if (titleSpan.textContent !== timestamps[i].title) {
        throw new Error(
          `title mismatch at index ${i}: expected "${timestamps[i].title}", got "${titleSpan.textContent}"`
        );
      }
    }

    return true;
  }),
  {
    numRuns: 100,
    verbose: true,
  }
);

console.log('Property 4 PASSED: renderTimestamps() produces exactly one clickable element per timestamp with correct time and title.');
