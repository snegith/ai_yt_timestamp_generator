/**
 * Property 3: Time String Format Invariant
 *
 * For any non-negative integer number of seconds `s`,
 * `_secondsToTimeStr(s)` SHALL return a string matching the regex
 * `^\d+:\d{2}(:\d{2})?$` (i.e., `M:SS` or `H:MM:SS` format).
 *
 * Validates: Requirements 10.2
 */

'use strict';

const fc = require('fast-check');

// ---------------------------------------------------------------------------
// Inline _secondsToTimeStr from content.js (browser content script — not a
// Node module, so we reproduce the implementation here verbatim).
// ---------------------------------------------------------------------------

/**
 * Convert an integer number of seconds to a "M:SS" or "H:MM:SS" string.
 * @param {number} s  Non-negative integer seconds.
 * @returns {string}
 */
function _secondsToTimeStr(s) {
  s = Math.floor(s);
  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const seconds = s % 60;

  const mm = String(minutes).padStart(2, '0');
  const ss = String(seconds).padStart(2, '0');

  if (hours > 0) {
    return `${hours}:${mm}:${ss}`;
  }
  return `${minutes}:${ss}`;
}

// ---------------------------------------------------------------------------
// Property 3 test
// ---------------------------------------------------------------------------

const FORMAT_REGEX = /^\d+:\d{2}(:\d{2})?$/;

// Generator: non-negative integers in [0, 359999] (0 to ~100 hours)
const nonNegativeSeconds = fc.integer({ min: 0, max: 359999 });

console.log('Running Property 3: Time String Format Invariant...');

fc.assert(
  fc.property(nonNegativeSeconds, (s) => {
    const result = _secondsToTimeStr(s);
    if (!FORMAT_REGEX.test(result)) {
      throw new Error(
        `_secondsToTimeStr(${s}) returned "${result}", which does not match ${FORMAT_REGEX}`
      );
    }
    return true;
  }),
  {
    numRuns: 1000, // well above the 100-iteration minimum
    verbose: true,
  }
);

console.log('Property 3 PASSED: all outputs match /^\\d+:\\d{2}(:\\d{2})?$/');
