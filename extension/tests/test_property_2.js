/**
 * Property 2: Time String Parsing Round-Trip
 *
 * For any non-negative integer number of seconds `s`, formatting `s` with
 * `_secondsToTimeStr(s)` and then parsing the result with
 * `timeStringToSeconds()` SHALL return `s`.
 *
 * Validates: Requirements 3.2, 10.2
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
// Inline timeStringToSeconds from content.js (verbatim).
// ---------------------------------------------------------------------------

/**
 * Convert a time string in "M:SS" or "H:MM:SS" format to integer seconds.
 * @param {string} timeStr
 * @returns {number}
 */
function timeStringToSeconds(timeStr) {
  const parts = timeStr.split(':').map(Number);
  if (parts.length === 2) {
    // M:SS
    return parts[0] * 60 + parts[1];
  } else if (parts.length === 3) {
    // H:MM:SS
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Property 2 test
// ---------------------------------------------------------------------------

// Generator: non-negative integers in [0, 359999] (0 to ~100 hours)
const nonNegativeSeconds = fc.integer({ min: 0, max: 359999 });

console.log('Running Property 2: Time String Parsing Round-Trip...');

fc.assert(
  fc.property(nonNegativeSeconds, (s) => {
    const timeStr = _secondsToTimeStr(s);
    const parsed = timeStringToSeconds(timeStr);
    if (parsed !== s) {
      throw new Error(
        `Round-trip failed for s=${s}: _secondsToTimeStr(${s}) = "${timeStr}", ` +
        `timeStringToSeconds("${timeStr}") = ${parsed} (expected ${s})`
      );
    }
    return true;
  }),
  {
    numRuns: 1000,
    verbose: true,
  }
);

console.log('Property 2 PASSED: timeStringToSeconds(_secondsToTimeStr(s)) === s for all tested s');
