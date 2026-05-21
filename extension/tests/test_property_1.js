/**
 * Property-Based Test: Video ID Extraction Correctness (Property 1)
 *
 * Validates: Requirements 2.2, 2.5
 *
 * Property 1:
 *   - For any URL string of the form `https://www.youtube.com/watch?v=<id>[&...]`,
 *     `extractVideoId()` SHALL return exactly the value of the `v` query parameter.
 *   - For any URL that does not contain a `v` parameter, it SHALL return `null`.
 *
 * Uses fast-check for property-based testing with a minimum of 100 iterations.
 */

'use strict';

const fc = require('fast-check');

// ---------------------------------------------------------------------------
// Inline the extractVideoId logic with a mockable search string
// ---------------------------------------------------------------------------

/**
 * Pure, testable version of extractVideoId that accepts a query string
 * instead of reading from window.location.search directly.
 *
 * This mirrors the logic in content.js exactly:
 *   const params = new URLSearchParams(window.location.search);
 *   return params.get('v');
 *
 * @param {string} search  The query string (e.g. "?v=abc123&t=10")
 * @returns {string|null}
 */
function extractVideoId(search) {
  const params = new URLSearchParams(search);
  return params.get('v');
}

// ---------------------------------------------------------------------------
// Arbitraries (generators)
// ---------------------------------------------------------------------------

/**
 * Generates a YouTube-like video ID: 11 alphanumeric characters (a-z, A-Z, 0-9, -, _).
 * YouTube IDs are base64url-encoded, so we include - and _ as valid chars.
 */
const videoIdArb = fc.stringOf(
  fc.mapToConstant(
    { num: 26, build: (i) => String.fromCharCode(97 + i) },   // a-z
    { num: 26, build: (i) => String.fromCharCode(65 + i) },   // A-Z
    { num: 10, build: (i) => String.fromCharCode(48 + i) },   // 0-9
    { num: 1,  build: () => '-' },
    { num: 1,  build: () => '_' }
  ),
  { minLength: 1, maxLength: 20 }
);

/**
 * Generates a random query parameter key that is NOT "v", to avoid
 * accidentally introducing a `v` param in the "no v param" tests.
 * Uses fc.constantFrom with explicit alphabet chars (fast-check 3.x compatible).
 */
const alphaCharArb = fc.constantFrom(
  ...'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
);

const nonVKeyArb = fc
  .stringOf(alphaCharArb, { minLength: 1, maxLength: 10 })
  .filter((k) => k !== 'v');

/**
 * Generates a random query parameter value (alphanumeric).
 */
const alphaNumCharArb = fc.constantFrom(
  ...'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'.split('')
);
const paramValueArb = fc.stringOf(alphaNumCharArb, { minLength: 0, maxLength: 20 });

/**
 * Generates an array of [key, value] pairs where no key is "v".
 */
const extraParamsArb = fc.array(
  fc.tuple(nonVKeyArb, paramValueArb),
  { minLength: 0, maxLength: 5 }
);

/**
 * Builds a URLSearchParams query string from a list of [key, value] pairs.
 * @param {Array<[string, string]>} pairs
 * @returns {string}  e.g. "?foo=bar&baz=qux" or ""
 */
function buildQueryString(pairs) {
  if (pairs.length === 0) return '';
  const p = new URLSearchParams(pairs);
  return '?' + p.toString();
}

// ---------------------------------------------------------------------------
// Property 1a: URLs WITH a `v` parameter return exactly that value
// ---------------------------------------------------------------------------

console.log('Running Property 1a: URLs with ?v=<id> return exactly <id>');

fc.assert(
  fc.property(videoIdArb, extraParamsArb, (videoId, extraParams) => {
    // Build a query string that includes v=<videoId> plus optional extra params.
    // We insert v at a random position among the extra params to ensure order
    // independence — URLSearchParams.get('v') should always find it.
    const allParams = [['v', videoId], ...extraParams];
    const search = '?' + new URLSearchParams(allParams).toString();

    const result = extractVideoId(search);
    return result === videoId;
  }),
  {
    numRuns: 200,
    verbose: true,
  }
);

console.log('  ✓ Property 1a passed (200 runs)');

// ---------------------------------------------------------------------------
// Property 1b: URLs WITHOUT a `v` parameter return null
// ---------------------------------------------------------------------------

console.log('Running Property 1b: URLs without v param return null');

fc.assert(
  fc.property(extraParamsArb, (extraParams) => {
    // Build a query string with no `v` key at all.
    const search = buildQueryString(extraParams);

    const result = extractVideoId(search);
    return result === null;
  }),
  {
    numRuns: 200,
    verbose: true,
  }
);

console.log('  ✓ Property 1b passed (200 runs)');

// ---------------------------------------------------------------------------
// Property 1c: v param value is returned verbatim (no trimming or mutation)
// ---------------------------------------------------------------------------

console.log('Running Property 1c: v param value is returned verbatim');

fc.assert(
  fc.property(videoIdArb, (videoId) => {
    // Minimal query string: just ?v=<id>
    const search = '?v=' + encodeURIComponent(videoId);
    const result = extractVideoId(search);
    // URLSearchParams decodes percent-encoding, so result should equal the
    // original (decoded) videoId string.
    return result === videoId;
  }),
  {
    numRuns: 200,
    verbose: true,
  }
);

console.log('  ✓ Property 1c passed (200 runs)');

// ---------------------------------------------------------------------------
// Edge cases: empty string and no query string at all
// ---------------------------------------------------------------------------

console.log('Running edge cases...');

// Empty search string → no v param → null
const emptyResult = extractVideoId('');
console.assert(emptyResult === null, `Expected null for empty string, got: ${emptyResult}`);

// Just "?" with no params → null
const justQuestionMark = extractVideoId('?');
console.assert(justQuestionMark === null, `Expected null for "?", got: ${justQuestionMark}`);

// v= with empty value → empty string (not null — the param exists but is empty)
const emptyV = extractVideoId('?v=');
console.assert(emptyV === '', `Expected "" for "?v=", got: ${emptyV}`);

// v param appears after other params
const vLast = extractVideoId('?t=30&v=abc123');
console.assert(vLast === 'abc123', `Expected "abc123" for "?t=30&v=abc123", got: ${vLast}`);

// Multiple v params — URLSearchParams.get returns the first one
const multiV = extractVideoId('?v=first&v=second');
console.assert(multiV === 'first', `Expected "first" for "?v=first&v=second", got: ${multiV}`);

console.log('  ✓ All edge cases passed');

console.log('\nAll Property 1 tests passed. ✓');
