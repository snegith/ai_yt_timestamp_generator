/**
 * Unit tests for formatApiError (mirrors extension/content.js).
 */

'use strict';

function formatApiError(detail, status) {
  if (typeof detail === 'string' && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (item && typeof item.msg === 'string' ? item.msg : null))
      .filter(Boolean);
    if (messages.length > 0) {
      return messages.join(' ');
    }
  }
  if (detail && typeof detail === 'object') {
    if (typeof detail.msg === 'string' && detail.msg.trim()) {
      return detail.msg;
    }
    if (typeof detail.message === 'string' && detail.message.trim()) {
      return detail.message;
    }
  }
  return `Error ${status}`;
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}

assertEqual(
  formatApiError('Transcript retrieval failed', 502),
  'Transcript retrieval failed',
  'string detail'
);

assertEqual(
  formatApiError(
    [{ type: 'value_error', loc: ['body', 'video_id'], msg: 'Value error, video_id must be a non-empty string' }],
    422
  ),
  'Value error, video_id must be a non-empty string',
  'FastAPI validation array'
);

assertEqual(formatApiError(undefined, 500), 'Error 500', 'fallback status');

assertEqual(
  formatApiError({ msg: 'Invalid API key' }, 401),
  'Invalid API key',
  'object detail with msg'
);

assertEqual(
  formatApiError({ message: 'Service unavailable' }, 503),
  'Service unavailable',
  'object detail with message'
);

console.log('formatApiError tests passed.');
