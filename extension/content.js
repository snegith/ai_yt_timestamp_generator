/**
 * YouTube Timestamp Generator — Content Script
 *
 * Injected into youtube.com/watch pages. Provides a floating sidebar panel
 * that lets users generate AI-powered chapter timestamps for the current video.
 */

'use strict';

const BACKEND_URL = 'http://127.0.0.1:8000'; // TODO: replace with your Railway URL before deploying (e.g. 'https://your-app.up.railway.app')
/** Set to match backend API_KEY when deployed; leave empty for local dev without auth. */
const API_KEY = '';

/** @type {string|null} Last video ID seen on a /watch page (for SPA navigation resets). */
let _lastVideoId = null;

/** @type {AbortController|null} In-flight /generate request, aborted on navigation or re-generate. */
let _generateAbortController = null;

/** Max wait for backend pipeline (transcript + Whisper + Groq). */
const FETCH_TIMEOUT_MS = 5 * 60 * 1000;

// ---------------------------------------------------------------------------
// URL / time utilities
// ---------------------------------------------------------------------------

/**
 * Extract the `v` query parameter from the current page URL.
 * @returns {string|null} The video ID, or null if not present.
 */
function extractVideoId() {
  const params = new URLSearchParams(window.location.search);
  return params.get('v');
}

/**
 * Convert a time string in "M:SS" or "H:MM:SS" format to integer seconds.
 * @param {string} timeStr
 * @returns {number}
 */
function timeStringToSeconds(timeStr) {
  const parts = timeStr.split(':').map(Number);
  if (parts.some((p) => Number.isNaN(p))) {
    return null;
  }
  if (parts.length === 2) {
    // M:SS
    return parts[0] * 60 + parts[1];
  } else if (parts.length === 3) {
    // H:MM:SS
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return null;
}

/**
 * Convert an integer number of seconds to a "M:SS" or "H:MM:SS" string.
 * Private helper used by property tests.
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
// Sidebar DOM management
// ---------------------------------------------------------------------------

/**
 * Inject the sidebar panel into the YouTube DOM.
 * Does nothing if the sidebar is already present.
 */
function injectSidebar() {
  if (document.getElementById('yt-ts-sidebar')) {
    return; // already injected
  }

  const sidebar = document.createElement('div');
  sidebar.id = 'yt-ts-sidebar';
  sidebar.innerHTML = `
    <div id="yt-ts-header">
      <span>Timestamps</span>
    </div>
    <div id="yt-ts-body">
      <button id="yt-ts-generate-btn">Generate Timestamps</button>
      <div id="yt-ts-status"></div>
      <ul id="yt-ts-list"></ul>
    </div>
  `;

  document.body.appendChild(sidebar);

  // Wire up the generate button
  document.getElementById('yt-ts-generate-btn').addEventListener('click', () => {
    const videoId = extractVideoId();
    if (!videoId || !videoId.trim()) {
      showError('Could not extract video ID from URL');
      return;
    }
    generateTimestamps(videoId.trim());
  });
}

/**
 * Remove the sidebar panel from the DOM if it exists.
 */
function removeSidebar() {
  const sidebar = document.getElementById('yt-ts-sidebar');
  if (sidebar) {
    sidebar.remove();
  }
}

/**
 * Abort any in-flight timestamp generation request.
 */
function cancelPendingGeneration() {
  if (_generateAbortController) {
    _generateAbortController.abort();
    _generateAbortController = null;
  }
}

/**
 * Clear timestamp results, errors, and loading state (e.g. after video change).
 */
function resetSidebarResults() {
  const list = document.getElementById('yt-ts-list');
  const status = document.getElementById('yt-ts-status');

  if (list) {
    list.innerHTML = '';
  }
  if (status) {
    status.textContent = '';
    status.style.color = '';
  }

  setLoadingState(false);
}

/**
 * Turn a FastAPI error `detail` (string or validation array) into readable text.
 * @param {unknown} detail
 * @param {number} status
 * @returns {string}
 */
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

// ---------------------------------------------------------------------------
// UI state helpers
// ---------------------------------------------------------------------------

/**
 * Show or hide the loading indicator and toggle the generate button.
 * @param {boolean} isLoading
 */
function setLoadingState(isLoading) {
  const btn = document.getElementById('yt-ts-generate-btn');
  const status = document.getElementById('yt-ts-status');

  if (!btn || !status) return;

  if (isLoading) {
    btn.disabled = true;
    status.textContent = 'Generating timestamps…';
  } else {
    btn.disabled = false;
    status.textContent = '';
  }
}

/**
 * Display an error message inside the sidebar.
 * @param {string} message
 */
function showError(message) {
  const status = document.getElementById('yt-ts-status');
  if (status) {
    status.textContent = message;
    status.style.color = '#e53935';
  }
}

// ---------------------------------------------------------------------------
// Timestamp rendering
// ---------------------------------------------------------------------------

/**
 * Render an array of timestamp objects as a clickable list in the sidebar.
 * Clicking a list item seeks the video to the corresponding time.
 *
 * @param {Array<{time: string, title: string}>} timestamps
 */
function renderTimestamps(timestamps) {
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

    btn.addEventListener('click', () => {
      const seconds = timeStringToSeconds(time);
      if (seconds === null) {
        return;
      }
      const video =
        document.querySelector('#movie_player video') ||
        document.querySelector('video.html5-main-video') ||
        document.querySelector('video');
      if (video) {
        video.currentTime = seconds;
      }
    });

    li.appendChild(btn);
    list.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// Backend communication
// ---------------------------------------------------------------------------

/**
 * POST to the backend /generate endpoint and handle all response states.
 * @param {string} videoId
 * @returns {Promise<void>}
 */
async function generateTimestamps(videoId) {
  cancelPendingGeneration();

  const controller = new AbortController();
  _generateAbortController = controller;
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, FETCH_TIMEOUT_MS);

  setLoadingState(true);
  try {
    /** @type {Record<string, string>} */
    const headers = { 'Content-Type': 'application/json' };
    if (API_KEY) {
      headers['X-API-Key'] = API_KEY;
    }

    const res = await fetch(`${BACKEND_URL}/generate`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ video_id: videoId, force_retry: true }),
      signal: controller.signal,
    });

    if (extractVideoId() !== videoId) {
      return;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(formatApiError(err.detail, res.status));
      return;
    }

    const data = await res.json();
    if (extractVideoId() !== videoId) {
      return;
    }

    if (!Array.isArray(data.timestamps)) {
      showError('Received an invalid response from the server.');
      return;
    }

    renderTimestamps(data.timestamps);
  } catch (e) {
    if (e.name === 'AbortError') {
      if (timedOut && extractVideoId() === videoId) {
        showError('Request timed out. The video may be too long — please try again.');
      }
      return;
    }
    if (extractVideoId() !== videoId) {
      return;
    }
    if (e instanceof SyntaxError || e instanceof TypeError) {
      showError('Received an invalid response from the server.');
      return;
    }
    showError('Could not reach the server. Please try again.');
  } finally {
    clearTimeout(timeoutId);
    if (_generateAbortController === controller) {
      _generateAbortController = null;
      // Always clear loading for this request; a newer generate call sets it again.
      setLoadingState(false);
    }
  }
}

// ---------------------------------------------------------------------------
// Navigation listeners (YouTube is a SPA)
// ---------------------------------------------------------------------------

function onWatchPageReady() {
  injectSidebar();

  // Defer URL read — yt-navigate-finish can fire before location.search updates.
  requestAnimationFrame(() => {
    const videoId = extractVideoId();
    if (videoId !== _lastVideoId) {
      cancelPendingGeneration();
      resetSidebarResults();
      _lastVideoId = videoId;
    }
  });
}

function onNavigate() {
  if (window.location.pathname === '/watch') {
    onWatchPageReady();
  } else {
    cancelPendingGeneration();
    _lastVideoId = null;
    removeSidebar();
  }
}

document.addEventListener('yt-navigate-finish', onNavigate);
window.addEventListener('popstate', onNavigate);

// ---------------------------------------------------------------------------
// Initial injection on script load
// ---------------------------------------------------------------------------

if (window.location.pathname === '/watch') {
  onWatchPageReady();
}
