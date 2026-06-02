/**
 * YouTube Timestamp Generator — Content Script
 *
 * Injected into youtube.com/watch pages. Provides a floating sidebar panel
 * that lets users generate AI-powered chapter timestamps for the current video.
 */

'use strict';

const BACKEND_URL = 'http://127.0.0.1:8000'; // TODO: replace with your Railway URL before deploying (e.g. 'https://your-app.up.railway.app')

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
  if (parts.length === 2) {
    // M:SS
    return parts[0] * 60 + parts[1];
  } else if (parts.length === 3) {
    // H:MM:SS
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return 0;
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
    if (!videoId) {
      showError('Could not extract video ID from URL');
      return;
    }
    generateTimestamps(videoId);
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
      const video = document.querySelector('video');
      if (video) {
        video.currentTime = timeStringToSeconds(time);
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
  setLoadingState(true);
  try {
    const res = await fetch(`${BACKEND_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showError(err.detail || `Error ${res.status}`);
      return;
    }

    const data = await res.json();
    renderTimestamps(data.timestamps);
  } catch (e) {
    showError('Could not reach the server. Please try again.');
  } finally {
    setLoadingState(false);
  }
}

// ---------------------------------------------------------------------------
// Navigation listeners (YouTube is a SPA)
// ---------------------------------------------------------------------------

function onNavigate() {
  if (window.location.pathname === '/watch') {
    injectSidebar();
  } else {
    removeSidebar();
  }
}

document.addEventListener('yt-navigate-finish', onNavigate);
window.addEventListener('popstate', onNavigate);

// ---------------------------------------------------------------------------
// Initial injection on script load
// ---------------------------------------------------------------------------

if (window.location.pathname === '/watch') {
  injectSidebar();
}
