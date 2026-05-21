# Requirements Document

## Introduction

This feature delivers an AI-powered YouTube timestamp generator consisting of two components: a Chrome Extension (Manifest V3, Vanilla JS) and a FastAPI backend. The extension injects a floating sidebar panel into YouTube watch pages, allowing users to trigger timestamp generation on demand. The backend processes the video transcript through a multi-stage pipeline — transcript retrieval, text preprocessing, topic boundary detection, and LLM-based title generation — returning a structured list of timestamped chapter titles. Users can click any timestamp to seek the video to that position.

## Glossary

- **Extension**: The Chrome Extension (Manifest V3, Vanilla JS) installed in the user's browser.
- **Content Script**: JavaScript injected by the Extension into YouTube watch pages.
- **Sidebar Panel**: The floating overlay UI element injected into the YouTube DOM by the Content Script.
- **Backend**: The FastAPI + Uvicorn server that processes timestamp generation requests.
- **Pipeline**: The sequential processing stages executed by the Backend to produce timestamps.
- **Transcript**: The text representation of a YouTube video's spoken content, with timing information.
- **Timestamp**: A data object containing a time string (e.g., "0:00") and a chapter title string.
- **Topic Boundary**: A detected point in the transcript where the subject matter changes significantly.
- **Cache**: An in-memory store on the Backend keyed by video_id to avoid redundant processing.
- **GROQ_API_KEY**: The environment variable holding the API key for the Groq API.
- **video_id**: The unique YouTube video identifier extracted from the watch page URL.

---

## Requirements

### Requirement 1: Extension — Content Script Injection

**User Story:** As a YouTube viewer, I want a timestamp panel to appear on YouTube watch pages, so that I can access timestamp generation without leaving the page.

#### Acceptance Criteria

1. WHEN the user navigates to a YouTube watch page (URL matching `youtube.com/watch`), THE Content Script SHALL inject the Sidebar Panel into the YouTube page DOM.
2. THE Sidebar Panel SHALL be rendered as a floating overlay element that does not obscure the video player controls.
3. WHEN the user navigates away from a YouTube watch page, THE Content Script SHALL remove the Sidebar Panel from the DOM.

---

### Requirement 2: Extension — Generate Timestamps Button

**User Story:** As a YouTube viewer, I want to click a button to generate timestamps, so that generation is triggered only when I choose.

#### Acceptance Criteria

1. THE Sidebar Panel SHALL contain a "Generate Timestamps" button visible to the user.
2. WHEN the user clicks the "Generate Timestamps" button, THE Content Script SHALL extract the `video_id` from the current page URL query parameter `v`.
3. WHEN the user clicks the "Generate Timestamps" button, THE Content Script SHALL send a POST request to the Backend `/generate` endpoint with the body `{ "video_id": "<extracted_video_id>" }`.
4. WHILE a generation request is in progress, THE Sidebar Panel SHALL display a loading indicator and disable the "Generate Timestamps" button.
5. IF the `video_id` cannot be extracted from the URL, THEN THE Content Script SHALL display an error message in the Sidebar Panel without sending a request to the Backend.

---

### Requirement 3: Extension — Timestamp Display and Video Seeking

**User Story:** As a YouTube viewer, I want to see a clickable list of timestamps, so that I can jump to specific sections of the video.

#### Acceptance Criteria

1. WHEN the Backend returns a successful response, THE Sidebar Panel SHALL render the timestamps as a scrollable, clickable list with each item showing the time string and chapter title.
2. WHEN the user clicks a timestamp list item, THE Content Script SHALL set `document.querySelector('video').currentTime` to the numeric seconds value corresponding to the clicked timestamp's time string.
3. IF the Backend returns an error response (non-2xx HTTP status), THEN THE Sidebar Panel SHALL display a human-readable error message and re-enable the "Generate Timestamps" button.
4. IF the network request to the Backend fails (e.g., connection refused, timeout), THEN THE Sidebar Panel SHALL display a human-readable error message and re-enable the "Generate Timestamps" button.

---

### Requirement 4: Backend — POST /generate Endpoint

**User Story:** As the Extension, I want a single endpoint to submit a video ID and receive timestamps, so that the backend processing is encapsulated behind a simple API contract.

#### Acceptance Criteria

1. THE Backend SHALL expose a POST endpoint at the path `/generate`.
2. WHEN the `/generate` endpoint receives a request with a JSON body containing a `video_id` string, THE Backend SHALL return a JSON response with the shape `{ "timestamps": [{ "time": "<HH:MM or M:SS>", "title": "<string>" }] }`.
3. IF the request body is missing the `video_id` field or the field is not a non-empty string, THEN THE Backend SHALL return an HTTP 422 response with a descriptive error message.
4. THE Backend SHALL include CORS headers permitting requests from Chrome Extension origins so that the Extension can call the API from the browser.

---

### Requirement 5: Backend — In-Memory Cache

**User Story:** As a user, I want repeated requests for the same video to return quickly, so that I do not wait for redundant processing.

#### Acceptance Criteria

1. THE Backend SHALL maintain an in-memory cache keyed by `video_id`.
2. WHEN the `/generate` endpoint receives a request for a `video_id` that exists in the Cache, THE Backend SHALL return the cached timestamps without re-executing the Pipeline.
3. WHEN the `/generate` endpoint receives a request for a `video_id` that does not exist in the Cache, THE Backend SHALL execute the full Pipeline and store the result in the Cache before returning the response.

---

### Requirement 6: Backend — Transcript Retrieval (Path A and Path B)

**User Story:** As the Backend, I want to retrieve a transcript for any YouTube video, so that the Pipeline has text input to process.

#### Acceptance Criteria

1. WHEN the Pipeline begins transcript retrieval, THE Backend SHALL first attempt to fetch the transcript using `youtube-transcript-api` (Path A).
2. IF Path A fails for any reason (e.g., no captions available, API error), THEN THE Backend SHALL fall back to downloading audio via `yt-dlp` and transcribing it using `faster-whisper` (Path B).
3. IF both Path A and Path B fail, THEN THE Backend SHALL return an HTTP 502 response with an error message indicating transcript retrieval failure.
4. WHEN Path B is used, THE Backend SHALL use the `faster-whisper` transcription output including word-level or segment-level timing information to preserve timestamp accuracy.

---

### Requirement 7: Backend — Transcript Preprocessing

**User Story:** As the Backend, I want to preprocess the raw transcript into structured windows, so that topic boundary detection operates on coherent text segments.

#### Acceptance Criteria

1. WHEN the transcript is retrieved, THE Backend SHALL merge transcript segments into windows of approximately 200 words each.
2. WHEN preprocessing transcript text, THE Backend SHALL strip filler words (e.g., "um", "uh", "like", "you know") from the transcript text.
3. WHEN splitting transcript text into sentences, THE Backend SHALL use `nltk.sent_tokenize` to perform sentence boundary detection.

---

### Requirement 8: Backend — Topic Boundary Detection

**User Story:** As the Backend, I want to detect where topics change in the transcript, so that timestamps correspond to meaningful chapter boundaries.

#### Acceptance Criteria

1. WHEN preprocessing is complete, THE Backend SHALL encode each text window into a sentence embedding using the `sentence-transformers` model `all-MiniLM-L6-v2`.
2. WHEN comparing adjacent text window embeddings, THE Backend SHALL compute cosine similarity between consecutive window embeddings.
3. WHEN the cosine similarity between two consecutive windows falls below the threshold of 0.35, THE Backend SHALL mark that boundary as a Topic Boundary.
4. THE Backend SHALL produce at least one Timestamp at time "0:00" regardless of topic boundary detection results.

---

### Requirement 9: Backend — Title Generation via Groq API

**User Story:** As the Backend, I want to generate descriptive chapter titles for each detected topic boundary, so that timestamps are human-readable and informative.

#### Acceptance Criteria

1. WHEN topic boundaries are detected, THE Backend SHALL send a single batched prompt to the Groq API containing all topic boundary text windows and requesting one concise title per window.
2. THE Backend SHALL authenticate with the Groq API using the value of the `GROQ_API_KEY` environment variable.
3. IF the `GROQ_API_KEY` environment variable is not set at startup, THEN THE Backend SHALL log an error message and refuse to start.
4. IF the Groq API returns an error, THEN THE Backend SHALL return an HTTP 502 response with an error message indicating title generation failure.
5. THE Backend SHALL NOT use any LLM provider other than Groq for title generation.

---

### Requirement 10: Backend — Output Format

**User Story:** As the Extension, I want timestamps in a consistent, predictable format, so that the UI can render them without additional parsing logic.

#### Acceptance Criteria

1. THE Backend SHALL return timestamps sorted in ascending chronological order.
2. THE Backend SHALL format each timestamp's time value as a string in `M:SS` or `H:MM:SS` format (e.g., "0:00", "1:23", "1:02:45").
3. THE Backend SHALL return at least one timestamp in every successful response.

---

### Requirement 11: Infrastructure — Docker and Deployment

**User Story:** As a developer, I want the backend containerized and deployable to Railway, so that the service is reproducible and easy to host.

#### Acceptance Criteria

1. THE Backend SHALL include a `Dockerfile` that builds a self-contained image capable of running the FastAPI application with Uvicorn.
2. THE Backend `Dockerfile` SHALL expose the application on port 8000.
3. WHERE the `GROQ_API_KEY` environment variable is provided at container runtime, THE Backend SHALL use it to authenticate with the Groq API without requiring the key to be baked into the image.
4. THE Backend SHALL include a `railway.toml` or equivalent Railway configuration file specifying the build and start commands.

---

### Requirement 12: Infrastructure — Chrome Extension Packaging

**User Story:** As a developer, I want the extension packaged for Chrome Web Store submission, so that users can install it from the store.

#### Acceptance Criteria

1. THE Extension SHALL include a `manifest.json` conforming to Manifest V3 specification with `content_scripts`, `permissions`, and `host_permissions` fields correctly configured for YouTube watch pages.
2. THE Extension `manifest.json` SHALL declare `host_permissions` for the Backend API origin so that the Content Script can make cross-origin requests.
3. THE Extension SHALL include all required Chrome Web Store assets (icons at 16×16, 48×48, and 128×128 pixels) referenced in `manifest.json`.
