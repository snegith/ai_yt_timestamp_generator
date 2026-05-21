# Implementation Plan: YouTube Timestamp Generator

## Overview

Implement a Chrome Extension (MV3, Vanilla JS) and a FastAPI backend that together generate AI-powered chapter timestamps for any YouTube video. The backend runs a multi-stage pipeline (transcript retrieval → preprocessing → topic boundary detection → LLM title generation) and the extension injects a sidebar UI into YouTube watch pages.

## Tasks

- [x] 1. Set up project structure and shared configuration
  - Create `extension/` and `backend/` directory trees as specified in the design
  - Create `backend/requirements.txt` with all dependencies: `fastapi`, `uvicorn`, `pydantic`, `youtube-transcript-api`, `yt-dlp`, `faster-whisper`, `sentence-transformers`, `nltk`, `groq`, `numpy`, `httpx`
  - Create `backend/Dockerfile` and `backend/railway.toml` exactly as specified in the design
  - Create `extension/manifest.json` conforming to MV3 with correct `content_scripts`, `permissions`, and `host_permissions`
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 12.1, 12.2, 12.3_

- [x] 2. Implement backend data models and cache
  - [x] 2.1 Create `backend/models.py` with `GenerateRequest`, `Timestamp`, and `GenerateResponse` Pydantic models including the `video_id` non-empty validator
    - _Requirements: 4.2, 4.3_
  - [x] 2.2 Create `backend/cache.py` with `get(video_id)` and `set(video_id, timestamps)` functions backed by a module-level `_cache` dict
    - _Requirements: 5.1, 5.2, 5.3_
  - [x] 2.3 Write property test for cache idempotence (Property 6)
    - **Property 6: Cache Idempotence**
    - Same `video_id` processed twice returns identical timestamps and pipeline runs only once
    - **Validates: Requirements 5.2, 5.3**

- [x] 3. Implement transcript retrieval pipeline module
  - [x] 3.1 Create `backend/pipeline/__init__.py` (empty) and `backend/pipeline/transcript.py` with `async get_transcript(video_id)` — Path A via `youtube-transcript-api`, fallback to Path B via `yt-dlp` + `faster-whisper`; raise `RuntimeError` if both fail
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 4. Implement transcript preprocessing pipeline module
  - [x] 4.1 Create `backend/pipeline/preprocess.py` with `TextWindow` dataclass, `strip_fillers(text)`, and `preprocess(segments)` functions; windows target ~200 words using `nltk.sent_tokenize`
    - _Requirements: 7.1, 7.2, 7.3_
  - [x] 4.2 Write property test for filler word stripping (Property 7)
    - **Property 7: Filler Word Stripping**
    - Any input string passed through `strip_fillers()` contains no standalone filler tokens and preserves non-filler content
    - **Validates: Requirements 7.2**
  - [x] 4.3 Write property test for windowing completeness and size (Property 8)
    - **Property 8: Windowing Completeness and Size**
    - For any segment list with total word count W, `preprocess()` windows contain all non-filler words and every window except the last has 150–250 words
    - **Validates: Requirements 7.1**

- [x] 5. Implement topic boundary detection pipeline module
  - [x] 5.1 Create `backend/pipeline/boundaries.py` with `cosine_similarity(a, b)` and `detect_boundaries(windows)` using `all-MiniLM-L6-v2` embeddings and a 0.35 threshold; always include `windows[0]`
    - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - [x] 5.2 Write property test for cosine similarity symmetry and bounds (Property 9)
    - **Property 9: Cosine Similarity Symmetry and Bounds**
    - For any two non-zero vectors, `cosine_similarity(a, b) == cosine_similarity(b, a)` and result is in `[-1.0, 1.0]`
    - **Validates: Requirements 8.2**
  - [x] 5.3 Write property test for boundary detection threshold and first-window invariant (Property 10)
    - **Property 10: Topic Boundary Detection Threshold and First-Window Invariant**
    - `detect_boundaries()` includes window at index `i > 0` iff similarity < 0.35, always includes `windows[0]`, and result is non-empty for non-empty input
    - **Validates: Requirements 8.3, 8.4, 10.3**

- [ ] 6. Implement title generation pipeline module
  - [x] 6.1 Create `backend/pipeline/titles.py` with `async generate_titles(boundary_windows)` that sends a single batched prompt to the Groq API, parses the JSON array response, and returns a list of `Timestamp` objects sorted by time ascending
    - _Requirements: 9.1, 9.2, 9.4, 9.5, 10.1, 10.2, 10.3_
  - [-] 6.2 Write property test for single batched Groq call (Property 11)
    - **Property 11: Single Batched Groq Call**
    - For any N ≥ 1 boundary windows, `generate_titles()` makes exactly 1 Groq API call and returns exactly N `Timestamp` objects
    - **Validates: Requirements 9.1**
  - [-] 6.3 Write property test for output chronological order (Property 12)
    - **Property 12: Output Chronological Order**
    - Any successful `/generate` response has `timestamps` sorted in non-decreasing order of time values converted to seconds
    - **Validates: Requirements 10.1**

- [x] 7. Implement FastAPI main application
  - [x] 7.1 Create `backend/main.py` with the FastAPI app, CORS middleware (`allow_origins=['*']`), lifespan startup validation for `GROQ_API_KEY`, and the `POST /generate` endpoint wiring cache lookup → pipeline → cache store → response; map pipeline errors to HTTP 502
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 9.3_
  - [x] 7.2 Write property test for invalid video_id rejection (Property 5)
    - **Property 5: Invalid video_id Rejected with 422**
    - Absent, null, empty, or whitespace-only `video_id` values return HTTP 422
    - **Validates: Requirements 4.3**

- [x] 8. Checkpoint — Backend complete
  - Ensure all backend tests pass, ask the user if questions arise.

- [x] 9. Implement Chrome Extension content script
  - [x] 9.1 Create `extension/content.js` with `extractVideoId()`, `timeStringToSeconds()`, `injectSidebar()`, `removeSidebar()`, `generateTimestamps(videoId)`, `renderTimestamps(timestamps)`, `setLoadingState()`, and `showError()` functions; wire `yt-navigate-finish` and `popstate` navigation listeners
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4_
  - [x] 9.2 Write property test for video ID extraction correctness (Property 1)
    - **Property 1: Video ID Extraction Correctness**
    - Any URL with `?v=<id>` returns exactly `<id>`; any URL without `v` param returns `null`
    - **Validates: Requirements 2.2, 2.5**
  - [x] 9.3 Write property test for time string parsing round-trip (Property 2)
    - **Property 2: Time String Parsing Round-Trip**
    - For any non-negative integer `s`, `timeStringToSeconds(_secondsToTimeStr(s)) === s`
    - **Validates: Requirements 3.2, 10.2**
  - [x] 9.4 Write property test for time string format invariant (Property 3)
    - **Property 3: Time String Format Invariant**
    - For any non-negative integer `s`, `_secondsToTimeStr(s)` matches `^\d+:\d{2}(:\d{2})?$`
    - **Validates: Requirements 10.2**
  - [x] 9.5 Write property test for timestamp list rendering completeness (Property 4)
    - **Property 4: Timestamp List Rendering Completeness**
    - For any non-empty timestamp array, `renderTimestamps()` produces exactly one clickable element per timestamp with correct `time` and `title`
    - **Validates: Requirements 3.1**

- [ ] 10. Create extension sidebar styles
  - Create `extension/sidebar.css` with styles for `#yt-ts-sidebar`, `#yt-ts-header`, `#yt-ts-body`, the generate button, loading indicator, timestamp list items, and error state; ensure the panel does not obscure video player controls
  - _Requirements: 1.2_

- [~] 11. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties defined in the design
- Unit tests validate specific examples and edge cases
- The backend must have `GROQ_API_KEY` set as an environment variable at runtime; it will refuse to start without it
- The extension's `manifest.json` `host_permissions` must include the deployed Railway backend URL before submission to the Chrome Web Store

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "2.2"] },
    { "id": 1, "tasks": ["2.3", "3.1"] },
    { "id": 2, "tasks": ["4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "6.1"] },
    { "id": 5, "tasks": ["6.2", "6.3", "7.1"] },
    { "id": 6, "tasks": ["7.2", "9.1"] },
    { "id": 7, "tasks": ["9.2", "9.3", "9.4", "9.5"] }
  ]
}
```
