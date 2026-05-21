# Design Document

## Overview

This document describes the technical architecture for the YouTube Timestamp Generator — a Chrome Extension (Manifest V3, Vanilla JS) paired with a FastAPI backend. The extension injects a floating sidebar into YouTube watch pages; the backend runs a multi-stage pipeline (transcript retrieval → preprocessing → topic boundary detection → LLM title generation) and returns structured, clickable timestamps.

The system has two independently deployable components:
- **Chrome Extension** — content script injected into `youtube.com/watch` pages, providing the UI and communicating with the backend.
- **FastAPI Backend** — deployed on Railway via Docker, exposing a single `POST /generate` endpoint that runs the full pipeline and caches results in memory.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Chrome Browser                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              YouTube Watch Page DOM                   │   │
│  │                                                       │   │
│  │   ┌─────────────────────────────────────────────┐    │   │
│  │   │           Content Script (JS)               │    │   │
│  │   │  - Injects / removes Sidebar Panel          │    │   │
│  │   │  - Extracts video_id from URL               │    │   │
│  │   │  - Sends POST /generate                     │    │   │
│  │   │  - Renders timestamp list                   │    │   │
│  │   │  - Seeks video via video.currentTime        │    │   │
│  │   └──────────────────┬──────────────────────────┘    │   │
│  │                       │  fetch POST /generate         │   │
│  └───────────────────────┼───────────────────────────────┘   │
└──────────────────────────┼──────────────────────────────────┘
                           │  HTTP (JSON)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Railway)                  │
│                                                              │
│  POST /generate                                              │
│       │                                                      │
│       ├─► Cache hit? ──► return cached timestamps            │
│       │                                                      │
│       └─► Pipeline:                                          │
│             1. Transcript Retrieval                          │
│                  Path A: youtube-transcript-api              │
│                  Path B: yt-dlp + faster-whisper             │
│             2. Preprocessing                                 │
│                  filler stripping, sent_tokenize,            │
│                  200-word windowing                          │
│             3. Topic Boundary Detection                      │
│                  all-MiniLM-L6-v2 embeddings                 │
│                  cosine similarity < 0.35 threshold          │
│             4. Title Generation                              │
│                  Groq API (single batched prompt)            │
│             5. Format & Cache result                         │
│                                                              │
│  In-Memory Cache: { video_id → [Timestamp] }                 │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
video_id (string)
    │
    ▼
Cache lookup ──hit──► [Timestamp] ──► HTTP 200
    │
   miss
    │
    ▼
Transcript Retrieval
    Path A: YouTubeTranscriptApi.get_transcript(video_id)
            → [{"text": str, "start": float, "duration": float}]
    Path B: yt-dlp download → faster-whisper transcribe
            → [{"text": str, "start": float, "duration": float}]
    │
    ▼
Preprocessing
    strip_fillers(text) → cleaned text
    nltk.sent_tokenize(cleaned) → sentences
    group into ~200-word TextWindow(text, start_time)
    → [TextWindow]
    │
    ▼
Boundary Detection
    SentenceTransformer.encode([window.text]) → embeddings
    cosine_similarity(embed[i-1], embed[i]) for each pair
    mark boundary where similarity < 0.35
    always include windows[0]
    → [TextWindow]  (boundary windows only)
    │
    ▼
Title Generation
    single batched prompt → Groq API
    parse JSON array response → [str]
    zip with boundary_windows → [Timestamp(time, title)]
    │
    ▼
Sort by time ascending
    │
    ▼
cache.set(video_id, timestamps)
    │
    ▼
HTTP 200 { "timestamps": [...] }
```

---

## Components and Interfaces

### Chrome Extension

#### File Structure

```
extension/
├── manifest.json          # MV3 manifest
├── content.js             # Content script — injected into youtube.com/watch
├── sidebar.css            # Sidebar panel styles
└── icons/
    ├── icon16.png
    ├── icon48.png
    └── icon128.png
```

#### `manifest.json`

```json
{
  "manifest_version": 3,
  "name": "YouTube Timestamp Generator",
  "version": "1.0.0",
  "description": "AI-powered chapter timestamps for any YouTube video.",
  "permissions": ["activeTab"],
  "host_permissions": [
    "https://www.youtube.com/*",
    "https://<railway-backend-domain>/*"
  ],
  "content_scripts": [
    {
      "matches": ["https://www.youtube.com/watch*"],
      "js": ["content.js"],
      "css": ["sidebar.css"],
      "run_at": "document_idle"
    }
  ],
  "icons": {
    "16": "icons/icon16.png",
    "48": "icons/icon48.png",
    "128": "icons/icon128.png"
  }
}
```

#### Content Script Interface (`content.js`)

Key functions exposed internally:

```javascript
// Extract video_id from current URL — returns string or null
function extractVideoId(): string | null

// Convert "M:SS" or "H:MM:SS" time string to integer seconds
function timeStringToSeconds(timeStr: string): number

// Inject sidebar panel into YouTube DOM
function injectSidebar(): void

// Remove sidebar panel from YouTube DOM
function removeSidebar(): void

// Send POST /generate and handle all response states
async function generateTimestamps(videoId: string): Promise<void>

// Render timestamp array into the sidebar list
function renderTimestamps(timestamps: Timestamp[]): void
```

**Navigation handling** — YouTube is a SPA; the content script listens for both `yt-navigate-finish` (YouTube's custom event) and `popstate`:

```javascript
document.addEventListener('yt-navigate-finish', onNavigate);
window.addEventListener('popstate', onNavigate);

function onNavigate() {
  if (window.location.pathname === '/watch') {
    injectSidebar();
  } else {
    removeSidebar();
  }
}
```

**Sidebar Panel HTML structure:**

```html
<div id="yt-ts-sidebar">
  <div id="yt-ts-header">
    <span>Timestamps</span>
  </div>
  <div id="yt-ts-body">
    <!-- States: idle | loading | results | error -->
    <button id="yt-ts-generate-btn">Generate Timestamps</button>
    <div id="yt-ts-status"></div>
    <ul id="yt-ts-list"></ul>
  </div>
</div>
```

**Request / response flow:**

```javascript
async function generateTimestamps(videoId) {
  setLoadingState(true);
  try {
    const res = await fetch('https://<backend>/generate', {
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
```

---

### FastAPI Backend

#### File Structure

```
backend/
├── main.py                # FastAPI app, /generate endpoint, CORS, startup validation
├── pipeline/
│   ├── __init__.py
│   ├── transcript.py      # Path A + Path B transcript retrieval
│   ├── preprocess.py      # Filler stripping, sent_tokenize, windowing
│   ├── boundaries.py      # Embedding, cosine similarity, boundary detection
│   └── titles.py          # Groq API title generation
├── cache.py               # In-memory cache
├── models.py              # Pydantic request/response models
├── requirements.txt
├── Dockerfile
└── railway.toml
```

#### API Interface

**Endpoint:** `POST /generate`

Request body:
```json
{ "video_id": "dQw4w9WgXcQ" }
```

Success response (HTTP 200):
```json
{
  "timestamps": [
    { "time": "0:00", "title": "Introduction" },
    { "time": "2:34", "title": "Main Topic" },
    { "time": "8:12", "title": "Conclusion" }
  ]
}
```

Error responses:
- `422` — missing or invalid `video_id`
- `502` — transcript retrieval, boundary detection, or title generation failure

#### Pipeline Module Interfaces

**`pipeline/transcript.py`**
```python
async def get_transcript(video_id: str) -> list[dict]:
    """Returns list of {"text": str, "start": float, "duration": float}"""
```

**`pipeline/preprocess.py`**
```python
@dataclass
class TextWindow:
    text: str
    start_time: float  # seconds

def strip_fillers(text: str) -> str: ...
def preprocess(segments: list[dict]) -> list[TextWindow]: ...
```

**`pipeline/boundaries.py`**
```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float: ...
def detect_boundaries(windows: list[TextWindow]) -> list[TextWindow]: ...
```

**`pipeline/titles.py`**
```python
async def generate_titles(boundary_windows: list[TextWindow]) -> list[Timestamp]: ...
```

**`cache.py`**
```python
def get(video_id: str) -> list[Timestamp] | None: ...
def set(video_id: str, timestamps: list[Timestamp]) -> None: ...
```

#### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],          # Chrome extensions send Origin: chrome-extension://...
    allow_methods=['POST', 'OPTIONS'],
    allow_headers=['Content-Type'],
)
```

#### Startup Validation

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get('GROQ_API_KEY'):
        logger.error('GROQ_API_KEY environment variable is not set. Refusing to start.')
        raise RuntimeError('GROQ_API_KEY is required')
    yield
```

#### Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### `railway.toml`

```toml
[build]
builder = "dockerfile"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port 8000"
healthcheckPath = "/docs"
healthcheckTimeout = 30
restartPolicyType = "on_failure"
```

---

## Data Models

### Extension (JavaScript)

```javascript
// Timestamp object returned by the backend
/** @typedef {{ time: string, title: string }} Timestamp */

// Sidebar UI state
/** @typedef {'idle' | 'loading' | 'results' | 'error'} SidebarState */
```

### Backend (Python / Pydantic)

```python
from pydantic import BaseModel, field_validator

class GenerateRequest(BaseModel):
    video_id: str

    @field_validator('video_id')
    @classmethod
    def video_id_must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('video_id must be a non-empty string')
        return v.strip()

class Timestamp(BaseModel):
    time: str   # "M:SS" or "H:MM:SS"
    title: str

class GenerateResponse(BaseModel):
    timestamps: list[Timestamp]
```

### Internal Pipeline Types

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class TextWindow:
    text: str
    start_time: float  # seconds from video start

# Transcript segment (from youtube-transcript-api or faster-whisper)
TranscriptSegment = dict  # {"text": str, "start": float, "duration": float}

# Embedding vector
Embedding = np.ndarray  # shape (384,) for all-MiniLM-L6-v2
```

### In-Memory Cache Schema

```python
# Simple dict — no TTL, lives for the lifetime of the process
_cache: dict[str, list[Timestamp]] = {}
```

---

## Error Handling

| Failure Point | Behavior |
|---|---|
| `video_id` missing or empty | FastAPI returns HTTP 422 (Pydantic validation) |
| Both transcript paths fail | HTTP 502 `"Transcript retrieval failed: ..."` |
| Boundary detection error | HTTP 502 `"Boundary detection failed: ..."` |
| Groq API error | HTTP 502 `"Title generation failed: ..."` |
| `GROQ_API_KEY` not set | Server refuses to start, logs error |
| Extension: `video_id` not in URL | Error shown in sidebar, no request sent |
| Extension: non-2xx response | Error message shown, button re-enabled |
| Extension: network failure | Error message shown, button re-enabled |

---

## Testing Strategy

### Unit Tests (Example-Based)

- `extractVideoId()` with valid and invalid YouTube URLs
- `timeStringToSeconds()` with `M:SS` and `H:MM:SS` inputs
- Sidebar DOM injection and removal on navigation events
- Loading state toggling during request lifecycle
- Error display on non-2xx and network failure responses
- `strip_fillers()` with known filler-containing strings
- Groq API error → HTTP 502 propagation
- Missing `GROQ_API_KEY` → server startup failure

### Property-Based Tests

Property tests use randomized inputs to verify universal invariants. Each test runs a minimum of 100 iterations.

- **Video ID extraction** — random URL strings with/without `v` param
- **Time string round-trip** — random non-negative integers → format → parse → compare
- **Time string format** — random seconds → verify regex match
- **Timestamp rendering** — random timestamp arrays → verify list length and content
- **Invalid video_id rejection** — random invalid inputs → verify 422
- **Cache idempotence** — same video_id twice → identical result, pipeline runs once
- **Filler stripping** — random text with injected fillers → verify removal
- **Windowing completeness** — random segment lists → verify word preservation and window sizes
- **Cosine similarity bounds** — random vectors → verify symmetry and `[-1, 1]` range
- **Boundary detection threshold** — controlled similarity sequences → verify boundary positions
- **Single Groq call** — N boundary windows → verify exactly 1 API call, N titles returned
- **Chronological ordering** — any pipeline output → verify ascending time order

### Integration Tests (Example-Based)

- End-to-end: POST `/generate` with a real (or mocked) video_id returns valid response shape
- CORS headers present on OPTIONS and POST responses
- Docker container starts, port 8000 responds to requests

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Video ID Extraction Correctness

*For any* URL string of the form `https://www.youtube.com/watch?v=<id>[&...]`, `extractVideoId()` SHALL return exactly the value of the `v` query parameter; and for any URL that does not contain a `v` parameter, it SHALL return `null`.

**Validates: Requirements 2.2, 2.5**

---

### Property 2: Time String Parsing Round-Trip

*For any* non-negative integer number of seconds `s`, formatting `s` with `_seconds_to_time_str(s)` and then parsing the result with `timeStringToSeconds()` SHALL return `s`.

**Validates: Requirements 3.2, 10.2**

---

### Property 3: Time String Format Invariant

*For any* non-negative integer number of seconds `s`, `_seconds_to_time_str(s)` SHALL return a string matching the regex `^\d+:\d{2}(:\d{2})?$` (i.e., `M:SS` or `H:MM:SS` format).

**Validates: Requirements 10.2**

---

### Property 4: Timestamp List Rendering Completeness

*For any* non-empty array of timestamp objects `[{ time, title }]`, `renderTimestamps()` SHALL produce a list containing exactly one clickable element per timestamp, each displaying the correct `time` and `title` strings.

**Validates: Requirements 3.1**

---

### Property 5: Invalid video_id Rejected with 422

*For any* request body where `video_id` is absent, `null`, an empty string, or a whitespace-only string, the `/generate` endpoint SHALL return HTTP 422.

**Validates: Requirements 4.3**

---

### Property 6: Cache Idempotence

*For any* `video_id` that has been successfully processed once, a second call to `/generate` with the same `video_id` SHALL return an identical `timestamps` array and the pipeline SHALL NOT be re-executed.

**Validates: Requirements 5.2, 5.3**

---

### Property 7: Filler Word Stripping

*For any* input string, `strip_fillers(text)` SHALL return a string containing no standalone occurrences of any filler word token (e.g., `"um"`, `"uh"`, `"like"`, `"you know"`), and SHALL preserve all non-filler content.

**Validates: Requirements 7.2**

---

### Property 8: Windowing Completeness and Size

*For any* list of transcript segments whose total word count is `W`, `preprocess(segments)` SHALL return windows such that (a) the concatenation of all window texts contains all non-filler words from the input, and (b) every window except possibly the last contains at least 150 and at most 250 words.

**Validates: Requirements 7.1**

---

### Property 9: Cosine Similarity Symmetry and Bounds

*For any* two non-zero embedding vectors `a` and `b`, `cosine_similarity(a, b)` SHALL equal `cosine_similarity(b, a)` and SHALL return a value in the closed interval `[-1.0, 1.0]`.

**Validates: Requirements 8.2**

---

### Property 10: Topic Boundary Detection Threshold and First-Window Invariant

*For any* sequence of text windows, `detect_boundaries(windows)` SHALL include a window at index `i > 0` in the result if and only if `cosine_similarity(embed[i-1], embed[i]) < 0.35`, SHALL always include `windows[0]`, and the result SHALL be non-empty for any non-empty input.

**Validates: Requirements 8.3, 8.4, 10.3**

---

### Property 11: Single Batched Groq Call

*For any* list of `N ≥ 1` boundary windows, `generate_titles(boundary_windows)` SHALL make exactly one call to the Groq API and SHALL return exactly `N` `Timestamp` objects.

**Validates: Requirements 9.1**

---

### Property 12: Output Chronological Order

*For any* successful `/generate` response, the `timestamps` array SHALL be sorted in strictly non-decreasing order of their time values (converted to seconds).

**Validates: Requirements 10.1**
