"""
main.py — FastAPI application entry point.

Exposes:
  POST /generate  — run the timestamp pipeline (or return cached result)

CORS allows YouTube origins (content scripts run in the page context) and
localhost for development.  Set API_KEY in production to require X-API-Key.

Startup validation ensures GROQ_API_KEY is present before the server
accepts any traffic.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import cache
from models import GenerateRequest, GenerateResponse
from pipeline.transcript import get_transcript
from pipeline.preprocess import preprocess
from pipeline.boundaries import detect_boundaries
from pipeline.titles import generate_titles

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup validation (Requirement 9.3)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate required environment variables before accepting traffic."""
    if not os.environ.get("GROQ_API_KEY"):
        logger.error(
            "GROQ_API_KEY environment variable is not set. Refusing to start."
        )
        raise RuntimeError("GROQ_API_KEY is required")
    logger.info("GROQ_API_KEY found — server starting.")
    api_key = os.environ.get("API_KEY")
    if api_key:
        logger.info("API_KEY is set — /generate requires X-API-Key header.")
    else:
        logger.warning(
            "API_KEY is not set — /generate is open to any caller. "
            "Set API_KEY in production."
        )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="YouTube Timestamp Generator", lifespan=lifespan)

# CORS — content scripts fetch from the YouTube page origin, not chrome-extension://.
_cors_origins = [
    "https://www.youtube.com",
    "https://youtube.com",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
_extra_origins = os.environ.get("ALLOWED_ORIGINS", "")
if _extra_origins:
    _cors_origins.extend(o.strip() for o in _extra_origins.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# One lock per video_id so concurrent cache misses run the pipeline only once.
_pipeline_locks: dict[str, asyncio.Lock] = {}


def _pipeline_lock(video_id: str) -> asyncio.Lock:
    if video_id not in _pipeline_locks:
        _pipeline_locks[video_id] = asyncio.Lock()
    return _pipeline_locks[video_id]


def _release_pipeline_lock(video_id: str) -> None:
    """Drop the per-video lock once no request holds it."""
    lock = _pipeline_locks.get(video_id)
    if lock is not None and not lock.locked():
        _pipeline_locks.pop(video_id, None)


def _verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require X-API-Key when API_KEY env var is configured."""
    expected = os.environ.get("API_KEY")
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def _run_pipeline(video_id: str):
    segments = await get_transcript(video_id)
    if not segments:
        raise RuntimeError("Transcript retrieval failed: no transcript segments were found.")

    windows = preprocess(segments)
    if not windows:
        raise RuntimeError("Transcript preprocessing failed: no usable transcript text was found.")

    loop = asyncio.get_running_loop()
    boundary_windows = await loop.run_in_executor(None, detect_boundaries, windows)
    if not boundary_windows:
        raise RuntimeError("Boundary detection failed: no timestamp boundaries were found.")

    timestamps = await generate_titles(boundary_windows)
    if not timestamps:
        raise RuntimeError("Title generation failed: no timestamps were generated.")

    return timestamps


# ---------------------------------------------------------------------------
# POST /generate  (Requirements 4.1, 4.2, 4.3, 5.1, 5.2, 5.3)
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    _: None = Depends(_verify_api_key),
) -> GenerateResponse:
    """Generate (or return cached) chapter timestamps for a YouTube video.

    Flow
    ----
    1. Cache lookup — return immediately on hit.
    2. Pipeline: transcript → preprocess → detect_boundaries → generate_titles
    3. Store result in cache.
    4. Return GenerateResponse.

    Any RuntimeError raised by the pipeline is mapped to HTTP 502.
    """
    video_id = request.video_id
    force_retry = request.force_retry

    if force_retry:
        cache.clear_failure(video_id)

    # --- Step 1: Cache lookup (Requirement 5.2) ---
    cached = cache.get(video_id)
    if cached is not None:
        logger.info("Cache hit for video_id=%s", video_id)
        return GenerateResponse(timestamps=cached)

    if not force_retry:
        cached_failure = cache.get_failure(video_id)
        if cached_failure is not None:
            logger.info("Failure cache hit for video_id=%s", video_id)
            raise HTTPException(status_code=502, detail=cached_failure)

    logger.info("Cache miss for video_id=%s — running pipeline", video_id)

    async with _pipeline_lock(video_id):
        # Another request may have finished the pipeline while we waited.
        cached = cache.get(video_id)
        if cached is not None:
            logger.info("Cache hit after lock wait for video_id=%s", video_id)
            return GenerateResponse(timestamps=cached)

        if not force_retry:
            cached_failure = cache.get_failure(video_id)
            if cached_failure is not None:
                logger.info("Failure cache hit after lock wait for video_id=%s", video_id)
                raise HTTPException(status_code=502, detail=cached_failure)

        # --- Step 2: Pipeline (Requirements 5.3, 6.x, 7.x, 8.x, 9.x) ---
        try:
            timestamps = await _run_pipeline(video_id)
        except RuntimeError as exc:
            logger.error("Pipeline error for video_id=%s: %s", video_id, exc)
            cache.set_failure(video_id, str(exc))
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception as exc:
            logger.exception("Unexpected pipeline error for video_id=%s", video_id)
            detail = f"Pipeline failed: {exc}"
            cache.set_failure(video_id, detail)
            raise HTTPException(status_code=502, detail=detail) from exc

        # --- Step 3: Store in cache (Requirement 5.3) ---
        cache.set(video_id, timestamps)
        logger.info(
            "Pipeline complete for video_id=%s — %d timestamps cached",
            video_id,
            len(timestamps),
        )

    _release_pipeline_lock(video_id)

    # --- Step 4: Return response ---
    return GenerateResponse(timestamps=timestamps)
