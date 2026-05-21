"""
main.py — FastAPI application entry point.

Exposes:
  POST /generate  — run the timestamp pipeline (or return cached result)

CORS is configured to allow requests from any origin (including
chrome-extension:// origins used by the Chrome Extension).

Startup validation ensures GROQ_API_KEY is present before the server
accepts any traffic.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="YouTube Timestamp Generator", lifespan=lifespan)

# CORS — allow Chrome Extension origins (chrome-extension://...) and any
# other origin.  Requirements 4.4.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# POST /generate  (Requirements 4.1, 4.2, 4.3, 5.1, 5.2, 5.3)
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
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

    # --- Step 1: Cache lookup (Requirement 5.2) ---
    cached = cache.get(video_id)
    if cached is not None:
        logger.info("Cache hit for video_id=%s", video_id)
        return GenerateResponse(timestamps=cached)

    logger.info("Cache miss for video_id=%s — running pipeline", video_id)

    # --- Step 2: Pipeline (Requirements 5.3, 6.x, 7.x, 8.x, 9.x) ---
    try:
        segments = await get_transcript(video_id)
        windows = preprocess(segments)
        boundary_windows = detect_boundaries(windows)
        timestamps = await generate_titles(boundary_windows)
    except RuntimeError as exc:
        logger.error("Pipeline error for video_id=%s: %s", video_id, exc)
        raise HTTPException(status_code=502, detail=str(exc))

    # --- Step 3: Store in cache (Requirement 5.3) ---
    cache.set(video_id, timestamps)
    logger.info(
        "Pipeline complete for video_id=%s — %d timestamps cached",
        video_id,
        len(timestamps),
    )

    # --- Step 4: Return response ---
    return GenerateResponse(timestamps=timestamps)
