"""
pipeline/transcript.py — Transcript retrieval

Path A (preferred): youtube-transcript-api
Path B (fallback):  yt-dlp audio download + faster-whisper transcription

Returns a list of {"text": str, "start": float, "duration": float} dicts.
Raises RuntimeError if both paths fail.
"""

import asyncio
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path A — youtube-transcript-api
# ---------------------------------------------------------------------------

def _fetch_via_youtube_transcript_api(video_id: str) -> list[dict[str, Any]]:
    """Synchronous helper; called in a thread pool to avoid blocking the event loop."""
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

    raw = YouTubeTranscriptApi.get_transcript(video_id)
    # The library already returns [{"text": str, "start": float, "duration": float}]
    return [
        {
            "text": seg["text"],
            "start": float(seg["start"]),
            "duration": float(seg["duration"]),
        }
        for seg in raw
    ]


# ---------------------------------------------------------------------------
# Path B — yt-dlp + faster-whisper
# ---------------------------------------------------------------------------

def _fetch_via_yt_dlp_and_whisper(video_id: str) -> list[dict[str, Any]]:
    """Synchronous helper; called in a thread pool to avoid blocking the event loop."""
    import yt_dlp  # type: ignore
    from faster_whisper import WhisperModel  # type: ignore

    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Locate the downloaded file (extension may vary)
        downloaded_file: str | None = None
        for fname in os.listdir(tmpdir):
            if fname.startswith("audio"):
                downloaded_file = os.path.join(tmpdir, fname)
                break

        if downloaded_file is None:
            raise RuntimeError("yt-dlp did not produce an audio file")

        # Transcribe with faster-whisper (base model, word-level timestamps)
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments_iter, _info = model.transcribe(
            downloaded_file,
            word_timestamps=True,  # Req 6.4: preserve word-level timing
        )

        result: list[dict[str, Any]] = []
        for seg in segments_iter:
            result.append(
                {
                    "text": seg.text.strip(),
                    "start": float(seg.start),
                    "duration": float(seg.end - seg.start),
                }
            )

    return result


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------

async def get_transcript(video_id: str) -> list[dict[str, Any]]:
    """
    Retrieve a transcript for the given YouTube video ID.

    Attempts Path A (youtube-transcript-api) first.
    Falls back to Path B (yt-dlp + faster-whisper) if Path A fails for any reason.
    Raises RuntimeError if both paths fail.

    Returns:
        list of {"text": str, "start": float, "duration": float}
    """
    loop = asyncio.get_event_loop()

    # --- Path A ---
    try:
        logger.info("Transcript retrieval: trying Path A (youtube-transcript-api) for %s", video_id)
        segments = await loop.run_in_executor(
            None, _fetch_via_youtube_transcript_api, video_id
        )
        logger.info("Transcript retrieval: Path A succeeded for %s (%d segments)", video_id, len(segments))
        return segments
    except Exception as exc_a:
        logger.warning(
            "Transcript retrieval: Path A failed for %s — %s. Falling back to Path B.",
            video_id,
            exc_a,
        )

    # --- Path B ---
    try:
        logger.info("Transcript retrieval: trying Path B (yt-dlp + faster-whisper) for %s", video_id)
        segments = await loop.run_in_executor(
            None, _fetch_via_yt_dlp_and_whisper, video_id
        )
        logger.info("Transcript retrieval: Path B succeeded for %s (%d segments)", video_id, len(segments))
        return segments
    except Exception as exc_b:
        logger.error(
            "Transcript retrieval: Path B also failed for %s — %s.",
            video_id,
            exc_b,
        )

    raise RuntimeError(
        f"Transcript retrieval failed for video '{video_id}': "
        f"both youtube-transcript-api and yt-dlp+faster-whisper failed."
    )
