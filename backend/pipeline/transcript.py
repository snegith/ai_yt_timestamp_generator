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
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Loaded once on first Path B use (same pattern as boundaries._get_model).
_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper_model():
    """Return the shared faster-whisper model, loading it on first call."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel  # type: ignore

                logger.info("Loading faster-whisper model (base, cpu, int8)")
                _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


# ---------------------------------------------------------------------------
# Path A — youtube-transcript-api
# ---------------------------------------------------------------------------

_PREFERRED_LANGUAGES = ("en", "en-US", "en-GB")


def _segments_from_raw(raw: Any) -> list[dict[str, Any]]:
    """Normalise youtube-transcript-api output to segment dicts."""
    if hasattr(raw, "to_raw_data"):
        raw = raw.to_raw_data()

    return [
        {
            "text": seg["text"] if isinstance(seg, dict) else seg.text,
            "start": float(seg["start"] if isinstance(seg, dict) else seg.start),
            "duration": float(seg["duration"] if isinstance(seg, dict) else seg.duration),
        }
        for seg in raw
    ]


def _select_transcript(transcript_list: Any) -> Any:
    """Pick the best available transcript, preferring English then any language."""
    for finder in (
        transcript_list.find_manually_created_transcript,
        transcript_list.find_generated_transcript,
        transcript_list.find_transcript,
    ):
        try:
            return finder(_PREFERRED_LANGUAGES)
        except Exception:
            continue

    available = list(transcript_list)
    if not available:
        raise RuntimeError("No transcripts available for this video")

    manual = [t for t in available if not t.is_generated]
    return manual[0] if manual else available[0]


def _fetch_via_youtube_transcript_api(video_id: str) -> list[dict[str, Any]]:
    """Synchronous helper; called in a thread pool to avoid blocking the event loop."""
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore

    api = YouTubeTranscriptApi()

    if hasattr(api, "list"):
        transcript_list = api.list(video_id)
        transcript = _select_transcript(transcript_list)
        raw = transcript.fetch()
        logger.info(
            "Transcript language %s (generated=%s) for %s",
            transcript.language_code,
            transcript.is_generated,
            video_id,
        )
        return _segments_from_raw(raw)

    if hasattr(api, "fetch"):
        raw = api.fetch(video_id)
        return _segments_from_raw(raw)

    raw = YouTubeTranscriptApi.get_transcript(video_id)
    return _segments_from_raw(raw)


# ---------------------------------------------------------------------------
# Path B — yt-dlp + faster-whisper
# ---------------------------------------------------------------------------

def _fetch_via_yt_dlp_and_whisper(video_id: str) -> list[dict[str, Any]]:
    """Synchronous helper; called in a thread pool to avoid blocking the event loop."""
    import yt_dlp  # type: ignore

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

        model = _get_whisper_model()
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
    loop = asyncio.get_running_loop()

    # --- Path A ---
    try:
        logger.info("Transcript retrieval: trying Path A (youtube-transcript-api) for %s", video_id)
        segments = await loop.run_in_executor(
            None, _fetch_via_youtube_transcript_api, video_id
        )
        if not segments:
            raise RuntimeError("youtube-transcript-api returned no transcript segments")
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
        if not segments:
            raise RuntimeError("yt-dlp+faster-whisper returned no transcript segments")
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
