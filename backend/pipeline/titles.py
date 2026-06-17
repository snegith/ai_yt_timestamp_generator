"""pipeline/titles.py — Groq API title generation for topic boundary windows."""

from __future__ import annotations

import json
import logging
import os
import re

from groq import AsyncGroq

from models import Timestamp
from pipeline.preprocess import TextWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Time formatting helper
# ---------------------------------------------------------------------------

def _seconds_to_time_str(seconds: float) -> str:
    """Format *seconds* as ``M:SS`` or ``H:MM:SS``.

    Examples
    --------
    >>> _seconds_to_time_str(0)
    '0:00'
    >>> _seconds_to_time_str(83)
    '1:23'
    >>> _seconds_to_time_str(3765)
    '1:02:45'
    """
    total = int(seconds)
    if total < 3600:
        m, s = divmod(total, 60)
        return f"{m}:{s:02d}"
    else:
        h, remainder = divmod(total, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates concise YouTube chapter titles. "
    "Given a list of transcript excerpts, return a JSON array of short, descriptive "
    "chapter titles — one title per excerpt, in the same order. "
    "Respond with ONLY the JSON array, no additional text or markdown."
)


def _build_user_prompt(boundary_windows: list[TextWindow]) -> str:
    """Build the batched user prompt containing all window texts."""
    lines = ["Generate one concise chapter title for each of the following transcript excerpts:\n"]
    for i, window in enumerate(boundary_windows, start=1):
        lines.append(f"Excerpt {i}:\n{window.text}\n")
    lines.append(
        f'\nReturn a JSON array of exactly {len(boundary_windows)} title strings, '
        'e.g. ["Title 1", "Title 2", ...]'
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Groq response parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n?(.*?)\n?```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_json_array_payload(raw: str) -> str:
    """Extract a JSON array string from plain or markdown-wrapped model output."""
    text = raw.strip()
    if not text:
        raise ValueError("empty response")

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    elif not text.lstrip().startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    return text


def _parse_titles_from_response(raw_content: str, expected_count: int) -> list[str]:
    """Parse Groq output into exactly *expected_count* title strings."""
    payload = _extract_json_array_payload(raw_content)
    titles = json.loads(payload)
    if not isinstance(titles, list):
        raise ValueError("Response is not a JSON array")
    if len(titles) != expected_count:
        raise ValueError(
            f"Expected {expected_count} titles, got {len(titles)}"
        )
    return [str(title) for title in titles]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

async def generate_titles(boundary_windows: list[TextWindow]) -> list[Timestamp]:
    """Generate chapter titles for *boundary_windows* via a single Groq API call.

    Parameters
    ----------
    boundary_windows:
        The topic-boundary :class:`~pipeline.preprocess.TextWindow` objects
        produced by :func:`~pipeline.boundaries.detect_boundaries`.

    Returns
    -------
    list[Timestamp]
        One :class:`~models.Timestamp` per boundary window, sorted in
        ascending chronological order.

    Raises
    ------
    RuntimeError
        If the Groq API returns an error or the response cannot be parsed.
    """
    if not boundary_windows:
        return []

    api_key = os.environ.get("GROQ_API_KEY")
    # The startup lifespan guard should have already caught a missing key,
    # but we defend here too for direct module usage.
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set; cannot call Groq API."
        )

    client = AsyncGroq(api_key=api_key)

    try:
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(boundary_windows)},
            ],
            temperature=0.3,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API request failed: {exc}") from exc

    if not response.choices:
        raise RuntimeError("Groq API returned an empty choices list")
    message = response.choices[0].message
    if message is None:
        raise RuntimeError("Groq API returned no message")
    raw_content = message.content or ""

    try:
        titles = _parse_titles_from_response(raw_content, len(boundary_windows))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to parse Groq response: %s — raw: %r", exc, raw_content)
        raise RuntimeError(
            f"Failed to parse Groq API response as JSON array: {exc}"
        ) from exc

    # Build Timestamp objects and sort by ascending start time.
    pairs = sorted(zip(boundary_windows, titles), key=lambda item: item[0].start_time)
    timestamps = [
        Timestamp(
            time=_seconds_to_time_str(window.start_time),
            title=title,
        )
        for window, title in pairs
    ]

    return timestamps


# ---------------------------------------------------------------------------
# Internal sort helper (mirrors timeStringToSeconds in the extension)
# ---------------------------------------------------------------------------

def _time_str_to_seconds(time_str: str) -> int:
    """Parse ``M:SS`` or ``H:MM:SS`` back to integer seconds for sorting."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Unrecognised time string format: {time_str!r}")
