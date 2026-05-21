"""pipeline/titles.py — Groq API title generation for topic boundary windows."""

from __future__ import annotations

import json
import os

from groq import AsyncGroq

from models import Timestamp
from pipeline.preprocess import TextWindow


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
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(boundary_windows)},
            ],
            temperature=0.3,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq API request failed: {exc}") from exc

    raw_content = response.choices[0].message.content or ""

    # Parse the JSON array from the response.
    try:
        titles: list[str] = json.loads(raw_content)
        if not isinstance(titles, list):
            raise ValueError("Response is not a JSON array")
        if len(titles) != len(boundary_windows):
            raise ValueError(
                f"Expected {len(boundary_windows)} titles, got {len(titles)}"
            )
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to parse Groq API response as JSON array: {exc}. "
            f"Raw response: {raw_content!r}"
        ) from exc

    # Build Timestamp objects and sort by ascending start time.
    timestamps = [
        Timestamp(
            time=_seconds_to_time_str(window.start_time),
            title=str(title),
        )
        for window, title in zip(boundary_windows, titles)
    ]

    timestamps.sort(key=lambda ts: _time_str_to_seconds(ts.time))

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
