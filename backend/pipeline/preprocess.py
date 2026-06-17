"""pipeline/preprocess.py — Filler stripping, sentence tokenization, windowing."""

from __future__ import annotations

import re
from dataclasses import dataclass

import nltk

# Ensure the punkt tokenizer data is available at import time.
# The Dockerfile pre-downloads it; this is a safety net for local dev.
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TextWindow:
    """A contiguous slice of transcript text with its start time in seconds."""

    text: str
    start_time: float  # seconds from video start


# ---------------------------------------------------------------------------
# Filler word stripping
# ---------------------------------------------------------------------------

# Matches standalone filler tokens (word-boundary anchored, case-insensitive).
# "you know" is a two-word phrase so it is handled separately.
_FILLER_PHRASE_RE = re.compile(
    r"\byou\s+know\b",
    re.IGNORECASE,
)

_FILLER_WORD_RE = re.compile(
    r"\b(?:um|uh|like)\b",
    re.IGNORECASE,
)


def strip_fillers(text: str) -> str:
    """Remove standalone filler words/phrases from *text*.

    Fillers removed: "um", "uh", "like", "you know".
    Non-filler content is preserved exactly.
    """
    # Remove two-word phrase first to avoid partial matches.
    text = _FILLER_PHRASE_RE.sub("", text)
    text = _FILLER_WORD_RE.sub("", text)
    # Collapse runs of whitespace introduced by removals.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

_TARGET_WORDS = 200
_MIN_WORDS = 150
_MAX_WORDS = 250


def preprocess(segments: list[dict]) -> list[TextWindow]:
    """Convert raw transcript segments into ~200-word :class:`TextWindow` objects.

    Parameters
    ----------
    segments:
        List of ``{"text": str, "start": float, "duration": float}`` dicts as
        returned by :func:`pipeline.transcript.get_transcript`.

    Returns
    -------
    list[TextWindow]
        Windows of approximately 200 words (150–250 words each, except
        possibly the last).  The first window always starts at time 0.
    """
    if not segments:
        return []

    # ------------------------------------------------------------------
    # Step 1: Build a flat list of (sentence, start_time) pairs.
    # Each segment may contain multiple sentences; we attribute the
    # segment's start time to every sentence it produces.
    # ------------------------------------------------------------------
    sentences: list[tuple[str, float]] = []  # (sentence_text, start_seconds)

    for seg in segments:
        raw_text: str = seg.get("text", "")
        start: float = float(seg.get("start", 0.0))
        duration: float = float(seg.get("duration", 0.0))

        cleaned = strip_fillers(raw_text)
        if not cleaned:
            continue

        tokenized = [s.strip() for s in nltk.sent_tokenize(cleaned) if s.strip()]
        if not tokenized:
            continue

        if len(tokenized) == 1:
            sentences.append((tokenized[0], start))
            continue

        # Spread sentence start times proportionally across the segment duration.
        weights = [len(s.split()) for s in tokenized]
        total_weight = sum(weights) or 1
        elapsed = 0.0
        for sent, weight in zip(tokenized, weights):
            sent_start = start + duration * (elapsed / total_weight)
            elapsed += weight
            sentences.append((sent, sent_start))

    if not sentences:
        return []

    # ------------------------------------------------------------------
    # Step 2: Greedily pack sentences into windows targeting ~200 words.
    # A window is "full" once it reaches _TARGET_WORDS words; we never
    # split a sentence across windows.
    # ------------------------------------------------------------------
    windows: list[TextWindow] = []
    current_sentences: list[str] = []
    current_start: float = sentences[0][1]
    current_word_count: int = 0

    for sent_text, sent_start in sentences:
        word_count = len(sent_text.split())

        if not current_sentences:
            current_start = sent_start

        # If adding this sentence would push us past _MAX_WORDS AND we
        # already have at least _MIN_WORDS, flush the current window first.
        if current_word_count + word_count > _MAX_WORDS and current_word_count >= _MIN_WORDS:
            windows.append(
                TextWindow(
                    text=" ".join(current_sentences),
                    start_time=current_start,
                )
            )
            current_sentences = []
            current_start = sent_start
            current_word_count = 0

        current_sentences.append(sent_text)
        current_word_count += word_count

        # Flush when we've hit or exceeded the target.
        if current_word_count >= _TARGET_WORDS:
            windows.append(
                TextWindow(
                    text=" ".join(current_sentences),
                    start_time=current_start,
                )
            )
            current_sentences = []
            current_word_count = 0

    # Flush any remaining sentences as the final (possibly short) window.
    if current_sentences:
        windows.append(
            TextWindow(
                text=" ".join(current_sentences),
                start_time=current_start,
            )
        )

    return windows
