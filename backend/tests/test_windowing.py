"""
Property-Based Test for Windowing Completeness and Size (Property 8)

**Validates: Requirements 7.1**

Property 8: Windowing Completeness and Size
For any list of transcript segments whose total word count is W,
`preprocess(segments)` SHALL return windows such that:
  (a) the concatenation of all window texts contains all non-filler words
      from the input, and
  (b) every window except possibly the last contains at least 150 and at
      most 250 words.
"""

from __future__ import annotations

import re
import sys
import os
from collections import Counter

# Allow imports from the backend directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from pipeline.preprocess import preprocess, strip_fillers


# ---------------------------------------------------------------------------
# Filler detection helpers (mirrors the logic in preprocess.py)
# ---------------------------------------------------------------------------

_FILLER_PHRASE_RE = re.compile(r"\byou\s+know\b", re.IGNORECASE)
_FILLER_WORD_RE = re.compile(r"\b(?:um|uh|like)\b", re.IGNORECASE)


def _is_filler_word(word: str) -> bool:
    """Return True if *word* is a standalone filler token."""
    return bool(_FILLER_WORD_RE.fullmatch(word.strip()))


def _extract_non_filler_words(text: str) -> list[str]:
    """Return the non-filler words from *text* in order, lowercased."""
    cleaned = strip_fillers(text)
    return [w.lower() for w in cleaned.split() if w.strip()]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A single "content" word: alphabetic, not a filler token.
_FILLER_TOKENS = {"um", "uh", "like"}

content_word = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz",
    min_size=2,
    max_size=12,
).filter(lambda w: w not in _FILLER_TOKENS and w != "you")

# A segment text: 1–40 content words joined by spaces.
segment_text_strategy = st.lists(
    content_word,
    min_size=1,
    max_size=40,
).map(lambda words: " ".join(words))

# A single segment dict.
segment_strategy = st.fixed_dictionaries(
    {
        "text": segment_text_strategy,
        "start": st.floats(min_value=0.0, max_value=7200.0, allow_nan=False, allow_infinity=False),
        "duration": st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False),
    }
)

# A list of 1–60 segments (enough to produce multiple windows in many cases).
segments_strategy = st.lists(segment_strategy, min_size=1, max_size=60)

# A list of segments guaranteed to produce at least 2 windows (≥ 300 words).
# Each segment has exactly 10 words; 30 segments = 300 words → ≥ 2 windows.
large_segments_strategy = st.lists(
    st.fixed_dictionaries(
        {
            "text": st.just(" ".join(["alpha"] * 10)),
            "start": st.floats(min_value=0.0, max_value=7200.0, allow_nan=False, allow_infinity=False),
            "duration": st.floats(min_value=0.5, max_value=30.0, allow_nan=False, allow_infinity=False),
        }
    ),
    min_size=30,
    max_size=60,
)


# ---------------------------------------------------------------------------
# Property 8a: All non-filler words from input appear in output windows
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(segments=segments_strategy)
def test_windowing_preserves_all_non_filler_words(segments: list[dict]) -> None:
    """
    **Validates: Requirements 7.1**

    For any segment list, the multiset of non-filler words across all output
    windows equals the multiset of non-filler words across all input segments.

    This verifies that preprocess() neither drops nor duplicates content words.
    """
    # Collect non-filler words from input
    input_words: list[str] = []
    for seg in segments:
        input_words.extend(_extract_non_filler_words(seg.get("text", "")))

    windows = preprocess(segments)

    # Collect words from output windows
    output_words: list[str] = []
    for window in windows:
        output_words.extend(w.lower() for w in window.text.split() if w.strip())

    assert Counter(input_words) == Counter(output_words), (
        f"Word multisets differ.\n"
        f"  Input  ({len(input_words)} words): {sorted(Counter(input_words).items())[:20]}\n"
        f"  Output ({len(output_words)} words): {sorted(Counter(output_words).items())[:20]}"
    )


# ---------------------------------------------------------------------------
# Property 8b: Every window except the last has 150–250 words
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(segments=large_segments_strategy)
def test_non_last_windows_have_150_to_250_words(segments: list[dict]) -> None:
    """
    **Validates: Requirements 7.1**

    For any segment list that produces multiple windows, every window except
    the last SHALL contain between 150 and 250 words (inclusive).
    """
    windows = preprocess(segments)

    # This strategy always produces enough words for multiple windows
    assert len(windows) >= 1, "Expected at least one window"

    for i, window in enumerate(windows[:-1]):  # all except the last
        word_count = len(window.text.split())
        assert 150 <= word_count <= 250, (
            f"Window {i} (non-last) has {word_count} words; expected 150–250.\n"
            f"  Window text (first 80 chars): {window.text[:80]!r}"
        )


# ---------------------------------------------------------------------------
# Property 8c: Combined — completeness AND size for varied inputs
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(segments=segments_strategy)
def test_windowing_completeness_and_size_combined(segments: list[dict]) -> None:
    """
    **Validates: Requirements 7.1**

    Combined property: for any segment list,
      (a) all non-filler input words appear in the output (completeness), and
      (b) every non-last window has 150–250 words (size constraint).

    This is the canonical statement of Property 8.
    """
    # --- Part (a): word preservation ---
    input_words: list[str] = []
    for seg in segments:
        input_words.extend(_extract_non_filler_words(seg.get("text", "")))

    windows = preprocess(segments)

    output_words: list[str] = []
    for window in windows:
        output_words.extend(w.lower() for w in window.text.split() if w.strip())

    assert Counter(input_words) == Counter(output_words), (
        f"Word preservation failed.\n"
        f"  Input  ({len(input_words)} words)\n"
        f"  Output ({len(output_words)} words)"
    )

    # --- Part (b): window size constraint ---
    for i, window in enumerate(windows[:-1]):
        word_count = len(window.text.split())
        assert 150 <= word_count <= 250, (
            f"Non-last window {i} has {word_count} words; expected 150–250."
        )


# ---------------------------------------------------------------------------
# Edge case: empty segment list produces no windows
# ---------------------------------------------------------------------------

def test_empty_segments_produces_no_windows() -> None:
    """
    **Validates: Requirements 7.1**

    An empty segment list must produce an empty window list.
    """
    assert preprocess([]) == []


# ---------------------------------------------------------------------------
# Edge case: single short segment produces exactly one window
# ---------------------------------------------------------------------------

def test_single_short_segment_produces_one_window() -> None:
    """
    **Validates: Requirements 7.1**

    A single segment with fewer than 150 words produces exactly one window
    (the last window, which may be short).
    """
    seg = {"text": "hello world this is a test", "start": 0.0, "duration": 3.0}
    windows = preprocess([seg])
    assert len(windows) == 1
    # The single window is also the last window, so no size constraint applies
    assert "hello" in windows[0].text
    assert "world" in windows[0].text
