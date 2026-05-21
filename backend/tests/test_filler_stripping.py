"""
Property-Based Test for Filler Word Stripping (Property 7)

**Validates: Requirements 7.2**

Property 7: Filler Word Stripping
For any input string, strip_fillers(text) SHALL return a string containing no
standalone occurrences of any filler word token ("um", "uh", "like", "you know"),
and SHALL preserve all non-filler content.

Two sub-properties are tested:

  7a — No filler tokens survive:
       For any string (including ones with injected fillers), the output of
       strip_fillers() contains no standalone "um", "uh", "like", or "you know".

  7b — Non-filler words are preserved:
       For any list of non-filler words assembled into a sentence, every word
       survives strip_fillers() unchanged.
"""

from __future__ import annotations

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from pipeline.preprocess import strip_fillers


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILLER_WORDS = ["um", "uh", "like", "you know"]

# Regex mirrors the implementation: word-boundary anchored, case-insensitive.
_FILLER_PHRASE_RE = re.compile(r"\byou\s+know\b", re.IGNORECASE)
_FILLER_WORD_RE = re.compile(r"\b(?:um|uh|like)\b", re.IGNORECASE)


def _contains_filler(text: str) -> bool:
    """Return True if *text* still contains any standalone filler token."""
    if _FILLER_PHRASE_RE.search(text):
        return True
    if _FILLER_WORD_RE.search(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Words that are definitely NOT fillers and do not contain filler substrings
# at word boundaries.  We avoid words like "likely" (contains "like") to keep
# the non-filler preservation check simple and unambiguous.
_SAFE_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

safe_word_strategy = st.text(
    alphabet=_SAFE_ALPHABET,
    min_size=2,
    max_size=12,
).filter(
    lambda w: not _FILLER_WORD_RE.fullmatch(w)
    and not _FILLER_PHRASE_RE.fullmatch(w)
    # Exclude words that ARE filler tokens (case-insensitive)
    and w.lower() not in {"um", "uh", "like"}
)

# A list of 1–10 safe (non-filler) words
safe_words_strategy = st.lists(safe_word_strategy, min_size=1, max_size=10)

# Filler tokens to inject (including mixed-case variants)
filler_token_strategy = st.sampled_from(
    ["um", "uh", "like", "you know", "Um", "UH", "Like", "You Know", "YOU KNOW"]
)

# A list of 0–5 filler tokens to inject
filler_list_strategy = st.lists(filler_token_strategy, min_size=0, max_size=5)


def _build_text(safe_words: list[str], fillers: list[str]) -> str:
    """Interleave safe words and filler tokens into a single string."""
    tokens: list[str] = list(safe_words)
    # Insert each filler at a random-ish position (deterministic given the
    # Hypothesis shrinking model — we just append then shuffle via sorted index).
    for i, filler in enumerate(fillers):
        insert_pos = i % (len(tokens) + 1)
        tokens.insert(insert_pos, filler)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Property 7a: No filler tokens survive strip_fillers()
# ---------------------------------------------------------------------------

@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    safe_words=safe_words_strategy,
    fillers=filler_list_strategy,
)
def test_no_filler_tokens_remain_after_strip(
    safe_words: list[str],
    fillers: list[str],
) -> None:
    """
    **Validates: Requirements 7.2**

    For any input string (including ones with injected filler tokens),
    strip_fillers() SHALL return a string that contains no standalone
    occurrences of "um", "uh", "like", or "you know".
    """
    text = _build_text(safe_words, fillers)
    result = strip_fillers(text)

    assert not _contains_filler(result), (
        f"Filler token found in output.\n"
        f"  Input:  {text!r}\n"
        f"  Output: {result!r}"
    )


# ---------------------------------------------------------------------------
# Property 7b: Non-filler words are preserved
# ---------------------------------------------------------------------------

@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(safe_words=safe_words_strategy)
def test_non_filler_words_are_preserved(safe_words: list[str]) -> None:
    """
    **Validates: Requirements 7.2**

    For any list of non-filler words, every word in the list SHALL appear
    in the output of strip_fillers() — i.e., non-filler content is not lost.
    """
    text = " ".join(safe_words)
    result = strip_fillers(text)
    result_words = result.split()

    for word in safe_words:
        assert word in result_words, (
            f"Non-filler word {word!r} was lost.\n"
            f"  Input:  {text!r}\n"
            f"  Output: {result!r}"
        )


# ---------------------------------------------------------------------------
# Property 7c: Filler-only input produces empty (or whitespace-only) output
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(fillers=st.lists(filler_token_strategy, min_size=1, max_size=8))
def test_filler_only_input_produces_empty_output(fillers: list[str]) -> None:
    """
    **Validates: Requirements 7.2**

    When the input consists entirely of filler tokens, strip_fillers() SHALL
    return an empty string (or a string that is blank after stripping).
    """
    text = " ".join(fillers)
    result = strip_fillers(text)

    assert result.strip() == "", (
        f"Expected empty output for filler-only input.\n"
        f"  Input:  {text!r}\n"
        f"  Output: {result!r}"
    )


# ---------------------------------------------------------------------------
# Property 7d: strip_fillers() is idempotent
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    safe_words=safe_words_strategy,
    fillers=filler_list_strategy,
)
def test_strip_fillers_is_idempotent(
    safe_words: list[str],
    fillers: list[str],
) -> None:
    """
    **Validates: Requirements 7.2**

    Applying strip_fillers() twice SHALL produce the same result as applying
    it once — i.e., the function is idempotent.
    """
    text = _build_text(safe_words, fillers)
    once = strip_fillers(text)
    twice = strip_fillers(once)

    assert once == twice, (
        f"strip_fillers() is not idempotent.\n"
        f"  Input:       {text!r}\n"
        f"  After once:  {once!r}\n"
        f"  After twice: {twice!r}"
    )
