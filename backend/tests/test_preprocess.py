"""Unit tests for pipeline/preprocess.py — TextWindow, strip_fillers, preprocess."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pipeline.preprocess import TextWindow, strip_fillers, preprocess


# ---------------------------------------------------------------------------
# strip_fillers — unit tests
# ---------------------------------------------------------------------------

class TestStripFillers:
    def test_removes_um(self):
        assert "um" not in strip_fillers("um that was great").split()

    def test_removes_uh(self):
        assert "uh" not in strip_fillers("uh I think so").split()

    def test_removes_like(self):
        result = strip_fillers("it was like really good")
        assert "like" not in result.split()

    def test_removes_you_know(self):
        result = strip_fillers("you know what I mean")
        assert "you know" not in result

    def test_preserves_non_filler_words(self):
        result = strip_fillers("the quick brown fox")
        assert "quick" in result
        assert "brown" in result
        assert "fox" in result

    def test_empty_string_returns_empty(self):
        assert strip_fillers("") == ""

    def test_only_fillers_returns_empty_or_whitespace(self):
        result = strip_fillers("um uh like you know")
        assert result.strip() == ""

    def test_case_insensitive_um(self):
        result = strip_fillers("Um that is correct")
        assert "Um" not in result.split() and "um" not in result.split()

    def test_case_insensitive_uh(self):
        result = strip_fillers("UH I see")
        assert "UH" not in result.split() and "uh" not in result.split()

    def test_no_double_spaces_after_removal(self):
        result = strip_fillers("it was um really good")
        assert "  " not in result

    def test_like_in_word_not_removed(self):
        # "likely" contains "like" but should NOT be stripped (word boundary)
        result = strip_fillers("it is likely true")
        assert "likely" in result

    def test_you_know_multiword_phrase(self):
        result = strip_fillers("you know this is important")
        assert "you know" not in result
        assert "important" in result

    def test_mixed_fillers_and_content(self):
        result = strip_fillers("um so uh the answer is like forty two you know")
        assert "forty" in result
        assert "two" in result
        assert "answer" in result


# ---------------------------------------------------------------------------
# preprocess — unit tests
# ---------------------------------------------------------------------------

def _make_segments(words_per_seg: int, num_segs: int, start_offset: float = 1.0) -> list[dict]:
    """Create synthetic segments each containing `words_per_seg` words."""
    word = "word"
    text = " ".join([word] * words_per_seg)
    return [
        {"text": text, "start": i * start_offset, "duration": start_offset}
        for i in range(num_segs)
    ]


class TestPreprocess:
    def test_empty_segments_returns_empty(self):
        assert preprocess([]) == []

    def test_returns_list_of_text_windows(self):
        segs = _make_segments(10, 5)
        result = preprocess(segs)
        assert isinstance(result, list)
        assert all(isinstance(w, TextWindow) for w in result)

    def test_first_window_start_time_is_zero(self):
        segs = _make_segments(10, 30)
        result = preprocess(segs)
        assert result[0].start_time == 0.0

    def test_single_short_segment_produces_one_window(self):
        segs = [{"text": "Hello world.", "start": 5.0, "duration": 2.0}]
        result = preprocess(segs)
        assert len(result) == 1
        assert result[0].start_time == 0.0  # forced to 0

    def test_window_text_contains_content(self):
        segs = [{"text": "The quick brown fox jumps.", "start": 0.0, "duration": 3.0}]
        result = preprocess(segs)
        assert "quick" in result[0].text
        assert "fox" in result[0].text

    def test_fillers_stripped_from_windows(self):
        segs = [{"text": "um the answer is uh forty two you know", "start": 0.0, "duration": 5.0}]
        result = preprocess(segs)
        assert result  # at least one window
        combined = " ".join(w.text for w in result)
        assert "um" not in combined.split()
        assert "uh" not in combined.split()
        assert "forty" in combined
        assert "two" in combined

    def test_large_input_creates_multiple_windows(self):
        # 300 words total → should produce at least 2 windows
        segs = _make_segments(10, 30)  # 300 words across 30 segments
        result = preprocess(segs)
        assert len(result) >= 2

    def test_window_sizes_within_bounds_except_last(self):
        # 600 words → ~3 windows; all but last should be 150–250 words
        segs = _make_segments(20, 30)  # 600 words
        result = preprocess(segs)
        for window in result[:-1]:
            word_count = len(window.text.split())
            assert 150 <= word_count <= 250, (
                f"Non-last window has {word_count} words, expected 150–250"
            )

    def test_all_words_preserved_across_windows(self):
        # Total non-filler words in input should equal total words in output
        segs = _make_segments(10, 25)  # 250 words, all "word"
        result = preprocess(segs)
        total_output_words = sum(len(w.text.split()) for w in result)
        assert total_output_words == 250

    def test_segments_with_only_fillers_produce_no_content(self):
        segs = [{"text": "um uh like you know", "start": 0.0, "duration": 2.0}]
        result = preprocess(segs)
        # Either empty or windows with no meaningful content
        if result:
            combined = " ".join(w.text for w in result)
            assert combined.strip() == ""

    def test_start_times_are_floats(self):
        segs = _make_segments(10, 30)
        result = preprocess(segs)
        for w in result:
            assert isinstance(w.start_time, float)

    def test_text_window_dataclass_fields(self):
        w = TextWindow(text="hello", start_time=1.5)
        assert w.text == "hello"
        assert w.start_time == 1.5
