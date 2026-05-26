from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from main import _run_pipeline
from models import Timestamp
from pipeline.preprocess import TextWindow


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_pipeline_rejects_empty_transcript_segments():
    with patch("main.get_transcript", new=AsyncMock(return_value=[])):
        try:
            _run(_run_pipeline("video123"))
        except RuntimeError as exc:
            assert "no transcript segments" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for empty transcript segments")


def test_pipeline_rejects_empty_preprocessed_windows():
    with patch("main.get_transcript", new=AsyncMock(return_value=[{"text": "um uh", "start": 0, "duration": 1}])):
        with patch("main.preprocess", return_value=[]):
            try:
                _run(_run_pipeline("video123"))
            except RuntimeError as exc:
                assert "no usable transcript text" in str(exc)
            else:
                raise AssertionError("Expected RuntimeError for empty preprocessed windows")


def test_pipeline_rejects_empty_boundary_windows():
    window = TextWindow(text="real content", start_time=0.0)
    with patch("main.get_transcript", new=AsyncMock(return_value=[{"text": "real content", "start": 0, "duration": 1}])):
        with patch("main.preprocess", return_value=[window]):
            with patch("main.detect_boundaries", return_value=[]):
                try:
                    _run(_run_pipeline("video123"))
                except RuntimeError as exc:
                    assert "no timestamp boundaries" in str(exc)
                else:
                    raise AssertionError("Expected RuntimeError for empty boundary windows")


def test_pipeline_rejects_empty_generated_timestamps():
    window = TextWindow(text="real content", start_time=0.0)
    with patch("main.get_transcript", new=AsyncMock(return_value=[{"text": "real content", "start": 0, "duration": 1}])):
        with patch("main.preprocess", return_value=[window]):
            with patch("main.detect_boundaries", return_value=[window]):
                with patch("main.generate_titles", new=AsyncMock(return_value=[])):
                    try:
                        _run(_run_pipeline("video123"))
                    except RuntimeError as exc:
                        assert "no timestamps were generated" in str(exc)
                    else:
                        raise AssertionError("Expected RuntimeError for empty timestamps")


def test_pipeline_returns_non_empty_timestamps():
    window = TextWindow(text="real content", start_time=0.0)
    timestamps = [Timestamp(time="0:00", title="Intro")]

    with patch("main.get_transcript", new=AsyncMock(return_value=[{"text": "real content", "start": 0, "duration": 1}])):
        with patch("main.preprocess", return_value=[window]):
            with patch("main.detect_boundaries", return_value=[window]):
                with patch("main.generate_titles", new=AsyncMock(return_value=timestamps)):
                    assert _run(_run_pipeline("video123")) == timestamps
