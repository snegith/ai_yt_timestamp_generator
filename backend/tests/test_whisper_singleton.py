"""Tests that faster-whisper is loaded once and reused on Path B."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pipeline.transcript as transcript


@pytest.fixture(autouse=True)
def reset_whisper_model():
    transcript._whisper_model = None
    yield
    transcript._whisper_model = None


def test_get_whisper_model_loads_once():
    mock_model = MagicMock()
    mock_whisper_cls = MagicMock(return_value=mock_model)
    fake_fw = MagicMock()
    fake_fw.WhisperModel = mock_whisper_cls

    with patch.dict(sys.modules, {"faster_whisper": fake_fw}):
        first = transcript._get_whisper_model()
        second = transcript._get_whisper_model()

    assert first is second
    assert mock_whisper_cls.call_count == 1
    mock_whisper_cls.assert_called_once_with("base", device="cpu", compute_type="int8")
