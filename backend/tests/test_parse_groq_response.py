"""Tests for Groq response JSON extraction in pipeline/titles.py."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.preprocess import TextWindow
from pipeline.titles import (
    _extract_json_array_payload,
    _parse_titles_from_response,
    generate_titles,
)


def _run(coro):
    return asyncio.run(coro)


def test_extract_plain_json_array():
    raw = '["Intro", "Main topic"]'
    assert _extract_json_array_payload(raw) == raw


def test_extract_json_inside_markdown_fence():
    raw = '```json\n["A", "B"]\n```'
    assert _extract_json_array_payload(raw) == '["A", "B"]'


def test_extract_json_with_preamble_and_fence():
    raw = 'Here are your titles:\n```\n["One", "Two"]\n```'
    assert _extract_json_array_payload(raw) == '["One", "Two"]'


def test_extract_json_with_preamble_no_fence():
    raw = 'Titles:\n["Alpha", "Beta"]\nThanks!'
    assert _extract_json_array_payload(raw) == '["Alpha", "Beta"]'


def test_parse_titles_from_response_count_mismatch():
    with pytest.raises(ValueError, match="Expected 2 titles"):
        _parse_titles_from_response('["only one"]', 2)


def test_generate_titles_parses_markdown_wrapped_response():
    windows = [
        TextWindow(text="first excerpt", start_time=0.0),
        TextWindow(text="second excerpt", start_time=60.0),
    ]
    titles = ["Introduction", "Deep dive"]
    wrapped = f"```json\n{json.dumps(titles)}\n```"

    message = MagicMock()
    message.content = wrapped
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]

    mock_create = AsyncMock(return_value=response)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create

    with patch("pipeline.titles.AsyncGroq", return_value=mock_client), patch.dict(
        os.environ, {"GROQ_API_KEY": "test-key"}
    ):
        result = _run(generate_titles(windows))

    assert len(result) == 2
    assert result[0].title == "Introduction"
    assert result[1].title == "Deep dive"
