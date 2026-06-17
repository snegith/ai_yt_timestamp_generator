from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from pipeline.transcript import _fetch_via_youtube_transcript_api, get_transcript


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@dataclass
class _Snippet:
    text: str
    start: float
    duration: float


class _FetchedTranscript:
    def __init__(self):
        self.snippets = [
            _Snippet("Hello world", 1.5, 2.0),
            _Snippet("Next line", 4.0, 1.25),
        ]

    def __iter__(self):
        return iter(self.snippets)


def _make_transcript_list(transcript: MagicMock) -> MagicMock:
    transcript_list = MagicMock()
    transcript_list.find_manually_created_transcript.side_effect = Exception(
        "no manual english"
    )
    transcript_list.find_generated_transcript.side_effect = Exception(
        "no generated english"
    )
    transcript_list.find_transcript.return_value = transcript
    return transcript_list


def test_fetches_transcripts_with_current_youtube_transcript_api_shape():
    transcript = MagicMock()
    transcript.language_code = "en"
    transcript.is_generated = False
    transcript.fetch.return_value = _FetchedTranscript()

    api = MagicMock()
    api.list.return_value = _make_transcript_list(transcript)

    with patch("youtube_transcript_api.YouTubeTranscriptApi", return_value=api):
        result = _fetch_via_youtube_transcript_api("video123")

    api.list.assert_called_once_with("video123")
    transcript.fetch.assert_called_once()
    assert result == [
        {"text": "Hello world", "start": 1.5, "duration": 2.0},
        {"text": "Next line", "start": 4.0, "duration": 1.25},
    ]


def test_fetches_non_english_transcript_when_english_unavailable():
    ko_transcript = MagicMock()
    ko_transcript.language_code = "ko"
    ko_transcript.is_generated = True
    ko_transcript.fetch.return_value = _FetchedTranscript()

    transcript_list = MagicMock()
    transcript_list.find_manually_created_transcript.side_effect = Exception(
        "no manual english"
    )
    transcript_list.find_generated_transcript.side_effect = Exception(
        "no generated english"
    )
    transcript_list.find_transcript.side_effect = Exception("no english")
    transcript_list.__iter__.return_value = iter([ko_transcript])

    api = MagicMock()
    api.list.return_value = transcript_list

    with patch("youtube_transcript_api.YouTubeTranscriptApi", return_value=api):
        result = _fetch_via_youtube_transcript_api("video123")

    ko_transcript.fetch.assert_called_once()
    assert len(result) == 2


def test_fetches_transcripts_with_legacy_raw_dict_shape():
    raw = [
        {"text": "Legacy hello", "start": 0, "duration": 3},
        {"text": "Legacy next", "start": 3.25, "duration": 2.5},
    ]

    class _LegacyApi:
        @staticmethod
        def get_transcript(video_id):
            assert video_id == "video123"
            return raw

    with patch("youtube_transcript_api.YouTubeTranscriptApi", _LegacyApi):
        result = _fetch_via_youtube_transcript_api("video123")

    assert result == [
        {"text": "Legacy hello", "start": 0.0, "duration": 3.0},
        {"text": "Legacy next", "start": 3.25, "duration": 2.5},
    ]


def test_get_transcript_falls_back_when_path_a_returns_empty():
    fallback_segments = [{"text": "Fallback text", "start": 0.0, "duration": 1.0}]

    with patch("pipeline.transcript._fetch_via_youtube_transcript_api", return_value=[]):
        with patch("pipeline.transcript._fetch_via_yt_dlp_and_whisper", return_value=fallback_segments):
            assert _run(get_transcript("video123")) == fallback_segments


def test_get_transcript_fails_when_both_paths_return_empty():
    with patch("pipeline.transcript._fetch_via_youtube_transcript_api", return_value=[]):
        with patch("pipeline.transcript._fetch_via_yt_dlp_and_whisper", return_value=[]):
            with pytest.raises(RuntimeError, match="Transcript retrieval failed"):
                _run(get_transcript("video123"))
