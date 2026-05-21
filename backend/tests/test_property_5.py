"""
Property-Based Test for Invalid video_id Rejection (Property 5)

**Validates: Requirements 4.3**

Property 5: Invalid video_id Rejected with 422
For any request body where video_id is absent, null, an empty string, or a
whitespace-only string, the /generate endpoint SHALL return HTTP 422.

This test uses FastAPI's TestClient to POST to /generate with invalid inputs
and asserts that the response status code is always 422.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

# Allow imports from the backend directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set GROQ_API_KEY before importing the app so the lifespan startup check passes
os.environ["GROQ_API_KEY"] = "test-key"

# Stub out the 'groq' module before any pipeline imports pull it in,
# since it may not be installed in the test environment.
if "groq" not in sys.modules:
    _groq_stub = MagicMock()
    sys.modules["groq"] = _groq_stub

import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from main import app


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app with GROQ_API_KEY set."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Strategies for invalid video_id values
# ---------------------------------------------------------------------------

# Whitespace-only strings: spaces, tabs, newlines, carriage returns, etc.
_whitespace_strategy = st.text(
    alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c"]),
    min_size=1,
    max_size=50,
)

# Any invalid video_id value: None, empty string, or whitespace-only
_invalid_video_id_strategy = st.one_of(
    st.none(),                  # null / None
    st.just(""),                # empty string
    _whitespace_strategy,       # whitespace-only strings
)


# ---------------------------------------------------------------------------
# Property 5a: Missing video_id key returns 422
# ---------------------------------------------------------------------------

def test_missing_video_id_returns_422(client: TestClient) -> None:
    """
    **Validates: Requirements 4.3**

    A request body with no video_id key at all SHALL return HTTP 422.
    """
    response = client.post("/generate", json={})
    assert response.status_code == 422, (
        f"Expected 422 for missing video_id, got {response.status_code}. "
        f"Response body: {response.text}"
    )


def test_empty_body_returns_422(client: TestClient) -> None:
    """
    **Validates: Requirements 4.3**

    A completely empty JSON body SHALL return HTTP 422.
    """
    response = client.post(
        "/generate",
        content=b"{}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422, (
        f"Expected 422 for empty body, got {response.status_code}. "
        f"Response body: {response.text}"
    )


# ---------------------------------------------------------------------------
# Property 5b: null video_id returns 422
# ---------------------------------------------------------------------------

def test_null_video_id_returns_422(client: TestClient) -> None:
    """
    **Validates: Requirements 4.3**

    A request body with video_id set to null SHALL return HTTP 422.
    """
    response = client.post("/generate", json={"video_id": None})
    assert response.status_code == 422, (
        f"Expected 422 for null video_id, got {response.status_code}. "
        f"Response body: {response.text}"
    )


# ---------------------------------------------------------------------------
# Property 5c: Empty string video_id returns 422
# ---------------------------------------------------------------------------

def test_empty_string_video_id_returns_422(client: TestClient) -> None:
    """
    **Validates: Requirements 4.3**

    A request body with video_id set to "" SHALL return HTTP 422.
    """
    response = client.post("/generate", json={"video_id": ""})
    assert response.status_code == 422, (
        f"Expected 422 for empty string video_id, got {response.status_code}. "
        f"Response body: {response.text}"
    )


# ---------------------------------------------------------------------------
# Property 5d: Whitespace-only video_id returns 422 (example-based)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("whitespace_id", [
    " ",
    "   ",
    "\t",
    "\n",
    "\r\n",
    "  \t  \n  ",
])
def test_whitespace_video_id_returns_422(
    client: TestClient, whitespace_id: str
) -> None:
    """
    **Validates: Requirements 4.3**

    A request body with a whitespace-only video_id SHALL return HTTP 422.
    """
    response = client.post("/generate", json={"video_id": whitespace_id})
    assert response.status_code == 422, (
        f"Expected 422 for whitespace video_id {whitespace_id!r}, "
        f"got {response.status_code}. Response body: {response.text}"
    )


# ---------------------------------------------------------------------------
# Property 5 (PBT): All invalid video_id values return 422
# ---------------------------------------------------------------------------

@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(invalid_video_id=_invalid_video_id_strategy)
def test_invalid_video_id_always_returns_422(invalid_video_id) -> None:
    """
    **Validates: Requirements 4.3**

    For any request body where video_id is null, an empty string, or a
    whitespace-only string, the /generate endpoint SHALL return HTTP 422.

    Uses a fresh TestClient per test invocation to avoid state leakage.
    """
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/generate", json={"video_id": invalid_video_id})
        assert response.status_code == 422, (
            f"Expected 422 for invalid video_id={invalid_video_id!r}, "
            f"got {response.status_code}. Response body: {response.text}"
        )


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    extra_fields=st.dictionaries(
        keys=st.text(min_size=1, max_size=20).filter(lambda k: k != "video_id"),
        values=st.one_of(st.text(), st.integers(), st.booleans(), st.none()),
        max_size=5,
    )
)
def test_missing_video_id_with_extra_fields_returns_422(extra_fields: dict) -> None:
    """
    **Validates: Requirements 4.3**

    A request body that contains other fields but is missing video_id entirely
    SHALL return HTTP 422, regardless of what other fields are present.
    """
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/generate", json=extra_fields)
        assert response.status_code == 422, (
            f"Expected 422 for body without video_id (extra_fields={extra_fields!r}), "
            f"got {response.status_code}. Response body: {response.text}"
        )
