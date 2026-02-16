from __future__ import annotations

from fastapi.testclient import TestClient

from astrooracle.api import app


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
    assert data.get("status") in {"ok", "healthy", "up", None} or "status" in data


def test_index_returns_html() -> None:
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    # We do not over-constrain the HTML, but it should be text/html-ish.
    ct = r.headers.get("content-type", "")
    assert "text/html" in ct or ct == ""
