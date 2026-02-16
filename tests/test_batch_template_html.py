from __future__ import annotations

from pathlib import Path


def test_batch_index_is_real_html() -> None:
    # This prevents regressions where batch_index.html is accidentally replaced by plain text.
    p = Path(__file__).resolve().parents[1] / "templates" / "batch_index.html"
    text = p.read_text(encoding="utf-8").lstrip().lower()
    assert text.startswith("<!doctype html") or text.startswith("<html"), text[:80]
    assert "candidates.json" in text
