from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


def normalize_to_uint8(img2d: np.ndarray, *, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Normalize a 2D float array to uint8 using robust percentiles."""

    arr = np.asarray(img2d, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = float(np.percentile(finite, p_low))
    hi = float(np.percentile(finite, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0

    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def png_bytes_from_array(img2d: np.ndarray) -> bytes:
    """Encode a 2D array into PNG bytes."""

    u8 = normalize_to_uint8(img2d)
    out = io.BytesIO()
    Image.fromarray(u8).save(out, format="PNG")
    return out.getvalue()


def png_b64_data_uri(img2d: np.ndarray) -> str:
    """Return a data URI (data:image/png;base64,...) for a 2D array."""

    raw = png_bytes_from_array(img2d)
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def strip_data_uri(data_uri: str) -> str:
    """Return only the base64 payload of a data URI."""

    if "," in data_uri:
        return data_uri.split(",", 1)[1]
    return data_uri


def parse_hex_color(color: str) -> Tuple[int, int, int]:
    """Parse '#RRGGBB' into (r,g,b)."""

    s = color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {color}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return r, g, b
