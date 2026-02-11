"""Compatibility shim for the historical ``null_trace`` entrypoint.

EchoNull no longer vendors the ``nulltrace`` package. This module keeps a stable
CLI entrypoint, but degrades gracefully when ``nulltrace`` is not installed.
"""

from __future__ import annotations

import importlib
from typing import Optional, Sequence, cast, Callable, Any


_MainFn = Callable[[Optional[Sequence[str]]], int]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the external ``nulltrace`` entrypoint if available.

    Returns:
        0 on success. Non-zero when ``nulltrace`` is missing or invalid.
    """
    try:
        mod = importlib.import_module("nulltrace.null_trace")
    except ModuleNotFoundError:
        print(
            "nulltrace is not installed. Install it to use this entrypoint, "
            "or run EchoNull via orchestrator/sweep workflows."
        )
        return 2

    main_obj: Any = getattr(mod, "main", None)
    if not callable(main_obj):
        print("nulltrace.null_trace.main is missing or not callable.")
        return 3

    main_fn = cast(_MainFn, main_obj)
    try:
        return int(main_fn(argv))
    except SystemExit as exc:
        # Preserve conventional CLI behavior.
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
