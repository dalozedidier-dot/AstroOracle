from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version as _pkg_version  # type: ignore

try:
    __version__ = _pkg_version("astrooracle")
except PackageNotFoundError:
    __version__ = "0.3.0"

__all__ = ["__version__"]
