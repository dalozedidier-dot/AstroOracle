from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:  # pragma: no cover
    Observer = None  # type: ignore
    FileSystemEventHandler = object  # type: ignore


class CandidatesHandler(FileSystemEventHandler):
    def __init__(self, candidates_path: Path, on_change: Callable[[], None]):
        self.candidates_path = candidates_path.resolve()
        self.on_change = on_change

    def on_modified(self, event):
        if getattr(event, "is_directory", False):
            return
        src = Path(getattr(event, "src_path", ""))
        if src and src.resolve() == self.candidates_path:
            self.on_change()

    def on_created(self, event):
        self.on_modified(event)

    def on_moved(self, event):
        dst = Path(getattr(event, "dest_path", ""))
        if dst and dst.resolve() == self.candidates_path:
            self.on_change()


def watch_candidates(candidates_path: Path, on_change: Callable[[], None]) -> None:
    if Observer is None:
        raise RuntimeError("watchdog not installed. Install extras: pip install -e '.[watch]'")

    handler = CandidatesHandler(candidates_path, on_change)
    obs = Observer()
    obs.schedule(handler, str(candidates_path.parent), recursive=False)
    obs.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
        obs.join()
