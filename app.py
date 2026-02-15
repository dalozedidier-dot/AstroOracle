from __future__ import annotations

from astrooracle.api import create_app
from astrooracle.config import OracleConfig

# Uvicorn entrypoint:
#   uvicorn app:app --reload
# or via CLI:
#   astrooracle serve --host 127.0.0.1 --port 8000
cfg = OracleConfig.default()
app = create_app(cfg)

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
