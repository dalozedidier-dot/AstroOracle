from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "AstroOracle"
author = "AstroOracle Contributors"
current_year = datetime.utcnow().year
copyright = f"{current_year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "AstroOracle"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

autodoc_typehints = "description"
