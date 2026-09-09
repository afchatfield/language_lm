"""Project paths and configuration loading.

Every module resolves filesystem locations through this one place so that
nothing hard-codes a relative path and the layout can move without a hunt.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

# `src/langlm/config.py` -> `src/langlm` -> `src` -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
RULES_DIR = PROJECT_ROOT / "rules"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Specific artefact locations, named once here.
FALKO_MERLIN_DIR = RAW_DIR / "falko_merlin"
LEIPZIG_DIR = RAW_DIR / "leipzig"
NGRAMS_DIR = RAW_DIR / "ngrams"
SPLIT_MANIFEST = SPLITS_DIR / "manifest.json"
PHASE0_REPORT_DIR = REPORTS_DIR / "phase0"

# The LanguageTool server from docker/docker-compose.yml.
LT_BASE_URL = "http://localhost:8010"


@cache
def load_config(name: str = "phase0") -> dict[str, Any]:
    """Load `configs/<name>.yaml`."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No config at {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs() -> None:
    """Create the data and report directories if they are missing."""
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        SPLITS_DIR,
        FALKO_MERLIN_DIR,
        LEIPZIG_DIR,
        PHASE0_REPORT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
