"""Central place for filesystem locations (all relative to the repository root)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
EXTERNAL_RAW = EXTERNAL / "raw"          # cached API responses (git-ignored)
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
FEATURE_CACHE = CACHE / "features"

ARTIFACTS = ROOT / "artifacts"           # OOF / test predictions of every model (git-ignored)
OOF_DIR = ARTIFACTS / "oof"
REPORTS = ROOT / "reports"
SUBMISSIONS = ROOT / "submissions"
CONFIGS = ROOT / "configs"


def ensure_dirs() -> None:
    for p in (RAW, EXTERNAL, EXTERNAL_RAW, PROCESSED, CACHE, FEATURE_CACHE, OOF_DIR, REPORTS, SUBMISSIONS):
        p.mkdir(parents=True, exist_ok=True)


def resolve(p: str | Path) -> Path:
    """Resolve a (possibly relative) path from a config against the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
