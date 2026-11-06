"""YAML configs with inheritance (`extends:`) and dotted command-line overrides.

    cfg = load_config("configs/exp/fast.yaml", overrides=["cv.n_repeats=2", "external.use=true"])
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import ROOT


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge `upd` into a copy of `base` (dicts are merged, everything else replaced)."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = _read((path.parent / parent).resolve())
        cfg = deep_update(base, cfg)
    return cfg


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def get_by_path(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    cfg = _read(path)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"override must look like a.b.c=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    cfg["_config_path"] = str(path)
    return cfg
