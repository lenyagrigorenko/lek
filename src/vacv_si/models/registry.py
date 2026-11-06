"""Model registry: maps the `type:` field of a model spec to a class.

To add a model family: subclass `BaseModel` (see base.py), then add it to MODEL_REGISTRY below (or call
`register_model("my_type")(MyClass)` from your own module before running the CV).
"""
from __future__ import annotations

import logging

from .base import BaseModel
from .kernel_models import KERNEL_MODELS
from .matrix_models import MATRIX_MODELS

log = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type] = {}
MODEL_REGISTRY.update(MATRIX_MODELS)
MODEL_REGISTRY.update(KERNEL_MODELS)


def register_model(name: str):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return deco


def build_model(store, task: str, spec: dict, seed: int = 0) -> BaseModel:
    typ = spec["type"]
    if typ not in MODEL_REGISTRY:
        raise KeyError(f"unknown model type {typ!r}; known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[typ](store, task, spec, seed)
