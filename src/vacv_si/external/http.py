"""Tiny cached + retrying HTTP helper: every response is stored on disk so re-runs are free and reproducible."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests

from ..paths import EXTERNAL_RAW

log = logging.getLogger(__name__)
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "vacv-si-hackathon/0.1 (academic research; contact: see repository)"})


def cached_get(url: str, params: Optional[dict] = None, cache_dir: Path = EXTERNAL_RAW, as_text: bool = False,
               refresh: bool = False, retries: int = 5, sleep: float = 0.25, timeout: int = 120) -> Any:
    params = dict(params or {})
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:20]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.{'txt' if as_text else 'json'}"
    if path.exists() and not refresh:
        txt = path.read_text(encoding="utf-8")
        return txt if as_text else json.loads(txt)
    last = None
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return "" if as_text else {}
            r.raise_for_status()
            time.sleep(sleep)
            path.write_text(r.text, encoding="utf-8")
            return r.text if as_text else r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 2.0 * (attempt + 1)
            log.warning("GET %s failed (%s); retry in %.0fs", url, str(e)[:80], wait)
            time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")
