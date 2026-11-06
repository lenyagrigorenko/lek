"""Repeated-CV runner.

For every model of every task it stores, under  artifacts/oof/<tag>/<task>/<name>.{npz,json}:
    oof   (n_repeats, n_train)    out-of-fold predictions (one full pass over the training rows per repeat)
    test  (n_repeats*n_splits, n_test)   predictions of every fold model on the test rows
so that the ensembling stage never has to re-train anything.

Tasks
    clf : P(SI == 10000)                           trained on all labelled rows
    reg : log10(SI) of non-sentinel records        trained on non-sentinel rows, predicted for every row
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..data.prepare import load_all_table
from ..features.store import FEATURE_CODE_VERSION, FeatureStore
from ..models import build_model
from ..paths import OOF_DIR, ensure_dirs
from .folds import Fold, make_fold_plan
from .metrics import clf_metrics, proxy_raw_mse, reg_metrics

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------------------------ setup
def exp_tag(cfg: dict) -> str:
    return cfg.get("tag") or Path(cfg.get("_config_path", "base.yaml")).stem


def exp_dir(cfg: dict) -> Path:
    return OOF_DIR / exp_tag(cfg)


def build_context(cfg: dict):
    ensure_dirs()
    df = load_all_table(cfg)
    store = FeatureStore(df, cfg["features"], n_jobs=int(cfg.get("n_jobs", 8)))
    cat, _ = store.meta()
    plan = make_fold_plan(store.df, cat["c_source_norm"], cfg["cv"])
    return store, plan


def _expand(names, fcfg: dict) -> list[str]:
    groups = fcfg.get("groups") or {}
    out: list[str] = []
    for n in names or []:
        out.extend(_expand(groups[n[1:]], fcfg) if isinstance(n, str) and n.startswith("@") else [n])
    return out


def _referenced_feature_specs(spec: dict, fcfg: dict) -> dict:
    blocks = _expand(spec.get("blocks"), fcfg)
    kernels = []
    if spec.get("kernel"):
        kernels.append(spec["kernel"]["base"])
    if spec.get("knn"):
        kernels.extend(spec["knn"].get("kernels") or [spec["knn"]["kernel"]])
    out = {"blocks": {b: fcfg["blocks"][b] for b in blocks}, "kernels": {}}
    for k in kernels:
        kspec = fcfg["kernels"][k]
        out["kernels"][k] = {"kernel": kspec, "block": fcfg["blocks"][kspec["block"]]}
    return out


def model_hash(task: str, spec: dict, cfg: dict, store: FeatureStore) -> str:
    h = hashlib.sha1()
    payload = {
        "task": task,
        "spec": spec,
        "feat": _referenced_feature_specs(spec, cfg["features"]),
        "cv": cfg["cv"],
        "ext": cfg.get("external", {}),
        "seed": cfg.get("seed", 0),
        "code": FEATURE_CODE_VERSION,
        "data": hashlib.sha1(("|".join(store.df["row_id"] + store.df["ik14"].fillna(""))).encode()
                             + np.nan_to_num(store.df["si"].values.astype(float), nan=-1.0).tobytes()).hexdigest(),
    }
    h.update(json.dumps(payload, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def _warm(store: FeatureStore, spec: dict) -> None:
    """Touch everything a model will need *before* going multi-threaded (the store is lazy and not thread-safe)."""
    for b in store.expand(spec.get("blocks") or []):
        store.block(b)
    store.meta()
    if spec.get("kernel"):
        from ..models.kernel_models import composite_kernel

        composite_kernel(store, spec["kernel"])
    if spec.get("knn"):
        for kname in spec["knn"].get("kernels") or [spec["knn"]["kernel"]]:
            store.kernel(kname)
        store.group_codes(spec["knn"].get("source_col", "c_source_raw"))


# ------------------------------------------------------------------------------------------------------------------ one CV
def _label(df: pd.DataFrame, task: str) -> np.ndarray:
    return (df["is_sentinel"] if task == "clf" else df["log_si"]).values.astype(float)


def _fit_rows(store: FeatureStore, task: str, fold: Fold, ext_rows: np.ndarray, test_rows: np.ndarray, cfg: dict):
    """Rows to train on (+ optional reference-only rows) for one fold.

    external.mode = train_rows      external records are ordinary (down-weighted) training rows
                    reference_only  external records are only *neighbours* (a separate pool of neighbour features);
                                    models are trained on competition rows only
    """
    df = store.df
    rows = fold.train
    ext_cfg = cfg.get("external", {})
    keep_ext = np.zeros(0, dtype=np.int64)
    if ext_cfg.get("use") and len(ext_rows):
        keep = np.ones(len(ext_rows), dtype=bool)
        if ext_cfg.get("drop_overlap_with_eval", True):
            banned = pd.unique(np.concatenate([df["ik14"].values[fold.val], df["ik14"].values[test_rows]]))
            keep = ~np.isin(df["ik14"].values[ext_rows], banned)
        keep_ext = ext_rows[keep]
    lab = _label(df, task)
    ref = None
    if len(keep_ext) and ext_cfg.get("mode", "train_rows") == "reference_only":
        ok_r = np.isfinite(lab[keep_ext])
        ref = (keep_ext[ok_r], lab[keep_ext][ok_r])
    elif len(keep_ext):
        rows = np.concatenate([rows, keep_ext])
    y = lab[rows]
    ok = np.isfinite(y)
    w = df["weight"].values[rows].copy()
    if "weight" in ext_cfg:                                   # model-level override of the external sample weight
        w[df["split"].values[rows] == "ext"] = float(ext_cfg["weight"])
    return rows[ok], y[ok], w[ok], ref


def model_external_cfg(spec: dict, cfg: dict) -> dict:
    """Effective external-data settings of one model: global `external:` overridden by the model's own `external:`.

        external: false              -> the model ignores external records
        external: reference_only     -> shorthand for {mode: reference_only}
        external: {mode: train_rows, weight: 0.5}
    """
    ext = dict(cfg.get("external", {}))
    o = spec.get("external")
    if o is False:
        ext["use"] = False
    elif isinstance(o, str):
        ext["mode"] = o
    elif isinstance(o, dict):
        ext.update(o)
    return ext


def _prepare_spec(spec: dict, cfg: dict) -> dict:
    """Inject the external-data policy into neighbour-feature specs (pools + same-molecule masking)."""
    ext_cfg = cfg.get("external", {})
    if not spec.get("knn") or not ext_cfg.get("use"):
        return spec
    spec = {**spec, "knn": dict(spec["knn"])}
    if not spec["knn"].get("pools"):
        if ext_cfg.get("mode", "train_rows") == "reference_only":
            spec["knn"]["pools"] = [{"name": "comp", "rows": "train"}, {"name": "ext", "rows": "ext"}]
        else:
            spec["knn"]["pools"] = [{"name": "all"}]
    strict = bool(ext_cfg.get("drop_overlap_with_eval", True))
    spec["knn"]["pools"] = [{**p, "hide_same_molecule": p.get("hide_same_molecule", strict and p.get("rows") in (None, "ext"))}
                            if p.get("rows") != "train" else p for p in spec["knn"]["pools"]]
    return spec


def run_cv_one(task: str, name: str, spec: dict, store: FeatureStore, plan: list[Fold], cfg: dict) -> dict:
    df = store.df
    train_rows = np.flatnonzero((df["split"] == "train").values)
    test_rows = np.flatnonzero((df["split"] == "test").values)
    ext_rows = np.flatnonzero((df["split"] == "ext").values)
    pos = np.full(len(df), -1, dtype=np.int64)
    pos[train_rows] = np.arange(len(train_rows))
    n_rep = max(f.repeat for f in plan) + 1
    seed0 = int(cfg.get("seed", 0))
    _warm(store, spec)

    cfg_global = cfg
    cfg = {**cfg, "external": model_external_cfg(spec, cfg)}       # per-model external-data policy
    spec_run = _prepare_spec(spec, cfg)

    def job(j: int, fold: Fold):
        rows, y, w, ref = _fit_rows(store, task, fold, ext_rows, test_rows, cfg)
        model = build_model(store, task, spec_run, seed=seed0 + 1000 * fold.repeat + fold.fold)
        model.fit(rows, y, w, ref=ref)
        return j, model.predict(fold.val), model.predict(test_rows)

    t0 = time.time()
    backend = "threading" if cfg["cv"].get("parallel", "threads") == "threads" else "loky"
    res = Parallel(n_jobs=int(cfg.get("n_jobs", 8)), backend=backend)(delayed(job)(j, f) for j, f in enumerate(plan))
    oof = np.full((n_rep, len(train_rows)), np.nan)
    test_pred = np.zeros((len(plan), len(test_rows)))
    for j, pv, pt in res:
        f = plan[j]
        oof[f.repeat, pos[f.val]] = pv
        test_pred[j] = pt
    elapsed = time.time() - t0
    summary = summarize(task, name, oof, df, train_rows)
    summary.update({"hash": model_hash(task, spec, cfg_global, store), "seconds": round(elapsed, 1), "spec": spec})
    # ---- full-data refits (all training rows, several seeds): used for the test predictions of the final submission
    n_full = int(cfg["cv"].get("full_fit_seeds", 3))
    test_full = np.zeros((0, len(test_rows)))
    if n_full > 0:
        full_fold = Fold(repeat=0, fold=-1, train=train_rows, val=np.zeros(0, dtype=np.int64))

        def job_full(s_i: int):
            rows, y, w, ref = _fit_rows(store, task, full_fold, ext_rows, test_rows, cfg)
            model = build_model(store, task, spec_run, seed=seed0 + 777 + 31 * s_i)
            model.fit(rows, y, w, ref=ref)
            return model.predict(test_rows)

        test_full = np.vstack(Parallel(n_jobs=min(n_full, int(cfg.get("n_jobs", 8))), backend=backend)(
            delayed(job_full)(i) for i in range(n_full)))
        summary["seconds"] = round(time.time() - t0, 1)
    return {"oof": oof, "test": test_pred, "test_full": test_full, "summary": summary,
            "train_rows": train_rows, "test_rows": test_rows}


def summarize(task: str, name: str, oof: np.ndarray, df: pd.DataFrame, train_rows: np.ndarray) -> dict:
    y_s = df["is_sentinel"].values[train_rows].astype(int)
    si = df["si"].values[train_rows]
    ns = y_s == 0
    m0 = float(si[ns].mean())
    per_rep, mean_oof = [], np.nanmean(oof, axis=0)
    if task == "clf":
        for r in range(oof.shape[0]):
            m = clf_metrics(y_s, oof[r])
            m["proxy_mse"] = proxy_raw_mse(si, oof[r], m0)
            per_rep.append(m)
        agg = {k: float(np.mean([m[k] for m in per_rep])) for k in per_rep[0]}
        sd = {k + "_sd": float(np.std([m[k] for m in per_rep])) for k in per_rep[0]}
        bag = clf_metrics(y_s, mean_oof)
        bag["proxy_mse"] = proxy_raw_mse(si, mean_oof, m0)
        agg.update({"bag_" + k: v for k, v in bag.items()})
    else:
        ly = df["log_si"].values[train_rows][ns]
        for r in range(oof.shape[0]):
            per_rep.append(reg_metrics(ly, oof[r][ns]))
        agg = {k: float(np.mean([m[k] for m in per_rep])) for k in per_rep[0]}
        sd = {k + "_sd": float(np.std([m[k] for m in per_rep])) for k in per_rep[0]}
        agg.update({"bag_" + k: v for k, v in reg_metrics(ly, mean_oof[ns]).items()})
    return {"task": task, "name": name, "metrics": {**agg, **sd}}


# ------------------------------------------------------------------------------------------------------------------ driver
def _is_available(spec: dict) -> bool:
    typ = spec["type"]
    try:
        if typ == "lightgbm":
            import lightgbm  # noqa: F401
        elif typ == "xgboost":
            import xgboost  # noqa: F401
    except Exception as e:  # missing libomp etc.
        log.warning("skipping %s model: %s", typ, str(e).splitlines()[0][:90])
        return False
    return True


def run_experiments(cfg: dict, tasks: Optional[list[str]] = None, only: Optional[list[str]] = None, force: bool = False):
    store, plan = build_context(cfg)
    out_root = exp_dir(cfg)
    log.info("tag=%s | rows=%d (train=%d test=%d ext=%d) | folds=%d", exp_tag(cfg), len(store.df),
             (store.df.split == "train").sum(), (store.df.split == "test").sum(), (store.df.split == "ext").sum(), len(plan))
    for task in tasks or [t for t in cfg["models"] if t in ("clf", "reg")]:
        for name, spec in (cfg["models"].get(task) or {}).items():
            if only and name not in only:
                continue
            if spec.get("enabled", True) is False or not _is_available(spec):
                continue
            d = out_root / task
            d.mkdir(parents=True, exist_ok=True)
            h = model_hash(task, spec, cfg, store)
            jpath = d / f"{name}.json"
            if jpath.exists() and not force and json.loads(jpath.read_text()).get("hash") == h:
                log.info("[%s/%s] cached", task, name)
                continue
            res = run_cv_one(task, name, spec, store, plan, cfg)
            np.savez_compressed(d / f"{name}.npz", oof=res["oof"], test=res["test"], test_full=res["test_full"],
                                train_rows=res["train_rows"], test_rows=res["test_rows"])
            jpath.write_text(json.dumps(res["summary"], indent=1, default=str))
            m = res["summary"]["metrics"]
            if task == "clf":
                print(f"[clf/{name:18s}] auc={m['auc']:.4f} brier={m['brier']:.4f} logloss={m['logloss']:.4f} "
                      f"proxy_mse={m['proxy_mse']/1e6:7.2f}M (bag {m['bag_proxy_mse']/1e6:7.2f}M)  {res['summary']['seconds']}s", flush=True)
            else:
                print(f"[reg/{name:18s}] rmse_log={m['rmse_log']:.4f} mae={m['mae_log']:.4f} spearman={m['spearman']:.4f} "
                      f"(bag rmse {m['bag_rmse_log']:.4f})  {res['summary']['seconds']}s", flush=True)
    print_leaderboard(cfg)


def load_summaries(cfg: dict) -> pd.DataFrame:
    rows = []
    for jp in sorted(exp_dir(cfg).glob("*/*.json")):
        s = json.loads(jp.read_text())
        rows.append({"task": s["task"], "name": s["name"], **s["metrics"], "seconds": s.get("seconds")})
    return pd.DataFrame(rows)


def print_leaderboard(cfg: dict) -> None:
    t = load_summaries(cfg)
    if t.empty:
        print("no stored models yet")
        return
    pd.set_option("display.width", 200)
    for task, key, cols in [("clf", "proxy_mse", ["name", "auc", "brier", "logloss", "proxy_mse", "bag_proxy_mse", "proxy_mse_sd"]),
                            ("reg", "rmse_log", ["name", "rmse_log", "mae_log", "spearman", "bag_rmse_log", "rmse_log_sd"])]:
        sub = t[t.task == task].sort_values(key)
        if len(sub):
            print(f"\n=== {task.upper()} leaderboard ({exp_tag(cfg)}) ===")
            show = sub[cols].copy()
            for c in ("proxy_mse", "bag_proxy_mse", "proxy_mse_sd"):
                if c in show:
                    show[c] = (show[c] / 1e6).round(2)
            print(show.round(4).to_string(index=False))
