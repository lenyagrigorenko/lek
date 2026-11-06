"""Stage 2: blend stored OOF predictions, evaluate the full pipeline on the competition metric and write the submission.

    p_hat(x)  = calibrated blend of the clf models              (P[SI == 10000])
    mu_hat(x) = blend of the reg models                         (log10 SI if the record is NOT a sentinel)
    M(x)      = retransform(mu_hat)                             (E[SI | x, non-sentinel])
    SI_hat(x) = p_hat * 10000 + (1 - p_hat) * M(x)              (the MSE-optimal point prediction)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ..cv.metrics import mse
from ..cv.runner import exp_dir, exp_tag
from ..data.prepare import load_all_table
from ..data.schema import SENTINEL_VALUE
from ..features.meta import build_meta
from ..paths import REPORTS, SUBMISSIONS, ensure_dirs, resolve
from .blend import (RETRANSFORMS, MeanBlend, fit_clf_stack, fit_reg_stack, nested_scores_clf, sigmoid, logit)

log = logging.getLogger(__name__)


def _load_task(cfg: dict, task: str, names=None):
    d = exp_dir(cfg) / task
    store = {}
    for npz in sorted(d.glob("*.npz")):
        n = npz.stem
        if names and n not in names:
            continue
        z = np.load(npz)
        t_fold = z["test"].mean(axis=0)
        t_full = z["test_full"].mean(axis=0) if "test_full" in z.files and len(z["test_full"]) else t_fold
        mode = (cfg.get("ensemble", {}) or {}).get("test_source", "mix")
        t = {"folds": t_fold, "full": t_full, "mix": 0.5 * (t_fold + t_full)}[mode]
        store[n] = (np.nanmean(z["oof"], axis=0), t)
    return store


def _score_single(task, store, y, ns_mask, ly):
    rows = []
    for n, (oof, _) in store.items():
        if task == "clf":
            rows.append((n, float(np.mean((oof - y) ** 2))))
        else:
            rows.append((n, float(np.sqrt(np.mean((oof[ns_mask] - ly) ** 2)))))
    return sorted(rows, key=lambda t: t[1])


def run_ensemble(cfg: dict, write_submission: bool = True) -> dict:
    ensure_dirs()
    ecfg = cfg.get("ensemble", {}) or {}
    df = load_all_table(cfg)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    y = tr["is_sentinel"].values.astype(float)
    si = tr["si"].values
    ns = y == 0
    ly = tr["log_si"].values[ns]
    seed = int(cfg.get("seed", 0))

    grp_all = build_meta(df)[0]["c_source_norm"].values           # source series (heavy-tailed series need their own factor)
    grp_tr, grp_te = grp_all[(df["split"] == "train").values], grp_all[(df["split"] == "test").values]
    clf = _load_task(cfg, "clf", ecfg.get("clf_models"))
    reg = _load_task(cfg, "reg", ecfg.get("reg_models"))
    if not clf or not reg:
        raise RuntimeError("run `vacv-si cv` first: no stored OOF predictions for clf and/or reg")
    out_tag = ecfg.get("out_tag") or exp_tag(cfg)       # name used for report / submission files (diagnostic runs differ)
    report: dict = {"tag": out_tag}

    # ---------------------------------------------------------------- classification blend
    singles = _score_single("clf", clf, y, ns, ly)
    print("\n== sentinel classifier: single models (Brier, 1e8*Brier ~ MSE share) ==")
    for n, s in singles:
        print(f"  {n:20s} brier={s:.4f}")
    ranked = [n for n, _ in singles]
    P = np.column_stack([clf[n][0] for n in ranked])
    Pt = np.column_stack([clf[n][1] for n in ranked])
    strategies: dict[str, tuple] = {}
    kmax = len(ranked)
    for k in sorted({1, 3, 5, 8, kmax}):
        if k <= kmax:
            cols = list(range(k))
            strategies[f"mean_top{k}"] = (lambda Pa, ya, c=cols: MeanBlend(c), cols)
    for l2 in ecfg.get("l2_grid", [0.0, 0.01, 0.1, 1.0]):
        strategies[f"stack_l2={l2}"] = ((lambda Pa, ya, l=l2: fit_clf_stack(Pa, ya, l2=l, loss=ecfg.get("loss", "brier"))), None)
    results = {}
    print("\n== sentinel classifier: blends (nested CV Brier on OOF) ==")
    for name, (fit_fn, _) in strategies.items():
        sc = nested_scores_clf(P, y, fit_fn, seed=seed)
        results[name] = (float(sc.mean()), float(sc.std()))
        print(f"  {name:16s} brier={sc.mean():.4f} ± {sc.std():.4f}")
    best_clf = min(results, key=lambda k: results[k][0])
    clf_blender = strategies[best_clf][0](P, y)
    p_oof = clf_blender.predict(P)
    p_test = clf_blender.predict(Pt)
    print(f"  -> chosen: {best_clf}  (in-sample brier {np.mean((p_oof - y) ** 2):.4f})")
    if hasattr(clf_blender, "w"):
        print("     weights:", {n: round(float(w), 3) for n, w in zip(ranked, clf_blender.w) if w > 1e-3}, "b=%.3f" % clf_blender.b)
    report["clf"] = {"chosen": best_clf, "nested_brier": results, "singles": dict(singles)}

    # ---------------------------------------------------------------- regression blend
    singles_r = _score_single("reg", reg, y, ns, ly)
    print("\n== log10(SI) regressors: single models (RMSE on non-sentinel rows) ==")
    for n, s in singles_r:
        print(f"  {n:20s} rmse_log={s:.4f}")
    ranked_r = [n for n, _ in singles_r]
    R = np.column_stack([reg[n][0] for n in ranked_r])
    Rt = np.column_stack([reg[n][1] for n in ranked_r])
    Rn = R[ns]
    rstrat = {}
    for k in sorted({1, 3, 5, kmax}):
        if k <= len(ranked_r):
            rstrat[f"mean_top{k}"] = lambda Ra, ya, k=k: type("MB", (), {"predict": staticmethod(lambda X, k=k: X[:, :k].mean(1))})()
    for l2 in ecfg.get("l2_grid", [0.0, 0.01, 0.1, 1.0]):
        rstrat[f"stack_l2={l2}"] = lambda Ra, ya, l=l2: fit_reg_stack(Ra, ya, l2=l)
    rres = {}
    print("\n== log10(SI) regressors: blends (nested CV RMSE) ==")
    for name, fit_fn in rstrat.items():
        errs = []
        for r in range(5):
            pred = np.zeros(len(ly))
            for a, b in KFold(5, shuffle=True, random_state=seed + r).split(Rn):
                pred[b] = fit_fn(Rn[a], ly[a]).predict(Rn[b])
            errs.append(np.sqrt(np.mean((pred - ly) ** 2)))
        rres[name] = (float(np.mean(errs)), float(np.std(errs)))
        print(f"  {name:16s} rmse_log={np.mean(errs):.4f} ± {np.std(errs):.4f}")
    best_reg = min(rres, key=lambda k: rres[k][0])
    reg_blender = rstrat[best_reg](Rn, ly)
    mu_oof, mu_test = reg_blender.predict(R), reg_blender.predict(Rt)
    print(f"  -> chosen: {best_reg}")
    report["reg"] = {"chosen": best_reg, "nested_rmse": rres, "singles": dict(singles_r)}

    # ---------------------------------------------------------------- retransformation + full pipeline metric
    print("\n== full pipeline: raw MSE on training rows (nested over the retransform calibration) ==")
    rt_results = {}
    for name, make in RETRANSFORMS.items():
        mses = []
        for r in range(8):
            Moof = np.zeros(len(si))
            for a, b in KFold(5, shuffle=True, random_state=seed + 100 + r).split(si):
                a_ns = a[ns[a]]
                f = make().fit(mu_oof[a_ns], si[a_ns], grp_tr[a_ns])
                Moof[b] = f(mu_oof[b], grp_tr[b])
            mses.append(mse(si, p_oof * SENTINEL_VALUE + (1 - p_oof) * Moof))
        rt_results[name] = (float(np.mean(mses)), float(np.std(mses)))
        print(f"  {name:10s} MSE={np.mean(mses):>12,.0f} ± {np.std(mses):>9,.0f}   RMSE={np.sqrt(np.mean(mses)):8.1f}")
    forced = ecfg.get("retransform", "auto")
    best_rt = min(rt_results, key=lambda k: rt_results[k][0]) if forced == "auto" else forced
    retrans = RETRANSFORMS[best_rt]().fit(mu_oof[ns], si[ns], grp_tr[ns])
    M_oof, M_test = retrans(mu_oof, grp_tr), retrans(mu_test, grp_te)
    final_oof = p_oof * SENTINEL_VALUE + (1 - p_oof) * M_oof
    ideal = np.where(ns, M_oof, SENTINEL_VALUE)          # perfect sentinel classifier, same M
    print(f"  -> retransform: {best_rt}")
    print(f"  OOF raw MSE = {mse(si, final_oof):,.0f}   (constant-mean baseline: {mse(si, np.full(len(si), si.mean())):,.0f};"
          f" perfect-classifier floor with this M: {mse(si, ideal):,.0f})")
    print(f"  split of the error: sentinel rows {np.sum((si - final_oof)[~ns] ** 2) / len(si):,.0f} + non-sentinel rows "
          f"{np.sum((si - final_oof)[ns] ** 2) / len(si):,.0f}")
    report["pipeline"] = {"retransform": best_rt, "retransform_cv": rt_results, "oof_mse": mse(si, final_oof)}

    # ---------------------------------------------------------------- submission
    final_test = p_test * SENTINEL_VALUE + (1 - p_test) * M_test
    print(f"\n  test: mean p={p_test.mean():.3f} (train sentinel rate {y.mean():.3f}) | mean SI_hat={final_test.mean():,.0f} "
          f"(train mean {si.mean():,.0f}) | p>0.5: {(p_test > 0.5).mean():.3f}")
    ensure_dirs()
    sub = pd.DataFrame({"index": te["test_index"].astype(int).values, "SI VACV": final_test})
    sub = sub.sort_values("index").reset_index(drop=True)
    _validate_submission(sub, cfg)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"ensemble_{out_tag}.json").write_text(json.dumps(report, indent=1, default=float))
    pd.DataFrame({"row_id": tr["row_id"], "si": si, "p": p_oof, "M": M_oof, "pred": final_oof}).to_csv(
        REPORTS / f"oof_final_{out_tag}.csv", index=False)
    if write_submission:
        path = SUBMISSIONS / f"submission_{out_tag}.csv"
        sub.to_csv(path, index=False)
        print(f"  wrote {path}")
    return report


def _validate_submission(sub: pd.DataFrame, cfg: dict) -> None:
    sample = pd.read_csv(resolve(cfg.get("paths", {}).get("raw_dir", "data/raw")) / "sample_submit.csv")
    assert list(sub.columns) == list(sample.columns), (list(sub.columns), list(sample.columns))
    assert len(sub) == len(sample), (len(sub), len(sample))
    assert set(sub["index"]) == set(sample["index"]), "index set differs from sample_submit.csv"
    assert np.isfinite(sub["SI VACV"]).all(), "non-finite predictions"
    assert (sub["SI VACV"] >= 0).all(), "negative predictions"
