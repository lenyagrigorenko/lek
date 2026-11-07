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
from ..cv.runner import RAW_SCALE, exp_dir, exp_tag
from ..data.prepare import load_all_table
from ..data.schema import SENTINEL_VALUE
from ..features.meta import build_meta
from ..paths import OOF_DIR, REPORTS, SUBMISSIONS, ensure_dirs, resolve
from .blend import (RETRANSFORMS, TopKIsoBlend, TopKMeanBlend, TopKRegMean, fit_clf_stack, fit_reg_stack, nested_scores_clf)

log = logging.getLogger(__name__)


def _load_task(cfg: dict, task: str, names=None):
    """Stored OOF / test predictions of this config's own tag plus, with `ensemble.include_tags: [tag, ...]`, those of other
    tags (e.g. the baseline feature tier an embedding experiment builds on). A name found in the own tag wins."""
    ecfg = cfg.get("ensemble", {}) or {}
    dirs = [exp_dir(cfg) / task] + [OOF_DIR / str(t) / task for t in ecfg.get("include_tags") or []]
    mode = ecfg.get("test_source", "mix")
    store = {}
    for d in dirs:
        for npz in sorted(d.glob("*.npz")):
            n = npz.stem
            if n in store or (names and n not in names):
                continue
            z = np.load(npz)
            t_fold = z["test"].mean(axis=0)
            t_full = z["test_full"].mean(axis=0) if "test_full" in z.files and len(z["test_full"]) else t_fold
            t = {"folds": t_fold, "full": t_full, "mix": 0.5 * (t_fold + t_full)}[mode]
            store[n] = (np.nanmean(z["oof"], axis=0), t)
    return store


def _score_single(task, store, y, ns_mask, ly, si=None):
    rows = []
    for n, (oof, _) in store.items():
        if task == "clf":
            rows.append((n, float(np.mean((oof - y) ** 2))))
        elif task == "rawns":
            rows.append((n, float(np.mean((oof[ns_mask] - si[ns_mask]) ** 2))))
        elif task == "raw":
            rows.append((n, float(np.mean((oof * RAW_SCALE - si) ** 2))))
        else:
            rows.append((n, float(np.sqrt(np.mean((oof[ns_mask] - ly) ** 2)))))
    return sorted(rows, key=lambda t: t[1])


def _mix_search(si: np.ndarray, base: np.ndarray, alts: list, seed: int = 0, grid=(0.0, 0.25, 0.5, 0.75, 1.0), n_rep: int = 8):
    """Search mixing weights  F = (1 - sum w) * base + sum w_i * alt_i  on the OOF predictions.

    Returns the weight tuple with the lowest full-data MSE, its nested-CV MSE (weights chosen on 4/5 of the rows,
    scored on the rest), the baseline MSE and the per-combination table. Nesting keeps the estimate honest.
    """
    import itertools

    k = len(alts)
    combos = [w for w in itertools.product(grid, repeat=k) if sum(w) <= 1.0 + 1e-9]
    errs = {w: (si - ((1.0 - sum(w)) * base + sum(wi * a for wi, a in zip(w, alts)))) ** 2 for w in combos}
    nested = []
    for r in range(n_rep):
        tot = 0.0
        for a, b in KFold(5, shuffle=True, random_state=seed + 300 + r).split(si):
            best = min(combos, key=lambda w: errs[w][a].mean())
            tot += errs[best][b].sum()
        nested.append(tot / len(si))
    zero = (0.0,) * k
    best_full = min(combos, key=lambda w: errs[w].mean())
    worst = int(np.argmax(errs[zero]))                      # the single most expensive row of the baseline
    robust = {w: float(np.delete(errs[w], worst).mean()) for w in combos}
    return best_full, float(np.mean(nested)), float(errs[zero].mean()), {w: float(errs[w].mean()) for w in combos}, robust


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
    for k in sorted({1, 3, 5, 8, 12, kmax}):          # model selection happens INSIDE fit(), hence the nested CV is honest
        if k <= kmax:
            strategies[f"top{k}"] = (lambda Pa, ya, kk=k: TopKMeanBlend(kk).fit(Pa, ya), None)
            if k in (3, 5, 8, 12):
                strategies[f"top{k}+iso"] = (lambda Pa, ya, kk=k: TopKIsoBlend(kk).fit(Pa, ya), None)
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
    # cross-fitted probabilities (the blender is refitted inside every fold): honest scores + choice of the clipping level.
    # The costly mistakes are confident ones (p ~ 0.95 on a non-sentinel row costs ~1e8 * 0.9), so p is clipped to [eps, 1-eps].
    PP = []
    for r in range(8):
        pp = np.zeros(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed + 500 + r).split(P, y):
            pp[b] = strategies[best_clf][0](P[a], y[a]).predict(P[b])
        PP.append(pp)
    brier_eps = {e: float(np.mean([np.mean((np.clip(pp, e, 1 - e) - y) ** 2) for pp in PP])) for e in (0.0, 0.003, 0.01, 0.02, 0.03, 0.05)}
    eps = min(brier_eps, key=brier_eps.get)
    if brier_eps[eps] > 0.99 * brier_eps[0.0]:               # adopt only a clear (>1%) gain
        eps = 0.0
    print("  clipping p to [eps, 1-eps], cross-fitted Brier:", {e: round(v, 5) for e, v in brier_eps.items()}, "-> eps =", eps)
    if eps:
        p_oof, p_test = np.clip(p_oof, eps, 1 - eps), np.clip(p_test, eps, 1 - eps)
        PP = [np.clip(pp, eps, 1 - eps) for pp in PP]
    print(f"  -> chosen: {best_clf}  (in-sample brier {np.mean((p_oof - y) ** 2):.4f})")
    if hasattr(clf_blender, "w"):
        print("     weights:", {n: round(float(w), 3) for n, w in zip(ranked, clf_blender.w) if w > 1e-3}, "b=%.3f" % clf_blender.b)
    if hasattr(clf_blender, "cols"):
        print("     members:", [ranked[i] for i in clf_blender.cols])
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
    for k in sorted({1, 3, 5, len(ranked_r)}):
        if k <= len(ranked_r):
            rstrat[f"top{k}"] = lambda Ra, ya, kk=k: TopKRegMean(kk).fit(Ra, ya)
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

    # ---- HONEST end-to-end estimate: the sentinel blender and the retransformation are re-fitted inside every CV fold
    cf = []
    for r in range(8):
        MM = np.zeros(len(si))
        for a, b in KFold(5, shuffle=True, random_state=seed + 600 + r).split(si):
            a_ns = a[ns[a]]
            MM[b] = RETRANSFORMS[best_rt]().fit(mu_oof[a_ns], si[a_ns], grp_tr[a_ns])(mu_oof[b], grp_tr[b])
        cf.append(mse(si, PP[r] * SENTINEL_VALUE + (1 - PP[r]) * MM))
    print(f"  HONEST cross-fitted OOF raw MSE = {np.mean(cf):,.0f} ± {np.std(cf):,.0f}   (RMSE {np.sqrt(np.mean(cf)):,.0f})")
    report["pipeline"]["oof_mse_crossfit"] = float(np.mean(cf))
    report["pipeline"]["clip_eps"] = eps

    final_test = p_test * SENTINEL_VALUE + (1 - p_test) * M_test

    # ---------------------------------------------------------------- optional members: raw-scale models of the mean
    if ecfg.get("use_raw_models", True):
        k_top = int(ecfg.get("raw_top_k", 3))
        alts_oof, alts_test, alts_names = [], [], []
        rawns = _load_task(cfg, "rawns", ecfg.get("rawns_models"))
        if rawns:
            sr = _score_single("rawns", rawns, y, ns, ly, si)
            print("\n== raw-scale models of E[SI | non-sentinel] (MSE on non-sentinel rows, M) ==")
            for n, v in sr:
                print(f"  {n:20s} mse_ns={v / 1e6:8.3f}M")
            top = [n for n, _ in sr[:k_top]]
            Mo = np.clip(np.mean([rawns[n][0] for n in top], axis=0), 0, None)
            Mt = np.clip(np.mean([rawns[n][1] for n in top], axis=0), 0, None)
            alts_oof.append(p_oof * SENTINEL_VALUE + (1 - p_oof) * Mo)
            alts_test.append(p_test * SENTINEL_VALUE + (1 - p_test) * Mt)
            alts_names.append("rawns[" + "+".join(top) + "]")
        raw = _load_task(cfg, "raw", ecfg.get("raw_models"))
        if raw:
            sr = _score_single("raw", raw, y, ns, ly, si)
            print("\n== direct raw-scale models of SI (MSE on all rows) ==")
            for n, v in sr:
                print(f"  {n:20s} mse={v / 1e6:8.3f}M")
            top = [n for n, _ in sr[:k_top]]
            alts_oof.append(np.clip(np.mean([raw[n][0] for n in top], axis=0) * RAW_SCALE, 0, None))
            alts_test.append(np.clip(np.mean([raw[n][1] for n in top], axis=0) * RAW_SCALE, 0, None))
            alts_names.append("raw[" + "+".join(top) + "]")
        if alts_oof:
            w, nested_mse, base_mse, table, robust = _mix_search(si, final_oof, alts_oof, seed)
            zero = (0.0,) * len(alts_oof)
            print("\n== mixing the two-part prediction with the raw-scale members ==")
            print("  members:", alts_names)
            for combo in sorted(table, key=table.get)[:6]:
                print(f"  weights {combo}  MSE={table[combo]:>12,.0f}  without-worst-row={robust[combo]:>12,.0f}")
            print(f"  baseline (two-part only): MSE={base_mse:,.0f} | best combo {w}: nested-CV MSE={nested_mse:,.0f}")
            adopt = nested_mse < 0.995 * base_mse and robust[w] <= 1.01 * robust[zero]
            print(f"  -> {'ADOPTED' if adopt else 'not adopted'} (needs nested MSE < 0.995 x baseline and no loss on typical rows)")
            report["raw_mix"] = {"members": alts_names, "weights": list(w), "nested_mse": nested_mse, "baseline_mse": base_mse,
                                 "adopted": bool(adopt)}
            if adopt:
                final_oof = (1.0 - sum(w)) * final_oof + sum(wi * a for wi, a in zip(w, alts_oof))
                final_test = (1.0 - sum(w)) * final_test + sum(wi * a for wi, a in zip(w, alts_test))
                report["pipeline"]["oof_mse"] = mse(si, final_oof)
                print(f"  OOF raw MSE after mixing = {mse(si, final_oof):,.0f}")

    # ---------------------------------------------------------------- the metric hangs on a handful of rows: show it without the worst one
    worst = int(np.argmax((si - final_oof) ** 2))
    rest = np.delete(np.arange(len(si)), worst)
    top5 = np.sort((si - final_oof) ** 2)[::-1][:5].sum() / np.sum((si - final_oof) ** 2)
    print(f"\n  metric without the single worst row ({tr['title'].iloc[worst]}, SI={si[worst]:,.0f}): MSE {mse(si[rest], final_oof[rest]):,.0f} "
          f"vs constant-mean {mse(si[rest], np.full(len(rest), si[rest].mean())):,.0f}; the 5 worst rows carry {100 * top5:.0f}% of the squared error")
    report["pipeline"]["mse_without_worst_row"] = mse(si[rest], final_oof[rest])
    report["pipeline"]["const_mse_without_worst_row"] = mse(si[rest], np.full(len(rest), si[rest].mean()))
    report["pipeline"]["share_of_error_top5_rows"] = float(top5)

    # ---------------------------------------------------------------- submission
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
