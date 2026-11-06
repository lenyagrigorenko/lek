"""Command line entry point:  python -m vacv_si <command> [--config configs/base.yaml] [-o key=value ...]"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def cmd_prepare(cfg, args):
    from .data.prepare import load_all_table

    df = load_all_table(cfg, rebuild=True)
    print(df.groupby("split").size().to_string())
    print("sentinel rate (train):", df.loc[df.split == "train", "is_sentinel"].mean().round(4))


def cmd_features(cfg, args):
    from .data.prepare import load_all_table
    from .features.store import FeatureStore

    df = load_all_table(cfg)
    store = FeatureStore(df, cfg["features"], n_jobs=cfg.get("n_jobs", 8))
    blocks = args.blocks or list(cfg["features"]["blocks"])
    for b in blocks:
        print(f"{b:12s} {store.block(b).shape}")
    for k in (cfg["features"].get("kernels") or {}):
        if args.kernels:
            print(f"kernel {k:8s} {store.kernel(k).shape}")


def cmd_cv(cfg, args):
    from .cv.runner import run_experiments

    run_experiments(cfg, tasks=args.tasks, only=args.only, force=args.force)


def cmd_ensemble(cfg, args):
    from .ensemble.assemble import run_ensemble

    run_ensemble(cfg, write_submission=not args.no_submit)


def cmd_collect(cfg, args):
    from .external.collect import collect_all

    collect_all(cfg, sources=args.sources, refresh=args.refresh)


def cmd_run_all(cfg, args):
    """The whole reproducible pipeline: (collect) -> prepare -> cv -> ensemble -> submissions/submission.csv."""
    import shutil

    from .paths import SUBMISSIONS, resolve

    ext_csv = resolve(cfg.get("paths", {}).get("external_csv", "data/external/vacv_external.csv"))
    if cfg.get("external", {}).get("use") and (args.collect or not ext_csv.exists()):
        from .external.collect import collect_all

        collect_all(cfg, sources=["chembl"], refresh=False)
    cmd_prepare(cfg, args)
    from .cv.runner import exp_tag, run_experiments

    run_experiments(cfg, force=args.force)
    from .ensemble.assemble import run_ensemble

    run_ensemble(cfg, write_submission=True)
    src = SUBMISSIONS / f"submission_{exp_tag(cfg)}.csv"
    shutil.copyfile(src, SUBMISSIONS / "submission.csv")
    print(f"\nDONE -> {SUBMISSIONS / 'submission.csv'}")


def cmd_report(cfg, args):
    from .cv.runner import print_leaderboard

    print_leaderboard(cfg)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vacv-si", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--config", "-c", default="configs/base.yaml")
        sp.add_argument("-o", "--override", action="append", default=[], help="dotted override, e.g. cv.n_repeats=2")

    sp = sub.add_parser("prepare", help="clean + standardise the competition (and external) tables")
    add_common(sp); sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("features", help="build & cache feature blocks")
    add_common(sp); sp.add_argument("--blocks", nargs="*"); sp.add_argument("--kernels", action="store_true")
    sp.set_defaults(func=cmd_features)

    sp = sub.add_parser("cv", help="run repeated CV for every model of the config and store OOF/test predictions")
    add_common(sp)
    sp.add_argument("--tasks", nargs="*", default=None, help="subset of tasks (clf, reg, ...)")
    sp.add_argument("--only", nargs="*", default=None, help="subset of model names")
    sp.add_argument("--force", action="store_true", help="recompute even if cached")
    sp.set_defaults(func=cmd_cv)

    sp = sub.add_parser("report", help="print the leaderboard of all stored models")
    add_common(sp); sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("ensemble", help="blend stored OOF predictions, evaluate and write the submission")
    add_common(sp); sp.add_argument("--no-submit", action="store_true"); sp.set_defaults(func=cmd_ensemble)

    sp = sub.add_parser("run-all", help="full pipeline: collect (if needed) -> prepare -> cv -> ensemble -> submission.csv")
    add_common(sp)
    sp.add_argument("--collect", action="store_true", help="re-run the external-data collection even if the CSV exists")
    sp.add_argument("--force", action="store_true", help="recompute every model even if cached")
    sp.set_defaults(func=cmd_run_all)

    sp = sub.add_parser("collect", help="download records from open databases and assemble the external dataset")
    add_common(sp)
    sp.add_argument("--sources", nargs="*", default=None, help="chembl pubchem")
    sp.add_argument("--refresh", action="store_true", help="ignore the on-disk cache of API responses")
    sp.set_defaults(func=cmd_collect)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config, args.override)
    args.func(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
