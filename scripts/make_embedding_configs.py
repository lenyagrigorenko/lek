"""Write configs/exp/emb_<tier>.yaml for every feature tier, using whichever embeddings exist in data/embeddings/*.npz.

    python scripts/make_embedding_configs.py                       # all tiers, all embeddings found
    python scripts/make_embedding_configs.py --names chemeleon chemberta77m_mtr --tiers chem

For every embedding X it declares
    blocks   emb_X (raw, used for the cosine kernel)  and  emb_X_pca (24 principal components, for tree models)
    kernel   k_emb_X  (cosine similarity of centred embeddings)
    models   et_emb_X       ExtraTrees on [PCA(X) + the usual blocks of the tier]
             et_embknn_X    ExtraTrees + neighbour-label features from the Morgan, AtomPair AND embedding kernels
             cat_emb_X      CatBoost on [PCA(X) + the usual blocks of the tier]
             krr_emb_X      kernel ridge regression on the embedding kernel
plus et_emb_all / et_embknn_all that use every embedding at once. The same model specs are used for `clf` and `reg`.

The generated config extends the tier's config but DISABLES every inherited model, so `cv` computes only the embedding
models (the tier's own results are not recomputed or touched). `ensemble.include_tags: [<tier>]` makes the ensemble blend the
tier's stored baseline models together with the new ones, under its own tag (reports/ensemble_emb_<tier>.json).

    python -m vacv_si cv -c configs/exp/emb_chem.yaml                       # add -o cv.full_fit_seeds=0 for a quick screening
    python -m vacv_si ensemble -c configs/exp/emb_chem.yaml --no-submit     # honest blended MSE of baseline + embedding models
    python -m vacv_si ensemble -c configs/exp/emb_final.yaml                # also writes submissions/submission_emb_final.csv

(run the tier's own `cv` first: python -m vacv_si cv -c configs/exp/<tier>.yaml)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vacv_si.config import load_config
from vacv_si.cv.runner import TASKS

ROOT = Path(__file__).resolve().parents[1]
ET = {"n_estimators": 800, "min_samples_leaf": 1, "max_features": 0.2, "n_jobs": 1}
CAT = {"iterations": 500, "depth": 5, "learning_rate": 0.05, "l2_leaf_reg": 5}


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):          # the same dict object appears under clf and reg: write it out twice, no &id001
        return True


def new_models(names: list[str], pca: int) -> tuple[dict, dict, dict]:
    blocks, kernels, models = {}, {}, {}
    for n in names:
        blocks[f"emb_{n}"] = {"type": "npz_embedding", "path": f"data/embeddings/{n}.npz"}
        blocks[f"emb_{n}_pca"] = {"type": "npz_embedding", "path": f"data/embeddings/{n}.npz", "pca": pca}
        kernels[f"k_emb_{n}"] = {"block": f"emb_{n}", "type": "cosine"}
        base = ["apc", "desc", "@meta", "@style"]
        models[f"et_emb_{n}"] = {"type": "extra_trees", "blocks": [f"emb_{n}_pca"] + base, "params": ET}
        models[f"et_embknn_{n}"] = {"type": "extra_trees", "blocks": [f"emb_{n}_pca", "desc", "@meta", "@style"],
                                    "knn": {"kernels": ["k_m2", "k_ap", f"k_emb_{n}"], "ks": [1, 3, 8]}, "params": ET}
        models[f"cat_emb_{n}"] = {"type": "catboost", "blocks": [f"emb_{n}_pca", "desc", "maccs", "@meta_n", "@style_n"],
                                  "cat_cols": ["@meta", "@style"], "params": CAT}
        models[f"krr_emb_{n}"] = {"type": "kernel_ridge", "kernel": {"base": f"k_emb_{n}", "gamma": 3.0, "mult": {"@src": 1.0}, "add": {"@src": 0.5}},
                                  "params": {"alpha": 0.2}}
    if len(names) > 1:
        allp = [f"emb_{n}_pca" for n in names]
        models["et_emb_all"] = {"type": "extra_trees", "blocks": allp + ["apc", "desc", "@meta", "@style"], "params": ET}
        models["et_embknn_all"] = {"type": "extra_trees", "blocks": allp + ["desc", "@meta", "@style"],
                                   "knn": {"kernels": ["k_m2", "k_ap"] + [f"k_emb_{n}" for n in names], "ks": [1, 3, 8]}, "params": ET}
    return blocks, kernels, models


def build(tier: str, names: list[str], pca: int) -> dict:
    blocks, kernels, models = new_models(names, pca)
    tier_models = load_config(ROOT / "configs" / "exp" / f"{tier}.yaml").get("models") or {}
    out_models: dict = {}
    for task in [t for t in TASKS if t in tier_models or t in ("clf", "reg")]:
        disabled = {name: {"enabled": False} for name in (tier_models.get(task) or {})}     # merged into the inherited specs
        out_models[task] = {**disabled, **models} if task in ("clf", "reg") else disabled
    return {
        "extends": f"{tier}.yaml",
        "tag": f"emb_{tier}",
        "ensemble": {"include_tags": [tier]},
        "features": {"blocks": blocks, "kernels": kernels},
        "models": out_models,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*", help="embedding names (default: every data/embeddings/*.npz)")
    ap.add_argument("--tiers", nargs="*", default=["final", "assay", "chem"])
    ap.add_argument("--pca", type=int, default=24)
    args = ap.parse_args()
    names = args.names or sorted(p.stem for p in (ROOT / "data/embeddings").glob("*.npz"))
    if not names:
        raise SystemExit("no embeddings in data/embeddings/*.npz (run scripts/colab_embeddings.py first)")
    for tier in args.tiers:
        cfg = build(tier, names, args.pca)
        path = ROOT / "configs" / "exp" / f"emb_{tier}.yaml"
        header = (f"# GENERATED by scripts/make_embedding_configs.py (embeddings: {', '.join(names)}). Do not edit by hand.\n"
                  f"# Pretrained-embedding features on top of the `{tier}` feature tier: only the embedding models are computed (every\n"
                  f"# inherited model is disabled); the ensemble blends them with the stored `{tier}` models (ensemble.include_tags).\n")
        path.write_text(header + yaml.dump(cfg, Dumper=_NoAliasDumper, sort_keys=False, default_flow_style=None, width=140))
        n_new = sum(1 for m in cfg["models"]["clf"].values() if "type" in m)
        print(f"wrote {path.relative_to(ROOT)}  ({n_new} embedding models per task)")


if __name__ == "__main__":
    main()
