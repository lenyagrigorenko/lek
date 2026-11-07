"""Сложная группа Copenhagen/Vero (ручные SMILES): состав train/test и хвост lab_code.

Запуск из корня репозитория: python experiments/analysis/13_hard_group_listing.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import pandas as pd, numpy as np
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",70); pd.set_option("display.max_rows",300)
from vacv_si.features.meta import smiles_style_features
df=pd.read_pickle("data/processed/all_ext.pkl"); df=df[df.split!="ext"].reset_index(drop=True)
sty=smiles_style_features(df.smiles_raw); df["can"]=sty.s_is_rdkit_can.values
tr=df[df.split=="train"].reset_index(drop=True)
o=pd.read_csv("reports/oof_final_final.csv"); tr["p"]=o.p.values; tr["M"]=o.M.values
sub=pd.read_csv("submissions/submission.csv"); te=df[df.split=="test"].reset_index(drop=True); te["pred"]=sub.set_index("index").loc[te.test_index.astype(int),"SI VACV"].values
G="Vaccinia virus (strain Copenhagen). culture Vero"
g=tr[tr.strain.str.strip()==G].copy(); g["tn"]=pd.to_numeric(g.title.str.extract(r"^(\d+)")[0],errors="coerce")
print("=== train rows of the Copenhagen/Vero group (n=%d): sorted by title number"%len(g))
print(g.sort_values(["can","tn"])[["title","can","is_sentinel","p","si"]].round(2).to_string(index=False))
t=te[te.strain.str.strip()==G]
print("\n=== test rows in the group: n=%d, canonical share %.2f, predicted mean SI %.0f"%(len(t),t.can.mean(),t.pred.mean()))
print("test titles (canonical):", sorted(t[t.can==1].title.tolist())[:40])
print("test titles (non-canonical):", sorted(t[t.can==0].title.tolist())[:70])
# ---- tail
ns=tr[(tr.is_sentinel==0)&(tr.si>300)].sort_values("si",ascending=False)
print("\n=== train non-sentinel rows with SI>300 (OOF M shown)")
print(ns[["title","strain","si","M","p"]].assign(strain=ns.strain.str.slice(0,40)).round(1).to_string(index=False))
lab=tr[tr.title.str.count("-")>=2]
print("\nlab_code-like titles in train: n=%d  sentinel=%.2f  non-sentinel SI: %s"%(len(lab),lab.is_sentinel.mean(),sorted(lab[lab.is_sentinel==0].si.round(0).tolist())))
labt=te[te.title.str.count("-")>=2]
print("lab_code-like titles in test: n=%d  predicted SI quantiles:"%len(labt), labt.pred.quantile([.1,.5,.9]).round(0).tolist())
print("\ntest rows with highest predicted SI below sentinel level (p small):")
print(te[te.pred<5000].sort_values("pred",ascending=False).head(12)[["title","strain","pred"]].assign(strain=lambda d:d.strain.str.slice(0,45)).round(0).to_string(index=False))
