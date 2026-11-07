"""Инициалы автора в лабораторных кодах (as, ks, qs, ya) и потенция серии.

Запуск из корня репозитория: python experiments/analysis/19_lab_code_tags.py   [после run-all]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import re, numpy as np, pandas as pd
pd.set_option("display.width",200)
df = pd.read_pickle("data/processed/all_ext.pkl"); k = df[df.split!="ext"].reset_index(drop=True)
sub = pd.read_csv("submissions/submission.csv").set_index("index")["SI VACV"]
def tag(t):
    m = re.fullmatch(r"(\d+[a-z]?)-([a-z]{2,3})-(\d+)(?:-.*)?", str(t).strip())
    return m.group(2) if m else None
k["tag"] = k.title.map(tag)
lab = k[k.tag.notna()]
print("lab-code rows: train %d, test %d | tags:"%((lab.split=='train').sum(), (lab.split=='test').sum()), lab.tag.value_counts().to_dict())
tr = lab[lab.split=="train"]
print("\nTRAIN lab-code rows by tag (sentinel, SI):")
for t, d in tr.groupby("tag"): print(f"  {t:4s} n={len(d):2d}  sentinel={int(d.is_sentinel.sum())}  SI: {sorted(d.si.round(1).tolist())}")
te = lab[lab.split=="test"].copy(); te["pred"] = sub.loc[te.test_index.astype(int)].values
print("\nTEST lab-code rows by tag: n and predicted SI quantiles (10/50/90):")
for t, d in te.groupby("tag"): print(f"  {t:4s} n={len(d):2d}  pred q10/50/90 = {d.pred.quantile([.1,.5,.9]).round(0).tolist()}")
print("\ntitles of the train lab-code rows with SI>=500:", tr[tr.si>=500][["title","si"]].round(0).values.tolist())
