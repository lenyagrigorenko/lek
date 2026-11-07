"""Источники данных: сырые строки strain/Culture и доля «потолка» в каждой группе.

Запуск из корня репозитория: python experiments/analysis/02_sources_and_cultures.py
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import pandas as pd, numpy as np, re
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110); pd.set_option("display.max_rows", 500)
D="data/raw"
tr=pd.read_csv(f"{D}/train.csv"); te=pd.read_csv(f"{D}/test.csv")
tr["sent"]=(tr["SI VACV"]==10000).astype(int)
tr["strain"]=tr["strain"].str.strip(); te["strain"]=te["strain"].str.strip()
tr["Culture"]=tr["Culture"].str.strip(); te["Culture"]=te["Culture"].str.strip()

print("=== strain string groups (train: n, sentinel rate, median SI; test n) ===")
g=tr.groupby("strain").agg(n=("sent","size"), sent=("sent","mean"), medSI=("SI VACV","median"), si1=("SI VACV", lambda s:(s==1).mean()))
g["test_n"]=te["strain"].value_counts().reindex(g.index).fillna(0).astype(int)
g=g.sort_values("n",ascending=False)
print(g.round(2).to_string())
only_test=te[~te.strain.isin(tr.strain)]["strain"].value_counts()
print("\nstrain strings only in TEST:", len(only_test), "rows:", only_test.sum()); print(only_test.head(25).to_string())

print("\n=== Culture (train n, sentinel rate; test n) ===")
c=tr.groupby("Culture").agg(n=("sent","size"), sent=("sent","mean"), medSI=("SI VACV","median"))
c["test_n"]=te["Culture"].value_counts().reindex(c.index).fillna(0).astype(int)
print(c.sort_values("n",ascending=False).round(2).to_string())
print("Culture only in test:", te[~te.Culture.isin(tr.Culture)]["Culture"].value_counts().to_dict())
