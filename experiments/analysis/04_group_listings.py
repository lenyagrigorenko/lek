"""Построчный просмотр неоднородных групп (generic «Vaccinia virus», Copenhagen/Vero).

Запуск из корня репозитория: python experiments/analysis/04_group_listings.py
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import pandas as pd, numpy as np
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 80); pd.set_option("display.max_rows", 500)
D="data/raw"
tr=pd.read_csv(f"{D}/train.csv")
tr["sent"]=(tr["SI VACV"]==10000).astype(int)
cols=["Title","Culture","CC50_uM","IC50_uM_VACV","SI VACV","SMILES"]
def show(mask, name, n=45):
    d=tr[mask].sort_values(["sent","SI VACV"], ascending=[False,False])
    print(f"\n##### {name}  (n={len(d)}, sent={d.sent.sum()})")
    print(d[cols].head(n).to_string(index=False))
show(tr.strain.str.strip()=="Vaccinia virus", "generic 'Vaccinia virus'")
show(tr.strain.str.strip()=="Vaccinia virus (strain Copenhagen). culture Vero", "Copenhagen/Vero")
