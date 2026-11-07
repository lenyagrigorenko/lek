"""Шаблоны Title (sid…, числа, лаб-коды) и «опечатки» в сырых строках (пробелы, запятые) как маркеры источника.

Запуск из корня репозитория: python experiments/analysis/03_title_patterns_and_string_quirks.py
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import pandas as pd, numpy as np, re
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 500)
D="data/raw"
tr=pd.read_csv(f"{D}/train.csv"); te=pd.read_csv(f"{D}/test.csv")
tr["sent"]=(tr["SI VACV"]==10000).astype(int)

# whitespace quirks in RAW strings
for col in ["strain","Culture","Title","SMILES"]:
    for df,name in [(tr,"train"),(te,"test")]:
        s=df[col].astype(str)
        print(f"{col:8s} {name:5s} lead_ws={s.str.match(r'^\s').sum():4d} trail_ws={s.str.contains(r'\s$').sum():4d} dbl_space={s.str.contains(r'  ').sum():4d} trailing_comma={s.str.contains(r',\s*$').sum():4d}")
print()
tr["lead_ws_culture"]=tr.Culture.str.match(r"^\s").astype(int); tr["trail_ws_strain"]=tr.strain.str.contains(r"\s$").astype(int)
print(tr.groupby("lead_ws_culture").sent.agg(["size","mean"])); print(tr.groupby("trail_ws_strain").sent.agg(["size","mean"]))

def tpat(t):
    t=str(t)
    if re.fullmatch(r"sid\d+",t): return "sid#"
    if re.fullmatch(r"\d+",t): return "int"
    if re.fullmatch(r"\d+[a-zA-Z]",t): return "int+letter"
    if re.fullmatch(r"[A-Za-z]{1,4}-?\d+[a-zA-Z]?",t): return "code(letters+num)"
    if re.search(r"[A-Za-z]{4,}",t) and " " not in t and len(t)<25: return "word/name"
    return "other"
tr["tp"]=tr.Title.map(tpat); te["tp"]=te.Title.map(tpat)
g=tr.groupby("tp").agg(n=("sent","size"), sent=("sent","mean"), medSI=("SI VACV","median"), si1=("SI VACV",lambda s:(s==1).mean()))
g["test_n"]=te.tp.value_counts().reindex(g.index).fillna(0).astype(int)
print("\n=== Title pattern ===\n", g.round(2).to_string())
print("\nTitle samples per pattern (train):")
for p,grp in tr.groupby("tp"): print(f"  {p:18s}", grp.Title.head(12).tolist())
print("\nTitle duplicates in train:", tr.Title.duplicated(keep=False).sum(), " in test:", te.Title.duplicated(keep=False).sum(), " overlap train/test titles:", len(set(tr.Title.astype(str))&set(te.Title.astype(str))))
print("\nsid-compounds: strain/culture mix (train)"); print(tr[tr.tp=="sid#"].groupby(["strain"]).agg(n=("sent","size"), sent=("sent","mean"), medSI=("SI VACV","median")).sort_values("n",ascending=False).head(12).round(2).to_string())
