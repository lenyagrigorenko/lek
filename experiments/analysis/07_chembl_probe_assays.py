"""Разведка ChEMBL API: цели и ассеи по вирусу осповакцины.

Запуск из корня репозитория: python experiments/analysis/07_chembl_probe_assays.py   [нужна сеть]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import requests, json, collections, time
B="https://www.ebi.ac.uk/chembl/api/data"
def get(path, **params):
    params.setdefault("format","json")
    r=requests.get(f"{B}/{path}", params=params, timeout=60); r.raise_for_status(); return r.json()

# 1) targets matching vaccinia
t=get("target/search", q="vaccinia", limit=50)
print("targets:", t["page_meta"]["total_count"])
for x in t["targets"][:15]: print("  ", x["target_chembl_id"], x["target_type"], x["organism"], "|", x["pref_name"])

# 2) assays by organism / description
for key,val in [("assay_organism__icontains","vaccinia"),("assay_tax_id","10245"),("description__icontains","vaccinia"),("description__icontains","VACV")]:
    a=get("assay", limit=1, **{key:val})
    print(f"assays {key}={val}: {a['page_meta']['total_count']}")
