"""Разведка ChEMBL API: объём активностей и типы значений (IC50, EC50, MIC …).

Запуск из корня репозитория: python experiments/analysis/08_chembl_probe_activities.py   [нужна сеть]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import requests, collections, time
B="https://www.ebi.ac.uk/chembl/api/data"
def get(path, **params):
    params.setdefault("format","json")
    for i in range(4):
        try:
            r=requests.get(f"{B}/{path}", params=params, timeout=90); r.raise_for_status(); return r.json()
        except Exception as e:
            time.sleep(2*(i+1)); err=e
    raise err
# all activities against the vaccinia organism targets
tg="CHEMBL613493,CHEMBL613168"
a=get("activity", target_chembl_id__in=tg, limit=1)
print("activities on organism targets:", a["page_meta"]["total_count"])
# page through everything for these targets and tally types
types=collections.Counter(); assays=set(); docs=set(); mols=set(); n=0
off=0
while True:
    d=get("activity", target_chembl_id__in=tg, limit=1000, offset=off)
    for x in d["activities"]:
        types[(x["standard_type"], x["standard_units"], x["standard_relation"])]+=1
        assays.add(x["assay_chembl_id"]); docs.add(x["document_chembl_id"]); mols.add(x["molecule_chembl_id"]); n+=1
    if not d["page_meta"]["next"]: break
    off+=1000
print("n",n,"assays",len(assays),"docs",len(docs),"molecules",len(mols))
for k,v in types.most_common(25): print("  ",k,v)
# assays by organism text (not necessarily on organism target)
b=get("assay", assay_organism__icontains="vaccinia", limit=1000)
print("\nassay organism strings:", collections.Counter(x["assay_organism"] for x in b["assays"]).most_common(6))
print("assay types:", collections.Counter(x["assay_type"] for x in b["assays"]).most_common())
print("assay target ids:", collections.Counter(x["target_chembl_id"] for x in b["assays"]).most_common(8))
