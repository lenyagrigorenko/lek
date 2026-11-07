"""Откуда идентификаторы sid… из Title: ассеи PubChem — зеркала ChEMBL.

Запуск из корня репозитория: python experiments/analysis/09_pubchem_sid_probe.py   [нужна сеть]
Это разведочный скрипт: он только читает данные и печатает числа (результаты — в experiments/README.md).
"""
import requests, json, time, collections, pandas as pd
H={"User-Agent":"vacv-si-hackathon/0.1"}
df=pd.read_pickle("data/processed/all_ext.pkl")
sids=df[df.title.str.match(r"^sid\d+$")].copy()
sids["sid"]=sids.title.str[3:].astype(int)
print("sid rows:", len(sids), "| train:", (sids.split=="train").sum(), "test:", (sids.split=="test").sum())
# sample a few SIDs from different batches / cultures
samp=sids[sids.split=="train"].drop_duplicates("culture").head(10)
aid_counter=collections.Counter(); per_sid={}
for _,r in samp.iterrows():
    url=f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{r.sid}/assaysummary/JSON"
    try:
        j=requests.get(url,headers=H,timeout=60).json()
    except Exception as e:
        print("ERR",r.sid,e); continue
    rows=j.get("Table",{}).get("Row",[]); cols=j.get("Table",{}).get("Columns",{}).get("Column",[])
    ci={c:i for i,c in enumerate(cols)}
    out=[]
    for row in rows:
        c=row["Cell"]; name=c[ci["Assay Name"]] if "Assay Name" in ci else ""
        if any(k in name.lower() for k in ("vaccinia","poxvirus","vacv","antiviral","cowpox")):
            out.append((c[ci["AID"]], name[:110], c[ci.get("Activity Outcome")] if "Activity Outcome" in ci else ""))
    per_sid[r.sid]=out
    print(f"\nsid{r.sid} ({r.culture.strip()}, si={r.si}): {len(rows)} assay rows; vaccinia-ish: {len(out)}")
    for o in out[:6]: print("   ",o)
    for o in out: aid_counter[(o[0],o[1])]+=1
    time.sleep(0.4)
print("\nAIDs seen:", aid_counter.most_common(12))
