"""
(TÙY CHỌN) Điền tên thuốc vào data/drugs.csv bằng PubChem (cần Internet, chạy trên máy bạn).
Chạy:  python add_drug_names.py
Chạy lại được nhiều lần: chỉ tra những thuốc còn thiếu tên. Mất khoảng 5–10 phút cho 1.608 thuốc.
"""
import time
from pathlib import Path

import pandas as pd
import requests

PATH = Path(__file__).parent / "data" / "drugs.csv"
URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/DrugBank/{}/synonyms/JSON"

df = pd.read_csv(PATH)
todo = df.index[df["name"].fillna(df["drugbank_id"]) == df["drugbank_id"]]
print(f"Cần tra {len(todo)}/{len(df)} thuốc")

for k, i in enumerate(todo, 1):
    did = df.at[i, "drugbank_id"]
    try:
        r = requests.get(URL.format(did), timeout=15)
        if r.ok:
            syn = r.json()["InformationList"]["Information"][0].get("Synonym", [])
            if syn:
                df.at[i, "name"] = syn[0]
    except Exception as e:
        print(did, "lỗi:", e)
    time.sleep(0.25)                      # PubChem giới hạn ~5 yêu cầu/giây
    if k % 50 == 0:
        df.to_csv(PATH, index=False)
        print(f"  {k}/{len(todo)} …")

df.to_csv(PATH, index=False)
print(f"Xong. Có tên cho {(df['name'] != df['drugbank_id']).sum()}/{len(df)} thuốc.")
