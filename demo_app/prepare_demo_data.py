"""
Bước 2: chuyển kết quả từ Kaggle (thư mục demo_files do save_rf_for_demo.py tạo ra)
thành dữ liệu cho app trong thư mục data/.
  - Lọc 439.290 cặp thuốc của tập kiểm tra -> data/test_pairs.parquet
  - Tính mean |SHAP| trên 500 cặp tập kiểm tra -> data/shap_global.csv
  - Chép mô hình, danh sách thuốc, thông tin mô hình

Chạy:  python prepare_demo_data.py <thư_mục_demo_files>
"""
import json, shutil, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "demo_files")
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

info = json.loads((SRC / "model_info.json").read_text(encoding="utf-8"))
F = info["features"]

pairs = pd.read_parquet(SRC / "pairs_features.parquet")
test = pairs[pairs["split"] == "test"].drop(columns="split").reset_index(drop=True)
test["label"] = test["label"].astype("int8")
test.to_parquet(OUT / "test_pairs.parquet", index=False)
print("Tập kiểm tra:", test.shape)

model = joblib.load(SRC / "rf_model.joblib")
sample = test.sample(500, random_state=0)[F].to_numpy(np.float32)
sv = shap.TreeExplainer(model).shap_values(sample, check_additivity=False)
sv = np.asarray(sv[1]) if isinstance(sv, list) else np.asarray(sv)
if sv.ndim == 3:
    sv = sv[..., 1]
(pd.DataFrame({"feature": F, "mean_abs_shap": np.abs(sv).mean(0)})
   .sort_values("mean_abs_shap", ascending=False)
   .to_csv(OUT / "shap_global.csv", index=False))

for f in ["rf_model.joblib", "drugs.csv", "model_info.json"]:
    shutil.copy(SRC / f, OUT / f)
print("Xong ->", OUT)
