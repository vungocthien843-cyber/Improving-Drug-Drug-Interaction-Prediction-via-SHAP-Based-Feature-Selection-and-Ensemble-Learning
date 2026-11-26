# ============================================================================
#  LƯU MÔ HÌNH RANDOM FOREST + DỮ LIỆU CHO DEMO
#  Dán vào 1 cell MỚI ở cuối notebook Kaggle rồi chạy.
#  Yêu cầu: đã có Final_Dataset_DDI.csv (bước 6) và drug_node_final.csv (bước 2).
#  Dùng sklearn (CPU), KHÔNG dùng cuML -> file .joblib chạy được trên mọi máy / HF Spaces.
# ============================================================================
import os, json, time, joblib, sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

# ---------- 1. ĐƯỜNG DẪN (sửa nếu file của bạn nằm chỗ khác) ----------
DATASET   = "/kaggle/input/datasets/vungocthien/dataset-dont-split/Final_Dataset_DDI.csv"
# Nguồn TÊN thuốc (không bắt buộc). Cell tự thử lần lượt; không có thì dùng mã DrugBank làm tên.
NODE_FILE = "/kaggle/working/drug_node_final.csv"   # đã mất -> bỏ qua
XML_FILE  = None   # nếu còn DrugBank XML trong input, điền đường dẫn, vd "/kaggle/input/.../full database.xml"
VOCAB_CSV = None   # hoặc file "drugbank vocabulary.csv" (DrugBank Open Data, miễn phí)
OUT_DIR   = "/kaggle/working/demo_files"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
TOPO_FEATS = ['cn', 'jc', 'aai', 'pa', 'rai', 'ccn', 'cra', 'wic']
# Mô hình đề xuất: bỏ ADE + ATC (10 đặc trưng) -> F1 test ~78,43%
# Muốn dùng đủ 12 đặc trưng thì thêm 'sim_atc', 'sim_ade' vào danh sách dưới.
FEATURES = TOPO_FEATS + ['sim_mesh', 'sim_chemical']

# ---------- 2. CHIA LẠI TRAIN/TEST Y HỆT NOTEBOOK (nhưng GIỮ u, v) ----------
# Cùng random_state=42 và cùng thứ tự thao tác -> ra đúng phép chia đã dùng trong báo cáo.
df = pd.read_csv(DATASET)
need = ['u', 'v', 'label', 'pos_split'] + FEATURES
missing = [c for c in need if c not in df.columns]
assert not missing, f"Final_Dataset_DDI.csv thiếu cột: {missing}"
df['pos_split'] = df['pos_split'].fillna('')
pos = df[df['label'] == 1]
neg = df[df['label'] == 0]
pos_train = pos[pos['pos_split'] == 'train']
pos_test  = pos[pos['pos_split'] == 'test']
neg_train, neg_test = train_test_split(neg, test_size=0.34, random_state=SEED, shuffle=True)
df_train = pd.concat([pos_train, neg_train], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df_test  = pd.concat([pos_test,  neg_test],  ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
df_train['split'] = 'train'
df_test['split']  = 'test'
print(f"Train: {df_train.shape} | Test: {df_test.shape}")

Xtr = df_train[FEATURES].fillna(0).to_numpy(np.float32); ytr = df_train['label'].to_numpy(np.int32)
Xte = df_test[FEATURES].fillna(0).to_numpy(np.float32);  yte = df_test['label'].to_numpy(np.int32)

# ---------- 3. TRAIN RF (cùng siêu tham số trong báo cáo) ----------
t0 = time.time()
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=SEED, n_jobs=-1)
rf.fit(Xtr, ytr)
print(f"Train xong sau {time.time() - t0:.1f}s")

# ---------- 4. KIỂM TRA LẠI TRÊN TẬP TEST (phải gần F1≈78,4%, AUPR≈87,6%) ----------
proba_te = rf.predict_proba(Xte)[:, 1]
pred_te = (proba_te >= 0.5).astype(int)
metrics = {
    'F1':        round(100 * f1_score(yte, pred_te), 2),
    'Precision': round(100 * precision_score(yte, pred_te), 2),
    'Recall':    round(100 * recall_score(yte, pred_te), 2),
    'AUPR':      round(100 * average_precision_score(yte, proba_te), 2),
}
print("Kết quả test:", metrics)

# ---------- 5. LƯU MÔ HÌNH ----------
joblib.dump(rf, f"{OUT_DIR}/rf_model.joblib", compress=3)
with open(f"{OUT_DIR}/model_info.json", "w", encoding="utf-8") as f:
    json.dump({'features': FEATURES, 'n_estimators': 100, 'max_depth': 15, 'seed': SEED,
               'test_metrics': metrics, 'sklearn_version': sklearn.__version__},
              f, ensure_ascii=False, indent=2)

# ---------- 6. BẢNG TRA CỨU CẶP THUỐC (để app chọn 2 thuốc -> lấy đặc trưng) ----------
pairs = pd.concat([df_train, df_test], ignore_index=True)
pairs['proba'] = rf.predict_proba(pairs[FEATURES].fillna(0).to_numpy(np.float32))[:, 1]
keep = ['u', 'v', 'label', 'split'] + FEATURES + ['proba']
pairs = pairs[keep]
for c in FEATURES + ['proba']:
    pairs[c] = pairs[c].astype(np.float32)
pairs['label'] = pairs['label'].astype(np.int8)
try:
    pairs.to_parquet(f"{OUT_DIR}/pairs_features.parquet", index=False)   # nhẹ hơn CSV rất nhiều
except ImportError:
    pairs.to_csv(f"{OUT_DIR}/pairs_features.csv.gz", index=False, compression='gzip')

# ---------- 7. DANH SÁCH THUỐC (id + tên) ----------
drug_ids = sorted(set(df['u']) | set(df['v']))
print(f"Số thuốc trong dataset: {len(drug_ids)}")   # phải là 1.608
names = {}
if NODE_FILE and os.path.exists(NODE_FILE):
    nodes = pd.read_csv(NODE_FILE)
    names = dict(zip(nodes['drugbank_id'], nodes['name']))
    print("Lấy tên thuốc từ", NODE_FILE)
elif XML_FILE and os.path.exists(XML_FILE):
    # Chỉ đọc mã + tên (nhanh, không cần chạy lại pipeline)
    import xml.etree.ElementTree as ET
    ns = '{http://www.drugbank.ca}'
    depth = 0
    for ev, el in ET.iterparse(XML_FILE, events=('start', 'end')):
        if el.tag == ns + 'drug':
            if ev == 'start':
                depth += 1
            else:
                depth -= 1
                if depth == 0:   # thẻ <drug> cấp cao nhất
                    pid = el.find(ns + "drugbank-id[@primary='true']")
                    nm = el.find(ns + 'name')
                    if pid is not None and nm is not None:
                        names[pid.text] = nm.text
                    el.clear()
    print("Lấy tên thuốc từ", XML_FILE)
elif VOCAB_CSV and os.path.exists(VOCAB_CSV):
    voc = pd.read_csv(VOCAB_CSV)
    names = dict(zip(voc['DrugBank ID'], voc['Common name']))
    print("Lấy tên thuốc từ", VOCAB_CSV)
else:
    print("[CHÚ Ý] Không có nguồn tên thuốc -> dùng mã DrugBank làm tên. Có thể bổ sung tên sau.")
drugs = pd.DataFrame({'drugbank_id': drug_ids})
drugs['name'] = drugs['drugbank_id'].map(names).fillna(drugs['drugbank_id'])
drugs.to_csv(f"{OUT_DIR}/drugs.csv", index=False)
print(f"Có tên cho {(drugs['name'] != drugs['drugbank_id']).sum()}/{len(drugs)} thuốc")

# ---------- 8. TOP TƯƠNG TÁC TIỀM NĂNG (cặp chưa có trong DrugBank, điểm cao nhất) ----------
id2name = dict(zip(drugs['drugbank_id'], drugs['name']))
top = pairs[pairs['label'] == 0].nlargest(200, 'proba')[['u', 'v', 'proba']].copy()
top.insert(1, 'name_u', top['u'].map(id2name))
top.insert(3, 'name_v', top['v'].map(id2name))
top.to_csv(f"{OUT_DIR}/top_potential_ddi.csv", index=False)

# ---------- 9. TỔNG KẾT ----------
print("\nĐã lưu vào", OUT_DIR)
for fn in sorted(os.listdir(OUT_DIR)):
    print(f"  {fn:28s} {os.path.getsize(os.path.join(OUT_DIR, fn)) / 1e6:8.1f} MB")
print("sklearn version:", sklearn.__version__, "(ghi lại để cài đúng bản khi deploy)")
print(top.head(10).to_string(index=False))
