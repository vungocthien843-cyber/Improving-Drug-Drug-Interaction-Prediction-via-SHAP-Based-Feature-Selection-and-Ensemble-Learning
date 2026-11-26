# Demo dự đoán tương tác thuốc–thuốc (Streamlit)

Demo chỉ dùng các cặp thuốc thuộc **tập kiểm tra** (439.290 cặp mô hình chưa từng thấy khi huấn luyện).

| Tab | Nội dung |
|---|---|
| Dự đoán cặp thuốc | Chọn 2 thuốc → xác suất tương tác (tính trực tiếp bằng mô hình), so với nhãn DrugBank, giải thích SHAP |
| Đánh giá trên tập kiểm tra | F1, Precision, Recall, Specificity, AUPR, ma trận nhầm lẫn, đường PR – thay đổi theo ngưỡng |
| Tương tác tiềm năng | Các cặp chưa được ghi nhận nhưng mô hình chấm điểm cao |

## Chạy trên máy (Windows)

```bash
cd demo_app
pip install -r requirements.txt
streamlit run app.py
```
Trình duyệt tự mở `http://localhost:8501`.

> `scikit-learn` phải đúng bản **1.6.1** (bản đã dùng để lưu mô hình trên Kaggle), nên dùng Python 3.10–3.12.

## (Tùy chọn) Hiện tên thuốc thay vì mã DrugBank

```bash
python add_drug_names.py
```

## Đưa lên mạng để có link (Streamlit Community Cloud – miễn phí)

1. Đưa thư mục `demo_app` lên GitHub (ví dụ vào repo NCKH hiện có). Các file đều < 100 MB nên đẩy bình thường được.
2. Vào https://share.streamlit.io → đăng nhập bằng GitHub → **Create app**.
3. Chọn repo, nhánh `main`, *Main file path*: `demo_app/app.py`.
4. **Advanced settings → Python version: 3.12** → **Deploy**.
5. Sau vài phút sẽ có link dạng `https://<ten-app>.streamlit.app`.

## Tái lập dữ liệu và mô hình

1. `save_rf_for_demo.py` – dán vào cuối notebook Kaggle (cần `Final_Dataset_DDI.csv`), chạy để train lại Random Forest
   (100 cây, độ sâu 15, 10 đặc trưng) và xuất thư mục `demo_files`.
2. `python prepare_demo_data.py <đường_dẫn_demo_files>` – lọc các cặp tập kiểm tra, tính SHAP toàn cục và ghi vào `data/`.

## Cấu trúc

```
demo_app/
├── app.py                 # giao diện Streamlit
├── requirements.txt
├── save_rf_for_demo.py    # (Kaggle) train và lưu mô hình
├── prepare_demo_data.py   # tạo dữ liệu trong data/
├── add_drug_names.py      # (tùy chọn) điền tên thuốc từ PubChem
└── data/
    ├── rf_model.joblib       # Random Forest, 10 đặc trưng
    ├── test_pairs.parquet    # 439.290 cặp tập test: u, v, label, 10 đặc trưng, proba
    ├── drugs.csv             # mã + tên thuốc
    ├── shap_global.csv       # mean |SHAP| trên 500 cặp tập test
    └── model_info.json
```
