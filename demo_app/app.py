"""
Demo: Dự đoán tương tác thuốc–thuốc (DDI) bằng Random Forest + giải thích SHAP.
Chỉ dùng các cặp thuốc thuộc TẬP KIỂM TRA (mô hình chưa từng thấy khi huấn luyện).

Chạy:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Dự đoán tương tác thuốc", page_icon="💊", layout="wide")

# ---------------------------------------------------------------- mô tả đặc trưng
FEAT_INFO = {
    "cn":           ("CN",   "Tô-pô",     "Số láng giềng chung"),
    "jc":           ("JC",   "Tô-pô",     "Hệ số Jaccard"),
    "aai":          ("AAI",  "Tô-pô",     "Chỉ số Adamic/Adar"),
    "pa":           ("PA",   "Tô-pô",     "Gắn kết ưu tiên (tích bậc)"),
    "rai":          ("RAI",  "Tô-pô",     "Chỉ số phân bổ tài nguyên"),
    "ccn":          ("CCN",  "Cộng đồng", "Láng giềng chung có nhận biết cộng đồng"),
    "cra":          ("CRA",  "Cộng đồng", "Phân bổ tài nguyên có nhận biết cộng đồng"),
    "wic":          ("WIC",  "Cộng đồng", "Láng giềng chung trong/ngoài cộng đồng"),
    "sim_mesh":     ("MeSH", "Ngữ nghĩa", "Tương đồng thuật ngữ MeSH"),
    "sim_chemical": ("CHEM", "Ngữ nghĩa", "Tương đồng cấu trúc hóa học (Tanimoto)"),
    "sim_atc":      ("ATC",  "Ngữ nghĩa", "Tương đồng phân loại ATC"),
    "sim_ade":      ("ADE",  "Ngữ nghĩa", "Tương đồng tác dụng phụ"),
}


def short(f):
    return FEAT_INFO.get(f, (f,))[0]


# ---------------------------------------------------------------- nạp dữ liệu
@st.cache_resource(show_spinner="Đang nạp mô hình…")
def load_model():
    model = joblib.load(DATA / "rf_model.joblib")
    return model, shap.TreeExplainer(model)


@st.cache_data(show_spinner="Đang nạp dữ liệu tập kiểm tra…")
def load_data():
    info = json.loads((DATA / "model_info.json").read_text(encoding="utf-8"))
    pairs = pd.read_parquet(DATA / "test_pairs.parquet")
    drugs = pd.read_csv(DATA / "drugs.csv")
    names = dict(zip(drugs["drugbank_id"], drugs["name"].fillna(drugs["drugbank_id"])))
    shap_global = pd.read_csv(DATA / "shap_global.csv")
    return info, pairs, names, shap_global


@st.cache_data
def partner_index(pairs):
    """Thuốc -> danh sách (thuốc kia, chỉ số dòng) trong tập test."""
    a = pd.DataFrame({"drug": pairs["u"], "other": pairs["v"], "row": pairs.index, "label": pairs["label"]})
    b = pd.DataFrame({"drug": pairs["v"], "other": pairs["u"], "row": pairs.index, "label": pairs["label"]})
    return pd.concat([a, b], ignore_index=True)


model, explainer = load_model()
info, pairs, NAMES, shap_global = load_data()
FEATURES = info["features"]
idx = partner_index(pairs)
ALL_DRUGS = sorted(idx["drug"].unique(), key=lambda d: (NAMES.get(d, d) == d, NAMES.get(d, d)))
HAS_NAMES = any(NAMES.get(d, d) != d for d in ALL_DRUGS)


def label_of(d):
    n = NAMES.get(d, d)
    return f"{n} ({d})" if n != d else d


def link(d):
    return f"[{label_of(d)}](https://go.drugbank.com/drugs/{d})"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("💊 Dự đoán tương tác thuốc")
    st.caption("Dự đoán tương tác thuốc–thuốc trên DrugBank 5.0 bằng Random Forest, "
               "chọn lọc đặc trưng dựa trên SHAP.")
    threshold = st.slider("Ngưỡng phân loại", 0.05, 0.95, 0.50, 0.05,
                          help="Xác suất ≥ ngưỡng → dự đoán CÓ tương tác.")
    st.divider()
    st.markdown("**Mô hình**")
    st.markdown(f"- Random Forest, {info['n_estimators']} cây, độ sâu {info['max_depth']}\n"
                f"- {len(FEATURES)} đặc trưng (đã bỏ ATC, ADE theo SHAP)\n"
                f"- Tập kiểm tra: **{len(pairs):,}** cặp thuốc")
    m = info["test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("F1", f"{m['F1']:.2f}%")
    c2.metric("AUPR", f"{m['AUPR']:.2f}%")
    st.caption("🔒 Mọi cặp thuốc trong demo đều thuộc **tập kiểm tra** – mô hình chưa từng thấy khi huấn luyện.")

tab_pred, tab_eval, tab_top = st.tabs(
    ["🔍 Dự đoán cặp thuốc", "📊 Đánh giá trên tập kiểm tra", "🧪 Tương tác tiềm năng"])

# ================================================================ TAB 1: DỰ ĐOÁN
rng = np.random.default_rng()


def pick_random(kind):
    if kind == "pos":
        pool = pairs.index[pairs["label"] == 1]
    elif kind == "neg":
        pool = pairs.index[pairs["label"] == 0]
    else:
        pool = pairs.index
    r = pairs.loc[rng.choice(pool)]
    st.session_state["drug_a"], st.session_state["drug_b"] = r["u"], r["v"]
    st.session_state["filter_b"] = "Tất cả"


with tab_pred:
    st.subheader("Chọn hai thuốc để dự đoán")
    b1, b2, b3, _ = st.columns([1.3, 1.3, 1, 2])
    b1.button("🎲 Cặp CÓ tương tác", on_click=pick_random, args=("pos",), width="stretch")
    b2.button("🎲 Cặp KHÔNG tương tác", on_click=pick_random, args=("neg",), width="stretch")
    b3.button("🎲 Ngẫu nhiên", on_click=pick_random, args=("any",), width="stretch")

    if "drug_a" not in st.session_state:   # mở đầu bằng một cặp dương dự đoán đúng, tự tin
        r0 = pairs[(pairs["label"] == 1) & (pairs["proba"] > 0.9)].iloc[0]
        st.session_state["drug_a"], st.session_state["drug_b"] = r0["u"], r0["v"]

    ca, cb = st.columns(2)
    with ca:
        drug_a = st.selectbox("Thuốc A", ALL_DRUGS, key="drug_a", format_func=label_of)
    part = idx[idx["drug"] == drug_a]
    with cb:
        filt = st.radio("Lọc thuốc B theo nhãn DrugBank", ["Tất cả", "Có tương tác", "Không tương tác"],
                        horizontal=True, key="filter_b")
        if filt == "Có tương tác":
            part = part[part["label"] == 1]
        elif filt == "Không tương tác":
            part = part[part["label"] == 0]
        options = sorted(part["other"], key=lambda d: (NAMES.get(d, d) == d, NAMES.get(d, d)))
        if st.session_state.get("drug_b") not in options:
            st.session_state["drug_b"] = options[0] if options else None
        drug_b = st.selectbox(f"Thuốc B ({len(options):,} thuốc tạo cặp với A trong tập test)",
                              options, key="drug_b", format_func=label_of)

    if drug_b is None:
        st.warning("Không có thuốc B phù hợp với bộ lọc.")
        st.stop()

    row_id = int(part.loc[part["other"] == drug_b, "row"].iloc[0])
    row = pairs.loc[row_id]
    X = row[FEATURES].to_numpy(np.float32).reshape(1, -1)
    proba = float(model.predict_proba(X)[0, 1])        # tính trực tiếp bằng mô hình
    pred = int(proba >= threshold)
    truth = int(row["label"])

    st.divider()
    st.markdown(f"#### {link(drug_a)} + {link(drug_b)}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Xác suất tương tác", f"{proba:.1%}")
    k2.metric("Dự đoán", "CÓ" if pred else "KHÔNG")
    k3.metric("Nhãn thật (DrugBank)", "CÓ" if truth else "KHÔNG")
    if pred == truth:
        k4.success("✅ Dự đoán đúng")
    elif pred == 1:
        k4.warning("⚠️ Dương tính giả – có thể là tương tác tiềm năng chưa được ghi nhận")
    else:
        k4.error("❌ Âm tính giả – mô hình bỏ sót")

    st.progress(proba, text=f"Ngưỡng hiện tại: {threshold:.2f}")

    # ---------- SHAP cho cặp đang chọn
    sv = explainer.shap_values(X, check_additivity=False)
    if isinstance(sv, list):    # shap bản cũ: [lớp 0, lớp 1]
        contrib = np.asarray(sv[1])[0]
    else:
        sv = np.asarray(sv)
        contrib = sv[0, :, 1] if sv.ndim == 3 else sv[0]   # (1, n_feat, 2)
    base = explainer.expected_value
    base = float(np.ravel(base)[1] if np.size(base) > 1 else np.ravel(base)[0])

    exp_df = pd.DataFrame({
        "feature": FEATURES,
        "Đặc trưng": [short(f) for f in FEATURES],
        "Nhóm": [FEAT_INFO.get(f, ("", "?"))[1] for f in FEATURES],
        "Ý nghĩa": [FEAT_INFO.get(f, ("", "", ""))[2] for f in FEATURES],
        "Giá trị": row[FEATURES].astype(float).values,
        "SHAP": contrib,
    }).sort_values("SHAP", key=np.abs, ascending=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("##### Vì sao mô hình đưa ra dự đoán này? (giá trị SHAP)")
        fig = go.Figure(go.Bar(
            x=exp_df["SHAP"], y=[f"{a} = {v:.3g}" for a, v in zip(exp_df["Đặc trưng"], exp_df["Giá trị"])],
            orientation="h",
            marker_color=["#d62728" if s > 0 else "#1f77b4" for s in exp_df["SHAP"]],
            text=[f"{s:+.3f}" for s in exp_df["SHAP"]], textposition="outside", cliponaxis=False,
        ))
        lim = float(np.abs(exp_df["SHAP"]).max()) * 1.35 + 1e-9
        fig.update_xaxes(range=[-lim, lim])
        fig.update_layout(height=420, margin=dict(l=10, r=40, t=10, b=10),
                          xaxis_title="Đóng góp vào xác suất (đỏ: tăng, xanh: giảm)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Xác suất nền (trung bình trên tập huấn luyện) = {base:.3f}; "
                   f"cộng tổng các đóng góp SHAP = {base + contrib.sum():.3f} ≈ xác suất dự đoán.")
    with right:
        st.markdown("##### Giá trị đặc trưng của cặp thuốc")
        st.dataframe(exp_df.iloc[::-1][["Đặc trưng", "Nhóm", "Giá trị", "SHAP", "Ý nghĩa"]]
                     .style.format({"Giá trị": "{:.4f}", "SHAP": "{:+.4f}"}),
                     hide_index=True, width="stretch", height=420)

# ================================================================ TAB 2: ĐÁNH GIÁ
with tab_eval:
    st.subheader("Hiệu năng trên toàn bộ tập kiểm tra")
    y = pairs["label"].to_numpy()
    p = pairs["proba"].to_numpy()
    yp = (p >= threshold).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum()); fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum()); tn = int(((yp == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    spec = tn / (tn + fp) if tn + fp else 0

    @st.cache_data
    def pr_data(y, p):
        pr, rc, _ = precision_recall_curve(y, p)
        step = max(1, len(pr) // 2000)
        return pr[::step], rc[::step], average_precision_score(y, p)

    pr, rc, aupr = pr_data(y, p)
    st.caption(f"{len(y):,} cặp: {int(y.sum()):,} có tương tác, {int((1 - y).sum()):,} không "
               f"(tỷ lệ mất cân bằng tự nhiên ≈ {(1 - y).sum() / y.sum():.2f}:1). Ngưỡng = {threshold:.2f}.")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("F1", f"{f1:.2%}"); m2.metric("Precision", f"{prec:.2%}"); m3.metric("Recall", f"{rec:.2%}")
    m4.metric("Specificity", f"{spec:.2%}"); m5.metric("AUPR", f"{aupr:.2%}")

    g1, g2 = st.columns(2)
    with g1:
        z = [[tn, fp], [fn, tp]]
        fig = go.Figure(go.Heatmap(z=z, x=["Dự đoán: Không", "Dự đoán: Có"], y=["Thật: Không", "Thật: Có"],
                                   colorscale="Blues", showscale=False,
                                   text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}",
                                   textfont=dict(size=18)))
        fig.update_layout(title="Ma trận nhầm lẫn", height=380, yaxis_autorange="reversed",
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with g2:
        fig = go.Figure(go.Scatter(x=rc, y=pr, mode="lines", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=[rec], y=[prec], mode="markers", marker=dict(size=12, color="red"),
                                 name="Ngưỡng hiện tại"))
        fig.update_layout(title=f"Đường Precision–Recall (AUPR = {aupr:.2%})", height=380, showlegend=False,
                          xaxis_title="Recall", yaxis_title="Precision", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    st.info("Kéo thanh **Ngưỡng phân loại** ở thanh bên để thấy sự đánh đổi giữa Precision và Recall.")

# ================================================================ TAB 3: TIỀM NĂNG
with tab_top:
    st.subheader("Các cặp chưa được DrugBank ghi nhận nhưng mô hình chấm điểm cao")
    st.markdown("Đây là ứng dụng thực tế của mô hình: **sàng lọc sơ bộ** để ưu tiên kiểm chứng "
                "những cặp thuốc có nguy cơ tương tác cao nhất. Danh sách chỉ lấy từ tập kiểm tra.")
    n = st.slider("Số cặp hiển thị", 10, 200, 30, 10)
    top = pairs[pairs["label"] == 0].nlargest(n, "proba")
    show = pd.DataFrame({
        "Thuốc A": [label_of(d) for d in top["u"]],
        "Thuốc B": [label_of(d) for d in top["v"]],
        "Xác suất": top["proba"].values,
        "DrugBank A": [f"https://go.drugbank.com/drugs/{d}" for d in top["u"]],
        "DrugBank B": [f"https://go.drugbank.com/drugs/{d}" for d in top["v"]],
    })
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={
                     "Xác suất": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                     "DrugBank A": st.column_config.LinkColumn(display_text="mở"),
                     "DrugBank B": st.column_config.LinkColumn(display_text="mở"),
                 })
    st.caption("Lưu ý: đây là giả thuyết của mô hình, cần chuyên gia và tài liệu y khoa thẩm định.")
