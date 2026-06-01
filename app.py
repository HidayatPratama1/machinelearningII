# =============================================================
# app.py — Tahap 10: Aplikasi Streamlit
# IoT Vulnerability Detection System
# Pipeline: SelectKBest (k=38) + RandomForestClassifier
# =============================================================

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["streamlit", "scikit-learn", "pandas", "numpy", "joblib",
            "imbalanced-learn", "matplotlib", "seaborn"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

# ── Import ────────────────────────────────────────────────────
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "IoT Vulnerability Detector",
    page_icon   = "🛡️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS KUSTOM
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Font & warna dasar */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header utama */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 { font-size: 2rem; margin: 0; font-weight: 700; }
    .main-header p  { margin: 0.4rem 0 0 0; opacity: 0.8; font-size: 0.95rem; }

    /* Kartu metrik */
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h3 { font-size: 0.85rem; color: #555; margin: 0 0 0.4rem 0; text-transform: uppercase; }
    .metric-card p  { font-size: 1.7rem; font-weight: 700; margin: 0; color: #1a1a2e; }

    /* Hasil prediksi */
    .result-safe {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 6px solid #28a745;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
    }
    .result-attack {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 6px solid #dc3545;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
    }
    .result-safe h2, .result-attack h2 {
        margin: 0 0 0.3rem 0; font-size: 1.6rem;
    }
    .result-safe p, .result-attack p { margin: 0; font-size: 0.95rem; }

    /* Divider */
    hr { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }

    /* Sidebar */
    .css-1d391kg { background-color: #f0f2f6 !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL — pipeline utuh dari .pkl
# ─────────────────────────────────────────────────────────────
MODEL_PATH = r"F:\Teknik informarika\Semester 6 03PT6\Machine Learning II\MID\MID\pipeline_terbaik.pkl"
LABEL_PATH = "label_encoder.pkl"   # sesuaikan jika berbeda lokasi

@st.cache_resource(show_spinner="Memuat model pipeline...")
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_label_encoder(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

pipeline = load_model(MODEL_PATH)
le       = load_label_encoder(LABEL_PATH)

# ─────────────────────────────────────────────────────────────
# DAFTAR 87 FITUR (urutan persis seperti saat training)
# ─────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "dur", "Protocol", "Length", "Source Host", "Destination Host",
    "Sender IP address", "Target IP address", "Opcode", "checksom(ICMP)",
    "Sequence Number (LE)", "Sequence Number (BE)", "File Data",
    "Content length", "Request URI Query", "Request Method",
    "Full Request URI", "Request Version", "Response", "Ack No",
    "Ack No (RAW)", "Checksum(TCP)", "Connection Finish",
    "Connection Reset", "Connection Establish Request",
    "Connection Establish Ack", "Source Port", "Destination Port",
    "TCPFlags", "Acknowledgment", "TCP Segment Length", "TCP Options",
    "TCP Payload", "TCP Seq No", "Src or Drc port", "Stream index",
    "Time since previous frame", "Query Name", "DNS retransmission",
    "DNS query retransmission", "DNS query retransmission in",
    "LG bit", "IG bit", "LG bit.1",
    "Duplicate IP address configured", "Time to Live",
    "Conversation completeness", "Push", "Content Type",
    "This is an ACK to the segment in frame", "ECN-Echo",
    "Mode", "Type", "Type.1", "Window", "Echo data", "Accept",
    "Status Code", "Transaction ID", "Handshake Type", "Flags",
    "Packet Type", "MSS Value", "Message type", "Timestamp value",
    "TSecr", "TCP Option - SACK permitted", "Response time",
    "No response seen", "Kind", "Duplicate ACK #",
    "This frame is a (suspected) retransmission",
    "Previous segment(s) not captured (common at capture start)",
    "FTP Data", "Bytes in flight", "Request command",
    "Request command.1", "BER Error: length is not valid", "Length.1",
    "Flags.1", "Packet Length (encrypted)", "Direction",
    "TCP Option - Maximum segment size", "Data", "Checksum",
    "CDATA", "Attack_Category", "Attack_sub_category",
]

# Tipe data per fitur (untuk form input)
FLOAT_FEATURES = {
    "dur", "Content length", "Ack No", "Ack No (RAW)", "Source Port",
    "Destination Port", "TCP Segment Length", "TCP Seq No", "Stream index",
    "Time since previous frame", "Type", "Window", "Status Code",
    "MSS Value", "Timestamp value", "TSecr",
    "TCP Option - SACK permitted", "Response time", "Duplicate ACK #",
    "Bytes in flight", "Length.1",
    "This is an ACK to the segment in frame",
}

# Fitur yang dipilih SelectKBest (k=38) — urutan dari hasil feature importance
# (didapat dari feature_importance.csv yang dihasilkan notebook/mlflow)
SELECTED_FEATURES_TOP = [
    "dur", "Length", "Source Host", "Destination Host",
    "Sender IP address", "Target IP address", "Sequence Number (LE)",
    "Sequence Number (BE)", "Content length", "Source Port",
    "Destination Port", "TCP Segment Length", "TCP Seq No",
    "Stream index", "Time since previous frame", "Time to Live",
    "Conversation completeness", "Status Code", "Timestamp value",
    "TSecr", "TCP Option - SACK permitted", "Response time",
    "Bytes in flight", "Protocol", "Ack No", "Ack No (RAW)",
    "TCP Options", "TCP Payload", "Query Name", "LG bit",
    "IG bit", "Window", "MSS Value", "Flags",
    "Handshake Type", "Response", "Checksum(TCP)", "Type",
]

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ IoT Vulnerability Detection System</h1>
    <p>Machine Learning II — Mid Semester Project | Pipeline: SelectKBest (k=38) + Random Forest</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# STATUS MODEL DI SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Status Sistem")

    if pipeline is not None:
        st.success("✅ Model ter-load")
        st.markdown(f"""
        | Info | Nilai |
        |------|-------|
        | **Method** | SelectKBest (Filter) |
        | **k features** | 38 dari 87 |
        | **Model** | RandomForest |
        | **n_estimators** | 100 |
        | **max_depth** | None |
        | **Accuracy** | 100.00% |
        | **F1 Macro** | 1.0000 |
        """)
    else:
        st.error("❌ Model tidak ditemukan!")
        st.code(MODEL_PATH, language=None)
        st.warning("Pastikan path file .pkl sudah benar dan file ada.")

    st.markdown("---")
    st.markdown("## 📋 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini mendeteksi apakah traffic jaringan IoT termasuk:
    - **🟢 Normal** — Traffic aman
    - **🔴 Attack** — Terdeteksi ancaman
    
    **Dataset:** IoT Vulnerability  
    **Samples:** 1,048,575 records  
    **Features:** 87 → dipilih 38
    """)

# ─────────────────────────────────────────────────────────────
# TABS UTAMA
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Manual", "📊 Info Model", "📖 Panduan"])

# ═════════════════════════════════════════════════════════════
# TAB 1: PREDIKSI MANUAL
# ═════════════════════════════════════════════════════════════
with tab1:
    if pipeline is None:
        st.error("⚠️ Model belum ter-load. Periksa path di sidebar.")
        st.stop()

    st.markdown("### 📝 Input Data Traffic Jaringan")
    st.markdown(
        "Isi nilai fitur jaringan di bawah ini. Kolom yang ditampilkan adalah "
        "**38 fitur terpilih** oleh SelectKBest. Fitur lainnya akan otomatis diisi 0."
    )

    # ── Preset contoh ──────────────────────────────────────
    st.markdown("#### 🚀 Preset Cepat")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        normal_clicked = st.button("🟢 Isi Contoh: Normal Traffic", use_container_width=True)
    with col_p2:
        attack_clicked = st.button("🔴 Isi Contoh: Attack Traffic", use_container_width=True)
    with col_p3:
        reset_clicked  = st.button("🔄 Reset ke Nol", use_container_width=True)

    # Default preset values
    DEFAULT_NORMAL = {
        "dur": 0.001, "Length": 74, "Source Host": 1, "Destination Host": 2,
        "Sender IP address": 192, "Target IP address": 10, "Sequence Number (LE)": 1000,
        "Sequence Number (BE)": 1000, "Content length": 0.0, "Source Port": 52345.0,
        "Destination Port": 443.0, "TCP Segment Length": 20.0, "TCP Seq No": 1234567.0,
        "Stream index": 1.0, "Time since previous frame": 0.001, "Time to Live": 64,
        "Conversation completeness": 1, "Status Code": 200.0, "Timestamp value": 1000.0,
        "TSecr": 500.0, "TCP Option - SACK permitted": 1.0, "Response time": 0.002,
        "Bytes in flight": 100.0, "Protocol": 6, "Ack No": 1234568.0,
        "Ack No (RAW)": 1234568.0, "TCP Options": 1, "TCP Payload": 40,
        "Query Name": 0, "LG bit": 0, "IG bit": 0, "Window": 65535.0,
        "MSS Value": 1460.0, "Flags": 18, "Handshake Type": 0, "Response": 0,
        "Checksum(TCP)": 12345, "Type": 0.0,
    }
    DEFAULT_ATTACK = {
        "dur": 0.0, "Length": 1500, "Source Host": 999, "Destination Host": 888,
        "Sender IP address": 0, "Target IP address": 255, "Sequence Number (LE)": 0,
        "Sequence Number (BE)": 0, "Content length": 0.0, "Source Port": 80.0,
        "Destination Port": 80.0, "TCP Segment Length": 1480.0, "TCP Seq No": 0.0,
        "Stream index": 0.0, "Time since previous frame": 0.0, "Time to Live": 1,
        "Conversation completeness": 0, "Status Code": 0.0, "Timestamp value": 0.0,
        "TSecr": 0.0, "TCP Option - SACK permitted": 0.0, "Response time": 0.0,
        "Bytes in flight": 1480.0, "Protocol": 17, "Ack No": 0.0,
        "Ack No (RAW)": 0.0, "TCP Options": 0, "TCP Payload": 1480,
        "Query Name": 9999, "LG bit": 1, "IG bit": 1, "Window": 0.0,
        "MSS Value": 0.0, "Flags": 0, "Handshake Type": 0, "Response": 0,
        "Checksum(TCP)": 0, "Type": 0.0,
    }
    DEFAULT_ZERO = {feat: 0.0 for feat in SELECTED_FEATURES_TOP}

    # State management preset
    if "preset" not in st.session_state:
        st.session_state.preset = "zero"
    if normal_clicked:
        st.session_state.preset = "normal"
    if attack_clicked:
        st.session_state.preset = "attack"
    if reset_clicked:
        st.session_state.preset = "zero"

    preset_map = {
        "normal": DEFAULT_NORMAL,
        "attack": DEFAULT_ATTACK,
        "zero"  : DEFAULT_ZERO,
    }
    current_preset = preset_map[st.session_state.preset]

    st.markdown("---")
    st.markdown("#### 🎛️ Input Nilai Fitur (38 Fitur Terpilih)")

    # ── Form input — tampilkan dalam 3 kolom ───────────────
    input_values = {}
    cols_per_row = 3
    feat_chunks  = [SELECTED_FEATURES_TOP[i:i+cols_per_row]
                    for i in range(0, len(SELECTED_FEATURES_TOP), cols_per_row)]

    for chunk in feat_chunks:
        cols = st.columns(cols_per_row)
        for idx, feat in enumerate(chunk):
            default_val = float(current_preset.get(feat, 0.0))
            if feat in FLOAT_FEATURES:
                input_values[feat] = cols[idx].number_input(
                    feat, value=default_val,
                    format="%.4f", key=f"feat_{feat}"
                )
            else:
                input_values[feat] = float(cols[idx].number_input(
                    feat, value=int(default_val),
                    step=1, key=f"feat_{feat}"
                ))

    st.markdown("---")

    # ── Tombol Prediksi ────────────────────────────────────
    predict_col, _ = st.columns([1, 3])
    with predict_col:
        predict_btn = st.button("🔍 Deteksi Sekarang", type="primary",
                                use_container_width=True)

    if predict_btn:
        # Bangun DataFrame lengkap 87 fitur
        row = {feat: 0.0 for feat in FEATURE_COLUMNS}
        row.update(input_values)  # overwrite dengan nilai yang diinput
        X_input = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # Pipeline otomatis: scaler → SelectKBest → RF
        try:
            prediction = pipeline.predict(X_input)[0]
            proba      = pipeline.predict_proba(X_input)[0]

            is_attack  = (prediction == 1)
            conf       = proba[prediction] * 100
            label_name = "ATTACK" if is_attack else "NORMAL"

            if is_attack:
                st.markdown(f"""
                <div class="result-attack">
                    <h2>🚨 {label_name} TERDETEKSI!</h2>
                    <p>Model mendeteksi <strong>ancaman/serangan</strong> pada traffic jaringan ini.</p>
                    <p>Confidence: <strong>{conf:.2f}%</strong> | Class: <strong>{prediction}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                    <h2>✅ Traffic {label_name}</h2>
                    <p>Model mendeteksi traffic jaringan ini <strong>aman</strong>.</p>
                    <p>Confidence: <strong>{conf:.2f}%</strong> | Class: <strong>{prediction}</strong></p>
                </div>
                """, unsafe_allow_html=True)

            # Detail probabilitas
            st.markdown("#### 📊 Detail Probabilitas")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Probabilitas Normal (0)", f"{proba[0]*100:.2f}%")
            col_b.metric("Probabilitas Attack (1)", f"{proba[1]*100:.2f}%")
            col_c.metric("Confidence Prediksi", f"{conf:.2f}%")

            # Bar chart probabilitas
            fig, ax = plt.subplots(figsize=(5, 2.5))
            colors = ["#28a745" if i == 0 else "#dc3545" for i in range(2)]
            bars = ax.barh(["Normal (0)", "Attack (1)"], proba * 100,
                           color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilitas (%)")
            ax.set_title("Distribusi Probabilitas Prediksi")
            for bar, val in zip(bars, proba * 100):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
            ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"❌ Error saat prediksi: {e}")
            st.exception(e)

# ═════════════════════════════════════════════════════════════
# TAB 2: INFO MODEL
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Informasi Model & Performa")

    # Kartu metrik
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown('<div class="metric-card"><h3>Accuracy</h3><p>100.00%</p></div>',
                unsafe_allow_html=True)
    c2.markdown('<div class="metric-card"><h3>F1 Macro</h3><p>1.0000</p></div>',
                unsafe_allow_html=True)
    c3.markdown('<div class="metric-card"><h3>Precision</h3><p>1.0000</p></div>',
                unsafe_allow_html=True)
    c4.markdown('<div class="metric-card"><h3>Recall</h3><p>1.0000</p></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🔧 Konfigurasi Pipeline")
        st.markdown("""
        | Step | Komponen | Parameter |
        |------|----------|-----------|
        | 1️⃣ Scaler | StandardScaler | default |
        | 2️⃣ Feature Selection | SelectKBest | score_func=f_classif, k=**38** |
        | 3️⃣ Classifier | RandomForestClassifier | n_estimators=**100**, max_depth=**None** |
        """)

        st.markdown("#### 📋 Perbandingan Metode Feature Selection (CV)")
        cv_comp = pd.DataFrame({
            "Metode"         : ["Filter (SelectKBest) ✅", "Wrapper (RFE)", "Embedded (SelectFromModel)"],
            "F1 Macro (mean)": [1.0000, 1.0000, 1.0000],
            "Dipilih"        : ["✅ Terbaik", "—", "—"],
        })
        st.dataframe(cv_comp, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("#### 📈 Performa GridSearchCV")
        st.markdown("""
        | Parameter | Nilai Diuji | Terbaik |
        |-----------|-------------|---------|
        | k (fitur) | 38, 43, 48 | **38** |
        | n_estimators | 100, 200 | **100** |
        | max_depth | None, 10, 20 | **None** |
        | min_samples_split | 2, 5 | **2** |
        """)
        st.markdown("#### 📦 Informasi Dataset")
        st.markdown("""
        | Info | Nilai |
        |------|-------|
        | Total Samples | 1,048,575 |
        | Train Split | 838,860 (80%) |
        | Test Split | 209,715 (20%) |
        | Fitur Asli | 87 |
        | Fitur Dipilih | 38 |
        | Kelas | 2 (Normal / Attack) |
        | Normal (0) | 88,650 |
        | Attack (1) | 959,925 |
        """)

    st.markdown("---")
    st.markdown("#### 🏆 Top 10 Fitur Terpenting")
    top10 = [
        ("dur", 0.180), ("Length", 0.095), ("Source Host", 0.082),
        ("Destination Host", 0.078), ("Stream index", 0.065),
        ("TCP Seq No", 0.058), ("Time since previous frame", 0.052),
        ("Bytes in flight", 0.048), ("Source Port", 0.041),
        ("Destination Port", 0.037),
    ]
    fi_df = pd.DataFrame(top10, columns=["Feature", "Importance"])
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bars2 = ax2.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
                     color=sns.color_palette("viridis", 10)[::-1],
                     edgecolor="black", linewidth=0.4)
    ax2.set_xlabel("Feature Importance")
    ax2.set_title("Top 10 Feature Importances (RandomForest)")
    for bar, val in zip(bars2, fi_df["Importance"][::-1]):
        ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.caption("*Nilai importance estimasi — jalankan notebook untuk nilai eksak dari model Anda.*")

# ═════════════════════════════════════════════════════════════
# TAB 3: PANDUAN
# ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📖 Panduan Penggunaan")

    st.markdown("""
    #### 🚀 Cara Menjalankan Aplikasi
    ```bash
    # Install Streamlit (jika belum)
    pip install streamlit

    # Jalankan dari direktori yang sama dengan app.py
    streamlit run app.py
    ```

    #### 📁 Struktur File yang Dibutuhkan
    ```
    MID/
    ├── app.py                          ← File ini
    ├── mlflow_tracking.py              ← Script MLflow
    ├── pipeline_terbaik.pkl            ← Model pipeline (dari notebook)
    ├── label_encoder.pkl               ← LabelEncoder (dari notebook)
    └── Preprocessed_Balanced_dataset.csv
    ```

    #### 🔍 Cara Prediksi Manual
    1. Buka tab **Prediksi Manual**
    2. Gunakan tombol **Preset** untuk mengisi nilai contoh, atau isi manual
    3. Klik tombol **🔍 Deteksi Sekarang**
    4. Hasil prediksi dan confidence akan ditampilkan

    #### 📊 Cara Menjalankan MLflow UI
    ```bash
    # Jalankan tracking dulu
    python mlflow_tracking.py

    # Buka MLflow UI
    mlflow ui

    # Buka browser ke:
    # http://127.0.0.1:5000
    ```

    #### ⚙️ Mengubah Path Model
    Edit baris berikut di `app.py`:
    ```python
    MODEL_PATH = r"F:\\Teknik informarika\\...\\pipeline_terbaik.pkl"
    ```
    Dan di `mlflow_tracking.py`:
    ```python
    MODEL_PKL = r"F:\\Teknik informarika\\...\\pipeline_terbaik.pkl"
    ```

    #### 🔬 Tentang Pipeline
    Pipeline memuat **3 langkah otomatis**:
    1. **StandardScaler** — normalisasi fitur input
    2. **SelectKBest** (k=38) — seleksi 38 fitur terbaik dari 87
    3. **RandomForestClassifier** — klasifikasi binary (Normal/Attack)

    Anda cukup memasukkan nilai fitur mentah — pipeline akan memproses secara otomatis.
    """)

    st.info("💡 **Tip:** Gunakan tombol preset '🟢 Normal Traffic' atau '🔴 Attack Traffic' "
            "untuk melihat contoh prediksi tanpa harus mengisi semua field manual.")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "🛡️ IoT Vulnerability Detection System | Machine Learning II — Mid Semester | "
    "Pipeline: SelectKBest (k=38) + RandomForest | Accuracy: 100.00%"
    "</p>",
    unsafe_allow_html=True
)
