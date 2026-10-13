import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. PAGE CONFIG
st.set_page_config(page_title="Pantau Tumbuh Balita", page_icon="🩺", layout="centered")

# 2. UNIVERSAL CSS
st.markdown("""
<style>
    .main-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #f9fbfc;
        margin-bottom: 20px;
    }
    .result-box {
        text-align: center;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: bold; 
        background-color: #007bff; 
        color: white; 
        border: none; 
        padding: 10px;
    }
    .stButton>button:hover { 
        background-color: #0056b3; 
        color: white; 
    }
</style>
""", unsafe_allow_html=True)

# 3. LOAD ASSETS
@st.cache_resource
def load_assets():
    # Load pipeline utuh menggunakan joblib
    pipeline = joblib.load("model_stunted_Dengan_SMOTE.joblib")
    le_jk = joblib.load("le_jk.joblib")
    le_target = joblib.load("le_target.joblib")
    return pipeline, le_jk, le_target

pipeline, le_jk, le_target = load_assets()

# 4. UI INPUT
st.title("Deteksi Dini Stunted Balita")
st.markdown("Website skrining indikasi stunted (tinggi/panjang badan menurut usia) pada balita.")

with st.form("input_form"):
    jk = st.selectbox("Jenis Kelamin", list(le_jk.classes_))

    col1, col2 = st.columns(2)
    with col1:
        thn = st.number_input("Umur (Tahun)", min_value=1, max_value=4, value=1)
    with col2:
        bln = st.number_input("Umur (Bulan)", min_value=0, max_value=11, value=0)

    bb = st.number_input("Berat (kg)", min_value=0.0, max_value=50.0, value=0.0, help="Wajib diisi")
    tb = st.number_input("Tinggi (cm)", min_value=0.0, max_value=150.0, value=0.0, help="Wajib diisi")
    lila = st.number_input("LiLA (cm)", min_value=0.0, max_value=30.0, value=0.0, help="Wajib diisi")
    submitted = st.form_submit_button("Analisis Status Pertumbuhan")

# 5. PREDICTION LOGIC
if submitted:
    if bb <= 0 or tb <= 0 or lila <= 0:
        st.error("⚠️ **Mohon lengkapi semua data pengukuran dengan angka yang valid!**")
    else:
        usia_total = (thn * 12) + bln
        jk_enc = le_jk.transform([jk])[0]

        # Buat DataFrame untuk input agar sesuai dengan format training XGBoost Pipeline
        input_data = pd.DataFrame(
            [[jk_enc, usia_total, bb, tb, lila]], 
            columns=["JK_enc", "Usia_Bulan", "Berat", "Tinggi", "LiLA"]
        )

        # Prediksi menggunakan pipeline
        probs = pipeline.predict_proba(input_data)[0]
        idx = np.argmax(probs)
        label = le_target.inverse_transform([idx])[0]

        is_stunted = label in ["Pendek", "Sangat Pendek"]
        color_hex = "#dc3545" if is_stunted else "#28a745"

        st.markdown(f'<div class="result-box" style="background-color: {color_hex}; color: white;"><h2>Status: {label}</h2></div>', unsafe_allow_html=True)

        st.markdown("### 📋 Rekomendasi")
        if is_stunted:
            st.warning("⚠️ **Perlu Perhatian:** Status pertumbuhan mengindikasikan kondisi stunted. Disarankan untuk segera berkonsultasi dengan dokter spesialis anak atau ahli gizi di faskes terdekat untuk penanganan klinis lebih lanjut.")
        else:
            st.success("✅ **Pertahankan:** Status pertumbuhan anak tergolong baik. Tetap berikan pola makan bergizi seimbang dan pantau secara rutin.")

        st.info(f"Detail Data: Usia {usia_total} bulan, BB {bb}kg, TB {tb}cm, LiLA {lila}cm.")

# 6. DISCLAIMER
st.markdown("---")
st.warning("⚠️ **Disclaimer:** Hasil klasifikasi pada aplikasi ini hanya bersifat sebagai **informasi awal** dan **tidak menggantikan** pemeriksaan serta diagnosis oleh tenaga kesehatan profesional. Untuk penanganan lebih lanjut, segera konsultasikan kondisi anak ke dokter atau petugas kesehatan terdekat.")