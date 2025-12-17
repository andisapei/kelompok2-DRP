import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD ARTIFACTS (Model dan Scaler) ---
try:
    # Memuat objek Standard Scaler (Wajib untuk menstandardisasi input pengguna)
    scaler = joblib.load('scaler_final.joblib')
    # Memuat model KNN terbaik (Model Klasifikasi)
    model = joblib.load('knn_final_model.joblib')
except FileNotFoundError:
    st.error("Error: File model atau scaler (.joblib) tidak ditemukan. Pastikan file berada di folder yang sama.")
    st.stop()

# --- 2. DEFINISI GLOBAL ---
# Fitur yang digunakan (Sesuai dengan urutan saat pelatihan model)
FEATURES = ['harga_cabai', 'gkg', 'kemiskinan', 'stunting', 'air_bersih']
# Peta label terbalik (Sesuai dengan encoding saat pelatihan: 0=Aman, 1=Rentan, 2=Rawan)
LABEL_MAP = {0: 'Aman', 1: 'Rentan', 2: 'Rawan'}

# --- 3. FUNGSI PREDIKSI ---
def predict_kerentanan(input_data):
    """Membuat prediksi label kerentanan dari input data."""
    
    # Konversi input ke DataFrame
    input_df = pd.DataFrame([input_data], columns=FEATURES)
    
    # 1. Standardisasi Input (Wajib menggunakan scaler yang sudah dilatih)
    input_scaled = scaler.transform(input_df)
    
    # 2. Prediksi Model
    prediction_numeric = model.predict(input_scaled)[0]
    
    # 3. Konversi hasil numerik ke label teks
    prediction_label = LABEL_MAP.get(prediction_numeric, "Label Tidak Dikenal")
    
    return prediction_label

# --- 4. TAMPILAN APLIKASI STREAMLIT ---
st.set_page_config(page_title="Prediksi Kerentanan Regional", layout="centered")

st.title("Sistem Prediksi Daerah Rawan Pangan")
st.subheader("Berdasarkan Model Klasifikasi KNN")
st.markdown("Masukkan indikator sosial-ekonomi dan harga komoditas untuk memprediksi tingkat kerentanan suatu wilayah.")

# Kolom Input dari Pengguna
st.header("Input Indikator Wilayah")

# Input dalam kolom untuk tampilan yang lebih rapi
col1, col2 = st.columns(2)

# Kolom 1
with col1:
    harga_cabai = st.number_input("1. Harga Cabai Rata-rata (Rp.)", min_value=100000.0, max_value=20000000.0, value=3500000.0, step=100000.0)
    gkg = st.number_input("2. GKG (Gabah Kering Giling)", min_value=500.0, max_value=800.0, value=650.0, step=1.0)
    kemiskinan = st.number_input("3. Persentase Kemiskinan (%)", min_value=1.0, max_value=20.0, value=9.5, step=0.1)

# Kolom 2
with col2:
    stunting = st.number_input("4. Persentase Stunting (%)", min_value=5.0, max_value=35.0, value=25.0, step=0.1)
    air_bersih = st.number_input("5. Akses Air Bersih (%)", min_value=70.0, max_value=100.0, value=90.0, step=0.1)

# Tombol Prediksi
if st.button("Prediksi Tingkat Kerentanan"):
    # Kumpulkan semua input
    input_data = [harga_cabai, gkg, kemiskinan, stunting, air_bersih]
    
    # Lakukan prediksi
    result_label = predict_kerentanan(input_data)
    
    st.markdown("---")
    st.header("Hasil Prediksi")
    
    # Logika Tampilan Hasil (Feedback instan)
    if result_label == 'Aman':
        st.success(f"Wilayah ini diprediksi berada dalam kategori **{result_label}**.")
        st.balloons()
        st.write("Indikator menunjukkan risiko kerentanan yang sangat rendah, mirip dengan Ciamis.")
    elif result_label == 'Rentan':
        st.warning(f"Wilayah ini diprediksi berada dalam kategori **{result_label}**.")
        st.write("Perlu pemantauan, terdapat beberapa indikator di batas rata-rata. Wilayah ini adalah mayoritas klaster.")
    elif result_label == 'Rawan':
        st.error(f"Wilayah ini diprediksi berada dalam kategori **{result_label}**.")
        st.write("Wilayah ini membutuhkan intervensi segera. Indikatornya serupa dengan Bogor, Sukabumi, dan Garut.")