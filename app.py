import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# --- 1. CONFIG ---
st.set_page_config(
    page_title="Analisis Kerentanan Jabar",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. CSS ADAPTIF ---
st.markdown("""
    <style>
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #2563eb;
        color: white;
    }
    /* Warna kustom untuk info box agar konsisten */
    .blue-box { background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3; color: #0d47a1; }
    .orange-box { background-color: #fff3e0; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9800; color: #e65100; }
    .red-box { background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #f44336; color: #b71c1c; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD DATA & MODELS ---
@st.cache_data
def get_data():
    try:
        data = pd.read_csv('dataset_final_label.csv')
        if data['harga_cabai'].dtype == 'object':
            data['harga_cabai'] = data['harga_cabai'].str.replace('.', '', regex=False).astype(float)
        return data
    except:
        return None

df = get_data()
# Memastikan file joblib ada
try:
    scaler = joblib.load('scaler_final.joblib')
    model = joblib.load('knn_final_model.joblib')
except:
    st.error("File model atau scaler tidak ditemukan!")

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🛡️ Navigator")
    menu = st.radio("Pilih Menu:", ["🏠 Beranda", "📊 Visualisasi", "🔮 Prediksi"])
    st.markdown("---")

# --- 5. LOGIKA HALAMAN ---

if menu == "🏠 Beranda":
    st.title("Sistem Analisis Kerentanan Wilayah")
    st.subheader("Tentang Dataset")
    st.markdown("""
    Dataset ini berisi indikator ekonomi dan sosial dari berbagai Kabupaten/Kota di Jawa Barat. 
    Tujuannya adalah mengklasifikasikan wilayah ke dalam status **Aman, Rentan, atau Rawan**.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Fitur Ekonomi:**\n- Harga Cabai\n- Produksi GKG")
    with col2:
        st.info("**Fitur Sosial:**\n- Kemiskinan\n- Stunting\n- Air Bersih")

elif menu == "📊 Visualisasi":
    st.title("📊 Eksplorasi Data")
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.write("### Sebaran Label")
        # Mengatur warna pie chart agar sesuai: Aman (Blue), Rentan (Orange), Rawan (Red)
        fig_pie = px.pie(df, names='Label', hole=0.4, 
                         color='Label',
                         color_discrete_map={'Aman':'blue', 'Rentan':'orange', 'Rawan':'red'})
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with c2:
        st.write("### Detail Data")
        st.dataframe(df, use_container_width=True)

elif menu == "🔮 Prediksi":
    st.title("🔮 Kalkulator Klasifikasi")
    
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            h_cabai = st.number_input("Harga Cabai", value=3500000)
            v_gkg = st.number_input("GKG", value=636.0)
            v_miskin = st.number_input("Kemiskinan (%)", value=9.5)
        with c2:
            v_stunting = st.number_input("Stunting (%)", value=24.5)
            v_air = st.number_input("Air Bersih (%)", value=90.0)
        
        if st.button("🚀 Jalankan Prediksi"):
            input_df = pd.DataFrame([[h_cabai, v_gkg, v_miskin, v_stunting, v_air]], 
                                     columns=['harga_cabai', 'gkg', 'kemiskinan', 'stunting', 'air_bersih'])
            scaled = scaler.transform(input_df)
            res = model.predict(scaled)[0]
            
            label_map = {0: 'Aman', 1: 'Rentan', 2: 'Rawan'}
            hasil = label_map[res]
            
            st.divider()
            
            # Pengkondisian warna output sesuai permintaan
            if hasil == 'Aman':
                st.markdown(f'<div class="blue-box"><h3>Hasil: {hasil}</h3> Wilayah cenderung stabil dan memiliki ketahanan pangan yang baik.</div>', unsafe_allow_html=True)
                st.balloons()
            elif hasil == 'Rentan':
                st.markdown(f'<div class="orange-box"><h3>Hasil: {hasil}</h3> Wilayah memerlukan pengawasan berkala terhadap indikator sosial-ekonomi.</div>', unsafe_allow_html=True)
            elif hasil == 'Rawan':
                st.markdown(f'<div class="red-box"><h3>Hasil: {hasil}</h3> Wilayah memerlukan intervensi kebijakan segera untuk mencegah krisis.</div>', unsafe_allow_html=True)