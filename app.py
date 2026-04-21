import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from src.main import CitrusClassifier

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Citrus Classifier ML",
    page_icon="🍊",
    layout="wide"
)

# --- CACHE DATA & MODEL TRAINING ---
# Menggunakan cache agar model tidak dilatih ulang setiap kali slider digeser
@st.cache_resource
def init_system():
    current_dir = Path(__file__).parent
    data_path = current_dir / "data" / "citrus.csv"
    
    classifier = CitrusClassifier(data_path=data_path)
    classifier.load_data()
    classifier.preprocess_data()
    df_metrics = classifier.train_and_evaluate()
    return classifier, df_metrics

# Load sistem
try:
    classifier, df_metrics = init_system()
except Exception as e:
    st.error(f"Gagal memuat sistem: Pastikan file citrus.csv ada di folder data/. Detail error: {e}")
    st.stop()

# --- ANTARMUKA UTAMA ---
st.title("Dasbor Klasifikasi Sitrus (Jeruk vs Grapefruit)")
st.markdown("Aplikasi *Machine Learning* untuk mengklasifikasikan jenis buah berdasarkan karakteristik fisik menggunakan perbandingan 3 algoritma.")

# Membuat 2 Tab
tab1, tab2 = st.tabs(["Evaluasi & Komparasi Model", "Uji Coba Prediksi Interaktif"])

# TAB 1: EVALUASI
with tab1:
    st.subheader("Komparasi Akurasi Algoritma")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Membuat Grafik Bar
        fig = px.bar(
            df_metrics.sort_values("Accuracy (%)", ascending=True), 
            x="Accuracy (%)", 
            y="Model", 
            orientation='h',
            color="Accuracy (%)",
            color_continuous_scale="Viridis",
            text="Accuracy (%)"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.dataframe(df_metrics.sort_values("Accuracy (%)", ascending=False), hide_index=True)
        st.info("**Kesimpulan:**\nModel Support Vector Machine (SVM) umumnya memberikan performa terbaik untuk dataset ini.")

# TAB 2: UJI COBA INTERAKTIF
with tab2:
    st.subheader("Masukkan Karakteristik Buah")
    
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        st.markdown("#### Dimensi Fisik")
        diameter = st.slider("Diameter (cm)", min_value=2.0, max_value=20.0, value=8.0, step=0.1)
        weight = st.slider("Berat / Weight (gram)", min_value=50.0, max_value=300.0, value=150.0, step=1.0)
        
        st.markdown("#### Pilih Algoritma")
        selected_model = st.selectbox(
            "Gunakan model untuk menebak:",
            ("Support Vector Machine (SVM)", "Naive Bayes", "Decision Tree")
        )

    with input_col2:
        st.markdown("#### Komposisi Warna (RGB)")
        red = st.slider("Intensitas Merah (Red)", 0, 255, 150)
        green = st.slider("Intensitas Hijau (Green)", 0, 255, 75)
        blue = st.slider("Intensitas Biru (Blue)", 0, 255, 20)
        
        st.markdown(f"**Pratinjau Warna:**")
        st.markdown(f'<div style="background-color: rgb({red}, {green}, {blue}); height: 50px; border-radius: 10px; border: 1px solid #ccc;"></div>', unsafe_allow_html=True)
        
    st.divider()
    
    if st.button("Lakukan Prediksi", type="primary", use_container_width=True):
        input_data = {
            'diameter': diameter,
            'weight': weight,
            'red': red,
            'green': green,
            'blue': blue
        }
        
        # Eksekusi prediksi
        hasil = classifier.predict_new_data(input_data, selected_model)
        
        if hasil.lower() == 'orange':
            st.success(f"### Hasil Prediksi: Ini adalah JERUK (Orange)")
        else:
            st.warning(f"### Hasil Prediksi: Ini adalah ANGGUR (Grapefruit)")