# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from PIL import Image

# Daftarkan folder Source_Code ke sistem path Python agar modul internal bisa di-import
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'Source_Code'))

# Import modul internal Anda dengan penyesuaian fungsi
from config import CLASS_NAMES, IMG_SIZE
from models import get_model
from augmentation import get_transforms  # Menggunakan fungsi utama standar Anda

# Config Halaman Streamlit
st.set_page_config(
    page_title="Pneumonia Detection & XAI Dashboard",
    page_icon="🩺",
    layout="centered"
)

# ============================================================
# CACHED FUNCTIONS (Agar aplikasi web berjalan cepat)
# ============================================================

@st.cache_resource
def load_trained_model():
    """Fungsi untuk memuat model terbaik E5_EfficientNetB0_best"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # PERBAIKAN 1: Menyesuaikan nama file dengan akhiran _best.pth sesuai repositori
    checkpoint_path = os.path.join(BASE_DIR, "Results", "checkpoints", "E5_EfficientNetB0_best.pth")
    
    # Bangun arsitektur kosong
    model = get_model(model_name="EfficientNetB0", num_classes=3)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        st.error(f"File model '{checkpoint_path}' tidak ditemukan! Pastikan folder Results/checkpoints sudah benar di GitHub.")
        st.stop()
        
    model = model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def create_shap_explainer(_model, device):
    """Membuat basis reference data (background) tiruan untuk kestabilan kalkulasi SHAP"""
    background = torch.zeros(3, 3, IMG_SIZE, IMG_SIZE).to(device)
    explainer = shap.GradientExplainer(_model, background)
    return explainer

# Load Model dan Explainer secara global saat web dibuka
model, device = load_trained_model()
explainer = create_shap_explainer(model, device)

# ============================================================
# USER INTERFACE (Tampilan Web)
# ============================================================

st.title("🩺 Pneumonia Detection App with Explainable AI")
st.markdown("""
Aplikasi web ini mendeteksi penyakit **Pneumonia (Bakteri/Virus)** menggunakan model **EfficientNetB0 (Akurasi: 86.73%)** dan dilengkapi transparansi medis menggunakan **SHAP (Peta Kontribusi Piksel)**.
""")

st.write("---")

# Widget Upload Gambar
uploaded_file = st.file_uploader("Unggah Citra Rontgen Paru-Paru (X-Ray Dada)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Tampilkan Gambar Asli yang diunggah User
    image = Image.open(uploaded_file).convert("RGB")  # Convert ke RGB untuk kecocokan model 3-channel
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar Rontgen Unggahan", use_container_width=True)
    
    # PERBAIKAN 2: Mengambil transformasi pengujian dari fungsi get_transforms() Anda
    _, _, test_transform = get_transforms()
    img_tensor = test_transform(image) # Mengubah menjadi tensor 3-channel terstandardisasi
    img_tensor = img_tensor.unsqueeze(0) # Tambah dimensi batch -> [1, 3, 224, 224]
    
    with col2:
        st.write("### 📊 Hasil Diagnosis Model")
        run_prediction = st.button("Jalankan Diagnosis Otomatis", type="primary")
        
    if run_prediction:
        with st.spinner("Model sedang menganalisis karakteristik visual rontgen..."):
            # Prediksi Utama
            with torch.no_grad():
                outputs = model(img_tensor.to(device))
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_class = np.argmax(probabilities)
                
            # Tampilkan Hasil Utama
            predicted_label = CLASS_NAMES[pred_class]
            st.success(f"**Diagnosis Utama: {predicted_label}**")
            
            # Tampilkan Nilai Probabilitas Masing-Masing Kelas
            prob_df = pd.DataFrame({
                'Kategori': [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]],
                'Keyakinan Model (%)': [p * 100 for p in probabilities]
            })
            st.dataframe(prob_df.set_index('Kategori'))
            
        st.write("---")
        
        # 3. FITUR EXPLAINABLE AI (SHAP VISUALIZATION)
        st.write("### 👁️ Analisis Akuntabilitas Piksel (Explainable AI - SHAP)")
        st.info("💡 **Cara Membaca Visualisasi:** Piksel berwarna **merah/merah muda** menunjukkan area yang memperkuat keyakinan model dalam mengambil keputusan diagnosis tersebut.")
        
        with st.spinner("Sedang memproses perhitungan SHAP Values pada piksel gambar..."):
            # Jalankan kalkulasi SHAP Gradient
            shap_values, indexes = explainer.shap_values(img_tensor.to(device), ranked_outputs=1)
            
            # Siapkan gambar numpy untuk matplotlib (Denormalisasi ringan)
            img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # PERBAIKAN 3: Penanganan Dimensi SHAP Adaptif agar Tidak Memicu Error Axes
            if isinstance(shap_values, list):
                shap_val_target = shap_values[0]
            else:
                shap_val_target = shap_values

            if len(shap_val_target.shape) == 4:
                shap_val_target = shap_val_target[0]

            if shap_val_target.shape[0] in [1, 3]:
                shap_val_trans = np.transpose(shap_val_target, (1, 2, 0))
            else:
                shap_val_trans = shap_val_target
            
            # Render plot menggunakan Matplotlib
            fig, ax = plt.subplots(figsize=(6, 6))
            shap.image_plot([shap_val_trans], [img_np], show=False)
            
            # Tampilkan objek gambar matplotlib tadi langsung ke halaman web Streamlit!
            st.pyplot(plt.gcf())
            plt.close()
