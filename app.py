import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

# Import modul buatan sendiri
from models_inference import get_inference_model
from gradcam_inference import apply_gradcam

# ============================================================
# CONFIGURASI APP
# ============================================================
st.set_page_config(page_title="Pneumonia Detection & Grad-CAM App", layout="wide")

CLASS_NAMES = {
    0: "No Disease (Normal)",
    1: "Bacterial Pneumonia",
    2: "Viral Pneumonia"
}

# Fungsi cache agar model tidak di-download ulang setiap kali user klik tombol
@st.cache_resource
def load_model_from_hf():
    repo_id = "laisalkk/pneumonia-classification-deeplearning"
    filename = "E5_EfficientNetB0_best.pth"
    
    with st.spinner("Mengunduh model terbaik dari Hugging Face... Harap tunggu..."):
        # Download file pth dari HF ke cache server Streamlit
        model_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        # Bangun arsitektur EfficientNetB0 (3 kelas)
        model = get_inference_model("EfficientNetB0", num_classes=3)
        
        # Load weights checkpoint secara aman
        checkpoint = torch.load(model_file_path, map_location=torch.device('cpu'))
        
        # REVISI: Cek struktur penyimpanan checkpoint secara adaptif
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Jika file pth langsung berisi matriks bobot murni (tanpa dictionary wrapper)
            model.load_state_dict(checkpoint)
            
        model.eval()
        
    return model

# Load model
try:
    model = load_model_from_hf()
except Exception as e:
    st.error(f"Gagal memuat model dari Hugging Face: {e}")
    st.stop()

# ============================================================
# HEADER GUI
# ============================================================
st.title("🫁 Pneumonia Deep Learning Classification & XAI (Grad-CAM)")
st.write("Aplikasi interpretasi medis otomatis untuk mendeteksi tipe Pneumonia menggunakan arsitektur **EfficientNetB0**.")
st.markdown("---")

# ============================================================
# AREA UPLOAD GAMBAR
# ============================================================
uploaded_file = st.file_uploader("Silahkan pilih atau drag-and-drop gambar Rontgen Dada (X-Ray)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Buka Gambar Menggunakan PIL
    pil_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(pil_image)
    
    # 2. Transformasi Gambar untuk Input Model (Sesuai Pipeline Validation)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = eval_transform(pil_image).unsqueeze(0) # Tambah dimensi batch [1, 3, 224, 224]
    
    # ============================================================
    # PROSES PREDIKSI & GRAD-CAM
    # ============================================================
    with st.spinner("Sedang menganalisis struktur citra medis..."):
        # Jalankan inferensi forward pass
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).detach().numpy()
        pred_class = np.argmax(probabilities)
        
        # Target layer untuk EfficientNetB0 Grad-CAM biasanya adalah bagian akhir dari features block
        target_layer = model.features[-1]
        
        # Generate gambar hasil visualisasi Grad-CAM
        gradcam_img = apply_gradcam(model, target_layer, input_tensor, pred_class, original_np)

    # ============================================================
    # TAMPILAN OUTPUT UTAMA (BERSANDING)
    # ============================================================
    st.subheader("📸 Visualisasi Perbandingan Citra Medis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_np, caption="Gambar Rontgen Asli (User Upload)", use_container_width=True)
        
    with col2:
        st.image(gradcam_img, caption=f"Peta Aktivasi Grad-CAM (Prediksi: {CLASS_NAMES[pred_class]})", use_container_width=True)
        
    st.markdown("---")
    
    # ============================================================
    # DETAIL PRESENTASE AKURASI / PROBABILITAS (BAGIAN BAWAH)
    # ============================================================
    st.subheader("📊 Hasil Analisis Probabilitas Model")
    
    # Berikan highlight text untuk kelas dengan probabilitas tertinggi
    st.success(f"**Hasil Diagnosis Utama: {CLASS_NAMES[pred_class]} ({probabilities[pred_class]*100:.2f}%)**")
    
    # Tampilkan persentase masing-masing kelas menggunakan progress bar
    for class_idx, class_name in CLASS_NAMES.items():
        prob_value = probabilities[class_idx]
        prob_percentage = prob_value * 100
        
        # Tampilkan teks label dan persentasenya
        st.write(f"**{class_name}** : {prob_percentage:.2f}%")
        # Tampilkan progress bar visualnya
        st.progress(float(prob_value))

else:
    st.info("💡 Petunjuk: Silahkan upload gambar file Rontgen dada di atas untuk memulai analisis otomatis.")
