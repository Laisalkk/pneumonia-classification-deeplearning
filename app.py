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
# CONFIGURASI UTAMA GUI & TEMA
# ============================================================
st.set_page_config(
    page_title="Pneumonia Diagnosis AI Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menambahkan CSS Custom agar gaya tampilan mirip dengan dashboard medis premium
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px;
        color: #64748B;
        margin-bottom: 25px;
    }
    .card-normal {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .card-pneumonia {
        background-color: #FEF2F2;
        border-left: 5px solid #DC2626;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 5px;
    }
    </style>
""", unsafe_index=True)

CLASS_NAMES = {
    0: "No Disease (Normal)",
    1: "Bacterial Pneumonia",
    2: "Viral Pneumonia"
}

# ============================================================
# LOGIKA PENGUNDUHAN MODEL (CACHE)
# ============================================================
@st.cache_resource
def load_model_from_hf():
    repo_id = "laisalkk/pneumonia-classification-deeplearning"
    filename = "E5_EfficientNetB0_best.pth"
    
    with st.spinner("Mengunduh konfigurasi bobot model EfficientNetB0..."):
        model_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = get_inference_model("EfficientNetB0", num_classes=3)
        checkpoint = torch.load(model_file_path, map_location=torch.device('cpu'))
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
    return model

try:
    model = load_model_from_hf()
except Exception as e:
    st.error(f"Gagal sinkronisasi dengan Hugging Face: {e}")
    st.stop()

# ============================================================
# SIDEBAR CONTROL PANEL (SESUAI GAMBAR ACUAN)
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843110.png", width=70)
    st.markdown("### **Control Panel**")
    st.write("Unggah data citra rontgen pasien di bawah ini untuk melakukan klasifikasi otomatis.")
    
    uploaded_file = st.file_uploader(
        "Pilih file X-Ray Citra Paru", 
        type=["png", "jpg", "jpeg"],
        help="Menerima format standar JPG, JPEG, atau PNG"
    )
    
    st.markdown("---")
    st.markdown("### **Spesifikasi Sistem**")
    st.info("**Model:** EfficientNetB0\n\n**XAI Tech:** Grad-CAM Core\n\n**Dataset:** Pneumonia Dataset (3 Class)")

# ============================================================
# HALAMAN UTAMA / DASHBOARD VIEW
# ============================================================
st.markdown('<p class="main-title">🫁 Clinical Intelligence Dashboard</p>', unsafe_index=True)
st.markdown('<p class="sub-title">Sistem deteksi Pneumonia berbasis Deep Learning & Explainable AI (XAI)</p>', unsafe_index=True)

if uploaded_file is not None:
    # 1. Loading Gambar & Transformasi
    pil_image = Image.open(uploaded_file).convert("RGB")
    original_np = np.array(pil_image)
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = eval_transform(pil_image).unsqueeze(0)
    
    # 2. Proses Komputasi Model
    with st.spinner("Menganalisis matriks dan pola opasitas citra rontgen..."):
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).detach().numpy()
        pred_class = np.argmax(probabilities)
        
        target_layer = model.features[-1]
        gradcam_img = apply_gradcam(model, target_layer, input_tensor, pred_class, original_np)

    # ============================================================
    # TAMPILAN CONTAINER UTAMA (CARD HASIL MEDIS)
    # ============================================================
    # Tentukan style kartu berdasarkan apakah pasien normal atau sakit
    card_style = "card-normal" if pred_class == 0 else "card-pneumonia"
    status_icon = "✅" if pred_class == 0 else "⚠️"
    
    st.markdown(f"""
        <div class="{card_style}">
            <p class="metric-title">{status_icon} Diagnosis Utama Hasil Analisis Sistem AI</p>
            <p class="metric-value">{CLASS_NAMES[pred_class]} ({probabilities[pred_class]*100:.2f}%)</p>
        </div>
    """, unsafe_index=True)

    # ============================================================
    # TAMPILAN INFRASTRUKTUR CITRA BERSANDING
    # ============================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### **Citra Asli Pasien**")
        st.image(original_np, use_container_width=True)
        
    with col2:
        st.markdown("#### **Visualisasi Lokasi Infeksi (Grad-CAM)**")
        st.image(gradcam_img, use_container_width=True)
        
    st.markdown("---")
    
    # ============================================================
    # PROGRESS METRIC BAR SEPERTI GAMBAR ACUAN BARU
    # ============================================================
    st.markdown("#### **Distribusi Skor & Probabilitas Diagnosis**")
    
    # Membuat 3 kolom kartu metrik persentase di bagian bawah
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"**{CLASS_NAMES[0]}**")
        st.progress(float(probabilities[0]))
        st.caption(f"Skor Keyakinan: {probabilities[0]*100:.2f}%")
        
    with m_col2:
        st.markdown(f"**{CLASS_NAMES[1]}**")
        st.progress(float(probabilities[1]))
        st.caption(f"Skor Keyakinan: {probabilities[1]*100:.2f}%")
        
    with m_col3:
        st.markdown(f"**{CLASS_NAMES[2]}**")
        st.progress(float(probabilities[2]))
        st.caption(f"Skor Keyakinan: {probabilities[2]*100:.2f}%")

else:
    # Tampilan Standby Dashboard saat pertama kali dibuka (Sangat Bersih)
    st.info("👋 Selamat Datang! Silahkan gunakan menu panel di sebelah kiri untuk memasukkan file foto rontgen dada pasien.")
    
    # Tambah placeholder layout estetik saat kosong
    col_empty1, col_empty2 = st.columns(2)
    with col_empty1:
        st.subheader("Bagaimana sistem AI bekerja?")
        st.write("""
        1. **Ekstraksi Fitur:** Jaringan konvolusi EfficientNetB0 memindai area paru untuk mencari infiltrat atau opasitas.
        2. **XAI Mapping:** Grad-CAM akan menyorot area paling mencurigakan dengan warna hangat (merah/kuning).
        3. **Diferensiasi Medis:** AI memisahkan tanda infeksi akibat bakteri ataupun virus secara spesifik.
        """)
