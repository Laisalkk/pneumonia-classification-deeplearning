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
    page_title="Pneumonia Diagnosis Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS untuk merombak total tampilan mentah Streamlit
st.markdown("""
    <style>
    /* Mengatur latar belakang aplikasi agar clean */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Desain teks judul utama */
    .main-title {
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 2px;
        font-family: 'Inter', sans-serif;
    }
    .sub-title {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 20px;
    }
    
    /* Box Container Utama Hasil Diagnosis */
    .diagnosis-container {
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .status-normal {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        color: #065F46;
    }
    .status-warning {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        color: #991B1B;
    }
    
    /* Desain Grid Tampilan Gambar */
    .image-card {
        background-color: #FFFFFF;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
        text-align: center;
    }
    .image-label {
        font-size: 14px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Desain Indikator Probabilitas Minimalis di Bagian Bawah */
    .prob-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-top: 20px;
    }
    .prob-title {
        font-size: 15px;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 15px;
    }
    .prob-label {
        font-size: 13px;
        font-weight: 600;
        color: #475569;
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

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
    
    with st.spinner("Menghubungkan ke server model..."):
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
    st.error(f"Koneksi gagal: {e}")
    st.stop()

# ============================================================
# SIDEBAR CONTROL PANEL
# ============================================================
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843110.png", width=60)
    st.markdown("### **Input Medis Pasien**")
    st.write("Silahkan unggah file citra rontgen dada (X-Ray) hasil scan laboratorium.")
    
    uploaded_file = st.file_uploader(
        "Upload Citra Paru-Paru", 
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("💡 **Informasi Engine AI:**")
    st.caption("Arsitektur: **EfficientNetB0**")
    st.caption("Metode Penjelasan: **Grad-CAM Core**")

# ============================================================
# HALAMAN UTAMA / DASHBOARD VIEW
# ============================================================
st.markdown('<p class="main-title">🫁 Clinical Intelligence AI Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistem otomatis pemindaian citra rontgen dada untuk identifikasi Pneumonia dan interpretasi area infeksi.</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    # 1. Olah Gambar Input
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
    
    # 2. Forward Pass Model & Grad-CAM
    with st.spinner("Mengomparasi pola piksel dan opasitas paru..."):
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0).detach().numpy()
        pred_class = np.argmax(probabilities)
        
        target_layer = model.features[-1]
        gradcam_img = apply_gradcam(model, target_layer, input_tensor, pred_class, original_np)

    # ============================================================
    # TOP CONTAINER: HASIL DIAGNOSIS UTAMA
    # ============================================================
    status_class = "status-normal" if pred_class == 0 else "status-warning"
    status_icon = "🟢" if pred_class == 0 else "🔴"
    
    st.markdown(f"""
        <div class="diagnosis-container {status_class}">
            <span style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Hasil Diagnosis Utama</span>
            <h2 style="margin: 4px 0 0 0; font-weight: 800; font-size: 24px;">
                {status_icon} {CLASS_NAMES[pred_class]} — {probabilities[pred_class]*100:.2f}%
            </h2>
        </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # MIDDLE CONTAINER: IMAGE GRID (BENTUK KARTU INDEPENDEN)
    # ============================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown('<p class="image-label">📸 Citra Rontgen Asli</p>', unsafe_allow_html=True)
        st.image(original_np, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="image-label">🔥 Pemetaan Grad-CAM ({CLASS_NAMES[pred_class]})</p>', unsafe_allow_html=True)
        st.image(gradcam_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================
    # BOTTOM CONTAINER: SCORE DISTRIBUTION ANALYSIS
    # ============================================================
    st.markdown('<div class="prob-box">', unsafe_allow_html=True)
    st.markdown('<p class="prob-title">📊 Analisis Distribusi Nilai Keyakinan Model</p>', unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"""
            <div class="prob-label">
                <span>{CLASS_NAMES[0]}</span>
                <span style="color: #0F172A;">{probabilities[0]*100:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)
        st.progress(float(probabilities[0]))
        
    with m_col2:
        st.markdown(f"""
            <div class="prob-label">
                <span>{CLASS_NAMES[1]}</span>
                <span style="color: #0F172A;">{probabilities[1]*100:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)
        st.progress(float(probabilities[1]))
        
    with m_col3:
        st.markdown(f"""
            <div class="prob-label">
                <span>{CLASS_NAMES[2]}</span>
                <span style="color: #0F172A;">{probabilities[2]*100:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)
        st.progress(float(probabilities[2]))
        
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Tampilan Standby Dashboard (Sangat Clean & Modern)
    st.markdown("""
        <div style="background-color: #EFF6FF; border: 1px dashed #BFDBFE; padding: 30px; border-radius: 12px; text-align: center; margin-top: 40px;">
            <h3 style="color: #1E40AF; margin-top: 0;">Sistem Siap Digunakan</h3>
            <p style="color: #1E3A8A; font-size: 14px; max-width: 600px; margin: 0 auto;">
                Silahkan unggah file gambar rontgen dada (.png, .jpg, .jpeg) melalui panel menu <b>Input Medis Pasien</b> di sebelah kiri untuk melihat hasil analisis diagnosis serta peta Grad-CAM secara langsung.
            </p>
        </div>
    """, unsafe_allow_html=True)
