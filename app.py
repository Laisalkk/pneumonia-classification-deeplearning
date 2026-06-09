import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
 
from models_inference import get_inference_model
from gradcam_inference import apply_gradcam
 
# ============================================================
# PAGE CONFIG & GLOBAL CSS
# ============================================================
st.set_page_config(
    page_title="PneumoScan — Chest X-Ray AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');
 
/* ── ROOT THEME ── */
:root {
    --bg-deep:    #090d13;
    --bg-panel:   #0d1421;
    --bg-card:    #111827;
    --bg-hover:   #1a2438;
    --border:     #1e2d45;
    --border-lit: #1e4d7b;
    --accent:     #00b4d8;
    --accent-dim: #0077a8;
    --accent-glow:#00b4d820;
    --danger:     #ff4d6d;
    --danger-dim: #c9184a;
    --warn:       #ffd60a;
    --success:    #06d6a0;
    --text-1:     #e2e8f0;
    --text-2:     #94a3b8;
    --text-3:     #475569;
    --mono:       'JetBrains Mono', monospace;
    --sans:       'Inter', sans-serif;
}
 
/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-1) !important;
    font-family: var(--sans) !important;
}
 
.stApp {
    background: var(--bg-deep) !important;
}
 
/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
 
/* ── TOPBAR ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 32px;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-logo {
    font-family: var(--mono);
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: var(--text-1);
}
.topbar-logo span { color: var(--accent); }
.topbar-sub {
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--text-3);
    text-transform: uppercase;
    font-family: var(--mono);
    margin-top: 2px;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--success);
    padding: 6px 14px;
    border: 1px solid #06d6a030;
    border-radius: 4px;
    background: #06d6a008;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--success);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
 
/* ── MAIN LAYOUT ── */
.main-grid {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 0;
    min-height: calc(100vh - 57px);
}
 
/* ── SIDEBAR PANEL ── */
.side-panel {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.panel-label {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--text-3);
    text-transform: uppercase;
    margin-bottom: 10px;
}
 
/* ── UPLOAD ZONE ── */
.upload-zone {
    border: 1.5px dashed var(--border-lit);
    border-radius: 8px;
    padding: 28px 16px;
    text-align: center;
    background: var(--accent-glow);
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
}
.upload-zone:hover {
    border-color: var(--accent);
    background: #00b4d830;
}
.upload-icon { font-size: 28px; margin-bottom: 8px; }
.upload-text { font-size: 13px; color: var(--text-2); line-height: 1.5; }
.upload-hint { font-size: 11px; color: var(--text-3); margin-top: 6px; font-family: var(--mono); }
 
/* ── MODEL BADGE ── */
.model-badge {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 14px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
}
.model-name {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--accent);
    font-weight: 500;
}
.model-meta { font-size: 11px; color: var(--text-3); margin-top: 2px; }
.model-acc {
    font-family: var(--mono);
    font-size: 15px;
    font-weight: 600;
    color: var(--success);
}
 
/* ── CONTENT AREA ── */
.content-area {
    background: var(--bg-deep);
    padding: 28px 32px;
    display: flex;
    flex-direction: column;
    gap: 24px;
}
.section-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.section-title {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--text-3);
    text-transform: uppercase;
}
.section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
 
/* ── RESULT CARD ── */
.result-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 24px 28px;
}
.result-diagnosis {
    font-family: var(--mono);
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 6px;
}
.result-conf {
    font-family: var(--mono);
    font-size: 13px;
    color: var(--text-2);
    letter-spacing: 1px;
    text-transform: uppercase;
}
.result-desc {
    margin-top: 16px;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.6;
    border-left: 3px solid;
}
 
/* ── PROBABILITY BARS ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.prob-label {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text-2);
    width: 160px;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    height: 6px;
    background: var(--bg-card);
    border-radius: 3px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s ease;
}
.prob-pct {
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 500;
    width: 54px;
    text-align: right;
    flex-shrink: 0;
}
 
/* ── IMAGE CONTAINER ── */
.img-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}
.img-header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-3);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.img-tag {
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 3px;
    font-family: var(--mono);
}
 
/* ── STAT CHIPS ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.stat-chip {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.stat-value {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
}
.stat-label {
    font-size: 11px;
    color: var(--text-3);
    margin-top: 5px;
    letter-spacing: 0.5px;
}
 
/* ── STREAMLIT COMPONENT OVERRIDES ── */
.stFileUploader > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.stFileUploader label { display: none !important; }
[data-testid="stFileUploadDropzone"] {
    border: 1.5px dashed var(--border-lit) !important;
    border-radius: 8px !important;
    background: var(--accent-glow) !important;
    padding: 24px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
    background: #00b4d825 !important;
}
[data-testid="stFileUploadDropzone"] p { color: var(--text-2) !important; font-size: 13px !important; }
 
.stButton > button {
    width: 100%;
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    padding: 12px !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #00d4f5 !important;
}
 
.stSpinner > div { border-color: var(--accent) transparent transparent !important; }
 
/* Override image captions */
[data-testid="stImage"] p {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--text-3) !important;
    text-align: center !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-top: 6px !important;
}
 
/* Warning / info banners */
.stAlert {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-1) !important;
}
 
/* Divider */
hr { border-color: var(--border) !important; margin: 0 !important; }
</style>
""", unsafe_allow_html=True)
 
# ============================================================
# CONSTANTS
# ============================================================
CLASS_NAMES = {
    0: "Normal",
    1: "Bacterial Pneumonia",
    2: "Viral Pneumonia"
}
 
CLASS_COLORS = {
    0: "#06d6a0",   # success green
    1: "#ff4d6d",   # danger red
    2: "#ffd60a",   # warning amber
}
 
CLASS_ICONS = {
    0: "✓",
    1: "⚠",
    2: "⚠",
}
 
CLASS_DESC = {
    0: "Tidak ditemukan indikasi infeksi pada citra rontgen dada. Struktur paru tampak dalam batas normal.",
    1: "Terdeteksi pola konsolidasi konsisten dengan infeksi bakterial. Segera konsultasikan ke dokter untuk evaluasi lebih lanjut.",
    2: "Terdeteksi pola ground-glass opacity yang mengarah pada infeksi viral. Diperlukan pemeriksaan klinis lanjutan.",
}
 
PROB_COLORS = {
    0: "#06d6a0",
    1: "#ff4d6d",
    2: "#ffd60a",
}
 
# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_model_from_hf():
    repo_id = "laisalkk/pneumonia-classification-deeplearning"
    filename = "E5_EfficientNetB0_best.pth"
    with st.spinner("Mengunduh model dari Hugging Face..."):
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
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)
 
# ============================================================
# TOPBAR
# ============================================================
st.markdown("""
<div class="topbar">
    <div>
        <div class="topbar-logo">Pneumo<span>Scan</span></div>
        <div class="topbar-sub">Chest X-Ray · Deep Learning · Explainable AI</div>
    </div>
    <div class="status-pill">
        <div class="status-dot"></div>
        SYSTEM ONLINE · EfficientNetB0
    </div>
</div>
""", unsafe_allow_html=True)
 
if not model_loaded:
    st.error(f"❌ Gagal memuat model: {load_error}")
    st.stop()
 
# ============================================================
# LAYOUT: two-column via st.columns
# ============================================================
left_col, right_col = st.columns([1, 2.2], gap="small")
 
# ─── LEFT PANEL ─────────────────────────────────────────────
with left_col:
    st.markdown("""
    <div style="background:#0d1421; border-right:1px solid #1e2d45; padding:24px 20px; min-height:calc(100vh - 57px);">
    """, unsafe_allow_html=True)
 
    # Model info
    st.markdown('<div class="panel-label">Active Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-badge">
        <div>
            <div class="model-name">EfficientNetB0</div>
            <div class="model-meta">ImageNet pretrained · 3-class</div>
        </div>
        <div class="model-acc">—</div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<div style='margin:20px 0 0 0'></div>", unsafe_allow_html=True)
 
    # Stats
    st.markdown('<div class="panel-label">Architecture Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row">
        <div class="stat-chip">
            <div class="stat-value">3</div>
            <div class="stat-label">Classes</div>
        </div>
        <div class="stat-chip">
            <div class="stat-value" style="font-size:16px;">224²</div>
            <div class="stat-label">Input</div>
        </div>
        <div class="stat-chip">
            <div class="stat-value" style="color:#06d6a0;">XAI</div>
            <div class="stat-label">Grad-CAM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("<div style='margin:20px 0 0 0'></div>", unsafe_allow_html=True)
 
    # Upload
    st.markdown('<div class="panel-label">Input X-Ray</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )
 
    if uploaded_file:
        pil_image_thumb = Image.open(uploaded_file).convert("RGB")
        st.image(pil_image_thumb, use_container_width=True,
                 caption=f"↑ {uploaded_file.name[:28]}")
        uploaded_file.seek(0)
 
        # Analyze button
        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
        run_analysis = st.button("▶  Analisis Gambar", use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-zone">
            <div class="upload-icon">🫁</div>
            <div class="upload-text">Drag & drop citra rontgen dada<br>atau klik untuk browse</div>
            <div class="upload-hint">PNG · JPG · JPEG</div>
        </div>
        """, unsafe_allow_html=True)
        run_analysis = False
 
    st.markdown("</div>", unsafe_allow_html=True)
 
# ─── RIGHT PANEL ────────────────────────────────────────────
with right_col:
    st.markdown('<div style="padding: 28px 28px;">', unsafe_allow_html=True)
 
    if not uploaded_file:
        # Empty state
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                    min-height:70vh; text-align:center; gap:16px;">
            <div style="font-size:64px; opacity:0.15">🫁</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:18px; color:#475569; letter-spacing:-0.5px;">
                Belum ada citra untuk dianalisis
            </div>
            <div style="font-size:13px; color:#334155; max-width:320px; line-height:1.6;">
                Upload citra rontgen dada (X-Ray) di panel kiri untuk memulai klasifikasi otomatis dan visualisasi Grad-CAM.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    elif uploaded_file and not run_analysis:
        # Preview state — waiting for button click
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
                    min-height:70vh; text-align:center; gap:16px;">
            <div style="font-size:48px; opacity:0.3">▶</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:16px; color:#475569;">
                Citra siap dianalisis
            </div>
            <div style="font-size:13px; color:#334155;">
                Klik tombol <b style="color:#00b4d8;">Analisis Gambar</b> di panel kiri untuk memulai.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    else:
        # ── RUN INFERENCE ──
        pil_image = Image.open(uploaded_file).convert("RGB")
        original_np = np.array(pil_image)
 
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = eval_transform(pil_image).unsqueeze(0)
 
        with st.spinner("Menganalisis struktur citra medis..."):
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).detach().numpy()
            pred_class = int(np.argmax(probabilities))
            target_layer = model.features[-1]
            gradcam_img = apply_gradcam(model, target_layer, input_tensor, pred_class, original_np)
 
        pred_color = CLASS_COLORS[pred_class]
        pred_conf  = probabilities[pred_class] * 100
 
        # ── SECTION: Prediction Result ──
        st.markdown("""
        <div class="section-head">
            <span class="section-title">Prediction Result</span>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)
 
        desc_border = pred_color
        desc_bg = f"{pred_color}12"
 
        st.markdown(f"""
        <div class="result-card">
            <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:24px; flex-wrap:wrap;">
                <div>
                    <div class="result-diagnosis" style="color:{pred_color};">
                        {CLASS_ICONS[pred_class]} {CLASS_NAMES[pred_class]}
                    </div>
                    <div class="result-conf">CONFIDENCE: {pred_conf:.2f}%</div>
                    <div class="result-desc" style="border-color:{desc_border}; background:{desc_bg}; color:#cbd5e1;">
                        {CLASS_DESC[pred_class]}
                    </div>
                </div>
                <div style="text-align:right; flex-shrink:0;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:42px; font-weight:700;
                                color:{pred_color}; line-height:1;">{pred_conf:.1f}%</div>
                    <div style="font-size:11px; color:#475569; letter-spacing:1px; margin-top:4px;">CONFIDENCE SCORE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("<div style='margin:24px 0 0 0'></div>", unsafe_allow_html=True)
 
        # ── SECTION: Probability Distribution ──
        st.markdown("""
        <div class="section-head">
            <span class="section-title">Probability Distribution</span>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        for class_idx, class_name in CLASS_NAMES.items():
            prob_pct = probabilities[class_idx] * 100
            bar_color = PROB_COLORS[class_idx]
            is_pred = "font-weight:600;" if class_idx == pred_class else ""
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label" style="{is_pred}">{class_name}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{prob_pct:.1f}%; background:{bar_color};"></div>
                </div>
                <div class="prob-pct" style="color:{bar_color}; {is_pred}">{prob_pct:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
 
        st.markdown("<div style='margin:24px 0 0 0'></div>", unsafe_allow_html=True)
 
        # ── SECTION: Image Visualization ──
        st.markdown("""
        <div class="section-head">
            <span class="section-title">Image Visualization</span>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)
 
        img_col1, img_col2 = st.columns(2, gap="medium")
 
        with img_col1:
            st.markdown("""
            <div class="img-card">
                <div class="img-header">
                    <span>Original X-Ray</span>
                    <span class="img-tag" style="background:#1e2d45; color:#94a3b8;">RAW INPUT</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(original_np, use_container_width=True)
 
        with img_col2:
            st.markdown(f"""
            <div class="img-card">
                <div class="img-header">
                    <span>Grad-CAM Attention</span>
                    <span class="img-tag" style="background:{pred_color}18; color:{pred_color};">XAI OUTPUT</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(gradcam_img, use_container_width=True,
                     caption=f"Region of interest → {CLASS_NAMES[pred_class]}")
 
    st.markdown("</div>", unsafe_allow_html=True)
