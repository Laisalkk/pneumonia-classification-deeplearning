import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from models_inference import get_inference_model
from gradcam_inference import apply_gradcam

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="PneumoScan — Chest X-Ray AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# GLOBAL CSS — styling only, zero layout logic here
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-deep:   #010F1A;
    --bg-panel:  #002e2e;
    --bg-card:   #004242;
    --border:    #005a5a;
    --accent:    #00CED1;
    --success:   #00898B;
    --danger:    #00CED1;
    --warn:      #00898B;
    --text-1:    #d4f5f5;
    --text-2:    #7ec8c8;
    --text-3:    #3d8080;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Inter', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-1) !important;
    font-family: var(--sans) !important;
}
.stApp { background: var(--bg-deep) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 1.5rem 2rem !important; max-width: 100% !important; }
.stMainBlockContainer { padding-top: 0 !important; }
[data-testid="stAppViewContainer"] > section > div { padding-top: 0 !important; }

/* ── TOPBAR ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 0 16px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.logo { font-family: var(--mono); font-size: 20px; font-weight: 600; color: var(--text-1); }
.logo span { color: var(--accent); }
.sub { font-size: 10px; letter-spacing: 2px; color: var(--text-3); text-transform: uppercase; font-family: var(--mono); margin-top: 3px; }
.status-pill {
    display: flex; align-items: center; gap: 8px;
    font-family: var(--mono); font-size: 11px; color: var(--success);
    padding: 6px 14px; border: 1px solid #00898B30; border-radius: 4px; background: #00898B08;
}
.sdot { width: 7px; height: 7px; border-radius: 50%; background: var(--success); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

/* ── SECTION HEADERS ── */
.sechead {
    display: flex; align-items: center; gap: 10px;
    margin: 4px 0 12px 0;
}
.sectl { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; color: var(--text-3); text-transform: uppercase; white-space: nowrap; }
.secline { flex: 1; height: 1px; background: var(--border); }

/* ── PANEL LABELS ── */
.plabel {
    font-family: var(--mono); font-size: 10px; letter-spacing: 2px;
    color: var(--text-3); text-transform: uppercase; margin-bottom: 8px;
}

/* ── CARDS ── */
.mbadge {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: 6px;
}
.mname { font-family: var(--mono); font-size: 13px; color: var(--accent); font-weight: 500; }
.mmeta { font-size: 11px; color: var(--text-3); margin-top: 2px; }
.macc { font-family: var(--mono); font-size: 14px; font-weight: 600; color: var(--success); }

.stat-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }
.chip { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 10px; text-align: center; }
.cv { font-family: var(--mono); font-size: 18px; font-weight: 600; color: var(--accent); }
.cl { font-size: 10px; color: var(--text-3); margin-top: 3px; }

/* ── RESULT CARD ── */
.rcard { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 20px 22px; }
.rdiag { font-family: var(--mono); font-size: 26px; font-weight: 600; letter-spacing: -0.5px; line-height: 1.1; margin-bottom: 5px; }
.rconf { font-family: var(--mono); font-size: 12px; color: var(--text-2); letter-spacing: 1px; text-transform: uppercase; }
.rdesc { margin-top: 14px; padding: 10px 14px; border-radius: 6px; font-size: 13px; line-height: 1.6; border-left: 3px solid; }

/* ── PROB BARS ── */
.prob-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.prob-label { font-family: var(--mono); font-size: 12px; color: var(--text-2); width: 160px; flex-shrink: 0; }
.prob-track { flex: 1; height: 6px; background: var(--bg-card); border-radius: 3px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 3px; }
.prob-pct { font-family: var(--mono); font-size: 12px; font-weight: 500; width: 54px; text-align: right; flex-shrink: 0; }

/* ── IMAGE CARDS ── */
.icard { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 4px; }
.ihdr {
    padding: 8px 14px; border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 10px; color: var(--text-3);
    letter-spacing: 1.5px; text-transform: uppercase;
    display: flex; align-items: center; justify-content: space-between;
}
.itag { font-size: 9px; padding: 2px 8px; border-radius: 3px; }

/* ── EMPTY STATE ── */
.empty-state {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 60vh; text-align: center; gap: 14px;
}
.empty-icon { font-size: 60px; opacity: 0.12; }
.empty-title { font-family: var(--mono); font-size: 16px; color: var(--text-3); letter-spacing: -0.3px; }
.empty-sub { font-size: 13px; color: var(--text-3); max-width: 300px; line-height: 1.6; }

/* ── STREAMLIT COMPONENT OVERRIDES ── */
/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: #00CED110 !important;
    border: 1.5px dashed #00688B !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: #00CED122 !important;
    border-color: var(--accent) !important;
}
[data-testid="stFileUploadDropzone"] p { color: var(--text-2) !important; font-size: 12px !important; }
[data-testid="stFileUploadDropzone"] svg { stroke: var(--accent) !important; }
.stFileUploader label { color: var(--text-3) !important; font-size: 10px !important; letter-spacing: 1px !important; font-family: var(--mono) !important; }

/* Button */
.stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: #002222 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    padding: 11px !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { background: #00e8eb !important; }
.stButton > button:active { transform: scale(0.98) !important; }

/* Image captions */
[data-testid="stImage"] p {
    font-family: var(--mono) !important; font-size: 10px !important;
    color: var(--text-3) !important; text-align: center !important;
    letter-spacing: 1px !important; text-transform: uppercase !important; margin-top: 4px !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Vertical divider between columns via border */
div[data-testid="column"]:first-child {
    border-right: 1px solid var(--border);
    padding-right: 24px !important;
}
div[data-testid="column"]:last-child {
    padding-left: 24px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
CLASS_NAMES = {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}
CLASS_COLORS = {0: "#00CED1", 1: "#FF6B6B", 2: "#FFD93D"}
CLASS_ICONS  = {0: "✓", 1: "⚠", 2: "⚠"}
CLASS_DESC   = {
    0: "Tidak ditemukan indikasi infeksi. Struktur paru tampak dalam batas normal.",
    1: "Terdeteksi pola konsolidasi konsisten dengan infeksi bakterial. Segera konsultasikan ke dokter.",
    2: "Terdeteksi pola ground-glass opacity yang mengarah pada infeksi viral. Diperlukan pemeriksaan klinis lanjutan.",
}
PROB_COLORS  = {0: "#00CED1", 1: "#FF6B6B", 2: "#FFD93D"}

# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_model_from_hf():
    repo_id = "laisalkk/pneumonia-classification-deeplearning"
    filename = "E5_EfficientNetB0_best.pth"
    with st.spinner("Mengunduh model dari Hugging Face..."):
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = get_inference_model("EfficientNetB0", num_classes=3)
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
    return model

try:
    model = load_model_from_hf()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ============================================================
# TOPBAR
# ============================================================
st.markdown("""
<div class="topbar">
    <div>
        <div class="logo">Pneumo<span>Scan</span></div>
        <div class="sub">Chest X-Ray · Deep Learning · Explainable AI</div>
    </div>
    <div class="status-pill">
        <div class="sdot"></div>
        SYSTEM ONLINE · EfficientNetB0
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN LAYOUT — Streamlit columns (LEFT: input | RIGHT: output)
# ============================================================
left_col, right_col = st.columns([1, 2.4], gap="medium")

# ══════════════════════════════════════════════
# LEFT COLUMN — All native Streamlit widgets
# ══════════════════════════════════════════════
with left_col:

    # Model badge
    st.markdown('<div class="plabel">Active Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="mbadge">
        <div>
            <div class="mname">EfficientNetB0</div>
            <div class="mmeta">ImageNet pretrained · 3-class</div>
        </div>
        <div class="macc">XAI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture stats
    st.markdown('<div class="plabel">Architecture Info</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-row">
        <div class="chip"><div class="cv">3</div><div class="cl">Classes</div></div>
        <div class="chip"><div class="cv" style="font-size:14px;">224²</div><div class="cl">Input px</div></div>
        <div class="chip"><div class="cv" style="color:#00898B;font-size:14px;">CAM</div><div class="cl">Grad-CAM</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # File uploader — native widget
    st.markdown('<div class="plabel">Input X-Ray</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload chest X-Ray image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    # Preview thumbnail
    if uploaded_file:
        pil_thumb = Image.open(uploaded_file).convert("RGB")
        st.image(pil_thumb, use_container_width=True, caption=uploaded_file.name[:32])
        uploaded_file.seek(0)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Analisis Gambar", use_container_width=True)
    else:
        run_btn = False


# ══════════════════════════════════════════════
# RIGHT COLUMN — Results
# ══════════════════════════════════════════════
with right_col:

    # ── Empty state ──
    if not uploaded_file:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🫁</div>
            <div class="empty-title">Belum ada citra untuk dianalisis</div>
            <div class="empty-sub">Upload citra rontgen dada (X-Ray) di panel kiri untuk memulai klasifikasi otomatis dan visualisasi Grad-CAM.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Waiting state ──
    elif not run_btn:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon" style="opacity:0.2;">▶</div>
            <div class="empty-title">Citra siap dianalisis</div>
            <div class="empty-sub" style="color:#004242;">
                Klik <span style="color:#00b4d8;font-family:monospace;">▶ Analisis Gambar</span> di panel kiri untuk memulai inferensi.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Run inference ──
    else:
        pil_image  = Image.open(uploaded_file).convert("RGB")
        original_np = np.array(pil_image)

        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = eval_transform(pil_image).unsqueeze(0)

        with st.spinner("Menganalisis struktur citra medis..."):
            outputs      = model(input_tensor)
            probs        = torch.softmax(outputs, dim=1).squeeze(0).detach().numpy()
            pred_class   = int(np.argmax(probs))
            target_layer = model.features[-1]
            gradcam_img  = apply_gradcam(model, target_layer, input_tensor, pred_class, original_np)

        color = CLASS_COLORS[pred_class]
        conf  = probs[pred_class] * 100

        # ── Prediction Result ──
        st.markdown("""
        <div class="sechead"><span class="sectl">Prediction Result</span><div class="secline"></div></div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rcard">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;flex-wrap:wrap;">
                <div style="flex:1;">
                    <div class="rdiag" style="color:{color};">{CLASS_ICONS[pred_class]} {CLASS_NAMES[pred_class]}</div>
                    <div class="rconf">CONFIDENCE: {conf:.2f}%</div>
                    <div class="rdesc" style="border-color:{color};background:{color}12;color:#d4f5f5;">
                        {CLASS_DESC[pred_class]}
                    </div>
                </div>
                <div style="text-align:right;flex-shrink:0;">
                    <div style="font-family:var(--mono);font-size:40px;font-weight:700;color:{color};line-height:1;">{conf:.1f}%</div>
                    <div style="font-size:10px;color:var(--text-3);letter-spacing:1px;margin-top:4px;">CONFIDENCE SCORE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability Distribution ── (single HTML block — no per-row st.markdown)
        prob_rows_html = ""
        for idx, name in CLASS_NAMES.items():
            p    = probs[idx] * 100
            c    = PROB_COLORS[idx]
            w    = "600" if idx == pred_class else "400"
            prob_rows_html += (
                f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#7ec8c8;'
                f'width:168px;flex-shrink:0;font-weight:{w};">{name}</div>'
                f'<div style="flex:1;height:6px;background:#004242;border-radius:3px;overflow:hidden;">'
                f'<div style="width:{p:.1f}%;height:100%;background:{c};border-radius:3px;"></div>'
                f'</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{c};'
                f'width:54px;text-align:right;flex-shrink:0;font-weight:{w};">{p:.2f}%</div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;'
            f'color:#3d8080;text-transform:uppercase;display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
            f'PROBABILITY DISTRIBUTION'
            f'<div style="flex:1;height:1px;background:#005a5a;"></div></div>'
            f'<div style="background:#002e2e;border:1px solid #005a5a;border-radius:10px;padding:20px 22px 8px 22px;">'
            f'{prob_rows_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Image Visualization ──
        st.markdown("""
        <div class="sechead"><span class="sectl">Image Visualization</span><div class="secline"></div></div>
        """, unsafe_allow_html=True)

        ic1, ic2 = st.columns(2, gap="medium")
        with ic1:
            st.markdown(f"""
            <div class="icard">
                <div class="ihdr">
                    <span>Original X-Ray</span>
                    <span class="itag" style="background:#005a5a;color:#7ec8c8;">RAW INPUT</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(original_np, use_container_width=True)
            st.markdown(
                '<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                'color:#7ec8c8;text-align:center;margin-top:4px;">Base Image Input</p>',
                unsafe_allow_html=True
            )

        with ic2:
            st.markdown(f"""
            <div class="icard">
                <div class="ihdr">
                    <span>Grad-CAM Attention</span>
                    <span class="itag" style="background:{color}18;color:{color};">XAI OUTPUT</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(gradcam_img, use_container_width=True)
            st.markdown(
                f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'color:{color};font-weight:600;text-align:center;margin-top:4px;">'
                f'Region of interest → {CLASS_NAMES[pred_class]}</p>',
                unsafe_allow_html=True
            )
