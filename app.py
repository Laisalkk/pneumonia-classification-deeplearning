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
# GLOBAL CSS
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
    --success:   #00B4B6;
    --text-1:    #d4f5f5;
    --text-2:    #7ec8c8;
    --text-3:    #6ab8b8;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Inter', sans-serif;
    --cn: #38BDF8;
    --cb: #F87171;
    --cv: #C084FC;
    --warn-bg:  #2a1f00;
    --warn-bdr: #f59e0b;
    --warn-txt: #fcd34d;
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
    padding: 6px 14px; border: 1px solid #00B4B630; border-radius: 4px; background: #00B4B608;
}
.sdot { width: 7px; height: 7px; border-radius: 50%; background: var(--success); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

/* ── SECTION HEADERS ── */
.sechead { display: flex; align-items: center; gap: 10px; margin: 4px 0 12px 0; }
.sectl { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; color: var(--text-3); text-transform: uppercase; white-space: nowrap; }
.secline { flex: 1; height: 1px; background: var(--border); }

/* ── PANEL LABELS ── */
.plabel { font-family: var(--mono); font-size: 10px; letter-spacing: 2px; color: var(--text-3); text-transform: uppercase; margin-bottom: 8px; }

/* ── MODEL BADGE ── */
.mbadge {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: 6px;
}
.mname { font-family: var(--mono); font-size: 13px; color: var(--accent); font-weight: 500; }
.mmeta { font-size: 11px; color: var(--text-3); margin-top: 2px; }
.macc { font-family: var(--mono); font-size: 14px; font-weight: 600; color: var(--success); }

/* ── ARCH CHIPS ── */
.stat-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; }
.chip { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; padding: 10px; text-align: center; }
.cv { font-family: var(--mono); font-size: 18px; font-weight: 600; color: var(--accent); }
.cl { font-size: 10px; color: var(--text-3); margin-top: 3px; }

/* ── LEGEND ── */
.legend-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 5px;
    background: var(--bg-card); border: 1px solid var(--border);
    margin-bottom: 6px;
}
.legend-sym { font-family: var(--mono); font-size: 11px; font-weight: 700; min-width: 36px; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

/* ── DISCLAIMER KUNING ── */
.disclaimer {
    background: var(--warn-bg);
    border: 1px solid #92400e;
    border-left: 3px solid var(--warn-bdr);
    border-radius: 6px;
    padding: 11px 14px;
    margin-top: 10px;
}
.disclaimer .disc-title {
    font-family: var(--mono); font-size: 10px; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--warn-bdr);
    margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
}
.disclaimer .disc-body {
    font-size: 11px; color: var(--warn-txt); line-height: 1.65;
}
.disclaimer .disc-body em { color: #fbbf24; font-style: normal; font-weight: 600; }

/* ── IMAGE METADATA ── */
.img-meta { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px; }
.img-meta-item { background: var(--bg-card); border: 1px solid var(--border); border-radius: 5px; padding: 7px 10px; }
.img-meta-label { font-size: 9px; color: var(--text-3); letter-spacing: 1px; text-transform: uppercase; font-family: var(--mono); }
.img-meta-val { font-size: 13px; font-weight: 600; color: var(--text-1); font-family: var(--mono); margin-top: 2px; }

/* ── RESULT CARD ── */
.rcard { background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 20px 22px; }
.rdiag { font-family: var(--mono); font-size: 26px; font-weight: 600; letter-spacing: -0.5px; line-height: 1.1; margin-bottom: 5px; }
.rconf { font-family: var(--mono); font-size: 12px; color: var(--text-2); letter-spacing: 1px; text-transform: uppercase; }
.rdesc { margin-top: 14px; padding: 10px 14px; border-radius: 6px; font-size: 13px; line-height: 1.6; border-left: 3px solid; }

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

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stFileUploadDropzone"] {
    background: #00CED110 !important; border: 1.5px dashed #00688B !important; border-radius: 8px !important;
}
[data-testid="stFileUploadDropzone"]:hover { background: #00CED122 !important; border-color: var(--accent) !important; }
[data-testid="stFileUploadDropzone"] p { color: var(--text-2) !important; font-size: 12px !important; }
[data-testid="stFileUploadDropzone"] svg { stroke: var(--accent) !important; }
.stFileUploader label { color: var(--text-3) !important; font-size: 10px !important; letter-spacing: 1px !important; font-family: var(--mono) !important; }

.stButton > button {
    width: 100% !important; background: var(--accent) !important; color: #002222 !important;
    border: none !important; border-radius: 6px !important; font-family: var(--mono) !important;
    font-size: 12px !important; font-weight: 600 !important; letter-spacing: 1.5px !important;
    padding: 11px !important; text-transform: uppercase !important;
}
.stButton > button:hover { background: #00e8eb !important; }
.stButton > button:active { transform: scale(0.98) !important; }

[data-testid="stImage"] p {
    font-family: var(--mono) !important; font-size: 10px !important;
    color: var(--text-3) !important; text-align: center !important;
    letter-spacing: 1px !important; text-transform: uppercase !important; margin-top: 4px !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }

div[data-testid="column"]:first-child { border-right: 1px solid var(--border); padding-right: 24px !important; }
div[data-testid="column"]:last-child { padding-left: 24px !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
CLASS_NAMES   = {0: "Normal", 1: "Bacterial Pneumonia", 2: "Viral Pneumonia"}
CLASS_COLORS  = {0: "#38BDF8", 1: "#F87171", 2: "#C084FC"}
CLASS_ICONS   = {0: "✓", 1: "⚠", 2: "⚠"}
CLASS_SYMBOLS = {0: "[ N ]", 1: "[ B ]", 2: "[ V ]"}
CLASS_DESC    = {
    0: "Tidak ditemukan indikasi infeksi. Struktur paru tampak dalam batas normal.",
    1: "Terdeteksi pola konsolidasi konsisten dengan infeksi bakterial. Segera konsultasikan ke dokter.",
    2: "Terdeteksi pola ground-glass opacity yang mengarah pada infeksi viral. Diperlukan pemeriksaan klinis lanjutan.",
}

# ============================================================
# MODEL LOADER
# ============================================================
@st.cache_resource
def load_model_from_hf():
    repo_id  = "laisalkk/pneumonia-classification-deeplearning"
    filename = "E5_EfficientNetB0_best.pth"
    with st.spinner("Mengunduh model dari Hugging Face..."):
        path  = hf_hub_download(repo_id=repo_id, filename=filename)
        model = get_inference_model("EfficientNetB0", num_classes=3)
        ckpt  = torch.load(path, map_location="cpu")
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
# MAIN LAYOUT
# ============================================================
left_col, right_col = st.columns([1, 2.4], gap="medium")

# ══════════════════════════════════════════════
# LEFT COLUMN
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
        <div class="chip"><div class="cv" style="color:#00B4B6;font-size:14px;">CAM</div><div class="cl">Grad-CAM</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Class legend — nama kelas berwarna sesuai kelasnya
    st.markdown('<div class="plabel">Class Legend</div>', unsafe_allow_html=True)
    for idx, name in CLASS_NAMES.items():
        c   = CLASS_COLORS[idx]
        sym = CLASS_SYMBOLS[idx]
        st.markdown(f"""
        <div class="legend-item">
            <span class="legend-sym" style="color:{c};">{sym}</span>
            <span style="font-family:var(--mono);font-size:11px;color:{c};flex:1;">{name}</span>
            <span class="legend-dot" style="background:{c};"></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # File uploader
    st.markdown('<div class="plabel">Input X-Ray</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload chest X-Ray image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        pil_thumb = Image.open(uploaded_file).convert("RGB")
        w_px, h_px = pil_thumb.size
        file_kb = uploaded_file.size // 1024

        # Metadata gambar
        st.markdown(f"""
        <div class="img-meta">
            <div class="img-meta-item">
                <div class="img-meta-label">Resolusi</div>
                <div class="img-meta-val">{w_px}×{h_px}</div>
            </div>
            <div class="img-meta-item">
                <div class="img-meta-label">Ukuran</div>
                <div class="img-meta-val">{file_kb} KB</div>
            </div>
            <div class="img-meta-item">
                <div class="img-meta-label">Format</div>
                <div class="img-meta-val">{uploaded_file.name.split('.')[-1].upper()}</div>
            </div>
            <div class="img-meta-item">
                <div class="img-meta-label">Mode</div>
                <div class="img-meta-val">RGB</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.image(pil_thumb, use_container_width=True, caption=uploaded_file.name[:32])
        uploaded_file.seek(0)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Analisis Gambar", use_container_width=True)
    else:
        run_btn = False

    # Disclaimer — warna kuning warning
    st.markdown("""
    <div class="disclaimer">
        <div class="disc-title">⚠ Disclaimer Medis</div>
        <div class="disc-body">
            Hasil analisis ini bersifat skrining awal berbasis AI dan
            <em>bukan diagnosis medis resmi</em>.
            Selalu konsultasikan hasil ini kepada dokter atau tenaga medis yang berkompeten
            sebelum mengambil keputusan klinis apapun.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# RIGHT COLUMN
# ══════════════════════════════════════════════
with right_col:

    if not uploaded_file:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🫁</div>
            <div class="empty-title">Belum ada citra untuk dianalisis</div>
            <div class="empty-sub">Upload citra rontgen dada (X-Ray) di panel kiri untuk memulai klasifikasi otomatis dan visualisasi Grad-CAM.</div>
        </div>
        """, unsafe_allow_html=True)

    elif not run_btn:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon" style="opacity:0.2;">▶</div>
            <div class="empty-title">Citra siap dianalisis</div>
            <div class="empty-sub" style="color:#3d7070;">
                Klik <span style="color:#00b4d8;font-family:monospace;">▶ Analisis Gambar</span> di panel kiri untuk memulai inferensi.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        pil_image   = Image.open(uploaded_file).convert("RGB")
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

        color  = CLASS_COLORS[pred_class]
        conf   = probs[pred_class] * 100
        symbol = CLASS_SYMBOLS[pred_class]

        # ── Prediction Result ──
        st.markdown("""
        <div class="sechead"><span class="sectl">Prediction Result</span><div class="secline"></div></div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="rcard">
            <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;flex-wrap:wrap;">
                <div style="flex:1;">
                    <div class="rdiag" style="color:{color};">
                        {CLASS_ICONS[pred_class]} {CLASS_NAMES[pred_class]}
                        <span style="font-size:14px;opacity:0.55;margin-left:8px;">{symbol}</span>
                    </div>
                    <div class="rconf">CONFIDENCE: {conf:.2f}%</div>
                    <div class="rdesc" style="border-color:{color};background:{color}15;color:#d4f5f5;">
                        {CLASS_DESC[pred_class]}
                    </div>
                </div>
                <div style="text-align:right;flex-shrink:0;">
                    <div style="font-family:var(--mono);font-size:40px;font-weight:700;color:{color};line-height:1;">{conf:.1f}%</div>
                    <div style="font-size:10px;color:#6ab8b8;letter-spacing:1px;margin-top:4px;">CONFIDENCE SCORE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability Distribution ──
        # PERBAIKAN: simbol, nama kelas, DAN persentase semua pakai warna kelas masing-masing
        prob_rows_html = ""
        for idx, name in CLASS_NAMES.items():
            p   = probs[idx] * 100
            c   = CLASS_COLORS[idx]
            sym = CLASS_SYMBOLS[idx]
            is_pred = idx == pred_class
            bar_h   = "8px" if is_pred else "5px"
            weight  = "700" if is_pred else "400"
            # semua elemen teks pakai warna kelas: sym, nama, persen
            prob_rows_html += (
                f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:13px;">'
                # Simbol + nama — keduanya warna kelas
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
                f'width:240px;flex-shrink:0;font-weight:{weight};color:{c};">'
                f'<span style="margin-right:7px;font-size:10px;">{sym}</span>'
                f'{name}'
                f'</div>'
                # Progress bar
                f'<div style="flex:1;height:{bar_h};background:#004242;border-radius:4px;overflow:hidden;">'
                f'<div style="width:{p:.1f}%;height:100%;background:{c};border-radius:4px;"></div>'
                f'</div>'
                # Persentase — warna kelas
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
                f'width:56px;text-align:right;flex-shrink:0;font-weight:{weight};color:{c};">'
                f'{p:.2f}%'
                f'</div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;letter-spacing:2px;'
            f'color:#6ab8b8;text-transform:uppercase;display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
            f'PROBABILITY DISTRIBUTION'
            f'<div style="flex:1;height:1px;background:#005a5a;"></div></div>'
            f'<div style="background:#002e2e;border:1px solid #005a5a;border-radius:10px;padding:20px 22px 10px 22px;">'
            f'{prob_rows_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Ringkasan Klinis (fitur baru) ──
        # Menampilkan panduan tindak lanjut berdasarkan prediksi
        CLINICAL_FOLLOWUP = {
            0: [
                ("Pemantauan rutin", "Lakukan pemeriksaan rutin sesuai jadwal dokter."),
                ("Jaga kesehatan paru", "Hindari paparan asap rokok dan polusi udara."),
                ("Gejala perubahan", "Segera periksakan jika muncul demam, batuk, atau sesak napas."),
            ],
            1: [
                ("Konsultasi segera", "Segera temui dokter untuk konfirmasi dan terapi antibiotik jika diperlukan."),
                ("Pemeriksaan lanjutan", "Dokter mungkin merekomendasikan kultur sputum atau tes darah."),
                ("Isolasi & istirahat", "Istirahat cukup dan hindari kontak dengan orang rentan selama masa pemulihan."),
            ],
            2: [
                ("Konsultasi segera", "Segera temui dokter untuk konfirmasi dan penanganan antiviral jika diperlukan."),
                ("Pemantauan saturasi O₂", "Pantau kadar oksigen darah secara berkala, terutama jika ada sesak napas."),
                ("Isolasi ketat", "Isolasi mandiri untuk mencegah penularan virus ke orang lain."),
            ],
        }

        st.markdown("""
        <div class="sechead"><span class="sectl">Panduan Tindak Lanjut</span><div class="secline"></div></div>
        """, unsafe_allow_html=True)

        steps_html = ""
        for i, (title, desc) in enumerate(CLINICAL_FOLLOWUP[pred_class], 1):
            steps_html += (
                f'<div style="display:flex;gap:14px;margin-bottom:12px;'
                f'padding:12px 14px;background:#002e2e;border:1px solid #005a5a;'
                f'border-left:3px solid {color};border-radius:6px;">'
                f'<div style="font-family:var(--mono);font-size:18px;font-weight:700;'
                f'color:{color};opacity:0.5;min-width:24px;line-height:1.4;">{i}</div>'
                f'<div>'
                f'<div style="font-family:var(--mono);font-size:12px;font-weight:600;color:{color};margin-bottom:3px;">{title}</div>'
                f'<div style="font-size:12px;color:#7ec8c8;line-height:1.6;">{desc}</div>'
                f'</div>'
                f'</div>'
            )
        st.markdown(steps_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Image Visualization ──
        st.markdown("""
        <div class="sechead"><span class="sectl">Image Visualization</span><div class="secline"></div></div>
        """, unsafe_allow_html=True)

        ic1, ic2 = st.columns(2, gap="medium")
        with ic1:
            st.markdown("""
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
                    <span class="itag" style="background:{color}22;color:{color};">XAI OUTPUT</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.image(gradcam_img, use_container_width=True)
            st.markdown(
                f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
                f'color:{color};font-weight:600;text-align:center;margin-top:4px;">'
                f'Region of interest → {symbol} {CLASS_NAMES[pred_class]}</p>',
                unsafe_allow_html=True
            )

        # ── Interpretasi Grad-CAM ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0a1f1f;border:1px solid #1a4040;border-radius:8px;padding:14px 18px;">
            <div style="font-family:var(--mono);font-size:10px;letter-spacing:2px;color:#6ab8b8;
                text-transform:uppercase;margin-bottom:10px;">Interpretasi Grad-CAM</div>
            <div style="font-size:12px;color:#7ec8c8;line-height:1.7;">
                Area <span style="color:#FF6347;font-weight:600;">merah-oranye</span> pada peta panas menunjukkan
                region yang paling berkontribusi terhadap prediksi model.
                Area <span style="color:#4fa3e0;font-weight:600;">biru</span> menunjukkan region dengan
                kontribusi rendah. Perhatikan distribusi area panas di lapang paru
                untuk memvalidasi kesesuaian dengan gambaran klinis.
            </div>
        </div>
        """, unsafe_allow_html=True)
