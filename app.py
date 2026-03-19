import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import gdown
import time

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI — Breast Cancer Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #c8102e 0%, #00205b 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(200,16,46,0.2);
}
.header-banner h1 {
    color: white;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: rgba(255,255,255,0.85);
    font-size: 1.1rem;
    margin: 0.5rem 0 0 0;
}
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin-top: 0.8rem;
    border: 1px solid rgba(255,255,255,0.3);
}

/* ── Cancer result cards ── */
.cancer-yes {
    background: linear-gradient(135deg, #fff0f0, #ffe0e0);
    border: 2px solid #c8102e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(200,16,46,0.15);
}
.cancer-no {
    background: linear-gradient(135deg, #f0fff4, #e0ffe8);
    border: 2px solid #00875a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,135,90,0.15);
}
.cancer-title-yes {
    font-size: 2rem;
    font-weight: 700;
    color: #c8102e;
    margin: 0;
}
.cancer-title-no {
    font-size: 2rem;
    font-weight: 700;
    color: #00875a;
    margin: 0;
}
.cancer-subtitle {
    font-size: 1rem;
    color: #555;
    margin-top: 0.4rem;
}
.cancer-confidence {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 1rem 0 0.2rem 0;
}

/* ── Metric cards ── */
.metric-card {
    background: white;
    border: 1px solid #e8ecf0;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #00205b;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.2rem;
    font-weight: 600;
    color: #00205b;
    border-left: 4px solid #c8102e;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed #c8102e;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    background: #fff8f8;
    margin-bottom: 1rem;
}
.upload-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.upload-text {
    color: #c8102e;
    font-weight: 600;
    font-size: 1.1rem;
}
.upload-subtext {
    color: #888;
    font-size: 0.85rem;
}

/* ── Warning box ── */
.warning-box {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #856404;
    margin-top: 1rem;
}

/* ── Info pill ── */
.info-pill {
    display: inline-block;
    background: #e8f0fe;
    color: #00205b;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 2px;
}

/* ── Progress bar custom ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #c8102e, #00205b);
}

/* ── Sidebar ── */
.sidebar-logo {
    text-align: center;
    padding: 1rem 0;
    border-bottom: 1px solid #eee;
    margin-bottom: 1rem;
}
.sidebar-logo h2 {
    color: #00205b;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0.5rem 0 0 0;
}
.sidebar-logo p {
    color: #c8102e;
    font-size: 0.8rem;
    margin: 0;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.9rem;
}
.stat-label { color: #888; }
.stat-value { color: #00205b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────
@st.cache_resource
def load_model():
    MODEL_PATH       = "model/best_model.pt"
    CLASS_NAMES_PATH = "model/class_names.json"

    if not os.path.exists("model"):
        os.makedirs("model")

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Loading AI model... please wait"):
            url = "https://drive.google.com/uc?id=https://drive.google.com/file/d/1nIRDmii9wH7ZTgZGe6BrmWHAhZUBbHnq/view?usp=sharing"
            gdown.download(url, MODEL_PATH, quiet=False)

    if not os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "w") as f:
            json.dump(["benign", "malignant"], f)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model, class_names

# ── Transform ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Predict ───────────────────────────────────────────
def predict(image, model, CLASS_NAMES):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs    = model(tensor)
        probs      = torch.softmax(outputs, dim=1)[0]
        pred_idx   = probs.argmax().item()
        confidence = probs[pred_idx].item() * 100
    return CLASS_NAMES[pred_idx], confidence, probs.numpy()

# ── Grad-CAM ──────────────────────────────────────────
def get_gradcam(image, model):
    tensor       = transform(image).unsqueeze(0)
    target_layer = [model.features[-1]]
    cam          = GradCAM(model=model, target_layers=target_layer)
    targets      = [ClassifierOutputTarget(1)]
    grayscale    = cam(input_tensor=tensor, targets=targets)[0]
    img_array    = np.array(
        image.resize((224, 224))).astype(np.float32) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    return show_cam_on_image(img_array, grayscale, use_rgb=True)

# ── Save log ──────────────────────────────────────────
def save_to_log(filename, label, confidence, probs):
    log_path = "prediction_log.csv"
    new_row  = {
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename"       : filename,
        "prediction"     : label,
        "cancer_detected": "YES" if label == "malignant" else "NO",
        "confidence"     : round(confidence, 2),
        "benign_prob"    : round(float(probs[0]) * 100, 2),
        "malignant_prob" : round(float(probs[1]) * 100, 2)
    }
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([new_row])],
                       ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(log_path, index=False)
    return df

# ── PDF report ────────────────────────────────────────
def generate_pdf(filename, label, confidence, probs, timestamp):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        import io

        buffer   = io.BytesIO()
        doc      = SimpleDocTemplate(buffer, pagesize=A4)
        styles   = getSampleStyleSheet()
        elements = []

        title_style = ParagraphStyle(
            "title", parent=styles["Title"],
            fontSize=22, textColor=colors.HexColor("#00205b"),
            spaceAfter=6
        )
        heading_style = ParagraphStyle(
            "heading", parent=styles["Heading2"],
            fontSize=13, textColor=colors.HexColor("#c8102e"),
            spaceAfter=4
        )
        normal_style = styles["Normal"]
        normal_style.fontSize = 11

        elements.append(Paragraph("MediScan AI — Medical Report", title_style))
        elements.append(Paragraph("Breast Cancer Detection Analysis", styles["Heading2"]))
        elements.append(Spacer(1, 20))

        info_data = [
            ["Report Date", timestamp],
            ["Image File",  filename],
            ["AI Model",    "EfficientNet-B3"],
            ["Model Accuracy", "97.2%"],
        ]
        info_table = Table(info_data, colWidths=[150, 300])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f0f4ff")),
            ("TEXTCOLOR",  (0,0), (0,-1), colors.HexColor("#00205b")),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (0,0), (-1,-1),
             [colors.white, colors.HexColor("#f9f9f9")]),
            ("GRID",       (0,0), (-1,-1), 0.5,
             colors.HexColor("#dddddd")),
            ("PADDING",    (0,0), (-1,-1), 8),
        ]))
        elements.append(info_table)
        elements.append(Spacer(1, 24))

        elements.append(Paragraph("Diagnosis Result", heading_style))

        cancer   = "YES — MALIGNANT DETECTED" if label == "malignant" \
                   else "NO — BENIGN (NO CANCER)"
        bg_color = colors.HexColor("#ffe0e0") if label == "malignant" \
                   else colors.HexColor("#e0ffe8")
        tx_color = colors.HexColor("#c8102e") if label == "malignant" \
                   else colors.HexColor("#00875a")

        result_data = [
            ["Cancer Detected", cancer],
            ["Confidence",      f"{confidence:.1f}%"],
            ["Benign Probability",    f"{float(probs[0])*100:.1f}%"],
            ["Malignant Probability", f"{float(probs[1])*100:.1f}%"],
        ]
        result_table = Table(result_data, colWidths=[180, 270])
        result_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), bg_color),
            ("TEXTCOLOR",  (0,0), (-1,0), tx_color),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 11),
            ("GRID",       (0,0), (-1,-1), 0.5,
             colors.HexColor("#dddddd")),
            ("PADDING",    (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (1,0), (-1,-1),
             [colors.white, colors.HexColor("#f9f9f9")]),
        ]))
        elements.append(result_table)
        elements.append(Spacer(1, 24))

        elements.append(Paragraph("Medical Disclaimer", heading_style))
        disclaimer = (
            "This report is generated by an AI system for educational and "
            "research purposes only. It is NOT a substitute for professional "
            "medical advice, diagnosis, or treatment. Always consult a qualified "
            "healthcare provider for medical decisions."
        )
        elements.append(Paragraph(disclaimer, normal_style))

        doc.build(elements)
        buffer.seek(0)
        return buffer
    except Exception as e:
        return None

# ══════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="font-size:2.5rem">🏥</div>
        <h2>MediScan AI</h2>
        <p>Breast Cancer Detection System</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🔬 Scan & Predict", "📊 History & Analytics",
         "ℹ️ About the Model"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model Stats**")
    st.markdown("""
    <div class="stat-row">
        <span class="stat-label">Accuracy</span>
        <span class="stat-value">97.2%</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Architecture</span>
        <span class="stat-value">EfficientNet-B3</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Training Images</span>
        <span class="stat-value">8,132</span>
    </div>
    <div class="stat-row">
        <span class="stat-label">Classes</span>
        <span class="stat-value">2</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        ⚕️ <b>Medical Disclaimer</b><br>
        This tool is for educational purposes only.
        Always consult a qualified doctor.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  PAGE: SCAN & PREDICT
# ══════════════════════════════════════════════════════
if "Scan" in page:

    # Header
    st.markdown("""
    <div class="header-banner">
        <h1>🏥 MediScan AI</h1>
        <p>Advanced Breast Cancer Detection using Deep Learning</p>
        <span class="header-badge">EfficientNet-B3 · 97.2% Accuracy · Grad-CAM XAI</span>
    </div>
    """, unsafe_allow_html=True)

    # Model performance strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Validation Accuracy</div>
            <div class="metric-value">97.2%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value">EfficientNet-B3</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Training Images</div>
            <div class="metric-value">8,132</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Classes</div>
            <div class="metric-value">Benign / Malignant</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload zone
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">🩻</div>
        <div class="upload-text">Upload Breast Ultrasound Image</div>
        <div class="upload-subtext">Supports JPG, JPEG, PNG</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose image", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Animated loading
        with st.status("🔬 Analyzing your scan...", expanded=True) as status:
            st.write("Loading AI model...")
            model, CLASS_NAMES = load_model()
            time.sleep(0.3)
            st.write("Processing image...")
            label, confidence, probs = predict(image, model, CLASS_NAMES)
            time.sleep(0.3)
            st.write("Generating Grad-CAM heatmap...")
            heatmap = get_gradcam(image, model)
            time.sleep(0.3)
            st.write("Preparing results...")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_log = save_to_log(
                uploaded_file.name, label, confidence, probs)
            status.update(label="Analysis complete!", state="complete")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── CANCER RESULT — BIG CLEAR MESSAGE ─────────
        if label == "malignant":
            st.markdown(f"""
            <div class="cancer-yes">
                <div style="font-size:3rem">🔴</div>
                <p class="cancer-title-yes">CANCER DETECTED</p>
                <p class="cancer-subtitle">
                    The AI has detected signs of <b>Malignant</b> tissue.
                    Please consult an oncologist immediately.
                </p>
                <div class="cancer-confidence" style="color:#c8102e">
                    {confidence:.1f}% Confidence
                </div>
                <p style="color:#888;font-size:0.85rem;margin:0">
                    Malignant Probability: {float(probs[1])*100:.1f}%
                    &nbsp;|&nbsp;
                    Benign Probability: {float(probs[0])*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.error("⚠️ URGENT: Please visit a doctor or oncologist immediately for further diagnosis.")
        else:
            st.markdown(f"""
            <div class="cancer-no">
                <div style="font-size:3rem">🟢</div>
                <p class="cancer-title-no">NO CANCER DETECTED</p>
                <p class="cancer-subtitle">
                    The AI has classified this as <b>Benign</b> tissue.
                    Continue with regular medical checkups.
                </p>
                <div class="cancer-confidence" style="color:#00875a">
                    {confidence:.1f}% Confidence
                </div>
                <p style="color:#888;font-size:0.85rem;margin:0">
                    Benign Probability: {float(probs[0])*100:.1f}%
                    &nbsp;|&nbsp;
                    Malignant Probability: {float(probs[1])*100:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.success("✅ Good news! No malignant tissue detected. Keep up with regular checkups.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Image + Heatmap + Scores ───────────────────
        st.markdown('<p class="section-header">Scan Analysis</p>',
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("**Original Scan**")
            st.image(image, use_column_width=True)

        with col2:
            st.markdown("**Grad-CAM Heatmap**")
            st.image(heatmap, use_column_width=True)
            st.caption("🔴 Red = High attention area | 🔵 Blue = Low attention")

        with col3:
            st.markdown("**Probability Breakdown**")
            st.markdown("<br>", unsafe_allow_html=True)

            benign_val    = float(probs[0])
            malignant_val = float(probs[1])

            st.markdown(f"""
            <div style="margin-bottom:1rem">
                <div style="display:flex;justify-content:space-between;
                            margin-bottom:4px">
                    <span style="font-weight:600;color:#00875a">
                        Benign</span>
                    <span style="font-weight:700">
                        {benign_val*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(benign_val)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-bottom:1rem">
                <div style="display:flex;justify-content:space-between;
                            margin-bottom:4px">
                    <span style="font-weight:600;color:#c8102e">
                        Malignant</span>
                    <span style="font-weight:700">
                        {malignant_val*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(malignant_val)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Final Verdict</div>
                <div class="metric-value" style="color:{'#c8102e'
                     if label=='malignant' else '#00875a'}">
                    {'CANCER' if label=='malignant' else 'NO CANCER'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── PDF Download ───────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Download Report</p>',
                    unsafe_allow_html=True)

        pdf_buffer = generate_pdf(
            uploaded_file.name, label,
            confidence, probs, timestamp
        )

        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            if pdf_buffer:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"MediScan_Report_{timestamp[:10]}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("Install reportlab to enable PDF: "
                        "`pip install reportlab`")
        with col_b:
            csv_row = pd.DataFrame([{
                "timestamp"  : timestamp,
                "filename"   : uploaded_file.name,
                "prediction" : label,
                "confidence" : f"{confidence:.1f}%"
            }])
            st.download_button(
                label="📊 Download CSV",
                data=csv_row.to_csv(index=False),
                file_name="prediction.csv",
                mime="text/csv",
                use_container_width=True
            )

# ══════════════════════════════════════════════════════
#  PAGE: HISTORY & ANALYTICS
# ══════════════════════════════════════════════════════
elif "History" in page:

    st.markdown("""
    <div class="header-banner">
        <h1>📊 Prediction History</h1>
        <p>Analytics and trends from all scans</p>
    </div>
    """, unsafe_allow_html=True)

    log_path = "prediction_log.csv"

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)

        # Summary metrics
        total     = len(df)
        cancer    = len(df[df["prediction"] == "malignant"])
        no_cancer = len(df[df["prediction"] == "benign"])
        avg_conf  = df["confidence"].mean()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Total Scans</div>
                <div class="metric-value">{total}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Cancer Detected</div>
                <div class="metric-value" style="color:#c8102e">
                    {cancer}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">No Cancer</div>
                <div class="metric-value" style="color:#00875a">
                    {no_cancer}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value">{avg_conf:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts
        st.markdown('<p class="section-header">Distribution Chart</p>',
                    unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 4))
            counts  = df["prediction"].value_counts()
            colors_pie = ["#00875a", "#c8102e"]
            ax.pie(counts, labels=[c.capitalize() for c in counts.index],
                   autopct="%1.1f%%", colors=colors_pie,
                   startangle=90, textprops={"fontsize": 12})
            ax.set_title("Benign vs Malignant", fontsize=13,
                         color="#00205b", fontweight="bold")
            st.pyplot(fig)

        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.hist(df["confidence"], bins=10, color="#00205b",
                     edgecolor="white", alpha=0.85)
            ax2.set_xlabel("Confidence %", fontsize=11)
            ax2.set_ylabel("Count",        fontsize=11)
            ax2.set_title("Confidence Distribution",
                          fontsize=13, color="#00205b", fontweight="bold")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            st.pyplot(fig2)

        # Full table
        st.markdown('<p class="section-header">All Predictions</p>',
                    unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=300)

        # Download full log
        st.download_button(
            "📥 Download Full History CSV",
            data=df.to_csv(index=False),
            file_name="full_prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions yet. Go to Scan & Predict to analyze your first image!")

# ══════════════════════════════════════════════════════
#  PAGE: ABOUT THE MODEL
# ══════════════════════════════════════════════════════
elif "About" in page:

    st.markdown("""
    <div class="header-banner">
        <h1>ℹ️ About the Model</h1>
        <p>Technical details and methodology</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Model Architecture</p>',
                unsafe_allow_html=True)
    st.markdown("""
    This app uses **EfficientNet-B3**, a state-of-the-art convolutional
    neural network pretrained on ImageNet and fine-tuned on breast
    ultrasound images.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Training Details**")
        details = {
            "Architecture"   : "EfficientNet-B3",
            "Pretrained on"  : "ImageNet",
            "Fine-tuned on"  : "Breast Ultrasound Dataset",
            "Training Images": "8,132",
            "Val Images"     : "900",
            "Epochs"         : "15 (10 frozen + 5 unfrozen)",
            "Optimizer"      : "Adam",
            "Learning Rate"  : "0.001 → 0.0001",
            "Image Size"     : "224 × 224 px",
        }
        for k, v in details.items():
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-label">{k}</span>
                <span class="stat-value">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Performance Metrics**")
        metrics = {
            "Validation Accuracy" : "97.2%",
            "Best Val Loss"       : "0.0672",
            "Train Accuracy"      : "97.8%",
            "Classes"             : "Benign, Malignant",
            "XAI Method"          : "Grad-CAM",
        }
        for k, v in metrics.items():
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-label">{k}</span>
                <span class="stat-value">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">What is Grad-CAM?</p>',
                unsafe_allow_html=True)
    st.markdown("""
    **Gradient-weighted Class Activation Mapping (Grad-CAM)** highlights
    the regions of the image that the model focused on when making its
    prediction. Red areas indicate high attention — these are the regions
    most responsible for the model's decision.

    This makes the AI **explainable** and **trustworthy** for medical use.
    """)

    st.markdown('<p class="section-header">Disclaimer</p>',
                unsafe_allow_html=True)
    st.warning("""
    This application is developed for **educational and research purposes
    only** as part of an AIML engineering project. It is NOT intended for
    clinical use or as a substitute for professional medical diagnosis.
    Always consult a qualified healthcare professional for medical advice.
    """)

    st.markdown('<p class="section-header">Developer</p>',
                unsafe_allow_html=True)
    st.markdown("""
    Built by an **AIML Engineering Student (6th Semester)** using:
    PyTorch · EfficientNet · Grad-CAM · Streamlit · Python
    """)
