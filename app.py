import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import os
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🔬",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #e63946;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    .benign-box {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .malignant-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────
@st.cache_resource
def load_model():
    MODEL_PATH      = os.path.join("model", "best_model.pt")
    CLASS_NAMES_PATH = os.path.join("model", "class_names.json")

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH,
                          map_location=torch.device("cpu")))
    model.eval()
    return model, class_names

model, CLASS_NAMES = load_model()

# ── Image transform ───────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Prediction function ───────────────────────────────
def predict(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        pred_idx    = probs.argmax().item()
        confidence  = probs[pred_idx].item() * 100
    return CLASS_NAMES[pred_idx], confidence, probs.numpy()

# ── Grad-CAM function ─────────────────────────────────
def get_gradcam(image):
    tensor     = transform(image).unsqueeze(0)
    target_layer = [model.features[-1]]
    cam        = GradCAM(model=model, target_layers=target_layer)
    targets    = [ClassifierOutputTarget(1)]
    grayscale  = cam(input_tensor=tensor, targets=targets)
    grayscale  = grayscale[0]

    img_array  = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    heatmap    = show_cam_on_image(img_array, grayscale, use_rgb=True)
    return heatmap

# ── UI ────────────────────────────────────────────────
st.markdown('<p class="main-title">🔬 Breast Cancer Prediction</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a breast ultrasound image for instant AI-powered analysis</p>',
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Breast_cancer_ultrasound.jpg/320px-Breast_cancer_ultrasound.jpg",
             caption="Sample ultrasound image")
    st.markdown("### About this App")
    st.markdown("""
    - **Model** : EfficientNet-B3
    - **Accuracy** : 97.2%
    - **Classes** : Benign / Malignant
    - **XAI** : Grad-CAM heatmap
    """)
    st.markdown("### How to use")
    st.markdown("""
    1. Upload an ultrasound image
    2. Wait for prediction
    3. View confidence score
    4. View Grad-CAM heatmap
    """)
    st.warning("This app is for educational purposes only. Always consult a medical professional.")

# ── Main content ──────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Breast Ultrasound Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a breast ultrasound image in JPG or PNG format"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ── Run prediction
    with st.spinner("Analyzing image..."):
        label, confidence, probs = predict(image)
        heatmap                  = get_gradcam(image)

    # ── Layout: 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        st.image(heatmap, use_column_width=True)
        st.caption("Red areas = regions the model focused on")

    with col3:
        st.subheader("Prediction Result")

        # Result box
        box_class = "benign-box" if label == "benign" else "malignant-box"
        icon      = "✅" if label == "benign" else "⚠️"
        st.markdown(f"""
        <div class="result-box {box_class}">
            {icon} {label.upper()}<br>
            <span style="font-size:1rem">Confidence: {confidence:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bars
        st.markdown("#### Confidence Scores")
        for i, cls in enumerate(CLASS_NAMES):
            st.metric(label=cls.capitalize(),
                      value=f"{probs[i]*100:.1f}%")
            st.progress(float(probs[i]))

        # Warning for malignant
        if label == "malignant":
            st.error("⚠️ Malignant detected. Please consult a doctor immediately.")
        else:
            st.success("✅ Benign detected. Continue regular checkups.")

    # ── Prediction log ────────────────────────────────
    st.markdown("---")
    st.markdown("### Prediction Details")

    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
    with detail_col1:
        st.markdown('<div class="metric-card"><b>Prediction</b><br>' +
                    label.upper() + '</div>', unsafe_allow_html=True)
    with detail_col2:
        st.markdown('<div class="metric-card"><b>Confidence</b><br>' +
                    f'{confidence:.1f}%</div>', unsafe_allow_html=True)
    with detail_col3:
        st.markdown('<div class="metric-card"><b>Model</b><br>' +
                    'EfficientNet-B3</div>', unsafe_allow_html=True)
    with detail_col4:
        st.markdown('<div class="metric-card"><b>Accuracy</b><br>' +
                    '97.2%</div>', unsafe_allow_html=True)

    # ── Save to log ───────────────────────────────────
    import pandas as pd
    from datetime import datetime

    log_path = "prediction_log.csv"
    new_row  = {
        "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename"   : uploaded_file.name,
        "prediction" : label,
        "confidence" : round(confidence, 2),
        "benign_prob": round(float(probs[0]) * 100, 2),
        "malignant_prob": round(float(probs[1]) * 100, 2)
    }

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(log_path, index=False)

else:
    # ── Empty state ───────────────────────────────────
    st.info("👆 Upload an ultrasound image above to get started")

    st.markdown("---")
    st.markdown("### Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Validation Accuracy", "97.2%")
    m2.metric("Training Images",     "8,132")
    m3.metric("Model",               "EfficientNet-B3")
    m4.metric("Classes",             "Benign / Malignant")
