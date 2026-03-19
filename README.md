# Breast-Cancer-prediction
AI-powered breast cancer prediction app using EfficientNet-B3

# 🏥 MediScan AI — Breast Cancer Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-green)
![Accuracy](https://img.shields.io/badge/Accuracy-97.2%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

> An AI-powered breast cancer detection system using deep learning and explainable AI (Grad-CAM), built as part of an AIML Engineering project (6th Semester).

---

## 🔗 Live Demo

👉 **https://breast-cancer-prediction-wxkugtlry2nxdeuaroudyw.streamlit.app/**

---

## 📸 Screenshots

| Home Screen | Prediction Result | Grad-CAM Heatmap |

| Upload ultrasound image | CANCER / NO CANCER verdict | AI attention heatmap |

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | EfficientNet-B3 |
| Pretrained on | ImageNet |
| Fine-tuned on | Breast Ultrasound Dataset |
| Training Images | 8,132 |
| Validation Images | 900 |
| Validation Accuracy | **97.2%** |
| Classes | Benign / Malignant |
| XAI Method | Grad-CAM |

---

## 🚀 Features

- ✅ Upload breast ultrasound image
- ✅ Instant AI prediction — Benign or Malignant
- ✅ Big clear **CANCER / NO CANCER** result
- ✅ Confidence score for both classes
- ✅ Grad-CAM heatmap — shows where AI looked
- ✅ PDF medical report download
- ✅ Prediction history with analytics charts
- ✅ 3 pages — Scan, History, About the Model

---

## 🗂️ Project Structure
```
breast-cancer-prediction/
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── packages.txt            ← System dependencies
├── .python-version         ← Python 3.11
├── model/
│   └── class_names.json    ← Class labels
└── README.md               ← This file
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| PyTorch | Deep learning framework |
| EfficientNet-B3 | Model architecture |
| Grad-CAM | Explainable AI heatmaps |
| Streamlit | Web application |
| Google Drive | Model storage |
| gdown | Model download |
| ReportLab | PDF generation |
| Matplotlib | Charts and visualization |

---

## ⚙️ How to Run Locally

# Clone the repository
git clone https://github.com/amitisingh18/breast-cancer-prediction.git
cd breast-cancer-prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📊 Training Results
```
Phase 1 — Frozen backbone (10 epochs)
  Best Validation Accuracy: 82.4%

Phase 2 — Full fine-tuning (5 epochs)
  Epoch 1 → 92.8%
  Epoch 2 → 94.4%
  Epoch 3 → 96.7%
  Epoch 4 → 96.9%
  Epoch 5 → 97.2% ← Best
```

---

## 🔬 What is Grad-CAM?

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the regions of the ultrasound image that the AI focused on when making its prediction. Red areas = high attention = most responsible for the diagnosis.

This makes the AI **explainable** and **trustworthy** for medical applications.

---

## ⚕️ Medical Disclaimer

> This application is developed for **educational and research purposes only** as part of an AIML Engineering project. It is NOT intended for clinical use or as a substitute for professional medical diagnosis. Always consult a qualified healthcare professional.

---

## 👩‍💻 Developer

Built by an **AIML Engineering Student — 6th Semester**

- 🔧 Deep Learning · PyTorch · Transfer Learning
- 🌐 Web Deployment · Streamlit Cloud
- 📊 Explainable AI · Grad-CAM
- 📄 Automated PDF Reports

---

## 📄 License

This project is licensed under the MIT License.
``
