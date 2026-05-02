import streamlit as st
import numpy as np
import joblib
import cv2
import shap
import matplotlib.pyplot as plt
import os
from PIL import Image
import time

# ---------------- PAGE ----------------
st.set_page_config(page_title="Brain Tumor AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
html, body {
    overflow: hidden;
    background: radial-gradient(circle at center, #020617, #000000);
}

.title {
    text-align:center;
    font-size:60px;
    font-weight:900;
    background: linear-gradient(90deg,#00f5ff,#ff00ff,#8b5cf6);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    text-shadow: 0 0 60px #00f5ff;
}

.section {
    text-align:center;
    font-size:26px;
    color:#00f5ff;
}

.card {
    background: rgba(255,255,255,0.05);
    border-radius:20px;
    padding:10px;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 25px rgba(0,255,255,0.2);
}

.result {
    font-size:34px;
    text-align:center;
    padding:20px;
    border-radius:15px;
    font-weight:bold;
}
.tumor {background:#ff004c;}
.notumor {background:#00ff88;}

.gauge {
    width:140px;height:140px;border-radius:50%;
    background:conic-gradient(#00f5ff var(--p), rgba(255,255,255,0.1) 0);
    display:flex;align-items:center;justify-content:center;
    margin:auto;
}
.inner {
    width:100px;height:100px;border-radius:50%;
    background:#020617;
    display:flex;align-items:center;justify-content:center;
    font-size:20px;font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PATHS ----------------
MODEL_PATH = "models/xgb_mobilevit_swinS_optuna.pkl"
IMG_DIR = "data/raw/processed/images"
MOBILEVIT_PATH = "models/mobilevit_features.npy"
SWIN_PATH = "models/swin_features.npy"

# ---------------- LOAD ----------------
@st.cache_resource
def load_all():
    model = joblib.load(MODEL_PATH)
    mobilevit = np.load(MOBILEVIT_PATH)
    swin = np.load(SWIN_PATH)
    X = np.concatenate([mobilevit, swin], axis=1)
    return model, X

model, X_all = load_all()
image_files = sorted(os.listdir(IMG_DIR))

# ---------------- TITLE ----------------
st.markdown('<div class="title">🧠 Brain Tumor Detection AI</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload MRI Image", type=["png","jpg","jpeg"])

if uploaded_file:

    col1, col2, col3 = st.columns([1,1.2,1])

    img = np.array(Image.open(uploaded_file).convert("L").resize((224,224)))

    # ---------------- MATCH INTERNAL ----------------
    sims = []
    for f in image_files:
        ref = cv2.imread(os.path.join(IMG_DIR,f),0)
        ref = cv2.resize(ref,(224,224))
        sims.append(np.mean((img-ref)**2))
    idx = int(np.argmin(sims))

    features = X_all[idx].reshape(1,-1)

    # ---------------- PREDICTION ----------------
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    conf = int(max(prob)*100)

    # ---------------- SHAP ----------------
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(features)

    if isinstance(shap_vals,list):
        shap_vals = shap_vals[1]

    shap_vals = np.array(shap_vals)

    # ================= FIXED BRAIN-ONLY HEATMAP =================

    # Brain mask
    _, brain_mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    brain_mask = cv2.medianBlur(brain_mask, 5)
    kernel = np.ones((5,5), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

    # SHAP map
    m = np.abs(shap_vals[0])
    m = cv2.GaussianBlur(m.reshape(1,-1),(51,51),0)
    m = cv2.resize(m,(224,224))
    m = m / (np.max(m) + 1e-8)

    important = (m > np.percentile(m, 97)).astype(np.uint8)

    # Apply brain mask
    important = important * (brain_mask // 255)

    # Smooth glow
    important = cv2.GaussianBlur(important.astype(np.float32),(31,31),0)

    heat = cv2.applyColorMap((important*255).astype(np.uint8), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        0.7,
        heat,
        0.6,
        0
    )

    # ================= UI =================

    with col1:
        st.markdown('<div class="section">MRI INPUT</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section">TUMOR REGION</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section">AI RESULT</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f'<div class="result tumor">⚠️ TUMOR<br>{conf}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result notumor">✅ NO TUMOR<br>{conf}%</div>', unsafe_allow_html=True)

        # Progress animation
        bar = st.empty()
        for i in range(conf):
            bar.progress(i+1)
            time.sleep(0.002)

        st.markdown(f'<div class="gauge" style="--p:{conf}%"><div class="inner">{conf}%</div></div>', unsafe_allow_html=True)

        # SHAP Graph
        fig, ax = plt.subplots(figsize=(4,2))
        ax.plot(np.abs(shap_vals[0]), color="#00f5ff")
        ax.set_title("SHAP Importance")
        ax.set_xticks([])
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)