import os
import gdown
import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. MODEL DOWNLOAD & PATH SETTINGS ---
MODEL_PATH = "best_resnet18_occupancy.pth"
# Replace with your public Google Drive file ID for best_resnet18_occupancy.pth
GDRIVE_FILE_ID = "17LUrfE6KUPhYkDje1c439_mJJJ0LLClh" 

@st.cache_resource
def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

download_model_if_missing()

# --- 2. STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="Room Occupancy & Grad-CAM Analyzer", page_icon="🏫", layout="wide")

st.title("🏫 Real-Time Room Occupancy Density Classifier")
st.markdown("Upload a classroom image to inspect prediction probabilities, raw logits, and Grad-CAM heatmaps.")

# Alphabetical order used by PyTorch ImageFolder
CLASSES = ['Empty', 'High_Occupancy', 'Low_Occupancy']

@st.cache_resource
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model()
    st.sidebar.success("✓ ResNet-18 Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"Error loading model weights: {e}")
    st.stop()

st.sidebar.header("⚙️ Dashboard Controls")
cam_opacity = st.sidebar.slider("Grad-CAM Heatmap Opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

uploaded_file = st.file_uploader("Choose a classroom photo...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")

    # Exact validation transform matching PyTorch training pipeline
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = val_transform(raw_img).unsqueeze(0).to(device)

    # Preprocessed RGB array for Grad-CAM
    resized_img = raw_img.resize((224, 224))
    rgb_img = np.float32(resized_img) / 255.0

    with torch.no_grad():
        outputs = model(input_tensor)
        raw_logits = outputs[0].cpu().numpy()
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        predicted_idx = np.argmax(probabilities)
        predicted_class = CLASSES[predicted_idx]
        confidence = probabilities[predicted_idx] * 100

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=1.0 - cam_opacity)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Room Photo")
        st.image(raw_img, use_container_width=True)

    with col2:
        st.subheader(f"🔥 Grad-CAM Heatmap ({predicted_class})")
        st.image(visualization, use_container_width=True)

    st.markdown("---")

    if predicted_class == "High_Occupancy":
        st.error(f"🔴 **Status: High Occupancy Detected** ({confidence:.1f}% Confidence) — HVAC & Ventilation Boost Recommended.")
    elif predicted_class == "Low_Occupancy":
        st.warning(f"🟡 **Status: Low Occupancy Detected** ({confidence:.1f}% Confidence) — Space partially utilized.")
    else:
        st.success(f"🟢 **Status: Room Empty** ({confidence:.1f}% Confidence) — Automated lights/AC can be powered down.")

    st.subheader("📊 Detailed Class Confidence Breakdown")
    for idx, (cls_name, prob, logit) in enumerate(zip(CLASSES, probabilities, raw_logits)):
        st.write(f"**Index {idx} ({cls_name}):** Logit = `{logit:.3f}` | Confidence = **{prob * 100:.2f}%**")
        st.progress(float(prob))
