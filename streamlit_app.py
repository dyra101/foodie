import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import requests
from io import BytesIO

# ============================================
# 1. Load Class Names
# ============================================

@st.cache_resource
def load_class_names():
    if not os.path.exists('classes.txt'):
        st.error("classes.txt not found. Please upload it.")
        return []
    
    with open('classes.txt', 'r') as f:
        return [line.strip() for line in f.readlines()]

class_names = load_class_names()

# ============================================
# 2. Load Model
# ============================================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    
    if not os.path.exists('best_gradual_unfreezing.pth'):
        st.error("Model file 'best_gradual_unfreezing.pth' not found. Please upload it.")
        return None, device
    
    model.load_state_dict(torch.load('best_gradual_unfreezing.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ============================================
# 3. Define Transforms
# ============================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================
# 4. Prediction Function
# ============================================

def predict_food(image):
    if model is None:
        return "Model not loaded", 0.0, []
    
    if isinstance(image, str):
        if image.startswith('http'):
            response = requests.get(image)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image).convert('RGB')
    else:
        img = Image.fromarray(image).convert('RGB')
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    
    food_name = class_names[pred.item()]
    confidence = conf.item()
    
    top5_probs, top5_preds = torch.topk(probs, 5)
    top5 = [(class_names[top5_preds[0][i].item()], top5_probs[0][i].item()) for i in range(5)]
    
    return food_name, confidence, top5

# ============================================
# 5. Streamlit UI
# ============================================

st.set_page_config(
    page_title="🍕 Food Image Classifier",
    page_icon="🍕",
    layout="centered"
)

st.title("🍕 Food Image Classifier")
st.markdown("Upload a photo of food and the AI will identify it!")

# Sidebar info
st.sidebar.header("About")
st.sidebar.markdown(f"""
- **Model:** ResNet50 (fine-tuned on Food-101)
- **Classes:** {len(class_names)} food categories
- **Accuracy:** ~69% on validation set
- **Built with:** PyTorch, Streamlit
""")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a single dish"
)

# Example images (using URLs)
st.markdown("### Or try one of these examples:")
col1, col2, col3 = st.columns(3)

example_urls = {
    "🍕 Pizza": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=300",
    "🍣 Sushi": "https://images.unsplash.com/photo-1579871494447-9811cf80d66c?w=300",
    "🍜 Ramen": "https://images.unsplash.com/photo-1569718212165-3a8278d5f624?w=300"
}

with col1:
    if st.button("🍕 Pizza"):
        uploaded_file = example_urls["🍕 Pizza"]

with col2:
    if st.button("🍣 Sushi"):
        uploaded_file = example_urls["🍣 Sushi"]

with col3:
    if st.button("🍜 Ramen"):
        uploaded_file = example_urls["🍜 Ramen"]

# Process uploaded image
if uploaded_file is not None:
    if isinstance(uploaded_file, str):
        st.image(uploaded_file, caption='Example Image', use_column_width=True)
    else:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing food...'):
        food_name, confidence, top5 = predict_food(uploaded_file)
    
    st.success(f"🍽️ **{food_name}**")
    st.metric("Confidence", f"{confidence:.2%}")
    
    st.markdown("### Top 5 Predictions:")
    for i, (name, prob) in enumerate(top5, 1):
        st.progress(prob, text=f"{i}. {name}: {prob:.2%}")

else:
    st.info("👆 Upload an image or click an example above to get started!")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and PyTorch")