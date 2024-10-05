import streamlit as st
import json
import numpy as np
import faiss
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTModel, ViTFeatureExtractor
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
model.eval()

# Load and preprocess JSON data
with open('datasets/data_local.json') as f:
    data = json.load(f)["images"]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the FAISS index
index = faiss.read_index('faiss_index.index')

# Extract features for all images (once and save them)
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).last_hidden_state[:, 0, :]
    return features.squeeze().numpy()

# Store features
features = []
for entry in data:
    feature_vector = extract_features(entry['file'])
    features.append(feature_vector)

features_np = np.array(features).astype('float32')
image_paths = [entry['file'] for entry in data]

# Prediction function
def predict(image_path, k=5):
    query_feature = extract_features(image_path)
    query_feature = np.array([query_feature]).astype('float32')
    distances, indices = index.search(query_feature, k)

    results = []
    for i, idx in enumerate(indices[0]):
        match_percentage = (1 - distances[0][i] / np.max(distances)) * 100
        results.append((data[idx]['label'], data[idx]['product'], match_percentage))
        
    return results

# Streamlit app layout
st.title("Image Classification with ViT and FAISS")
st.write("Upload an image to classify it and find similar products.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict
    results = predict(uploaded_file)
    st.write("### Matching Results:")
    
    for product, subproduct, percentage in results:
        st.write(f"**Product:** {product}, **Subproduct:** {subproduct}, **Match Percentage:** {percentage:.2f}%")
