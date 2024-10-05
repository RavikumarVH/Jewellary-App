# app.py

import streamlit as st
import torch
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
import faiss
import json
import os
import pandas as pd
import glob
import base64
from io import BytesIO
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


st.set_page_config(layout="wide", page_title="Image Classifications")

st.write("## Image Classifications")
st.write(
    "Application supports **Bracelets**, **Kasumalai**, **Rings**"
)
st.sidebar.write("## Upload Image :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB



def get_thumbnail(path: str) -> Image:
    img = Image.open(path)
    img.thumbnail((200, 200))
    return img

def image_to_base64(img_path: str) -> str:
    img = get_thumbnail(img_path)
    with BytesIO() as buffer:
        img.save(buffer, 'png') # or 'jpeg'
        return base64.b64encode(buffer.getvalue()).decode()
    
def image_formatter(img_path: str) -> str:
    return f'<img src="data:image/png;base64,{image_to_base64(img_path)}">'

@st.cache_data 
def convert_df(input_df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return input_df.to_html(escape=False, formatters=dict(thumbnail=image_formatter))

# Load your model and FAISS index
class Model:
    def __init__(self):
        # Load your model
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.model.eval()
        
        # Load your FAISS index
        self.index = faiss.read_index(r"faiss_index.index")  # Adjust the path as necessary

        # Load file mapping
        with open("datasets/data.json") as f:
            self.file_mapping = json.load(f)["images"]

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        image = self.transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.model(image).last_hidden_state[:, 0, :].numpy()

        D, I = self.index.search(output.astype('float32'), k=5)

        _k=5

        # Define a threshold and calculate matching percentage
        threshold = 0.5  # Example threshold

        # Calculate match percentages for all k values
        match_percentages = []

        for k in range(1, _k + 1):
            # Count how many distances are below the threshold for the first k neighbors
            matches = np.sum(D[0, :k] < threshold)
            match_percentage = (matches / k) * 100
            match_percentages.append(round(match_percentage,2))
        
        predicted_metadata = [
            {
                "Product_Image": image_formatter(self.file_mapping[idx]["file"]),
                "Category": self.file_mapping[idx]["label"],
                "SubCategory": self.file_mapping[idx].get("product", "N/A"),
                "Scores": match_percentages[i]

            }
            for i,idx in enumerate(I[0])
        ]
        
        return predicted_metadata

# Instantiate the model
model = Model()

# Streamlit UI
uploaded_file = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])



if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("# Input Image:")
        st.image(image, caption='Uploaded Image.',  width=200)
        # Make predictions
        predictions = model.predict(image)
        df = pd.DataFrame(predictions)
        df.index = np.arange(1, len(df) + 1)
        html = convert_df(df)
    
        # Display predictions
        st.markdown("# Image Predictions Results:")
        st.markdown(f"##### Images Category is :green[{predictions[0]["Category"]}] and Sub category is a  :violet[{predictions[0]["SubCategory"]}]")
        #st.markdown(f"##### Images Sub Category is :violet[{predictions[0]["product"]}]")

        st.markdown(
            html,
            unsafe_allow_html=True
        )
    # for result in predictions:
    #     st.write(f"File: {result['file']}, Label: {result['label']}, Breed: {result['product']}")
