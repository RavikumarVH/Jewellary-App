import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
#from datasets import load_dataset, concatenate_datasets, load_from_disk
from PIL import Image
from io import BytesIO
import base64
from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset
import os
import warnings
warnings.filterwarnings("ignore")
import tempfile
from itertools import cycle

fileDir = os.path.dirname(os.path.realpath('__file__'))
train_path= fileDir+'/datasets/train'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

st.set_page_config(layout="wide", page_title="Image Classifications")

st.write("## Image Classifications")
st.write(
    "Application only will support for **Bracelets**, **Kasumalai**, **Rings**"
)
st.sidebar.write("## Upload Image :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# st.write("Username:", st.secrets["db_username"])
# st.write("Password:", st.secrets["db_password"])

# st.write(
#     "Has environment variables been set:",
#     os.environ["db_username"] == st.secrets["db_username"],
# )

model_ckpt = "google/vit-base-patch16-224"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

# load training dataset from Google Drive
dataset = load_dataset("imagefolder", data_dir=train_path)

train_dataset = dataset['train']

# resize all PIL images to 224x224
for i, example in enumerate(train_dataset):
    train_dataset[i]['image'] = example['image'].resize((224, 224), Image.BILINEAR)

# labels to names mapping for visualization purposes
folder_names = train_dataset.features['label'].names

# assign labels to the images from the folder names
labels_to_names = {i: folder_names[i] for i in range(len(folder_names))}

#Extract embeddings
def extract_embeddings(image):
    image_pp = extractor(image, return_tensors="pt")
    features = model(**image_pp).last_hidden_state[:, 0].detach().numpy()
    return features.squeeze()

dataset_with_embeddings = train_dataset.map(lambda example: {'embeddings': extract_embeddings(example["image"])})

# save dataset with embeddings variable to disk
dataset_with_embeddings.save_to_disk('old_embeddings')

# load dataset with embeddings from disk
dataset_with_embeddings.add_faiss_index(column='embeddings')

# save the faiss index to disk
dataset_with_embeddings.save_faiss_index('embeddings', 'old_index.faiss')
    

def get_neighbors(query_image, top_k=5):
    qi_embedding = model(**extractor(query_image, return_tensors="pt"))
    qi_embedding = qi_embedding.last_hidden_state[:, 0].detach().numpy().squeeze()
    scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples('embeddings', qi_embedding, k=top_k)
    return retrieved_examples


def image_grid(imgs, rows, cols):
    w,h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs): grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def generateImages(query_image):
    #print(query_image)
    retrieved_examples = get_neighbors(query_image)
    images = [query_image]
    images.extend(retrieved_examples["image"])
    #print(scores)
    find_Image_Category(retrieved_examples, images)
  
    # col1.image(image_grid(images, 1, len(images)))


def generateImage(images):
    cols = cycle(st.columns(len(images)))
    for idx, filteredImage in enumerate(images):
        new_image = filteredImage.resize((400, 300))
        next(cols).image(new_image)
    
    


def find_Image_Category(retrieved_examples,images):
    names = np.array(retrieved_examples['label'])
    ## Logic for finding the majority label: Select the label that occurs the most number of times and if there is a tie, select the first one with higher similarity score
    if labels_to_names[names[0]] != labels_to_names[np.argmax(np.bincount(names))]:
        st.markdown(f"### Images Category is :green[{labels_to_names[names[0]]}]")
    else:
        st.markdown(f"### Images Category is :green[{labels_to_names[np.argmax(np.bincount(names))]}]")

    generateImage(images)


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    pass
    # image = Image.open(upload)
    # col1.write("Original Image :camera:")
    # col1.image(image)

    # fixed = remove(image)
    # col2.write("Fixed Image :wrench:")
    # col2.image(fixed)
    # st.sidebar.markdown("\n")
    # st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])



if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        input_img=Image.open(my_upload)
        # st.write(path)
        generateImages(input_img)

else:
    pass
    #fix_image("./zebra.jpg")