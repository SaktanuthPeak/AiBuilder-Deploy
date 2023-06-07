from PIL import Image
import torch 
from torchvision import models, transforms
import streamlit as st

# set title of app
st.title("Simple Image Classification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")