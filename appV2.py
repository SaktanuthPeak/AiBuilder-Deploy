import torch

import urllib
from PIL import Image
import streamlit as st
import urllib.request
import os
import os.path as op
import pandas as pd
import numpy as np
from torchvision import datasets, models, transforms
import torch
from torch.utils.data import Dataset
import cv2

st.title("Thai food image classification")
st.write("")
file_up = st.file_uploader("Upload an image", type = "jpg")
transform_test = transforms.Compose([
    transforms.Resize((224, 224), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])




model = models.resnet50()  # Or the appropriate model architecture
model.load_state_dict(torch.load("Foodimgcls.pth", map_location=torch.device('cpu')))
model.eval()

# enable users to upload images for the model to make predictions
img = Image.open(urllib.request.urlopen(file_up))
scaled_img = transform_test(img)
torch_images = scaled_img.unsqueeze(0)

with torch.no_grad():
    top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(torch_images)

    _, predict = torch.max(concat_logits, 1)
    pred_id = predict.item()
    print(' ชนิดอาหาร :', model.food_class[pred_id])