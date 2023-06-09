import torch
import urllib
from PIL import Image
import streamlit as st
import os
import os.path as op
from torchvision import models, transforms

st.title("Thai food image classification")
st.write("")
file_up = st.file_uploader("Upload an image", type="jpg")
transform_test = transforms.Compose([
    transforms.Resize((224, 224), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

model = models.resnet34()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 54)  # Replace the last fully connected layer for 54 classes

model.load_state_dict(torch.load("Foodimgcls.pth", map_location=torch.device('cpu')))

model.eval()

# enable users to upload images for the model to make predictions
if file_up is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = "temp.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(file_up.getvalue())

    # Open the saved image file using PIL
    img = Image.open(temp_file_path)
    scaled_img = transform_test(img)
    torch_images = scaled_img.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        top_n_coordinates, concat_out, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob = model(torch_images)

        _, predict = torch.max(concat_logits, 1)
        pred_id = predict.item()
        print('ชนิดอาหาร:', model.food_class[pred_id])

    # Remove the temporary image file
    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")
