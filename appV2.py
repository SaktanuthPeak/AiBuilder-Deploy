from torchvision import transforms
import torch
import urllib
from PIL import Image
import streamlit as st
import urllib.request

st.title("Thai food image classification")
st.write("")
file_up = st.file_uploader("Upload an image", type = "jpg")
transform_test = transforms.Compose([
    transforms.Resize((224, 224), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
Model_URL = "https://github.com/SaktanuthPeak/AiBuilder-Deploy/blob/main/Foodimgcls.pth"
urllib.request.urlretrieve(Model_URL, "model.pth")

model = torch.load('model.pth')

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