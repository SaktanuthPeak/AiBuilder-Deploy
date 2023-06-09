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

# Load the class names from a dictionary or file
class_names = {
    0: "Class 0",
    1: "Class 1",
 "Vermicelli_Salad",
 'Tai_pla_curry',
 'moo_krob',
 'Omlette_with_rice',
 'Grilled_river_prawns',
 'Pla_kapong_nueng_manow',
 'massaman_curry',
 'Koong_chae_nampla',
 'Stewed_pork_leg',
 'tom_yam_kung',
 'Boat_noodle',
 'pad_thai',
 'Stir_Fried_Chicken_with_Cashew_Nuts',
 'pla_sam_rod',
 'tom_kha_kai',
 'grilled_squid',
 'pad_si_ew',
 'mango_with_sticky_rice',
 'khao_klook_kapi',
 'khao_soi',
 'Kanom_khai_tao',
 'Curry_puff',
 'kanom_tarn',
 'Deep_fried_crab_meat_roll',
 'khao_na_ped',
 'salt_grilled_fish',
 'khao_yam',
 'pineapple_fried_rice',
 'stir-fried_stink_bean',
 'khai_palo',
 'som_tam',
 'crab_curry',
 'Rad_na',
 'Mango_with_sweet_fish_sauce',
 'stir-fried_morning_glory',
 'fried_spring_roll',
 'Kanom_krok',
 'Crispy_Catfish_Salad',
 'Boiled_cockles',
 'sai_ua',
 'Yen_ta_fo',
 'khao_mok_kai',
 'Chives',
 'Stir_Fried_Baby_Clams',
 'son_in_law_egg',
 'fried_chicken',
 'pork_basil_with_rice',
 'kanom_chan',
 'Steamed_Fish_Curry',
 'american_fried_rice',
 'larb',
 'tod_mun_pla',
 'bitter_gourd_soup',
 'bua_loi'
    
}

model.food_class = class_names  # Assign the class names to the model's food_class attribute

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
        outputs = model(torch_images)

        _, predict = torch.max(outputs, 1)
        pred_id = predict.item()
        print('ชนิดอาหาร:', model.food_class[pred_id])

    # Remove the temporary image file
    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")
