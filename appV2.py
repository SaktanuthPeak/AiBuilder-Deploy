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
MODEL_URL = "https://github.com/SaktanuthPeak/AiBuilder-Deploy/blob/main/Foodimgcls.pth"
urllib.request.urlretrieve(MODEL_URL, "Foodimgcls.pkl")
model = models.resnet34()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 54)  
model.load_state_dict(torch.load("Foodimgcls.pth", map_location=torch.device('cpu')))

model.eval()


class_names = {
    
 0:"Vermicelli_Salad",
 1:"Tai_pla_curry",
 2:"moo_krob",
 3:"Omlette_with_rice",
 4:"Grilled_river_prawns",
 5:"Pla_kapong_nueng_manow",
 6:"massaman_curry",
 7:"Koong_chae_nampl",
 8:"Stewed_pork_leg",
 9:"tom_yam_kung",
 10:"Boat_noodle",
 11:"pad_thai",
 12:"Stir_Fried_Chicken_with_Cashew_Nuts",
 13:"pla_sam_rod",
 14:"tom_kha_kai",
 15:"grilled_squid",
 16:"pad_si_ew",
 17:"mango_with_sticky_rice",
 18:"khao_klook_kapi",
 19:"khao_soi",
 20:"Kanom_khai_tao",
 21:"Curry_puff",
 22:"kanom_tarn",
 23:"Deep_fried_crab_meat_roll",
 24:"khao_na_ped",
 25:"salt_grilled_fish",
 26:"khao_yam",
 27:"pineapple_fried_rice",
 28:"stir-fried_stink_bean",
 29:"khai_palo",
 30:"som_tam",
 31:"crab_curry",
 32:"Rad_na",
 33:"Mango_with_sweet_fish_sauce",
 34:"stir-fried_morning_glory",
 35:"fried_spring_roll",
 36:"Kanom_krok",
 37:"Crispy_Catfish_Salad",
 38:"Boiled_cockles",
 39:"sai_ua",
 40:"Yen_ta_fo",
 41:"khao_mok_kai",
 42:"Chives",
 43:"Stir_Fried_Baby_Clams",
 44:"son_in_law_egg",
 45:"fried_chicken",
 46:"pork_basil_with_rice",
 47:"kanom_chan",
 48:"Steamed_Fish_Curry",
 49:"american_fried_rice",
 50:"larb",
 51:"tod_mun_pla",
 52:"bitter_gourd_soup",
 53:"bua_loi"
    
}

model.food_class = class_names 


if file_up is not None:
    
    temp_file_path = "temp.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(file_up.getvalue())

    
    img = Image.open(temp_file_path)
    scaled_img = transform_test(img)
    torch_images = scaled_img.unsqueeze(0)  

    with torch.no_grad():
        outputs = model(torch_images)

        _, predict = torch.max(outputs, 1)
        pred_id = predict.item()
        print('ชนิดอาหาร:', model.food_class[pred_id])

    
    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")
