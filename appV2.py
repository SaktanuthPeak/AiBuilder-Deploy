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
 0:"ก๋วยเตี๋ยวเรือ",
 1:"bitter_gourd_soup",
 2:"กุ้ยช่าย",
 3:"ยําปลาดุกฟู",
 4:"กะหรี่พัฟ",
 5:"หอยจ๊อ",
 6:"กุ้งเเม่นํ้า",
 7:"ขนมไข่เต่า",
 8:"ขนมครก",
 9:"Deep_fried_crab_meat_roll",
 10:"fried_chicken",
 11:"fried_spring_roll",
 12:"Grilled_river_prawns",
 13:"grilled_squid",
 14:"kanom_chan",
 15:"Kanom_khai_tao",
 16:"Kanom_krok",
 17:"kanom_tarn",
 18:"khai_palo",
 19:"khao_klook_kapi",
 20:"khao_mok_kai",
 21:"ข้าวผัดอเมริกัน",
 22:"เเกงจืดมะระ",
 23:"ข้าวยํา",
 24:"ปูผัดผงกะหรี่",
 25:"ไก่ทอด",
 26:"ปอเปี้ยทอด",
 27:"ปลาหมึกย่าง",
 28:"ขนมชั้น",
 29:"ขนมตาล",
 30:"Omlette_with_rice",
 31:"pad_si_ew",
 32:"pad_thai",
 33:"pineapple_fried_rice",
 34:"Pla_kapong_nueng_manow",
 35:"pla_sam_rod",
 36:"pork_basil_with_rice",
 37:"Rad_na",
 38:"sai_ua",
 39:"salt_grilled_fish",
 40:"som_tam",
 41:"son_in_law_egg",
 42:"Steamed_Fish_Curry",
 43:"Stewed_pork_leg",
 44:"Stir_Fried_Baby_Clams",
 45:"Stir_Fried_Chicken_with_Cashew_Nuts",
 46:"stir-fried_morning_glory",
 47:"stir-fried_stink_bean",
 48:"Tai_pla_curry",
 49:"tod_mun_pla",
 50:"tom_kha_kai",
 51:"tom_yam_kung",
 52:"Vermicelli_Salad",
 53: "Yen_ta_fo"
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
        st.write('ชนิดอาหาร:', model.food_class[pred_id])

    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")
