import torch
import urllib
from PIL import Image
import streamlit as st
import os
import os.path as op
from torchvision import models, transforms
st.set_page_config(layout="centered")
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
model.fc = torch.nn.Linear(num_features, 54)  
model.load_state_dict(torch.load("Finalmodel.pth", map_location=torch.device('cpu')))
model.eval()

class_names = {
 0:"ก๋วยเตี๋ยวเรือ",
 1:"หอยเเครงลวก",
 2:"กุ้ยช่าย",
 3:"ยําปลาดุกฟู",
 4:"กะหรี่พัฟ",
 5:"หอยจ๊อ",
 6:"กุ้งเเม่นํ้า",
 7:"ขนมไข่เต่า",
 8:"ขนมครก",
 9:"กุ้งเเช่นํ้าปลา",
 10:"มะม่วงนํ้าปลาหวาน",
 11:"ข้าวไข่เจียว",
 12:"ปลากระพงนึ่งมะนาว",
 13:"ราดหน้า",
 14:"ห่อหมกปลา",
 15:"ข้าวขาหมู",
 16:"ผัดหอยเเครง",
 17:"ไก่ผัดเม็ดมะม่วงหิมพานต์",
 18:"เเกงไตปลา",
 19:"ยําวุ้นเส้น",
 20:"เย็นตาโฟ",
 21:"ข้าวผัดอเมริกัน",
 22:"เเกงจืดมะระ",
 23:"บัวลอย",
 24:"ปูผัดผงกะหรี่",
 25:"ไก่ทอด",
 26:"ปอเปี้ยทอด",
 27:"ปลาหมึกย่าง",
 28:"ขนมชั้น",
 29:"ขนมตาล",
 30:"ไข่พะโล้",
 31:"ข้าวคลุกกะปิ",
 32:"ข้าวหมกไก่",
 33:"ข้าวหน้าเป็ด",
 34:"ข้าวซอย",
 35:"ข้าวยํา",
 36:"ลาบ",
 37:"ข้าวเหนียวมะม่วง",
 38:"เเกงมัสมั่น",
 39:"หมูกรอบ",
 40:"ผัดซีอิ้ว",
 41:"ผัดไทย",
 42:"ข้าวผัดสับปะรด",
 43:"ปลาสามรส",
 44:"กระเพราหมู",
 45:"ไส้อั่ว",
 46:"ปลาย่างเกลือ",
 47:"ส้มตํา",
 48:"ไข่ลูกเขย",
 49:"ผัดผักบุ้ง",
 50:"ผัดสะตอ",
 51:"ทอดมันปลา",
 52:"เเกงข่าไก่",
 53:"ต้มยํากุ้ง"
}

model.food_class = class_names 

if file_up is not None:
    col1 , col2 ,col3 , col4 = st.columns(4)
    with col1:
        st.write(' ')

    with col2:
        st.image('cat.gif')

    with col3:
        st.image('cat-jump.gif')
    with col4:
        st.write(' ')    
        
        
    temp_file_path = "temp.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(file_up.getvalue())

    img = Image.open(temp_file_path)

    st.image(img, caption='รูปอาหาร', use_column_width=True)

    scaled_img = transform_test(img)
    torch_images = scaled_img.unsqueeze(0)

    with torch.no_grad():
        outputs = model(torch_images)

        _, predict = torch.max(outputs, 1)
        pred_id = predict.item()
        st.write('ชนิดอาหาร:', model.food_class[pred_id])
        st.balloons()

    os.remove(temp_file_path)
else:
    st.write("Please upload an image file.")
    col1 , col2 , col3 = st.columns(3)
    with col1:
        st.write(' ')

    with col2:
        st.image('banana-crying-cat.gif')

    with col3:
        st.write(' ')

