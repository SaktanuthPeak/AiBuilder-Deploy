!pip install torchvision
!pip install pytorch-lightning

from google.colab import drive
drive.mount('/content/drive')

import shutil
source_dir = "/content/drive/MyDrive/10classes.zip"
destination_dir = "/content"
shutil.copy(source_dir,destination_dir)

!unzip '/content/dataset.zip' -d '/content/dataset'

import os
import os.path as op
import shutil
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from torchvision import datasets, models, transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

train_transform = T.Compose([  
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
val_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),)
])

train_data = datasets.ImageFolder("/content/dataset/train", transform=train_transform)
val_data = datasets.ImageFolder("/content/dataset/val", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

n_train = len(train_loader.dataset)
n_val = len(val_loader.dataset)

n_train,n_val

images, labels = next(iter(train_loader))

idx2_class = {v: k for k, v in train_data.class_to_idx.items()}
fig = plt.figure(figsize=(25, 4))
for i in range(10):
    image = np.transpose(images.cpu()[i])
    label = idx2_class[labels.cpu().tolist()[i]]
    ax = fig.add_subplot(2, 8, i + 1, xticks=[], yticks=[])
    plt.imshow(image)
    ax.set_title(label)

model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
print(model)    

model.fc = nn.Linear(in_features=512, out_features=len(train_data.classes))
print(model)

cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)

gpu = torch.cuda.is_available()
print(gpu)
if gpu:
    model.cuda()

n_epochs = 70
for epoch in range(n_epochs):
    
    model.train()
    train_loss, val_loss = 0, 0
    for images, labels in tqdm(train_loader):
        if gpu:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        pred = model(images) 
        loss = cross_entropy(pred, labels)
        loss.backward() 
        optimizer.step() 
        train_loss += loss.item() * images.size(0)

    
    model.eval() 
    for images, labels in tqdm(val_loader):
        if gpu:
            images, labels = images.cuda(), labels.cuda()
        pred = model(images)
        loss = cross_entropy(pred, labels)
        val_loss += loss.item() * images.size(0)
    print("Training loss = {}, Validation loss = {}".format(train_loss / n_train, val_loss / n_val))    

 # คำนวณหา classification report สำหรับ validation set
y_pred, y_true = [], []
model.eval() 
for images, labels in tqdm(val_loader):
    if gpu:
        images, labels = images.cuda(), labels.cuda()
    pred = model(images)
    yp = pred.argmax(dim=1).tolist()
    yt = labels.tolist()
    y_pred.extend(yp)
    y_true.extend(yt)
print(classification_report(y_true, y_pred))

print("Accuracy on validation set = {}".format(
    accuracy_score(y_true, y_pred))
)