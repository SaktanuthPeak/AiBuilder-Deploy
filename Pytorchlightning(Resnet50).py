!pip install torchvision
!pip install pytorch-lightning

from google.colab import drive
drive.mount('/content/drive')

!pip install -q --pre pytorch-ignite==0.5.0.dev20230325
!pip install -q fastbook==0.0.29
!pip install --upgrade -q mxnet==1.9.1
!pip install -q autogluon==0.7.0
!pip install -q pythainlp==3.1.1
!pip install -q transformers==4.27.3
exit()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as T
from torchvision import models

from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise RuntimeError(
            "This module requires either tensorboardX or torch >= 1.2.0. "
            "You may install tensorboardX with command: \n pip install tensorboardX \n"
            "or upgrade PyTorch using your package manager of choice (pip or conda).")

probs = torch.tensor([[0.4, 0.6],
                      [0.1, 0.9],
                      [0.9, 0.1]], dtype=torch.float)
preds = probs.argmax(1)
targets = torch.tensor([0,1,0])

print(probs)
print(preds)
print(targets)


(preds==targets).float().mean()

train_dir = '/content/54cls/train'
valid_dir = '/content/54cls/val'
batch_size = 256


train_transform = T.Compose([
    T.RandomResizedCrop((224,224),scale=(0.5,1.2)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

valid_transform = T.Compose([
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
valid_ds = datasets.ImageFolder(valid_dir, transform=valid_transform)


train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=0, shuffle=False)

classes = train_ds.classes
n_classes = len(classes)

from PIL import Image
import glob


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, 
                            squeeze=False,)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

test_images = glob.glob('/content/10cls/test/Boat_noodle/064fd7c6-c793-4b41-9faf-801e816a1fad.jpg')
i = np.random.randint(0,len(test_images))
orig_img = Image.open(test_images[i])

augmenter = T.TrivialAugmentWide()
imgs = [augmenter(orig_img) for _ in range(2)]
plot(imgs)

def plot_row(imgs, img_per_row = 3):
    num_cols = img_per_row
    num_rows = len(imgs) // num_cols
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        ax = axs[row_idx // num_cols, row_idx % num_cols]
        ax.imshow(row.permute(1,2,0))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    
sample_imgs = next(iter(train_dl))
plot_row(sample_imgs[0][:9])

backbone = models.resnet50(pretrained=True)
backbone

probs = torch.tensor([[0.4, 0.6],
                      [0.1, 0.9],
                      [0.9, 0.1]], dtype=torch.float)
preds = probs.argmax(1)
targets = torch.tensor([0,1,0])

print(probs)
print(preds)
print(targets)

probs = torch.tensor([[0.4, 0.6],
                      [0.1, 0.9],
                      [0.9, 0.1]], dtype=torch.float)
preds = probs.argmax(1)
targets = torch.tensor([0,1,0])

print(probs)
print(preds)
print(targets)


(preds==targets).float().mean()

optimizer = torch.optim.AdamW(backbone.parameters(), lr=3e-4)
optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FoodResNet(n_classes=len(train_ds.classes)).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer = create_supervised_trainer(model, optimizer, loss_function, device=device)
val_metrics = {"accuracy": Accuracy(), "ce_loss": Loss(loss_function)}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

#logger
writer = SummaryWriter(log_dir='logs')
train_log_interval = 1 
eval_log_interval = len(train_dl) 
pbar = tqdm(initial=0, 
            leave=False, 
            total=len(train_dl), 
            desc=f"epoch {0} - loss: {0:.4f} - lr: {0:.4f}")


@trainer.on(Events.ITERATION_COMPLETED(every=train_log_interval))
def log_training_loss(engine):
    lr = optimizer.param_groups[0]["lr"]
    pbar.desc = f"epoch {engine.state.epoch} - train loss: {engine.state.output:.4f} - lr: {lr:.4f}"
    pbar.update(train_log_interval)
    writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)


@trainer.on(Events.ITERATION_COMPLETED(every=eval_log_interval))
def log_training_results(engine):
    evaluator.run(train_dl)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_loss = metrics["ce_loss"]
    tqdm.write(
        f"train results - epoch: {engine.state.epoch} avg accuracy: {avg_accuracy:.2f} avg loss: {avg_loss:.2f}"
    )
    writer.add_scalar("training/avg_loss", avg_loss, engine.state.iteration)
    writer.add_scalar("training/avg_accuracy", avg_accuracy, engine.state.iteration)
@trainer.on(Events.ITERATION_COMPLETED(every=eval_log_interval))
def log_validation_results(engine):
    evaluator.run(valid_dl)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics["accuracy"]
    avg_loss = metrics["ce_loss"]
    tqdm.write(
        f"valid results - epoch: {engine.state.epoch} avg accuracy: {avg_accuracy:.2f} avg loss: {avg_loss:.2f}"
    )
    writer.add_scalar("valdation/avg_loss", avg_loss, engine.state.iteration)
    writer.add_scalar("valdation/avg_accuracy", avg_accuracy, engine.state.iteration)
    pbar.n = pbar.last_print_n = 0

trainer.run(train_dl, max_epochs=3)
writer.close()

%reload_ext tensorboard
%tensorboard --logdir logs/ --host 0.0.0.0