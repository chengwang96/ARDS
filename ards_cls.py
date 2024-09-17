from __future__ import print_function, division
import os
import random
import numpy as np
from PIL import Image
from torch import optim
import torch.utils.data
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from PIL import Image

import monai.transforms as mtf

import random
import time
import datetime
import argparse
import datetime

from vit3d_pytorch import ViT3D
from dataset import ARDSDataset
from vit import ViT
from losses import FocalLoss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--dataset', default='ards_v2', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--model', default='vit3d', type=str)
    parser.add_argument('--initial_lr', default=1e-4, type=float)
    parser.add_argument('--modal', default='mediastinum_window', type=str)
    parser.add_argument('--loss', default='ce', type=str)

    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--intensity', action='store_true')
    parser.add_argument('--use_smote', action='store_true')
    
    config = parser.parse_args()

    return config


config = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################
# Setting the basic paramters of the model
#######################################################

image_size = (32, 256, 256)
batch_size = config.batch_size
max_epoch = config.max_epoch
num_workers = config.batch_size
pin_memory = True

#######################################################
# Setting up the model
#######################################################

if config.model == 'vit3d':
    patch_size = (4, 16, 16)
    model = ViT3D(image_size=image_size, patch_size=patch_size, num_classes=config.num_classes, dim=256, depth=2, heads=4, mlp_dim=512, dropout=0.2, emb_dropout=0.2)
elif config.model == 'med_clip':
    patch_size = (4, 16, 16)
    model = ViT(in_channels=1, img_size=image_size, patch_size=patch_size, pos_embed="perceptron", spatial_dims=len(patch_size), classification=True)
    model.load_state_dict(torch.load('./pretrained_ViT.bin', map_location='cpu'), strict=False)
model = model.cuda()
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model) / 1_000_000
print(f'Parameters: {num_params:.2f} M')

#######################################################
# Dataset
#######################################################

dataset_name = config.dataset
data_dir = os.path.join('data', dataset_name)
print('Using {} dataset'.format(dataset_name))
all_subfolders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
numeric_subfolders = [f for f in all_subfolders if f.isdigit()]
numeric_subfolders = sorted(numeric_subfolders, key=int)
# random.shuffle(numeric_subfolders)
split_point = int(len(numeric_subfolders) * 0.2)

train_img_ids = numeric_subfolders[split_point:]
val_img_ids = numeric_subfolders[:split_point]

transform_list = []
if config.rotate:
    transform_list.append(mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)))
if config.flip:
    transform_list.append(mtf.RandFlip(prob=0.10, spatial_axis=0))
    transform_list.append(mtf.RandFlip(prob=0.10, spatial_axis=1))
    transform_list.append(mtf.RandFlip(prob=0.10, spatial_axis=2))
if config.intensity:
    transform_list.append(mtf.RandScaleIntensity(factors=0.1, prob=0.5))
    transform_list.append(mtf.RandShiftIntensity(offsets=0.1, prob=0.5))
transform_list.append(mtf.ToTensor(dtype=torch.float))
train_transform = mtf.Compose(transform_list)

val_transform = mtf.Compose(
    [
        mtf.ToTensor(dtype=torch.float),
    ]
)

start_time = time.time()
train_dataset = ARDSDataset(data_root=data_dir, img_ids=train_img_ids, num_classes=config.num_classes, transform=train_transform, image_size=image_size, mode='train', modal=config.modal, use_smote=config.use_smote)
print('training dataset is loaded, has {} examples'.format(len(train_dataset)))
val_dataset = ARDSDataset(data_root=data_dir, img_ids=val_img_ids, num_classes=config.num_classes, transform=val_transform, image_size=image_size, mode='val', modal=config.modal, use_smote=config.use_smote)
print('validation dataset is loaded, has {} examples'.format(len(val_dataset)))
end_time = time.time()
elapsed_time = end_time - start_time
print("data time: {:.2f}s".format(elapsed_time))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False)

#######################################################
# Optimizer
#######################################################

initial_lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
#opt = optim.SGD(model.parameters(), lr = initial_lr, momentum=0.99)

# MAX_STEP = int(1e10)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEP, eta_min=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch)

now = datetime.datetime.now()
formatted_time = now.strftime("%m%d%H%M")
New_folder = './output/{}_{}_{}_{}'.format(dataset_name, config.modal, config.model, config.loss)
if config.rotate:
    New_folder += '_rotate'
if config.flip:
    New_folder += '_flip'
if config.intensity:
    New_folder += '_intensity'
if config.use_smote:
    New_folder += '_smote'
New_folder += '_{}'.format(formatted_time)

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    import ipdb; ipdb.set_trace()

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

my_writer = SummaryWriter(New_folder)

#######################################################
# Training loop
#######################################################

if config.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
elif config.loss == 'focal':
    criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
best_val_acc = 0

for epoch in range(max_epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = {i: 0 for i in range(config.num_classes)}
    class_total = {i: 0 for i in range(config.num_classes)}

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for label, prediction in zip(labels, predicted):
            if label == prediction:
                class_correct[label.item()] += 1
            class_total[label.item()] += 1

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    class_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(config.num_classes)}

    my_writer.add_scalar('train/loss', epoch_loss, global_step=epoch)
    my_writer.add_scalar('train/acc', epoch_acc, global_step=epoch)

    if (epoch+1) % 1 == 0:
        now = datetime.datetime.now()
        print('Epoch: {}/{} \t {:02d}:{:02d}:{:02d} \t Training Loss: {:.6f} Training Acc: {:.6f}'.format(epoch + 1, max_epoch, now.hour, now.minute, now.second, epoch_loss, epoch_acc))
        for i in range(config.num_classes):
            print(f'Class {i} Accuracy: {class_acc[i]:.4f}')
    
    if (epoch+1) % 5 == 0:
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        num_classes = config.num_classes
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        class_correct[label.item()] += 1
                    class_total[label.item()] += 1

        val_loss = running_loss / total
        val_acc = correct / total

        class_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)}

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(New_folder, 'best.pth'))

        my_writer.add_scalar('val/loss', val_loss, global_step=epoch)
        my_writer.add_scalar('val/acc', val_acc, global_step=epoch)

        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        for i in range(num_classes):
            print(f'Class {i} Accuracy: {class_acc[i]:.4f}')
        torch.cuda.empty_cache()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model.load_state_dict(torch.load(os.path.join(New_folder, 'best.pth')))

#######################################################
# Validation
#######################################################

model.eval()
running_loss = 0.0
correct = 0
total = 0

num_classes = config.num_classes
class_correct = {i: 0 for i in range(num_classes)}
class_total = {i: 0 for i in range(num_classes)}

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for label, prediction in zip(labels, predicted):
            if label == prediction:
                class_correct[label.item()] += 1
            class_total[label.item()] += 1

val_loss = running_loss / total
val_acc = correct / total

class_acc = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)}

my_writer.add_scalar('test/loss', val_loss, global_step=i)
my_writer.add_scalar('test/acc', val_acc, global_step=i)

print(f'Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.4f}')
for i in range(num_classes):
    print(f'Class {i} Accuracy: {class_acc[i]:.4f}')
torch.cuda.empty_cache()
