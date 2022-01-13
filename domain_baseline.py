import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from skimage.io import imread
import os
import numpy as np

from Unet import Unet as net
from Dataset import USDataset as newset
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = net().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = cfg.DiceLoss()

transform_train = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        #tf.Resize((442,565)),
        tf.Resize((256,256)),
        tf.RandomHorizontalFlip(),
        tf.RandomAffine(0, shear=[-15, 15, -15, 15]),
        tf.ToTensor()
     ])

transform_valid = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        #tf.Resize((442,565)),
        tf.Resize((256,256)),        
        tf.ToTensor(),
     ])


path1s = ["/data/ymh/US_segmentation/newset/wrist_train/wrist_HM70A",
          "/data/ymh/US_segmentation/newset/forearm_train/forearm_HM70A",
          "/data/ymh/US_segmentation/newset/wrist_train/wrist_miniSONO",
          "/data/ymh/US_segmentation/newset/forearm_train/forearm_miniSONO"]

path2s = ["/data/ymh/US_segmentation/newset/wrist_target/wrist_HM70A",
          "/data/ymh/US_segmentation/newset/forearm_target/forearm_HM70A",
          "/data/ymh/US_segmentation/newset/wrist_target/wrist_miniSONO",
          "/data/ymh/US_segmentation/newset/forearm_target/forearm_miniSONO"]

datasets = ['wrist_HM70A', 'forearm_HM70A', 'wrist_miniSONO', 'forearm_miniSONO']
domain = ['HM-wr', 'HM-fr', 'mi-wr', 'mi-fr']

batch_size = 4

i = 3

dataset_train = newset(datasets[i], path1s[i], path2s[i], transform_train, transform_train)
dataset_valid = newset(datasets[i], path1s[i], path2s[i], transform_valid, transform_valid)
    
if i == 0:
    train_idx = cfg.wrist_HM70A_train_idx
    valid_idx = cfg.wrist_HM70A_valid_idx
elif i == 1:
    train_idx = cfg.forearm_HM70A_train_idx
    valid_idx = cfg.forearm_HM70A_valid_idx
elif i == 2:
    train_idx = cfg.wrist_miniSONO_train_idx
    valid_idx = cfg.wrist_miniSONO_valid_idx
elif i == 3:
    train_idx = cfg.forearm_miniSONO_train_idx
    valid_idx = cfg.forearm_miniSONO_valid_idx
    
n_val_sample = len(valid_idx)
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
    
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(dataset=dataset_valid, batch_size=1, sampler=valid_sampler)
    
for epoch in range(300):
    print("="*100)
    
    running_loss = 0
    
    model.train()
    for x, y in train_loader:
        x = x.float().to(device)
        y = (y >= 0.5).float().to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
            
    running_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss), end=" ")

    if (epoch+1) % 1 == 0:
        model.eval()
        
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_dice = 0

        for x, y in valid_loader:
            x = x.float().to(device)
            y = (y >= 0.5).to(device)
    
            with torch.no_grad():
                output = model(x)
    
            pred = output >= 0.5        
            pred = pred.view(-1)
                    
            Trues = pred[pred == y.view(-1)]
            Falses = pred[pred != y.view(-1)]
                
            TP = (Trues == 1).sum().item()
            TN = (Trues == 0).sum().item()
            FP = (Falses == 1).sum().item()
            FN = (Falses == 0).sum().item()
        
            accuracy = (TP + TN)/(TP + TN + FP + FN)
            if TP == 0:
                precision = 0
                recall = 0
                dice = 0
            else:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                dice = (2 * TP) / ((2 * TP) + FP + FN)

            mean_accuracy += accuracy
            mean_precision += precision
            mean_recall += recall
            mean_dice += dice
    
        mean_accuracy /= n_val_sample
        mean_precision /= n_val_sample
        mean_recall /= n_val_sample
        mean_dice /= n_val_sample
                
        print("[%s]" % domain[i], end=" ")
        print("[Accuracy:%f]" % mean_accuracy, end=" ")
        print("[Precision:%f]" % mean_precision, end=" ")
        print("[Recall:%f]" % mean_recall, end=" ")
        print("[F1 score:%f]" % mean_dice)
        

