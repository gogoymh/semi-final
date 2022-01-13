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

from additive_decomp_var4 import Unet as net
from additive_decomp_var4 import HM70A, miniSONO, wrist, forearm, get_norm_weight, zeroshot
from Dataset import USDataset as newset
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scale = 0.1
model = net(scale).to(device)
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


base_root = ["/data/ymh/US_segmentation",
             "/DATA1/ymh/dataset",
             "/home/compu/ymh"]

base_num = 2

path1s = [base_root[base_num]+"/newset/wrist_train/wrist_HM70A",
          base_root[base_num]+"/newset/forearm_train/forearm_HM70A",
          base_root[base_num]+"/newset/wrist_train/wrist_miniSONO",
          base_root[base_num]+"/newset/forearm_train/forearm_miniSONO"]

path2s = [base_root[base_num]+"/newset/wrist_target/wrist_HM70A",
          base_root[base_num]+"/newset/forearm_target/forearm_HM70A",
          base_root[base_num]+"/newset/wrist_target/wrist_miniSONO",
          base_root[base_num]+"/newset/forearm_target/forearm_miniSONO"]

datasets = ['wrist_HM70A', 'forearm_HM70A', 'wrist_miniSONO', 'forearm_miniSONO']
domain = ['HM-wr', 'HM-fr', 'mi-wr', 'mi-fr']

train_loaders = []
valid_loaders = []
n_val_samples = []

batch_sizes = [2,2,2,2]# [1,1,1,1] # # # 
lambdas = torch.zeros(4) # [1,1,1,1]

for i in range(4):
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
    
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_sizes[i], sampler=train_sampler)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=1, sampler=valid_sampler)
    
    train_loaders.append(train_loader)
    valid_loaders.append(valid_loader)
    n_val_samples.append(n_val_sample)
    lambdas[i] = len(train_idx)

lambdas /= lambdas.sum()
lambdas.to(device)
repeat = max(len(train_loaders[0]),len(train_loaders[1]),len(train_loaders[2]),len(train_loaders[3]))

'''
X_init = []
for _ in range(1):
    for _, train_loader in enumerate(train_loaders):
        x, _ = train_loader.__iter__().next()
        x = x.float().to(device)
        X_init.append(x)
    
X_init = torch.cat(X_init, dim=0)
print(X_init.shape)
model.train()
zeroshot(model, True)
test = model(X_init)
del X_init, test
'''

for epoch in range(300):
    print("="*100)
    
    running_loss = 0
    running_loss1 = 0
    running_loss2 = 0
    running_loss3 = 0
    running_loss4 = 0
    
    zeroshot(model, False)
    model.train()
    for _ in range(repeat):
        loss = 0
        optimizer.zero_grad()
        
        for index, train_loader in enumerate(train_loaders):
            x, y = train_loader.__iter__().next()
            x = x.float().to(device)
            y = (y >= 0.5).float().to(device)
            
            if index == 0:
                HM70A(model)
                wrist(model)
                output = model(x)
                cur_loss = criterion(output, y)
                loss += cur_loss #* lambdas[index]
                running_loss1 += cur_loss.item()
                
            elif index == 1:
                HM70A(model)
                forearm(model)
                output = model(x)
                cur_loss = criterion(output, y)
                loss += cur_loss #* lambdas[index]
                running_loss2 += cur_loss.item()
                
            elif index == 2:
                miniSONO(model)
                wrist(model)
                output = model(x)
                cur_loss = criterion(output, y)
                loss += cur_loss #* lambdas[index]
                running_loss3 += cur_loss.item()
                
            elif index == 3:
                miniSONO(model)
                forearm(model)
                output = model(x)
                cur_loss = criterion(output, y)
                loss += cur_loss #* lambdas[index]
                running_loss4 += cur_loss.item()
        
        ## ---- Total ---- ##
        #norm = get_norm_weight(model) * 1e-5
        #print(norm.item())
        loss /= 4
        #loss += norm
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
            
    running_loss /= repeat
    running_loss1 /= repeat
    running_loss2 /= repeat
    running_loss3 /= repeat
    running_loss4 /= repeat
    print("[Epoch:%d] [Loss:%f] [HM-wr:%f] [HM-fr:%f] [mi-wr:%f] [mi-fr:%f]" % ((epoch+1), running_loss, running_loss1, running_loss2, running_loss3, running_loss4))

    if (epoch+1) % 1 == 0:
        model.eval()
        for index, valid_loader in enumerate(valid_loaders):
            if index == 0:
                HM70A(model)
                wrist(model)
            elif index == 1:
                HM70A(model)
                forearm(model)
            elif index == 2:
                miniSONO(model)
                wrist(model)
            elif index == 3:
                miniSONO(model)
                forearm(model)
        
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
    
            mean_accuracy /= n_val_samples[index]
            mean_precision /= n_val_samples[index]
            mean_recall /= n_val_samples[index]
            mean_dice /= n_val_samples[index]
                
            print("[%s]" % domain[index], end=" ")
            print("[Accuracy:%f]" % mean_accuracy, end=" ")
            print("[Precision:%f]" % mean_precision, end=" ")
            print("[Recall:%f]" % mean_recall, end=" ")
            print("[F1 score:%f]" % mean_dice)
        
        print("Zero-shot")
        zeroshot(model, True)
        for index, valid_loader in enumerate(valid_loaders):
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
    
            mean_accuracy /= n_val_samples[index]
            mean_precision /= n_val_samples[index]
            mean_recall /= n_val_samples[index]
            mean_dice /= n_val_samples[index]
                
            print("[%s]" % domain[index], end=" ")
            print("[Accuracy:%f]" % mean_accuracy, end=" ")
            print("[Precision:%f]" % mean_precision, end=" ")
            print("[Recall:%f]" % mean_recall, end=" ")
            print("[F1 score:%f]" % mean_dice)
