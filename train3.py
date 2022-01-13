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

from additive_decomp2 import Unet as net
from additive_decomp2 import sparse_one, sparse_two, get_norm_sparse1, get_norm_sparse2, num_parameter, get_norm_dense
from Dataset import USDataset as newset
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

scale = 0.1
model = net(1,1,scale=scale).to(device)

num_param = num_parameter(model)
print(num_param)

criterion = cfg.DiceLoss()

optimizer = optim.Adam(model.parameters(), lr=5e-4)

transform_train = tf.Compose([
        tf.ToTensor(),
        tf.ToPILImage(),
        #tf.Resize((442,565)),
        tf.Resize((256,256)),
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

#path1 = "/home/DATA/ymh/ultra/newset/wrist_train/wrist_HM70A"
#path1 = "/data/ymh/US_segmentation/newset/wrist_train/wrist_HM70A"
path1 = "/DATA/ymh_ksw/dataset/US_image/newset/wrist_train/wrist_HM70A"
#path2 = "/home/DATA/ymh/ultra/newset/wrist_target/wrist_HM70A"
#path2 = "/data/ymh/US_segmentation/newset/wrist_target/wrist_HM70A"
path2 = "/DATA/ymh_ksw/dataset/US_image/newset/wrist_target/wrist_HM70A"

dataset_labeled = newset('wrist_HM70A', path1, path2, transform_train, transform_train)
dataset_unlabaled = newset('wrist_HM70A', path1, path2, transform_train, transform_train)
dataset_val = newset('wrist_HM70A', path1, path2, transform_valid, transform_valid)

labeled_idx = cfg.wrist_HM70A_labeled_idx
unlabeled_idx = cfg.wrist_HM70A_unlabeled_idx
valid_idx = cfg.wrist_HM70A_valid_idx
n_val_sample = len(valid_idx)
labeled_sampler = SubsetRandomSampler(labeled_idx)
unlabeled_sampler = SubsetRandomSampler(unlabeled_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

labeled_batch = 4
unlabeled_batch = 4
labeled_loader = DataLoader(dataset=dataset_labeled, batch_size=labeled_batch, sampler=labeled_sampler)
unlabeled_loader = DataLoader(dataset=dataset_unlabaled, batch_size=unlabeled_batch, sampler=unlabeled_sampler)
valid_loader = DataLoader(dataset=dataset_val, batch_size=1, sampler=valid_sampler)

# torch.linspace(0., 0., 300)
lambda_1 = 1# torch.linspace(0, 15, 300).to(device)
lambda_2 = 0.00001
lambda_3 = 1e-7

for epoch in range(300):
    print("="*100)
    
    running_loss = 0
    running_loss1 = 0
    running_loss2 = 0
    running_loss3 = 0
    running_loss4 = 0
    
    model.train()
    for x_U, _ in unlabeled_loader:
        x_U = x_U.float().to(device)      
        
        x_L, y_L = labeled_loader.__iter__().next()
        x_L = x_L.float().to(device)
        y_L = (y_L >= 0.5).float().to(device)
        
        optimizer.zero_grad()
        
        ## ---- Labeled set ---- ##
        sparse_one(model)
        output_L1 = model(x_L)
        
        sparse_two(model)
        output_L2 = model(x_L)
        
        loss_L = (criterion(output_L1, y_L) + criterion(output_L2, y_L))/2
        
        ## ---- Unlabeled set ---- ##
        sparse_one(model)
        output_U1 = model(x_U)

        sparse_two(model)
        output_U2 = model(x_U)

        loss_U = (criterion(output_U1, (output_U2 >= 0.5).detach()) + criterion(output_U2, (output_U1 >= 0.5).detach()))/2
        
        ## ---- L1 norm for Sparsity ---- ##
        Sparse_norm = (get_norm_sparse1(model) + get_norm_sparse2(model))/2
        #L1_norm /= num_param # averaging with the number of parameters
        
        ## ---- L2 norm for Density ---- ##
        Dense_norm = get_norm_dense(model)
        #L2_norm /= num_param
        
        ## ---- Total ---- ##
        loss = loss_L + lambda_1 * loss_U + lambda_2 * Dense_norm + lambda_3 * Sparse_norm
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_loss1 += loss_L.item()
        running_loss2 += loss_U.item()
        running_loss3 += Sparse_norm.item()
        running_loss4 += Dense_norm.item()
            
    running_loss /= len(unlabeled_loader)
    running_loss1 /= len(unlabeled_loader)
    running_loss2 /= len(unlabeled_loader)
    running_loss3 /= len(unlabeled_loader)
    running_loss4 /= len(unlabeled_loader)
    print("[Epoch:%d] [Loss:%f] [Labeled:%f] [Norm:%f] [Unlabeled:%f] [Norm:%f]" % ((epoch+1), running_loss, running_loss1, running_loss4, running_loss2, running_loss3))

    if (epoch+1) % 1 == 0:
        ## ---- Sparse1 ---- ##
        model.eval()
        sparse_one(model)
        
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_dice = 0

        for x_V, y_V in valid_loader:
            x_V = x_V.float().to(device)
            y_V = y_V >= 0.5
            y_V = y_V.to(device)
    
            with torch.no_grad():
                output_V1 = model(x_V)
    
            pred = output_V1 >= 0.5        
            pred = pred.view(-1)
                    
            Trues = pred[pred == y_V.view(-1)]
            Falses = pred[pred != y_V.view(-1)]
                
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
                
        print("[Sparse1]", end=" ")
        print("[Accuracy:%f]" % mean_accuracy, end=" ")
        print("[Precision:%f]" % mean_precision, end=" ")
        print("[Recall:%f]" % mean_recall, end=" ")
        print("[F1 score:%f]" % mean_dice)
        
        ## ---- Sparse2 ---- ##
        model.eval()
        sparse_two(model)
        
        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_dice = 0

        for x_V, y_V in valid_loader:
            x_V = x_V.float().to(device)
            y_V = y_V >= 0.5
            y_V = y_V.to(device)
    
            with torch.no_grad():
                output_V2 = model(x_V)
    
            pred = output_V2 >= 0.5        
            pred = pred.view(-1)
                    
            Trues = pred[pred == y_V.view(-1)]
            Falses = pred[pred != y_V.view(-1)]
                
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
                
        print("[Sparse2]", end=" ")
        print("[Accuracy:%f]" % mean_accuracy, end=" ")
        print("[Precision:%f]" % mean_precision, end=" ")
        print("[Recall:%f]" % mean_recall, end=" ")
        print("[F1 score:%f]" % mean_dice)

