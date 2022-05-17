#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:21:21 2021

@author: M.R. Gonzalez
"""

import torch.nn as nn
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import PIL
import glob
import numpy as np
import json
import cv2

#F = nn.functional

#=====================Unnormalization========================================
#https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
#============CustomDataset for semantic segmentation=========================
class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths, extension, transform1, train=True):   # initial logic happens like transform

        self.image_paths = glob.glob(image_paths+'*.'+extension)
        self.target_paths = glob.glob(target_paths+'*.'+extension)
        self.transform1 = transform1
        self.size_img = transform1.transforms[0].size[0]

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = PIL.Image.open(self.target_paths[index])
        t_image = self.transform1(image)
        t_mask = mask.resize((self.size_img,self.size_img),PIL.Image.BILINEAR)
        return t_image, torch.from_numpy(np.array(t_mask)).long() , [self.image_paths[index], self.target_paths[index]]

    def __len__(self):  

        return len(self.image_paths)  
    
    
#============CustomDataset for validation=========================
class CustomDataset_val(Dataset):
    def __init__(self, image_paths, extension, transform1, train=True):   # initial logic happens like transform

        self.image_paths = glob.glob(image_paths+'*.'+extension)
        self.transform1 = transform1

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        t_image = self.transform1(image)
        return t_image,  self.image_paths[index]

    def __len__(self):  

        return len(self.image_paths)  
#============CustomDataset for dotted anottation=========================
class CustomDataset_dotted(Dataset):
    def __init__(self, image_paths, target_paths, extension, transform1, train=True):   # initial logic happens like transform

        self.image_paths = sorted(glob.glob(image_paths+'/*/*.'+extension))
        self.target_paths = sorted(glob.glob(target_paths+'/*/*.'+"json"))
        self.transform1 = transform1
        

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        t_image = self.transform1(image)
        
        data = json.load(open(self.target_paths[index]))
        pts = list(data.get('shapes'))
#        print(len(pts))
        points = []
        for p in pts:
            points.append(p["points"])
            
#         print(np.array(points).shape , self.target_paths[index])
            
        return t_image, np.array(points).reshape(len(points),2) , [self.image_paths[index], self.target_paths[index]]

    def __len__(self):  

        return len(self.image_paths)  
#============CustomDataset for dotted anottation_last experiment=========================
class CustomDataset_dotted_sb(Dataset):
    def __init__(self, image_paths, target_paths, extension, transform1, train=True):   # initial logic happens like transform

        self.image_paths = sorted(glob.glob(image_paths+'/*/*.'+extension))
        self.target_paths = sorted(glob.glob(target_paths+'/*/*.'+"json"))
        self.transform1 = transform1
        

    def __getitem__(self, index):
        A = cv2.imread(self.image_paths[index])
        A_gs = cv2.cvtColor(A,cv2.COLOR_BGR2GRAY)
        th,A_bin = cv2.threshold(A_gs,100,255,cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(A_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2:]
        max_area = 0
        bunch = [0,0,0,0]
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            center, rad = cv2.minEnclosingCircle(c)
            if h*w > max_area:
                max_area = h*w
                bunch = [center, rad]
            
        epsilon = 300
        center, rad = bunch
        x, y = center
        new_size = int(2*(rad + epsilon))
        y_new = y-rad - epsilon
        y = int(y_new) if  y_new > 0  else int(0)
        x_new = x-rad - epsilon
        x = int(x_new) if x > 0 else int(0)
        bunch = [x,y,new_size,new_size]
#         image = Image.open(self.image_paths[index])
#         t_image = self.transform1(image)
        image = Image.fromarray(A[bunch[1]:bunch[1]+bunch[-1],bunch[0]:bunch[0]+bunch[-1]])
        t_image = self.transform1(image)
        data = json.load(open(self.target_paths[index]))
        pts = list(data.get('shapes'))
        points = []
        for p in pts:
            points.append(np.array(p["points"]))
            
#         if self.target_paths[index] == "/media/cacie/Data/data_ricardo/CosmeDS/1_Merlot/012/DSC_0049.json":
#         print(len(np.array(points)), self.target_paths[index])
            
        return t_image, np.array(points).reshape(len(points),2) , np.array(bunch),  self.image_paths[index] , self.target_paths[index]

    def __len__(self):  

        return len(self.image_paths)
#---------------------Utils UNet-------------------------------
def mirroring(I,dif1,dif2):
    B,C,N,M = I.shape
    left_edge = torch.flip(I[:,:,:,:dif2], [0,3])
    right_edge = torch.flip(I[:,:,:,M-dif2:], [0,3])
    I = torch.cat((left_edge,I,right_edge),3)
    top_edge = torch.flip(I[:,:,:dif1,:], [0,2])
    bot_edge = torch.flip(I[:,:,N-dif1:,:], [0,2])
    return torch.cat((top_edge,I,bot_edge),2)

def mirroring1(I,dif1,dif2):
    C,N,M = I.shape
    left_edge = torch.flip(I[:,:,:dif2], [0,2])
    right_edge = torch.flip(I[:,:,M-dif2:], [0,2])
    I = torch.cat((left_edge,I,right_edge),2)
    top_edge = torch.flip(I[:,:dif1,:], [0,1])
    bot_edge = torch.flip(I[:,N-dif1:,:], [0,1])
    return torch.cat((top_edge,I,bot_edge),1)

def double_conv(in_c, out_c):
    conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3),
                nn.ReLU(inplace=True)
                        )
    return conv

def crop_img(tensor, target_tensor,labeled = False):
    target_size = target_tensor.size()[2]    
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    if labeled:
        return tensor[:,delta:tensor_size-delta, delta:tensor_size-delta]
    else:
        return tensor[:,:, delta:tensor_size-delta, delta:tensor_size-delta]  

#---------------------Utils Segnet-----------------------------------
def Conv_down(ic, oc, k, p, s, n_layers = 2, drop = 0.0):
    neurons = [nn.Conv2d(in_channels=ic,out_channels=oc,kernel_size=k,padding=p,stride=s),
                                    nn.BatchNorm2d(oc),
                                    nn.ReLU(inplace=True)]
    if n_layers > 1:
        for i in range(n_layers-1):
            neurons += [nn.Conv2d(in_channels=oc, out_channels=oc,kernel_size=k,padding=p,stride=s),
                                nn.BatchNorm2d(oc),
                                nn.ReLU(inplace=True)]
    if n_layers == 3 and drop > 0:
        neurons += [nn.Dropout(0.5)]

    conv = nn.Sequential(*neurons)
    return conv

def Conv_up(ic, oc, k, p, s, n_layers = 2, drop = 0.0):
    neurons = []
    for i in range(n_layers):
        neurons += [nn.Conv2d(in_channels = ic, out_channels = oc if i == n_layers-1 else ic, kernel_size=k,padding=p,stride=s),
                            nn.BatchNorm2d(oc if i == n_layers-1 else ic),
                            nn.ReLU(inplace=True)]
    if n_layers == 3 and drop > 0:
        neurons += [nn.Dropout(0.5)]

    conv = nn.Sequential(*neurons)
    return conv

#============================DS UNet=======================================================
def double_dsconvolution(ic,oc):
    ds_conv = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=ic, groups=ic, kernel_size=3),
            nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=oc, out_channels=oc, groups=oc, kernel_size=3),
            nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=1),
            nn.ReLU(inplace=True)
    )
    return ds_conv

#import torchvision.transforms as transforms
#
#folder_images_val_biv = "/media/cacie/Data/data_ricardo/External_datasets/counting_tests/Kicherer/test_tmps/images/"
#folder_targets_val_biv = "/media/cacie/Data/data_ricardo/External_datasets/counting_tests/Kicherer/test_tmps/targets/"
#normalized = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))# Normalización en el intervalo [-1,1]
#transform = transforms.Compose([transforms.Resize(size=(580, 580)), 
#                                transforms.ToTensor(),
#                                normalized])
#
#train_subset = CustomDataset_dotted(folder_images_val_biv, folder_targets_val_biv,'png',transform, train=False)
#trainloader = torch.utils.data.DataLoader(train_subset,batch_size=1,shuffle=True,num_workers=2)
#
#for data in trainloader:
#    images, labels , paths = data[0], data[1], data[2]
#    print(labels)

