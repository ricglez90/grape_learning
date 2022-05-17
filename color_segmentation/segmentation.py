#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:59:54 2021

@author: cacie
"""
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from color_index import normalizar8
#import colorsys
#import math

def number_objects(bin_img,ap):
    """
    bin_img = numpy array with a binary image where 0 is the bg and 1 (or 255) is the segmented area.
    ap = mean value of the round object (grape)
    """
    az = bin_img.shape[0]*bin_img.shape[1]
    q = (az - np.count_nonzero(bin_img)) / az
    objs = -(az/ap) * np.log(q)
    return objs

def create_circular_mask(img, circulos,eti):
    h, w, p = img.shape
    mask1 = np.zeros_like(img[:,:,0],dtype=np.uint8)

    X, Y = np.ogrid[:h, :w]
    for i in range(circulos.shape[0]):
        center = [circulos[i,0],circulos[i,1]]
        radius = circulos[i,2]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        mask1[mask==True] = eti 
    return mask1

def global_threshold_gs(I,delta):
    B = np.zeros((I.shape[0],I.shape[1]),dtype=np.uint8)
    B[I>=delta] = 255
    return B

def global_threshold_color(I,deltas,dmax):
    B = np.zeros((I.shape[0],I.shape[1]),dtype=np.uint8)
    d = np.linalg.norm(I-deltas,axis=2)
    B[d>=dmax] = 255
    return B

def k_means_segmentation(I,k,iterar):
    np.random.seed(None)
    mu = np.random.randint(0,255,(k,I.shape[2]))  
    d = np.zeros((I.shape[0],I.shape[1],k), dtype=np.float)
    
    for i in range(iterar):
        for j in range(k):
            d[:,:,j] = np.linalg.norm(I-mu[j],axis=2)
            
        S = np.argmin(d,axis=2)
        for l in range(k):
            tm1 = I[S==l]
            if len(tm1) == 0:
                mu[l] = np.zeros(I.shape[2])
            else:
                mu[l] = np.mean(tm1,axis=0)

    return normalizar8(S)

def fc_means_color_segmentation(I,k,p,iters):
    if I.shape[2] == 2:
        x =  np.concatenate((I[:,:,0].reshape(I.shape[0]*I.shape[1],1),I[:,:,1].reshape(I.shape[0]*I.shape[1],1)),axis=1)
    if I.shape[2] == 3:
        x =  np.concatenate((I[:,:,0].reshape(I.shape[0]*I.shape[1],1),I[:,:,1].reshape(I.shape[0]*I.shape[1],1),I[:,:,2].reshape(I.shape[0]*I.shape[1],1)),axis=1)
        
    else:
        x = I[:,:,0].reshape(I.shape[0]*I.shape[1],1)
        for i in range(1,I.shape[2],1):
            x = np.hstack((x,I[:,:,i].reshape(I.shape[0]*I.shape[1],1)))
            
    c =  np.random.randint(0,255,(k,x.shape[1]))
    for kc in range(iters):
        d = cdist(x, c, metric='euclidean').T
        exp = -2. / (p - 1)
        d2 = d ** exp
        w = d2 / np.sum(d2, axis=0, keepdims=1)
        wp = w**p
        c = (np.dot(x.T,wp.T)) / (np.atleast_2d(wp.sum(axis=1)))
        if np.isnan(c).any():
            wan = np.isnan(c)
            c[wan] = 0
            c = c.T
        else:
            c = c.T
    SSE = np.argmax(w,axis=0).reshape(I.shape[0],I.shape[1])
    return normalizar8(SSE)