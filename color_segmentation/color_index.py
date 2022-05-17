#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:51:50 2021

@author: cacie
"""

import numpy as np
import cv2

def normalizar1(imagen):
    s = imagen.shape
    mx = 255
    mn = 0
    norm = np.zeros((s), dtype=np.float)
    imagen = np.subtract(imagen,mn)
    norm = np.divide(imagen,float(mx-mn))
    return norm

def normalizar8(imagen):
    tam= np.shape(imagen)
    n = tam[0]
    m = tam[1]
    s = (n,m)
    if np.any(imagen)==float('inf'):
        mx = 255
    else:
        mx = np.max(imagen)
    mn = np.min(imagen)
    norm = np.zeros((s), dtype=np.float)
    imagen = np.subtract(imagen,mn)
    norm = np.uint8(np.divide(imagen,float(mx-mn)) *255)
    return norm

def NDI_compute(A):
    B,G,R = cv2.split(A)
    Gf = G.astype(float)
    Rf = R.astype(float)
    return normalizar8(np.nan_to_num((128 *(np.divide((Gf-Rf),(Gf+Rf))+1))))

def ExG_compute(A):
    B,G,R = cv2.split(A)
    
    Rf = R.astype(float)
    Gf = G.astype(float)
    Bf = B.astype(float)
    
    Rn = Rf/np.max(Rf)
    Gn = Gf/np.max(Gf)
    Bn = Bf/np.max(Bf)
    
    r = Rn/(Rn+Gn+Bn)
    g = Gn/(Rn+Gn+Bn)
    b = Bn/(Rn+Gn+Bn)
    
#    eq1 = r - g
#    eq2 = g - b
#    eq3 = eq2 / eq1
    eq4 = 2*g - r - b
    
    return normalizar8(np.nan_to_num(eq4))

def ExR_compute(A):
    B,G,R = cv2.split(A)
    Rf = R.astype(float)
    Gf = G.astype(float)
    
    return normalizar8(np.nan_to_num(1.3*Rf - Gf))

def CIVE_compute(A):
    B,G,R = cv2.split(A)
    return normalizar8(np.nan_to_num(0.441*R - 0.811*G + 0.385*B + 18.78754))

def ExGR_compute(A):
    ExG, ExR = ExG_compute(A) , ExR_compute(A)
    return normalizar8(np.nan_to_num(ExG - ExR))

def NGRDI_compute(A):
    B,G,R = cv2.split(A)
    Gf = G.astype(float)
    Rf = R.astype(float)
    return normalizar8(np.nan_to_num((Gf-Rf)/(Gf+Rf)))

def VEG_compute(A):
    B,G,R = cv2.split(A)
    Gf = G.astype(float)
    Rf = R.astype(float)
    Bf = B.astype(float)
    a = 0.667
    VEG = Gf/(Rf**(a)*Bf**(1-a))
    VEG[VEG == float('inf')] = np.nanmax(VEG[VEG != float('inf')])
    return normalizar8(np.nan_to_num(VEG))

def COM1_compute(A):
    return normalizar8(np.nan_to_num(ExG_compute(A) + CIVE_compute(A) + ExGR_compute(A) + VEG_compute(A)))

def MExG_compute(A):
    B,G,R = cv2.split(A)
    return normalizar8(np.nan_to_num(1.262*G - 0.884*R - 0.311*B))

def COM2_compute(A):
    return normalizar8(np.nan_to_num(((0.36*ExG_compute(A) + 0.47*CIVE_compute(A) +  0.17*VEG_compute(A)))))

