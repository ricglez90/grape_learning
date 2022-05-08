#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 13:00:31 2021

@author: cacie
"""
import sys    
sys.path.insert(0,"/home/cacie/Documents/CNN/modules/cnn_architectures/")
sys.path.insert(0,"/home/cacie/Documents/CNN/modules/color_segmentation/")
import torch
from cnn import SegNet, UNet, deeplabv3plus_mobilenet
from aids_cnn import mirroring, UnNormalize


from color_index import ExG_compute,ExR_compute,ExGR_compute,NDI_compute,CIVE_compute,NGRDI_compute,VEG_compute,COM1_compute,COM2_compute,MExG_compute
from color_index import normalizar1
import pickle

import cv2
import numpy as np

def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick]

def shallow_image(image):
    rows,cols,chan = image.shape
    nrows = rows*cols
    B,G,R = cv2.split(image)
    LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    YCC = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    NDI = NDI_compute(image)
    ExG = ExG_compute(image)
    ExR = ExR_compute(image)
    ExGR = ExGR_compute(image)
    CIVE = CIVE_compute(image)
    NGRDI = NGRDI_compute(image)
    VEG = VEG_compute(image)
    COM1 = COM1_compute(image)
    MExG = MExG_compute(image)
    COM2 = COM2_compute(image)
    list_tmp = np.concatenate((R.reshape(nrows,1),G.reshape(nrows,1),B.reshape(nrows,1),LAB[:,:,0].reshape(nrows,1),LAB[:,:,1].reshape(nrows,1),LAB[:,:,2].reshape(nrows,1),HSV[:,:,0].reshape(nrows,1),HSV[:,:,1].reshape(nrows,1),HSV[:,:,2].reshape(nrows,1),YCC[:,:,0].reshape(nrows,1),YCC[:,:,1].reshape(nrows,1),YCC[:,:,2].reshape(nrows,1),NDI.reshape(nrows,1), ExG.reshape(nrows,1),ExR.reshape(nrows,1),CIVE.reshape(nrows,1), ExGR.reshape(nrows,1),NGRDI.reshape(nrows,1),VEG.reshape(nrows,1),COM1.reshape(nrows,1), MExG.reshape(nrows,1),COM2.reshape(nrows,1)),axis=1)
    return normalizar1(list_tmp)

def shallow_predict(image, model_path, algorithm, morph_type, kernel, min_area):
    img_tmp = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    if model_path is None:
        if algorithm == "KNN":
            model_path = "/media/cacie/Data/data_ricardo/CosmeDS/saved_models/shallow_models/KNN_original.sav"
        if algorithm == "MLR":
            model_path = "/media/cacie/Data/data_ricardo/CosmeDS/saved_models/shallow_models/MLR_original.sav"
        if algorithm == "MLP":
            model_path = "/media/cacie/Data/data_ricardo/CosmeDS/saved_models/shallow_models/MLP_original.sav"
        if algorithm == "ML":
            model_path = "/media/cacie/Data/data_ricardo/CosmeDS/saved_models/shallow_models/ML_original.sav"
        
    
    model = pickle.load(open(model_path,"rb"))
    X_norm = shallow_image(image)
    Y_pred = model.predict(X_norm)
    
    kernel = np.ones((kernel,kernel),np.uint8)
    Y_pred = Y_pred.reshape(image.shape[:2])
    if morph_type == "Opening":
        morph_img = cv2.morphologyEx(Y_pred, cv2.MORPH_OPEN, kernel)
    elif morph_type == "Closing":
        morph_img = cv2.morphologyEx(Y_pred, cv2.MORPH_CLOSE, kernel)
    else:
        morph_img = cv2.morphologyEx(cv2.morphologyEx(Y_pred, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    
    morph_img1 = morph_img.astype(np.uint8).copy()
    morph_img1[morph_img1==1] = 255
    contours, hierarchy = cv2.findContours(morph_img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    boxes = []
    for i in range(len(contours)):
        c1 = contours[i]
        x,y,w,h = cv2.boundingRect(c1)
        rect = cv2.minAreaRect(c1)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        if min_area < rect[1][0]*rect[1][1]:
            boxes.append([x,y,int(x+w),int(y+h)])
#            cv2.drawContours(img_tmp,[box],0,(0,255,0),2)
            
    boxes_n = np.array(boxes)
    boxes_n = non_max_suppression_slow(boxes_n,0.6)
    for bx in boxes_n:
        start = (bx[0],bx[1])
        end = (bx[2],bx[3])
        cv2.rectangle(img_tmp,start,end,(0,255,0),2)
        cv2.putText(img_tmp,"Grape bunch", start,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA )
    return morph_img, img_tmp

def load_model(model_path, cnn_arc, ic, nc, device):
    if cnn_arc == "Segnet":
        model = SegNet(input_channels=ic, n_classes=nc, pretrained=True).to(device)
    elif cnn_arc == "UNet":
        model = UNet(input_channels= ic, n_classes=nc).to(device)
    else:
        model = deeplabv3plus_mobilenet(num_classes=nc,output_stride=6).to(device)
    
    if model_path is None:    
        if cnn_arc == "Segnet":
            model_path = "/media/cacie/Data/data_ricardo/CosmeDS/saved_models/version_2/segnet/no_dropout/segnet_elbueno_pretrain_fine_tuning_100_20218305"
        elif cnn_arc == "UNet":
            model_path = "/media/cacie/Data/data_ricardo/CosmeDS/saved_models/version_2/unet/resized/unet-vgg16_DA_resized_pretrain_fine_tuning_100_202110821"
        elif cnn_arc == "Zabnet":
            model_path = "/media/cacie/Data/data_ricardo/CosmeDS/saved_models/version_2/zabnet/resized_dataset/zabnet_mbnv2_DA_elbueno_2_pretrain_fine_tuning_100_202193010"

        
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def unnorm_img(tensor):
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    Iun = unorm(torch.tensor(tensor))
    return np.dstack((Iun.cpu().numpy()[0,...],Iun.cpu().numpy()[1,...],Iun.cpu().numpy()[2,...]))

def predict(input_image, model_path, cnn_arc, ic, nc):
    img_tmp = unnorm_img(input_image)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, cnn_arc,ic,nc,device)
    
    if cnn_arc == "UNet":
        input_image = mirroring(input_image.unsqueeze(0),92,92).to(device)
    else:
        input_image = input_image.unsqueeze(0).to(device)
    with torch.no_grad():
        _,pred = model(input_image)
        pred_mask = torch.argmax(pred,dim=1)
        pred_mask = pred_mask[0, ...].cpu().numpy()
        
    pred_mask1 = pred_mask.astype(np.uint8).copy()
    pred_mask1[pred_mask1==2] = 0
    contours, hierarchy = cv2.findContours(pred_mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for i in range(len(contours)):
        c1 = contours[i]
        rect = cv2.minAreaRect(c1)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_tmp,[box],0,(0,255,0),2)

    return pred_mask , img_tmp
