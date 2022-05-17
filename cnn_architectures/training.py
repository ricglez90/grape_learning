#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:45:09 2021

@author: M.R. Gonzalez
"""
import torch
import numpy as np
from aids import mirroring
#==========================Train module===================================
    
def train(model,trainlodader,valloader,device,criterion,optimizer,n_epochs=10,validation=True, unet_train = False):
    train_logs = []
    valid_logs = []
    for epoch in range(n_epochs):
        current_loss = []
        current_accs = []
        model.train() #model updates weights
        for i, data in enumerate(trainlodader,0):
            inputs, labels = data[0].to(device), data[1].to(device)
            #===================Epoch starts===========================
            #Ponemos a cero los gradientes de los parámetros
            optimizer.zero_grad()
            if unet_train:
                inputs = mirroring(inputs,92,92)
            # forward + backward + optimize
            outputs_pred,outputs_softmax = model(inputs)
            loss = criterion(outputs_pred, labels.long())
    
    
            loss.backward()
            optimizer.step()
            
            #===================epoch ends===========================
            current_loss.append(loss.item())
            if i % 10 == 0 :
                print('[%d / %d, %5d / %d] loss: %6f' % (epoch+1,n_epochs,i+1,len(trainlodader),loss.item()))
            
            #=========save accuracy=====================
            pred_mask = torch.argmax(outputs_softmax,dim=1)
            current_accs.append((pred_mask == labels).float().mean().item())
            
        #=====Save stats of training process================
        train_logs.append([epoch+1,np.array(current_loss).mean(),np.array(current_loss).std(),np.array(current_accs).mean(),np.array(current_accs).std()])
        
        if validation:
            loss_v_tmp = []
            acc_v_tmp = []
            model.eval()#model does not update weights
            for i, data in enumerate(valloader,0):
                 with torch.no_grad():
                        inputs, labels = data[0].to(device), data[1].to(device)
                        if unet_train:
                            inputs = mirroring(inputs,92,92)
                        outputs_pred,outputs_softmax = model(inputs)
                        pred_mask = torch.argmax(outputs_softmax,dim=1)
                        acc_v_tmp.append((pred_mask == labels).float().mean().item())
                        loss = criterion(outputs_pred, labels.long())
                        loss_v_tmp.append(loss.cpu().numpy())
            print("Accuracy: "+str(np.array(acc_v_tmp).mean()))
            valid_logs.append([np.array(loss_v_tmp).mean(),np.array(loss_v_tmp).std(),np.array(acc_v_tmp).mean(),np.array(acc_v_tmp).std()])
    
    print("Training finished!!")
    return model,train_logs,valid_logs