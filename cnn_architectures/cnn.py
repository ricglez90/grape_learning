#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:25:55 2021

@author: M.R. Gonzalez
"""

#------------General packages--------------------------
import torch
import torch.nn as nn
import torchvision.models as models
#----------Utils----------------------------------------
from aids_cnn import double_conv, crop_img #UNet's aids functions
from aids_cnn import Conv_down, Conv_up #Segnet's aids functions
from aids_cnn import double_dsconvolution #UNet_ds' aids functions
from utils import IntermediateLayerGetter
from _deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
import mobilenetv2

#========================UNet===========================================
# Original implementation: https://www.youtube.com/watch?v=u1loyDCoGbE
class UNet(nn.Module):
    def __init__(self, input_channels=1, n_classes=2, pretrained=False, debug = False):
        super(UNet, self).__init__()
        
        ecs = [input_channels,64,128,256,512,1024]
        dcs = ecs[::-1][:-1] + [n_classes]
        self.tensors_path = []
        self.debug = debug
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(5):
            self.encoder.append(double_conv(ecs[i],ecs[i+1]))
            if i < 4:
                self.decoder.append(nn.ConvTranspose2d(in_channels=dcs[i],out_channels=dcs[i+1], kernel_size=2, stride=2))
                self.decoder.append(double_conv(dcs[i],dcs[i+1]))
   
        self.out = nn.Conv2d(in_channels=dcs[-2] , out_channels=dcs[-1], kernel_size=1) #out_channels = no. classes
        
        if pretrained:
            print("Transfer from VGG16 model")
            self.transfer_vgg16()
            
        
    def forward(self, image):
        # (batchSize,channels,rows,cols)
        #encoder part
        skips = []
        x = image
        for i in range(5):
            x = self.encoder[i](x)
            if self.debug:
                self.tensors_path.append(x)
            if i < 4:
                skips.append(x)
                x = self.max_pool_2x2(x)
        
        #decoder part
        for i,j,k in zip(reversed(range(4)), range(0,7,2), range(1,8,2)):#i for skips, j for convtransposed and k for double_conv
            x = self.decoder[j](x)
            y = crop_img(skips[i],x)
#            print(x.shape,skips[i].shape,y.shape)
            x = self.decoder[k](torch.cat([x,y],1))            
            if self.debug:
                self.tensors_path.append(x)
        
        x = self.out(x)
        if self.debug:
            self.tensors_path.append(x)

        return x , F.softmax(x, dim=1)
                    
    def transfer_vgg16(self):
        vgg_indx = [0,2,5,7,10,12,17,19]
        vgg16 = models.vgg16(pretrained = True)
        for i,j,k in zip([0,0,1,1,2,2,3,3],[0,2,0,2,0,2,0,2],vgg_indx):# i double conv, j single conv, k vgg16 layer 
            assert self.encoder[i][j].weight.size() == vgg16.features[k].weight.size()
            self.encoder[i][j].weight.data = vgg16.features[k].weight.data
            assert self.encoder[i][j].bias.size() == vgg16.features[k].bias.size()
            self.encoder[i][j].bias.data = vgg16.features[k].bias.data
            
            
#==========================SegNet=======================================
# Original implementation: https://github.com/say4n/pytorch-segnet
F = nn.functional
class SegNet(nn.Module):
    def __init__(self,input_channels=1, n_classes=2, pretrained=False, debug = False):
        super(SegNet, self).__init__()
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        i_channels = [input_channels,64,128,256,512,512]
        o_channels = [512,512, 256, 128, 64,64]
        self.feat_index = [0,2,5,7,10,12,14,17,19,21,24,26,28]#indices of vgg16 for the original implementation
        n_encoder_neurons = [2,2,3,3,3]
        n_decoder_neurons = [3,3,3,2,1]
#        self.indices = []
#        self.dimensions = []
        
        for i in range(5):
            self.encoder.append(Conv_down(i_channels[i],i_channels[i+1],k=3,p=1,s=1,n_layers=n_encoder_neurons[i]))
            self.decoder.append(Conv_up(o_channels[i],o_channels[i+1],k=3,p=1,s=1,n_layers=n_decoder_neurons[i]))
            
        self.last_conv = nn.Conv2d(o_channels[-1],n_classes,kernel_size=3,stride=1,padding=1)
        
        if pretrained:
            self.transfer_procedure()
    
            
    def forward(self,input_image):
        dimensions = []
        indices = []
        dimensions.append(input_image.size())
        x = input_image
        for i in range(5):
            x = self.encoder[i](x)
            x,  ind = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
            dimensions.append(x.size())
            indices.append(ind)
        
        dimensions.pop()
        for i in reversed(range(5)):
            x = F.max_unpool2d(x, indices[i], kernel_size=2, stride=2, output_size=dimensions[i])
            x = self.decoder[4-i](x)
        
        x = self.last_conv(x)
        return x , F.softmax(x, dim=1)
    
    def transfer_procedure(self):
        k = 0
        vgg16 = models.vgg16(pretrained=True)
        for i in range(5):
            for name,p in self.encoder[i].named_parameters():
                if "0" in name or "3" in name or "6" in name: # convolutional neurons indexes
                    j = self.feat_index[k]
                    if "weight" in name:
                        assert p.size() == vgg16.features[j].weight.size()
                        p.data = vgg16.features[j].weight.data
                    elif "bias" in name:
                        assert p.size() == vgg16.features[j].bias.size()
                        p.data = vgg16.features[j].bias.data                    
                        k+=1
        
#===========================Zabnet (MobileNetV2 + DeepLabV3+)=====================
# Implementación original: https://github.com/VainF/DeepLabV3Plus-Pytorch
def _segm_vgg16(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = models.vgg16(pretrained = pretrained_backbone)
    # rename layers
    backbone.low_level_features = backbone.features[0:5]
    backbone.high_level_features = backbone.features[5:-1]
    backbone.features = None
    backbone.classifier = None
    backbone.avgpool = None

    inplanes = 512
    low_level_planes = 64

    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
        
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def deeplabv3plus_vgg16(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'vgg16', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone) 

def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)   

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone=='vgg16':
        model = _segm_vgg16(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model
#===============================================================================
class DSUNet(nn.Module):
    def __init__(self, input_channels=1, n_classes=2, pretrained=False, pretrain_model=None):
        super(DSUNet, self).__init__()
        
        ecs = [input_channels,64,128,256,512,1024]
        dcs = ecs[::-1][:-1] + [n_classes]

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(5):
            self.encoder.append(double_dsconvolution(ecs[i],ecs[i+1]))
            if i < 4:
                self.decoder.append(nn.ConvTranspose2d(in_channels=dcs[i],out_channels=dcs[i+1], kernel_size=2, stride=2))
                self.decoder.append(double_dsconvolution(dcs[i],dcs[i+1]))
   
        self.out = nn.Conv2d(in_channels=dcs[-2] , out_channels=dcs[-1], kernel_size=1) #out_channels = al no de clases
        
        if pretrained:
            if pretrain_model is not None:
                self.unet_pretrained = pretrain_model
                self.transfer_procedure()
            else:
                print("Missing model")
        
    def forward(self, image):
        # (tamañoDelBatch,channles,altura,anchura)
        #encoder part
        skips = []
        x = image
        for i in range(5):
            x = self.encoder[i](x)
            if i < 4:
                skips.append(x)
                x = self.max_pool_2x2(x)
        
        #decoder part
        for i,j,k in zip(reversed(range(4)), range(0,7,2), range(1,8,2)):#i for skips, j for convtransposed and k for double_conv
            x = self.decoder[j](x)
            y = crop_img(skips[i],x)
            x = self.decoder[k](torch.cat([x,y],1))
        
        return x , F.softmax(x, dim=1)

        return x
    
    def transfer_procedure(self):
        indexes = [0,1,3,4]
        for i in range(5):
            for j in indexes:
                assert self.encoder[i][j].weight.size() == self.unet_pretrained.encoder[i][j].weight.size()
                self.encoder[i][j].weight.data = self.unet_pretrained.encoder[i][j].weight.data
                assert self.encoder[i][j].bias.size() == self.unet_pretrained.encoder[i][j].bias.size()
                self.encoder[i][j].bias.data = self.unet_pretrained.encoder[i][j].bias.data

