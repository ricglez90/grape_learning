#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:42:13 2021

@author: cacie
"""

import cv2
import color_index as ci
import numpy as np
import pandas as pd


class AgroImage():
    def __init__(self,A):
        self.I = A.copy()
        self.CS = []
        self.CI = []
        self.names = ['NDI','ExG','ExR','CIVE','ExGR','NGRDI','VEG','MExG','COM1','COM2']
        self.spaces = ['HSV','LAB']
        self.descriptions = pd.read_csv("CI.csv")
        
    def color_index(self,option):
        self.CI = []
        functions = [ci.NDI_compute,ci.ExG_compute,ci.ExR_compute,
                     ci.CIVE_compute,ci.ExGR_compute,ci.NGRDI_compute,
                     ci.VEG_compute,ci.MExG_compute,ci.COM1_compute,
                     ci.COM2_compute]
        
        i = self.names.index(option)
        self.CI = functions[i](self.I)
        
    def concepts(self,option):
        i = self.names.index(option)
        return self.descriptions.iloc[i]
                
    def color_sapces(self,option):
        i = self.spaces.index(option)
        functions = [cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB]
        return cv2.split(cv2.cvtColor(self.I,functions[i]))
        
            
        