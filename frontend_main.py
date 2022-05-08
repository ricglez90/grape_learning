#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:08:49 2022

@author: ricardo
"""
import streamlit as st
import numpy as np
#import time
#import cv2
from PIL import Image
from aids_web import predict, shallow_predict
#import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#from matplotlib import cm
from matplotlib.colors import ListedColormap#, LinearSegmentedColormap


st.title("Grape learning app")
st.write("This app aims to count and locate single berries within a digital image.")

st.subheader("Select the input images")

predicted_img = None
file_model = None


st.subheader("Load the input image")
file_rbg = st.file_uploader("Image")


learn_type = st.sidebar.selectbox("Learning type", ["Shallow", "Deep"])