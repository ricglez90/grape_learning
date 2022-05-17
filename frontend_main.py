#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:08:49 2022

@author: ricardo
"""
import streamlit as st
import numpy as np
#import time
import cv2
from PIL import Image
import pickle
from aids_web import predict, shallow_predict
#import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from cnn import UNet
from matplotlib.patches import Rectangle, Circle
#from matplotlib import cm
from matplotlib.colors import ListedColormap#, LinearSegmentedColormap


# *************************Globals**************

#loading models
model_shallow_path = "models/sk_pickle_supervised_model.sav"
model_shallow = pickle.load(open(model_shallow_path, 'rb'))

model_weigth_path = "models/weigth_prediction.sav"
model_linear_w= pickle.load(open(model_weigth_path, 'rb'))

model_deep1_path = "models/pythorch_deep_model.pth"
device = torch.device('cpu')
model_deep = UNet(input_channels= 3, n_classes=3)
model_deep.load_state_dict(torch.load(model_deep1_path, map_location=device))

lab_plate = cv2.imread('images/lab_plate.png')

n_classes = 2

option = None

berry_circles = []
b_selected = None

pred4weigth = []

weigth = None
# *********************************************


st.title("Grape learning app")
st.write("This app aims to aids the viticulturist in the vineyard analysis.")

# st.subheader("Select the input images")

predicted_img = None
file_model = None
image1 = None



options_list = ["Bunch location", "Individual bunch","Weight prediction","Extra"]

main_opts = st.sidebar.selectbox("Image type", options_list)

if main_opts !=  options_list[2] and main_opts !=  options_list[3]:
    st.subheader("Load the input image")
    file_rbg = st.file_uploader("Image")

#Bunch location
if main_opts == options_list[0]:
    option = 0
#     pred_method = st.sidebar.radio("Method",["Millan"])
    if file_rbg is not None:
        image1 = Image.open(file_rbg).convert("RGB")
        r,g,b = image1.split()
        I_np = np.array(Image.merge("RGB", (r,g,b)))
        predicted_img, boxes = shallow_predict(I_np,model_shallow)
        
        


#Individual bunch analysis and grape count prediction
if main_opts == options_list[1]:
    option = 1
#     pred_method = st.sidebar.radio("Method",["Millan"])
    if file_rbg is not None:
        image1 = Image.open(file_rbg).convert("RGB")
        image2= Image.open(file_rbg)
        r,g,b = image1.split()
        I_np = np.array(Image.merge("RGB", (r,g,b)))
        
        
        options_analysis = ["Fast", "Integral", "Per berry"]
        stats_options = st.sidebar.selectbox("Analysis", options_analysis)
        
        predicted_img, boxes = predict(image2,model_deep,stats_options)
        
        
#Weith prediction
if main_opts == options_list[2]:
    option = 1
    uploaded_file = st.file_uploader("Please select 4 images and follow the protocol to predict the weight:", accept_multiple_files=True)
    if uploaded_file is not None:
#         st.write(uploaded_file)
#         st.write(len(uploaded_file)/3)
        assert len(uploaded_file) == 4
        areas_ = []
        for i in range(len(uploaded_file)):
            image1 = Image.open(uploaded_file[i]).convert("RGB")
            image2= Image.open(uploaded_file[i])
            r,g,b = image1.split()
            I_np = np.array(Image.merge("RGB", (r,g,b)))
            stats_options = "Per berry"
            predicted_img, boxes = predict(image2,model_deep,stats_options)
            pred4weigth.append([I_np,predicted_img, boxes])
            areas_.append(np.count_nonzero(predicted_img))
        
        X_w = np.array([ [c[2].grape_objects_amount for c in pred4weigth] + [a for a in areas_]])
        
        weigth  = model_linear_w.predict(X_w)[0]
#         st.write(weigth)
            
            
if main_opts == options_list[3]:
    option = 3
    st.subheader("Images samples:")
    st.markdown("[Repository link](https://duckduckgo.com)")
    st.subheader("Documentation:")
    st.markdown("[Guide tutorial](https://duckduckgo.com)")

        
        
#Display results
if main_opts !=  options_list[3]:
    st.subheader("Results")

#Bunch location
if predicted_img is not None and  main_opts == options_list[0]:
    left_col2, right_col2 = st.columns(2)
    with left_col2:
#         st.sidebar.subheader("Display options")
        color_select = []
     
    
        I_np = np.array(Image.merge("RGB", (r,g,b)))
        fig2 = plt.figure()

        ax2 = fig2.add_subplot(111)
        ax2.imshow(I_np)
        ax2.axis("off")
            
        st.write("Original image")
        st.pyplot(fig2)
    
    with right_col2:
        
        colors = [[0/255.0,0/255.0,0/255.0],[255/255.0, 0/255.0, 0/255.0]]
        cmap1 = ListedColormap(colors)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.imshow(I_np)
        for box in boxes:
            rect = Rectangle((box[0],box[1]),box[2],box[3],fill = None, ec='lime',lw = 3)
            rect2 = Rectangle((box[0],box[1]),box[2]*0.5,box[3]*0.15,ec='lime',fc='lime')
            ax2.add_patch(rect)
            ax2.add_patch(rect2)
            ax2.text(box[0],box[1]+box[3]*0.05, "Width: "+str(round(box[2],2))+" px.")
            ax2.text(box[0],box[1]+box[3]*0.1, "Heigth: "+str(round(box[3],2))+" px.")
        ax2.axis("off")
            
        st.write(options_list[option])
        st.pyplot(fig2)
        
    st.write("Here we include with bunch dimensions in pixels")
    
    


#Berry analysis
if predicted_img is not None and  main_opts == options_list[1]:
    if stats_options == options_analysis[0]:
        left_col2, right_col2 = st.columns(2)
        with left_col2:
    #         st.sidebar.subheader("Display options")
            color_select = []


            I_np = np.array(Image.merge("RGB", (r,g,b)))
            fig2 = plt.figure()

            ax2 = fig2.add_subplot(111)
            ax2.imshow(I_np)
            ax2.axis("off")

            st.write("Original image")
            st.pyplot(fig2)

        with right_col2:
            
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.imshow(I_np)
            for box in boxes:
                rect = Rectangle((box[0],box[1]),box[2],box[3],fill = None, ec='lime',lw = 3)
                rect2 = Rectangle((box[0],box[1]),box[2]*0.5,box[3]*0.15,ec='lime',fc='lime')
                ax2.add_patch(rect)
                ax2.add_patch(rect2)
                ax2.text(box[0],box[1]+box[3]*0.05, "Width: "+str(round(box[2],2))+" px.")
                ax2.text(box[0],box[1]+box[3]*0.1, "Heigth: "+str(round(box[3],2))+" px.")

            ax2.axis("off")

            st.write(main_opts+": \n "+stats_options + " analysis")
            st.pyplot(fig2)

        st.write("Here we include with bunch dimensions in pixels")

    else:
        left_col2, right_col2 = st.columns(2)
        with left_col2:
    #         st.sidebar.subheader("Display options")
            color_select = []


            I_np = np.array(Image.merge("RGB", (r,g,b)))
            fig2 = plt.figure()

            ax2 = fig2.add_subplot(111)
            ax2.imshow(I_np)
            ax2.axis("off")

            st.write("Original image")
            st.pyplot(fig2)

        with right_col2:
            
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.imshow(I_np)
#             st.write(len(boxes.circles))
            for b in range(len(boxes.circles)):
                box = boxes.circles[b]
                
                color = 'red' if b == b_selected else 'green'
           
                for c in box:
                    if ~np.isnan(c[1]):
                        berry_circles.append([b,c[0],c[1]])
                        circ = Circle(c[0],c[1],fill=None,color=color)
                        ax2.add_patch(circ)
                    
            ax2.axis("off")

            st.write(main_opts+": \n "+stats_options+ " analysis")
            st.pyplot(fig2)
            st.write("Count: "+str(boxes.grape_objects_amount)+ " berries.")
        
        if stats_options == options_analysis[1]:
            left_col2, right_col2 = st.columns(2)
            with left_col2:
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                ax2.hist(boxes.areas, bins = 'auto')
                ax2.set_title(r"Sizes $\mu+\sigma$: "+str(round(boxes.mu,2))+"+"+str(round(boxes.sd,2)))
                ax2.set_xlabel("Size in pixels")
                ax2.set_ylabel("Frequency")

                st.write("Size distribution")
                st.pyplot(fig2)

            with right_col2:
                HSV = cv2.cvtColor(I_np, cv2.COLOR_RGB2LAB)

                H,S,V = cv2.split(HSV)
 
                Class = predicted_img.reshape((predicted_img.shape[0]*predicted_img.shape[1]))

                HP = H.reshape((H.shape[0]*H.shape[1]))[Class == 1]# * 2   convervir a radianes si usas  proyeccion polar: np.deg2rad(HP)
                SP = S.reshape((S.shape[0]*S.shape[1]))[Class == 1] #- 128
                VP = V.reshape((V.shape[0]*V.shape[1]))[Class == 1]# - 128


                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                ax2.imshow(cv2.cvtColor(lab_plate,cv2.COLOR_BGR2RGB))
    #             ax2 = fig2.add_subplot(111,polar='True')
    #             ax2.pcolormesh(thetas, r, values, cmap='hsv')
                ax2.scatter(SP,VP, c='white', alpha=0.8,marker='o',  s=2**2)
                ax2.axis("off")
    #             ax2.scatter(SP-128,VP-128, c='r', alpha=0.5,marker='+')
    #             ax2.set_xlim(-127,128)
    #             ax2.set_ylim(-127,128)
    #             ax2.scatter(np.deg2rad(HP), SP, c='white', alpha=0.5,marker='+')

                st.write("Color distribution (LAB space)")
                st.pyplot(fig2)
        else:
            #boxes.circles:
            
            id_berry = st.selectbox("Select berry:", [str(i[0]+1) for i in berry_circles])
            
            id_berry_int = int(id_berry)-1
            
            b_selected = id_berry_int
            
            _,center, rad= berry_circles[id_berry_int]
#             st.write(center, rad)
            row_0, col_0 = int(center[1] - rad) ,  int(center[0] - rad)
            row_f, col_f = row_0 + int(2*rad), col_0 + int(2*rad)
            I_berry = I_np[row_0:row_f, col_0:col_f,:]
            I_b_pred = predicted_img[row_0:row_f, col_0:col_f]
#             st.write(row_0, col_0,row_f, col_f )
            HSV = cv2.cvtColor(I_berry, cv2.COLOR_RGB2LAB)

            H,S,V = cv2.split(HSV)

            Class = I_b_pred.reshape((I_b_pred.shape[0]*I_b_pred.shape[1]))

            HP = H.reshape((H.shape[0]*H.shape[1]))[Class == 1]# * 2   convervir a radianes si usas  proyeccion polar: np.deg2rad(HP)
            SP = S.reshape((S.shape[0]*S.shape[1]))[Class == 1] #- 128
            VP = V.reshape((V.shape[0]*V.shape[1]))[Class == 1]# - 128



        
            
            
            left_col2, right_col2 = st.columns(2)
            with left_col2:
                fig1 = plt.figure()
                ax2 = fig1.add_subplot(111)
    #             ax2.imshow(cv2.cvtColor(I_berry,cv2.COLOR_BGR2RGB))
                ax2.imshow(I_berry)

                ax2.axis("off")
                st.pyplot(fig1)
            with right_col2:
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                ax2.imshow(cv2.cvtColor(lab_plate,cv2.COLOR_BGR2RGB))
    #             ax2 = fig2.add_subplot(111,polar='True')
    #             ax2.pcolormesh(thetas, r, values, cmap='hsv')
                ax2.scatter(SP,VP, c='white', alpha=0.8,marker='o',  s=2**2)
                ax2.axis("off")
                st.pyplot(fig2)

         
        st.write("Here we include location per berry")
    
#Berry analysis
if predicted_img is not None and  main_opts == options_list[2]:
    plt.rcParams.update({'font.size': 8})

 
    fig2 = plt.figure()

    i = 1
    for res in pred4weigth:
        I , _, _ = res[0],res[1],res[2]
        ax2 = fig2.add_subplot(1,4,i)
        ax2.imshow(I)
        ax2.set_title("No. of Berries: "+str(res[2].grape_objects_amount))#
        ax2.axis("off")
        i+=1

    st.subheader("Original image")
    st.pyplot(fig2)
    st.subheader("Weight predicted: "+str(round(weigth,2))+" gr")
