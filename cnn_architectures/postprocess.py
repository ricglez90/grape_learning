#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:57:25 2021

@author: cacie
"""

import numpy as np
import cv2
from skimage.measure import label as LABEL
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from scipy import stats


class BerryArc():
    def __init__(self,pixChain, sample = 10, dims = (580,580)):
        pixChain = np.vstack(pixChain)[[np.arange(0,pixChain.shape[0],10)],:][0]
        pixChain = np.insert(pixChain,pixChain.shape[0],pixChain[:2], axis=0)
        self.pixChain = pixChain
        self.dims = dims
        self.arcs = []
        self.circles = []
        self.arcs_candidates = []
        
    def compute_arcs(self):
        signos = []
        old_direction = None
        j = 0 #arc number
        for i in range(0,self.pixChain.shape[0]-2,1):
            v1 = [self.pixChain[i+1,0] -self.pixChain[i,0], self.pixChain[i+1,1] - self.pixChain[i,1]]
            v2 = [self.pixChain[i+2,0] -self.pixChain[i+1,0], self.pixChain[i+2,1] - self.pixChain[i+1,1]]
            v1len = np.sqrt(v1[0]**2 + v1[1]**2)
            v2len = np.sqrt(v2[0]**2 + v2[1]**2)
            angle = np.rad2deg(np.arccos((v1[0]*v2[0] + v1[1]*v2[1])/(v1len*v2len)))
            direction = 1 if (v1[0]*v2[1] - v2[0]*v1[1])>=0 else -1
            if i > 0:
                if old_direction != direction:
                    j+=1
                else:
                    j=j
                        
            if angle <= 60 and angle >=6:
                signos.append([angle,self.pixChain[i],self.pixChain[i+1],self.pixChain[i+2],j])            
            old_direction = direction
            
        self.arcs = np.array(signos)
        
    def check_valid_arcs(self,circs):
        # binary mask of the polygon
        A1 = np.zeros(self.dims)
        cv2.fillPoly(A1,[self.pixChain.reshape((-1,1,2))],(255,255,255)) 
        # binary mask circle forme by the arc
        A2 = np.zeros(self.dims)
        cv2.circle(A2,(np.int0(circs[0][0]),np.int0(circs[0][1])),np.int0(circs[1]),(255,255,255),-1)
        #Area in pixel 
        a2 = np.count_nonzero(A2)
        # intersection compute
        intersec = cv2.bitwise_and(A1,A2)
        # union compute
        union = cv2.bitwise_or(A1,A2)
        contain = 1 if np.count_nonzero(union) < a2 else 0
        
        return np.count_nonzero(intersec) / a2, contain

    
    def least_squares_fit(self,points):
        x , y = points[:,0] , points[:,1]
        
        u , v = x - x.mean() , y - y.mean()
        Suu = (u**2).sum()
        Suv = (u*v).sum()
        Svv = (v**2).sum()
        Suuu = (u**3).sum()
        Svvv= (v**3).sum()
        Suvv = (u*(v**2)).sum()
        Svuu = (v*(u**2)).sum()
        
        A = np.array([[Suu, Suv],[Suv,Svv]])
        B = np.array([[Suuu + Suvv],[Svvv + Svuu]])/2
        
        uc, vc = np.linalg.solve(A,B)
        
        xc_1 = x.mean() + uc
        yc_1 = y.mean() + vc
        
        
        Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
        R_1      = np.mean(Ri_1)
        residu_1 = sum((Ri_1-R_1)**2)
        
        candidate = [(xc_1[0],yc_1[0]),R_1]
        A_hat = self.check_valid_arcs(candidate)
        return [(xc_1[0],yc_1[0]),R_1, residu_1,A_hat]
    
    def fit_circles(self):#https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
        try:
            arc_ind = np.unique(self.arcs[:,4])
            
            for i in arc_ind:
                points_tmp = self.arcs[:,[1,2,3]][self.arcs[:,4]==i]
                if points_tmp.shape[0]>=3:
                    points_tmp = np.unique(np.stack(np.hstack(points_tmp)), axis=0)
                    self.arcs_candidates.append(np.array(points_tmp))
                    self.circles.append(self.least_squares_fit(np.array(points_tmp)))
                
        except:
            self.arcs_candidates.append(np.array([np.nan,np.nan]))
            self.circles.append([(np.nan,np.nan),np.nan,[np.nan,np.nan]])
            
    def fit_circles_2(self):
        try:
            self.circles.append(self.least_squares_fit(self.pixChain))
        except:
            self.circles.append([(np.nan,np.nan),np.nan,np.nan])
            
# ------------------------------------------------------------

class GrapeBunch():
    def __init__(self,img_prediction):
        self.img_mask = np.array(img_prediction,np.uint8)
        
        #Size ratios
        self.As = 0
        self.Csf = 0
        self.Rd = 0
        
        #Bunches Lines
        self.L = []
        self.Ms = []
        
        #----------------Grapes---------------------
        self.circles = []
        self.arc_amount = 0
        self.grape_objects_amount = 0
        self.areas = []
        self.mu = 0
        self.sd = 0
        
    def compute_bunch_ratios(self):
        img_bunch = self.img_mask.copy()
        #we make all bunch and close hole
        img_bunch[img_bunch==2] = 1
        kernel = np.ones((20,20))
        closed = cv2.morphologyEx(img_bunch,cv2.MORPH_CLOSE,kernel)
        
        #Compute contours and sort to find L (vertical line from the bunch)
        contours, hierarchy = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2:]
        contours = np.vstack(contours[0])
        ind = np.lexsort((contours[:,0],contours[:,1])) #no jala
        contours = contours[ind]
        L = contours[[0,-1],:]
        m, b, r, p, se = stats.linregress(L[:,0], L[:,1])
        L_length = np.linalg.norm(L[1] - L[0],2)
        
        L_start = np.min([L[0,0],L[1,0]]) # inicio del dominio
        L_end = np.max([L[0,0],L[1,0]]) # fin del dominio 
        L_step = np.int0((L_end - L_start)/4) # pasos
        
        #Computing Mw (the largest axis from the bunche)
        Mw = []
        for p in range(L_start,L_end,L_step):
            if p >L_start and p<L_end-L_step-1:
                x1 , y1 = L[0,:]
                y3 = m*(p)+b
                #Compute perpendicular lines
                m_p = -1/m 
                b_p = y3-y1

                x_n = np.arange(0,self.img_mask.shape[0])
                y_n = m_p*x_n+b_p
                
                Per_Line = np.array([x_n,y_n]).T
                inx = []
                #checking intercepting points
                for j in range(Per_Line.shape[0]):
                    for k in range(contours.shape[0]):
                        if (np.int0(Per_Line[j]) == contours[k]).all():
                            inx.append(k)
                            
                
                #Saving bunch lines
                Mw_new = np.linalg.norm(contours[inx[1]]-contours[inx[0]],2)
#                 Mw = Mw_new if Mw_new >= Mw else Mw
                Mw.append(Mw_new)
                self.Ms.append([contours[inx],Mw_new])
        
        self.Ms = np.array(self.Ms)
        self.L = L
        Mw =  np.array(Mw).mean()
        #Computing indices
        P = contours.shape[0] #bunch perimeter
        A = np.count_nonzero(img_bunch) #bunch area
        print(P**2,A, P**2/float(A))
        #Size ratio
        self.As = Mw/float(L_length)
        
        #Area/perimeter ratio
#         self.Csf = P**2/float(A)#Artículo
        self.Csf = A/float(P**2)#referencia original
        
        #Circular ratio
        self.Rd = (4*np.pi*A)/float(P**2)
        
    def compute_grapes_size(self):
        a = self.img_mask.copy()
        a[a==2] = 0
        contours, hierarchy = cv2.findContours(a,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2:]
        circles = []
        for cont in contours:
            if cont.shape[0] > 10:
                sample = np.int0(cont.shape[0] * 0.05)
                arc = BerryArc(cont,sample, dims = a.shape)
                arc.compute_arcs()
                arc.fit_circles()
                if len(arc.circles)>0:
                    circles.append(arc.circles)
                    
        self.arc_amount = sum(len(c) for c in circles)
        self.grape_objects_amount = len(contours)
        self.circles = circles
        
        #computing areas
        areas = []
        for c in circles:
            for d in c:
                if d[-1][0] > 0.15 and d[-1][1] == 0:
                    areas.append(np.pi * d[1]**2)
            
        areas = np.array(areas)
        self.areas = areas[~np.isnan(areas)]
        self.mu = self.areas.mean()
        self.sd = self.areas.std()


#a = np.asarray(Image.open("/media/cacie/Data/data_ricardo/CosmeDS/CosmeDS_revision/CosmeDS_clean/CosmeDS_completeV2/s2/DSC_0046-3.png")).copy()
#a[a==2] = 0
#I = cv2.imread("/media/cacie/Data/data_ricardo/CosmeDS/CosmeDS_revision/CosmeDS_clean/CosmeDS_completeV2/originales/DSC_0046-3.png")
#contours, hierarchy = cv2.findContours(a,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2:]
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.imshow(a,cmap="bone")
#ax.imshow(cv2.cvtColor(I,cv2.COLOR_BGR2RGB))
#cont= contours[13]
#sample_rate = 0.05
#for cont in contours:
#    if cont.shape[0] > 10:
#        sample = np.int0(cont.shape[0] * sample_rate)
#        arc = BerryArc(cont,sample, dims = a.shape)
#        arc.compute_arcs()
#        arc.fit_circles()
        
#        for ar in arc.arcs_candidates:
#            ax.scatter(ar[:,0],ar[:,1])
            
#        for cir in arc.circles:
#            if cir[-1] > 0.25:
#                ax.scatter(cir[0][0],cir[0][1], marker = "x",color="g")
#                circ = Circle(cir[0],cir[1],fill=None,color="g",lw=2)
#                ax.add_patch(circ)
#        else:
#            ax.scatter(cir[0][0],cir[0][1], marker = "x",color="r")
#            circ = Circle(cir[0],cir[1],fill=None,color="r",lw=2,ls="--")
#            ax.add_patch(circ)
        
    

#A1,A2,A3,A_hat = arc.check_valid_arcs()      
#ax = fig.add_subplot(142)
#ax.imshow(A1, cmap="bone")
#      
#ax = fig.add_subplot(143)
#ax.imshow(A2, cmap="bone")
#      
#ax = fig.add_subplot(144)
#ax.imshow(A3, cmap="bone")
