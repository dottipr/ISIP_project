# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage import filters
import os
import glob
import time
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *

def findPupilCenter (img):
    #++++++++Detect Circle Center++++++++++++++
    # -- Aplying only simple thresholding -- 
    img = cv2.medianBlur(img,45)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #cv2.imshow('Original Image',img)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equal = clahe.apply(img)
    #cv2.imshow('Equalized Histogramm', equal)
    #cv2.imshow('feature detection', cimg)
    circle = cv2.HoughCircles(equal,cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=75,maxRadius=150)
    circle = np.uint16(np.around(circle))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #cv2.imshow('detected circles',cimg) 
    
    
#    #ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,9)
#    #ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    cv2.imshow('Thresholding', thresh)
    
#    kernel = np.ones((3,3),np.uint8)
#    er = cv2.erode(thresh,kernel,iterations=6)
#    dil = cv2.dilate(er,kernel,iterations=6)
#    cv2.imshow('Dilation', dil)
    
    
    # Calculate center of circle:
#    thresh_inv = np.zeros(thresh.shape, dtype = 'uint8')
#    thresh_inv[dil ==0] = 255
#    dist_transform = cv2.distanceTransform(thresh_inv,cv2.DIST_L2,5)
#    ind_center = np.unravel_index(np.argmax(dist_transform, axis=None), dist_transform.shape)
#    cv2.imshow('Distance Transform', np.uint8(dist_transform))
#    
#    sobel = cv2.Sobel(thresh,cv2.CV_8U,1,0,ksize=5)
#    ind_circle = np.unravel_index(np.argmax(sobel, axis=None), sobel.shape)
#    radius = np.int(np.ceil(np.sqrt(np.square(ind_center[1]-ind_circle[1])+np.square(ind_center[0]-ind_circle[0]))))
#    
#    cv2.circle(cimg,(ind_center[1],ind_center[0]),2,(0,0,255),3)
#    cv2.circle(cimg,(ind_center[1],ind_center[0]),radius,(0,255,0),3)
#    cv2.imshow('Center marked', cimg)
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return cimg, circle 

def findInstr(img, center_circle):
    
    x_center = center_circle[0][0][0]
    y_center = center_circle[0][0][1]
    radius = center_circle[0][0][2]
    
    mask = np.zeros(img.shape, dtype = 'uint8')
    cv2.circle(mask,(x_center,y_center),radius,255,-1)
    cv2.circle(mask,(x_center,y_center),np.uint16(np.ceil(0.75*radius)),0,-1)
    img[mask == 0] = 0
    
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.imshow('Mask',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return 0



# ++++++++++Search for images in selected folder+++++++++++++++

data  = Tk()
data.filepath = filedialog.askdirectory(initialdir = "/",title = "Select Directory to work on")

pathlist = []
filelist = []

for root, dirs, files in os.walk(data.filepath):
    for file in files:
        if file.endswith(".png"):
            filelist.append(file)
            pathlist.append(root)
             
data.destroy()             

imagelist = []
circles_vector = []
# +++++++++++++ Set save path ++++++++++
#data  = Tk()
#data.filepath = filedialog.askdirectory(initialdir = "/",title = "Select Directory to save results")
#savepath = data.filepath
#data.destroy()

# +++++++++++++++ Main +++++++++++++++++++
i = 0

for i in range(0,len(filelist)):
    if len(filelist) == len(pathlist):
        filepath = os.path.normpath(os.path.join(pathlist[i],filelist[i]))
    else:
        print('Filelist and Pathlist have not equal size')
    center = (0,0)
    img = cv2.imread(filepath,0)
    img_fill = cv2.medianBlur(img,51)
    #sobel = cv2.Sobel(img_fill,cv2.CV_8U,1,0,ksize=9)
#    laplacian = cv2.Laplacian(img_fill,cv2.CV_64F, ksize =17)
#    
#    cv2.imshow('original image', img)
#    cv2.imshow('filtered image', img_fill)
#    cv2.imshow('laplacian_filtered', laplacian)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
    cimg, circle = findPupilCenter(img)
    imagelist.append(cimg)
    if circle.shape == (1,1,3):
        circles_vector.append(circle)
    else:
        circles_vector.append([0,0,100])
    zero = findInstr(img,circles_vector[i])
    
    # ++++ Save annotated images +++++
#    save_img_path = os.path.normpath(os.path.join(savepath,filelist[i]))
#    print(save_img_path)
#    cv2.imwrite(save_img_path, cimg)