# -*- coding: utf-8 -*-
"""
Created on Thu May 24 08:48:32 2018

@author: Michael Stiefel and Prisca Dotti 

**** This Script should contain all our functions used in the final version 
of the Tracking allgorithm. Please try to keep it as clean as possible...:-)
Please don't delete any function just add functions we will clean up at the end.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import filters
from skimage import exposure
from skimage import data, color
from skimage.draw import circle_perimeter
from skimage.filters import threshold_otsu
import scipy

def findPupilCenter (img):
    #++++++++Detect Circle Center++++++++++++++
    # Aplying heavy image blurring in order to vanish all disturbing features 
    img = cv2.medianBlur(img,45)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    #cv2.imshow('Original Image',img)
    
    # Equalizing the histogramm in order to imporve the performance of the hough transform
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equal = clahe.apply(img)
    #cv2.imshow('Equalized Histogramm', equal)
    #cv2.imshow('feature detection', cimg)
    
    # Applying Hough Circle Transform to identify the pupil center 
    circle = cv2.HoughCircles(equal,cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=75,maxRadius=150)
    circle = np.uint16(np.around(circle))
    for i in circle[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    #cv2.imshow('detected circles',cimg) 
        
    # Press 0 to close all windows (only needed for testing)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return cimg, circle
    
def findPupilCenter2 (img):
    # Function to find circle center only with skimage functions
    # Works only with grayscale images 

    # Load picture
    #image_raw = io.imread(filepath, as_grey= True)
    image_raw = color.rgb2gray(image_color)
    image =np.uint8(255*(image_raw/image_raw.max()))
    plt.figure()
    plt.imshow(image, cmap = plt.cm.gray)
    image = filters.gaussian(image, sigma=31)
    plt.figure()
    plt.title('Gaussian Filtered Image')
    plt.imshow(image, cmap = plt.cm.gray)
    #image = exposure.equalize_hist(image)
    low, high = np.percentile(image, (50, 90))
    image = exposure.rescale_intensity(image, in_range=(low, high))
    plt.figure()
    plt.title('Rescaled intensity')
    plt.imshow(image, cmap = plt.cm.gray)
        
    thresh = threshold_otsu(image, nbins=256)
    image_threshold = image > thresh
    plt.figure()
    plt.title('Thresholded Image')
    plt.imshow(image_threshold, cmap =plt.cm.gray)
    
    dist = scipy.ndimage.morphology.distance_transform_edt(image_threshold, return_distances=True, return_indices=False)
    plt.figure()
    plt.title('Distance Transform')
    plt.imshow(dist, cmap =plt.cm.gray)
    
    indices = np.where(dist == dist.max())
    
    circy, circx = circle_perimeter(indices[0][0], indices[1][0], 10)
    image_raw[circy, circx] = 0
    plt.figure()
    plt.title('Marked Image')
    plt.imshow(image_raw, cmap =plt.cm.gray)
    
    return image_raw, indices[0][0], indices[1][0]

def cutpatch (img, x_center, y_center,width,height):
    # width, height should be odd numbers 
    # returns 2D-Array if Gray image or RGB-patch if the input was RGB-Image
    cut_img = img
    x_top_left = x_center - (width-1)/2
    y_top_left = y_center - (height-1)/2
    if len(cut_img.shape) == 2: 
        patch = cut_img[y_top_left:y_top_left+height,x_top_left:x_top_left+width]
    elif len(cut_img.shape) ==3:
        patch = cut_img[y_top_left:y_top_left+height,x_top_left:x_top_left+width,:]
    else:
        patch = np.zeros((width,height), dtype = 'uint8')
    
    return patch