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
    #image_raw = color.rgb2gray(image_color)
    #image =np.uint8(255*(image_raw/image_raw.max()))
#    plt.figure()
#    plt.imshow(image, cmap = plt.cm.gray)
    image = filters.gaussian(img, sigma=31)
#    plt.figure()
#    plt.title('Gaussian Filtered Image')
#    plt.imshow(image, cmap = plt.cm.gray)
    #image = exposure.equalize_hist(image)
    low, high = np.percentile(image, (50, 90))
    image = exposure.rescale_intensity(image, in_range=(low, high))
#    plt.figure()
#    plt.title('Rescaled intensity')
#    plt.imshow(image, cmap = plt.cm.gray)
        
    thresh = threshold_otsu(image, nbins=256)
    image_threshold = image > thresh
#    plt.figure()
#    plt.title('Thresholded Image')
#    plt.imshow(image_threshold, cmap =plt.cm.gray)
    
    dist = scipy.ndimage.morphology.distance_transform_edt(image_threshold, return_distances=True, return_indices=False)
#    plt.figure()
#    plt.title('Distance Transform')
#    plt.imshow(dist, cmap =plt.cm.gray)
    
    indices = np.where(dist == dist.max())
    
#    circy, circx = circle_perimeter(indices[0][0], indices[1][0], 10)
#    img[circy, circx] = 0
#    plt.figure()
#    plt.title('Marked Image')
#    plt.imshow(img, cmap =plt.cm.gray)
    return indices[0][0], indices[1][0]

def alignImages (imagelist):
    #Input: List of images (imagelist)
    #Output: list of aligned images (imagelist_corr)
    imagelist_corr = []
    center_vector = []
    shift_vector = []
    height = imagelist[0].shape[0]
    width =  imagelist[0].shape[1]   
    for ind, img in enumerate(imagelist):
        if ind == 0:
            y_center,x_center = findPupilCenter2(img)
            center_vector.append((y_center,x_center))
            imagelist_corr.append(img)
            shift_vector.append((0,0))
        elif ind > 0:
            y_center,x_center = findPupilCenter2(img)
            center_vector.append((y_center,x_center))
            #compute difference 
            x_diff = x_center - center_vector[0][1]
            y_diff = y_center - center_vector[0][0]
            shift_vector.append((y_diff,x_diff))
            #Shift image
            newImg = np.zeros((1001,1001), dtype='uint8')
            if x_diff >= 0 and y_diff >= 0:
                x_top_left = 501-width/2-x_diff
                y_top_left = 501-height/2-y_diff
            elif x_diff <= 0 and y_diff <= 0:
                x_top_left = 501-width/2-x_diff
                y_top_left = 501-height/2-y_diff
            elif x_diff > 0 and y_diff < 0:
                x_top_left = 501-width/2-x_diff
                y_top_left = 501-height/2-y_diff
            elif x_diff < 0 and y_diff > 0:
                x_top_left = 501-width/2+x_diff
                y_top_left = 501-height/2+y_diff
            newImg[y_top_left:y_top_left+height, x_top_left:x_top_left+width] = img
            expImg = newImg[501-height/2:501+height/2,501-width/2:501+width/2]
            imagelist_corr.append(expImg)
        
    return imagelist_corr, shift_vector
    
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

def compute_background_image(stack):
    # Input: Stack of images to compute the background image
    # Output: Background image dtype = uint8 
    avg_img_raw = np.zeros(img_corr[0].shape, dtype = 'uint16')
    for ind , img in enumerate(img_corr):
        avg_img_raw = avg_img_raw+img
    avg_img_raw =avg_img_raw/len(img_corr)
    avg_img = np.uint8(255*(avg_img_raw/avg_img_raw.max()))
    return avg_img