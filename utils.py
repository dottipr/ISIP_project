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
from scipy import signal
from scipy.signal import convolve2d


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

    image = filters.gaussian(img, sigma=31)
#    plt.figure()
#    plt.title('Gaussian Filtered Image')
#    plt.imshow(image, cmap = plt.cm.gray)
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
    # Marking the center of the eye needed for debuging only
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
    x_top_left = int(x_center - (width-1)/2)
    y_top_left = int(y_center - (height-1)/2)
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


#################### Homework 2 functions ####################


def gauss1d(sigma, filter_length=10):
    # INPUTS
    # @ sigma         : standard deviation of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    # if filter_length is even add one
    filter_length += ~filter_length % 2
    x = np.linspace(np.int(-filter_length/2),np.int(filter_length/2), filter_length)

    gauss_filter = np.exp(- (x ** 2) / (2 * (sigma ** 2)))

    gauss_filter = gauss_filter / np.sum(gauss_filter)

    return gauss_filter

def gauss2d(sigma, filter_size=10):
    # INPUTS
    # @ sigma           : standard deviation of gaussian distribution
    # @ filter_size     : integer denoting the filter size, default is 10
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    # create a 1D gaussian filter
    gauss1d_filter = gauss1d(sigma, filter_size)[np.newaxis, :]
    # convolve it with its transpose
    gauss2d_filter = convolve2d(gauss1d_filter, np.transpose(gauss1d_filter))

    return gauss2d_filter

def gconv(img, sigma, filter_size):
    # Function that filters an image with a Gaussian filter
    # INPUTS
    # @ img           : 2D image
    # @ sigma         : the standard deviation of gaussian distribution
    # @ size          : the size of the filter
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter

    filter = gauss2d(sigma, filter_size)

    return convolve2d(img, filter, mode='valid')


#################### Harris corners related functions ####################

    # PARAMETERS : [parameters kind of working]
    #   smoothing of the image: sigma and filter size [3, 10]
    #   sharpening of the image: sharpening filter [[0,-2,0], [-2,9,-2], [0,-2,0]]
    #   response for each pixel : k [0.04]
    #   response matrix : patch size [8]
    #   binary matrix : threshold [?]

def computeResponsePixel(patch, sigma=3, filter_size=10, k=0.4):
    # input: greyscale patch (2-dimensional array), sigma and size of the
    #        Gaussian filter, response value k
    # output: value of R over the patch
    #        (which will be the value of R at the center of the patch)

    # apply filters to patch
    smoothed_patch =  gconv(patch, sigma, filter_size)
    sharpening_filter = [[0,-2,0], [-2,9,-2], [0,-2,0]]
    filtered_patch = signal.convolve2d(smoothed_patch, sharpening_filter, mode='valid')

    # partial derivatives
    x_filter = [[-1, 1], [-1, 1]]
    y_filter = [[-1, -1], [1, 1]]
    I_x = signal.convolve2d(filtered_patch, x_filter, mode='valid')
    I_y = signal.convolve2d(filtered_patch, y_filter, mode='valid')

    # H matrix and response of the detector
    Sx2 = np.sum(I_x*I_x)
    Sxy = np.sum(I_x*I_y)
    Sy2 = np.sum(I_y*I_y)

    H = [[Sx2, Sxy], [Sxy, Sy2]]

    return np.linalg.det(H)-k*np.trace(H)*np.trace(H)

def computeResponseMatrix(image, patch_half_size=7):
    # input: greyscale image (2-dimensional array) of dimension NxM and patch half size
    # output: array containing response value for each pixel in the image
    #         (being defined only on valid pixels of the image w.r.t. the patch)
    #         the new array has dimension N-2*patch_half_size x M-2*patch_half_size

    n,m = np.shape(image)
    patch_size = 2*patch_half_size+1
    nNew, mNew = n-2*patch_half_size, m-2*patch_half_size
    responseMatrix = np.zeros((nNew, mNew))

    # extract patch around each pixel and compute response value
    for i in range(0, nNew):
        for j in range(0, mNew):
            patch = image[i : i+patch_size, j : j+patch_size]
            responseMatrix[i,j] = computeResponsePixel(patch)

    return responseMatrix

def binaryResponse(image, fraction=0.4):
    # input: greyscale image and fraction below of which we "keep" the points
    #        (between 0 and 1) the smaller the fraction, the more we keep from
    #        the image
    # output: binary image (same size as image) where only point above a certain
    #         threshold are =1 (the rest is =0)
    #         [threshold need to be improved]

    threshold = np.amin(image)*fraction

    return np.where(image<threshold, 1, 0)

def harrisCorners(image):
    # input: greyscale image
    # output: array containing masked region that correspond to corners
    #         the new array has dimension N-2*patch_half_size x M-2*patch_half_size
    #         (patch_half_size=7, if necessary it can be added as a parameter)

    response = responseMatrix(image)

    return binaryResponse(response)


#################### Tool detection functions ####################


def findTool(newImg, x, y, patch_half_size=17, sigma=5):
    # input: position of the tool in the previous image (x and y)
    #        and new image (newImg) where we want to find the position of the tool,
    #        patch half size and sigma value for Gaussian filtering
    # output: position x and y of the tool in the new image

    patch_size = 2*patch_half_size+1
    patch = cutpatch(newImg, x, y, patch_size, patch_size)
    response_half_size=7
    response = computeResponseMatrix(patch, response_half_size)
    mask0 = binaryResponse(response)


    #apply gaussian filter to binary mask to give priority to positions close
    #to the previous one
    gaussianFilter = gauss2d(sigma, mask0[0].size)
    mask = gaussianFilter*mask0

    #define new position of the tool
    maskedResponse = response * mask
    y_rel,x_rel = np.unravel_index(np.argmin(maskedResponse, axis=None), maskedResponse.shape)# relative to patch
    mask_half_size = int((mask[0].size-1)/2)
    x_top_left = x-mask_half_size
    y_top_left = y-mask_half_size
    x_new = x_top_left + x_rel
    y_new = y_top_left + y_rel

    '''
    plt.figure()

    plt.subplot(2,3,1)
    plt.imshow(patch)
    plt.axis("off")
    plt.title("original patch")
    plt.scatter((patch[0].size-mask[0].size)/2+x_rel,(patch[0].size-mask[0].size)/2+y_rel)

    plt.subplot(2,3,2)
    plt.imshow(response)
    plt.axis("off")
    plt.title("response matrix")

    plt.subplot(2,3,3)
    plt.imshow(mask0)
    plt.axis("off")
    plt.title("mask without gaussian")

    plt.subplot(2,3,4)
    plt.imshow(mask)
    plt.axis("off")
    plt.title("mask with gaussian")

    plt.subplot(2,3,5)
    plt.imshow(maskedResponse)
    plt.axis("off")
    plt.title("masked response with gaussian")
    plt.scatter(x_rel,y_rel)

    plt.show()
    '''
    return [x_new, y_new]
