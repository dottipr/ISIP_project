'''
Author: Prisca Dotti
Updated: 22.05.2018

script containing harris corner detection's functions
'''

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
from skimage import filters
import tools_hw2 as tools



# PARAMETERS : [parameters kind of working]
#   smoothing of the image: sigma and filter size [3, 10]
#   sharpening of the image: sharpening filter [[0,-2,0], [-2,9,-2], [0,-2,0]]
#   response for each pixel : k [0.04]
#   response matrix : patch size [8]
#   binary matrix : threshold [?]

########################### functions ###########################

def computeResponsePixel(patch):
    # input: greyscale patch (2-dimensional array)
    # output: value of R over the patch
    #        (which will be the value of R at the center of the patch)

    # apply filters to patch
    smoothed_patch =  tools.gconv(patch, 3, 10)
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
    k = 0.4

    return np.linalg.det(H)-k*np.trace(H)*np.trace(H)

def computeResponseMatrix(image):
    # input: greyscale image (2-dimensional array)
    # output: array containing response value for each pixel in the image
    #         (being defined only on valid pixels of the image w.r.t. the patch)

    n,m = np.shape(image)
    patch_half_size = 7
    patch_size = 2*patch_half_size+1
    nNew, mNew = n-2*patch_half_size, m-2*patch_half_size
    responseMatrix = np.zeros((nNew, mNew))

    # extract patch around each pixel and compute response value
    for i in range(0, nNew):
        for j in range(0, mNew):
            patch = image[i : i+patch_size, j : j+patch_size]
            responseMatrix[i,j] = computeResponsePixel(patch)

    return responseMatrix

def binaryResponse(image):
    # input: greyscale image
    # output: binary image (same size as image) where only point above a certain
    #         threshold are =1 (the rest is =0)
    #         [threshold need to be improved]

    threshold = np.amin(image)/2

    return np.where(image<threshold, 1, 0)
'''
########################### main ###########################

# load images
filenames = glob.glob(os.path.join('project_data', 'a_train_copy', '*.png'))
images = [np.asarray(Image.open(f).convert('L')) for f in filenames]
nb_images = np.shape(images)[0]

responseMatrices = [computeResponseMatrix(images[i]) for i in range(0,nb_images)]
binaryMatrices = [binaryResponse(responseMatrices[i]) for i in range(0,nb_images)]


# plot response matrices
plt.figure()
plt.subplot(2,4,1)
plt.imshow(responseMatrices[0])
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(responseMatrices[1])
plt.axis('off')
plt.subplot(2,4,3)
plt.imshow(responseMatrices[2])
plt.axis('off')
plt.subplot(2,4,4)
plt.imshow(responseMatrices[3])
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(responseMatrices[4])
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(responseMatrices[5])
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(responseMatrices[6])
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(responseMatrices[7])
plt.axis('off')
plt.show()

# plot unmasked regions
n, m = np.shape(images[0])
patch_half_size = int((n-np.shape(binaryMatrices[0])[0])/2)
unmaskedRegions = np.array([np.copy(images[i][patch_half_size : n-patch_half_size, patch_half_size : m-patch_half_size]) for i in range(0,nb_images)])
unmaskedRegions[np.where(binaryMatrices)] = 10

plt.figure()
plt.subplot(2,4,1)
plt.imshow(unmaskedRegions[0])
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(unmaskedRegions[1])
plt.axis('off')
plt.subplot(2,4,3)
plt.imshow(unmaskedRegions[2])
plt.axis('off')
plt.subplot(2,4,4)
plt.imshow(unmaskedRegions[3])
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(unmaskedRegions[4])
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(unmaskedRegions[5])
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(unmaskedRegions[6])
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(unmaskedRegions[7])
plt.axis('off')
plt.show()
'''
