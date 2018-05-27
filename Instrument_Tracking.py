# -*- coding: utf-8 -*-
"""
Author: Michael Stiefel and Prisca Dotti
Updated: 24.05.2018

This is the a script to do instrument tracking
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tkinter import filedialog
from tkinter import *
from skimage import io
from skimage import color
from skimage.segmentation import slic
from skimage import exposure
from skimage import restoration
from skimage import feature
from skimage import filters
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import draw
# Import of all our functions defined in the utils.py script
from utils import *


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

# +++++++++++++ Set save path ++++++++++
data  = Tk()
data.filepath = filedialog.askdirectory(initialdir = "/",title = "Select Directory to save results")
savepath = data.filepath
data.destroy()

# +++++ This function imports all images that are in the selected folder
# 1) files will be imported to the list called imagelist
# The imported RGB-images are numpy arrays with format MxNx3
# 2) files will be converted into gray scale and stored in imagelist_gray
# The imported Files are numpy array with format MxN
imagelist = []
imagelist_gray =[]
for i in range(0,len(filelist)):
    if len(filelist) == len(pathlist):
        filepath = os.path.normpath(os.path.join(pathlist[i],filelist[i]))
    else:
        print('Filelist and Pathlist have not equal size')
    center = (0,0)
    img = io.imread(filepath)
    imagelist.append(img)
    img_raw = color.rgb2gray(img)
    img_gray =np.uint8(255*(img_raw/img_raw.max()))
    imagelist_gray.append(img_gray)

# +++++++++++++++ Main +++++++++++++++++++
# Some Global Variable that we need:
# Data Set A
#first_center_x = 348  # x-Coordinate of the Instrument center in the first image
#first_center_y = 191  # y-Coordinate of the Instrument center in the first image

# Data Set B
first_center_x = 439  # x-Coordinate of the Instrument center in the first image
first_center_y = 272  # y-Coordinate of the Instrument center in the first image

# ++++ Image Preprocessing ++++
img_corr = []
# align image all images to the first one 
img_corr, shift_vector = alignImages(imagelist_gray)
# compute background 
avg_img = compute_background_image(img_corr)
# calculate difference from background image (contrast enhancement)
img_diff = []
for ind, img in enumerate(img_corr):
    img_diff.append(np.abs(np.int16(img)-np.int16(avg_img)))
    
# ++++ Tool detection ++++
tool_pos =[]    
for ind, img in enumerate(img_diff):
    if ind == 0 : 
        tool_pos.append((first_center_x,first_center_y))
    elif ind > 0:
        new_center_x,new_center_y = findTool(img,tool_pos[ind-1][0],tool_pos[ind-1][1],patch_half_size=22, sigma=2 )
        tool_pos.append((new_center_x,new_center_y))
    rr, cc = draw.circle(tool_pos[ind][1]+shift_vector[ind][0], tool_pos[ind][0]+shift_vector[ind][1], 5, imagelist[ind].shape)
    imagelist[ind][rr,cc,:] = (255, 255, 0)
    save_img_path = os.path.normpath(os.path.join(savepath,(str(ind)+'.png')))
    io.imsave(save_img_path, imagelist[ind])
    plt.figure()
    plt.imshow(imagelist[ind])
    

# ++++ Generating the text file output:
# Output: Text file Tool_Coordinates.txt located in the same directory as 
# this script 
file = open("Tool_Coordinates.txt","w")
for ind, coord in enumerate(tool_pos):
    file.writelines(filelist[ind]+"\t"+str(coord[0])+"\t"+str(coord[1])+"\n")
file.close()

# This are left overs from older trials. 

#img_harris = []
#for ind, img in enumerate(img_diff):
#    harris = feature.corner_harris(img, method = 'eps', sigma =2, k =0.01)
#    harris_int = np.uint8(255*(harris/harris.max()))
#    img_harris.append(harris_int)
#    plt.figure()
#    plt.imshow(img_diff[ind], cmap = 'gray')

#for ind, img in enumerate(imagelist_gray):
#    img_fill = filters.gaussian(img, sigma =2)
#    img_fill = np.int16(32768*(img_fill/img_fill.max()))
#    img_fill2 = filters.gaussian(img, sigma =5)
#    img_fill2 = np.int16(32768*(img_fill2/img_fill2.max()))
#    dog = img_fill-img_fill2
#    harris = feature.corner_harris(dog, method = 'eps', sigma =1, k =0.2)
#    harris_int = np.uint8(255*(harris/harris.max()))
#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.imshow(dog, cmap='gray')
#    plt.subplot(2,1,2)
#    plt.imshow(harris_int)
#    
#    save_img_path = os.path.normpath(os.path.join(savepath,(str(ind)+'.png')))
#    plt.savefig(save_img_path, dpi=200)
#    io.imsave(save_img_path, imagelist[ind])

#for ind, img in enumerate(img_harris):
#    #save_img_path = os.path.normpath(os.path.join(savepath,filelist[ind]))
#    save_img_path = os.path.normpath(os.path.join(savepath,(str(ind)+'.png')))
#    plt.figure()
#    plt.subplot(2,2,1)
#    plt.imshow(img)
#    plt.subplot(2,2,2)
#    plt.imshow(img_corr[ind], cmap = 'gray')
#    plt.subplot(2,2,3)
#    plt.imshow(img_diff[ind], cmap = 'gray')
#    plt.savefig(save_img_path, dpi = 300)
#    
#harris = feature.corner_harris(patch, method = 'eps', sigma =3, k =0.2)
#harris_int = np.uint8(255*(harris/harris.max()))
#plt.figure()
#plt.imshow(harris_int)

##patch = filters.gaussian_filter(patch, sigma=2)
#patch = exposure.equalize_hist(patch)
#for i in range(0,3):
#    patch = restoration.denoise_bilateral(patch, multichannel=True)
#
#
#plt.imshow(patch)
#xyzpatch = color.rgb2xyz(patch)
#segments = slic(xyzpatch, n_segments=100, compactness=20)
#plt.figure()
#plt.imshow(segments)

