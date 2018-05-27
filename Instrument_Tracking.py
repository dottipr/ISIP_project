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
from skimage import transform
from skimage import morphology
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
#avg_img = compute_background_image(img_corr)
# calculate difference from background image (contrast enhancement)
#img_diff = []
#for ind, img in enumerate(img_corr):
#    img_diff.append(np.abs(np.int16(img)-np.int16(avg_img)))
    
# ++++ Tool detection ++++
tool_pos =[]
img_harris = []
img_cannys = []
det_lines = []
points = []
angels1 = np.linspace(0,np.pi/4,45)
angels2 = np.linspace(np.pi*1.5,np.pi*2,45)
angels = np.concatenate((angels1,angels2), axis=0)

for ind, img in enumerate(img_corr):
    img = exposure.equalize_adapthist(img)
    img = np.uint8(255*(img/img.max()))
    patch = cutpatch(img,first_center_x,first_center_y, 101,101)
    skeleton = np.zeros(patch.shape)
    edges = filters.sobel(patch)
    edges = np.uint8(255*(edges/edges.max()))
    otsu = threshold_otsu(edges)
    skeleton[edges>= otsu] = 1
    skeleton = morphology.skeletonize(skeleton)
    edges[skeleton == True] = 255
    edges[skeleton == False] = 0
    
    plt.figure()
    plt.imshow(edges, cmap = 'gray')
    harris = feature.corner_harris(edges, method = 'k', sigma =2, k =0.05)
    peak_data = feature.corner_peaks(harris, min_distance=10, num_peaks=3, exclude_border=True)
    points.append(peak_data)
    for bla, peak in enumerate(peak_data):
        rr, cc = draw.circle(peak[1], peak[0], 3, patch.shape)
        patch[cc,rr] = 255
    plt.figure()
    plt.imshow(harris)
    lines = transform.probabilistic_hough_line(edges,threshold=10,line_gap= 3, line_length=20, theta=angels)
    det_lines.append(lines)
    for bla, line in enumerate(lines):
        rr,cc = draw.line(line[0][0],line[0][1],line[1][0],line[1][1])
        patch[cc,rr] = 255
    plt.figure()
    plt.imshow(patch, cmap = 'gray')

# Version with Priscas code    
#for ind, img in enumerate(img_diff):
#    if ind == 0 : 
#        tool_pos.append((first_center_x,first_center_y))
#    elif ind > 0:
#        new_center_x,new_center_y = findTool(img,tool_pos[ind-1][0],tool_pos[ind-1][1],patch_half_size=22, sigma=2 )
#        tool_pos.append((new_center_x,new_center_y))
#    rr, cc = draw.circle(tool_pos[ind][1]+shift_vector[ind][0], tool_pos[ind][0]+shift_vector[ind][1], 5, imagelist[ind].shape)
#    imagelist[ind][rr,cc,:] = (255, 255, 0)
#    save_img_path = os.path.normpath(os.path.join(savepath,(str(ind)+'.png')))
#    io.imsave(save_img_path, imagelist[ind])
#    plt.figure()
#    plt.imshow(imagelist[ind])
    

# ++++ Generating the text file output:
# Output: Text file Tool_Coordinates.txt located in the same directory as 
# this script 
#file = open("Tool_Coordinates.txt","w")
#for ind, coord in enumerate(tool_pos):
#    file.writelines(filelist[ind]+"\t"+str(coord[0])+"\t"+str(coord[1])+"\n")
#file.close()

# This are left overs from older trials. 



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

#for ind, img in enumerate(img_diff):
#    if ind == 0 : 
#        tool_pos.append((first_center_x,first_center_y))
#    elif ind > 0:
#        width = 71
#        height =71
#        patch = cutpatch(img, tool_pos[ind-1][0]-10, tool_pos[ind-1][1],width,height)
#        plt.figure()
#        plt.imshow(patch, cmap='gray')
#        harris = feature.corner_harris(patch, method = 'eps', sigma =3, k =0.1)
#        peak_data = feature.corner_peaks(harris, min_distance=5, num_peaks=2, exclude_border=True)
#        print(peak_data)
#        harris_int = np.uint8(255*(harris/harris.max()))
#        plt.figure()        
#        plt.imshow(harris_int)
#        img_harris.append(harris_int)
#        # Calulate line between two ends:
#        m = (peak_data[1][1]-peak_data[0][1])/(peak_data[1][0]-peak_data[0][0])
#        print("m= "+str(m))
#        x_middle = (peak_data[1][1]+peak_data[0][1])/2
#        y_middle = (peak_data[1][0]+peak_data[0][0])/2
#        m_ortho = -1/m
#        vector = np.zeros(30)
#        for x in range(0,30):
#            y = np.ceil(y_middle+(m_ortho*x))
#            vector[ind] = patch[x_middle+x,y]
#        gradient = np.gradient(vector)
#        x_max_grad = np.argmax(gradient)
#        y_max_grad = np.ceil(y_middle+(m_ortho*x_max_grad))
#        center_x = tool_pos[ind-1][0] - (width-1)/2 + x_max_grad+x_middle
#        center_y = y_middle + y_max_grad
#        tool_pos.append((center_x,center_y))
#        rr, cc = draw.circle(tool_pos[ind][1]+shift_vector[ind][0], tool_pos[ind][0]+shift_vector[ind][1], 5, imagelist[ind].shape)
#        imagelist[ind][rr,cc,:] = (255, 255, 0)
#        plt.figure()
#        plt.imshow(imagelist[ind])

