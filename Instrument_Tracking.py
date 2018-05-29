# -*- coding: utf-8 -*-
"""
Author: Michael Stiefel and Prisca Dotti
Updated: 29.05.2018
This is the a script to do instrument tracking
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog
from tkinter import *
from skimage import io
from skimage import color
from skimage.segmentation import slic
from skimage import exposure
from skimage import feature
from skimage import filters
from skimage import transform
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage import draw
import cv2
# Import of all our functions defined in the utils.py script
from utils import *



# ++++++++++Search for images in selected folder+++++++++++++++

data  = Tk()
data.filepath = filedialog.askdirectory(initialdir = "/",title = "Please select working directory")
homeDir = data.filepath
data.destroy()

# Set variables for Dataset A,B
datasetSelector = [[True, False, "/a", "Tool_PositionA.txt", "Dataset A"],[ False, True, "/b", "Tool_PositionB.txt", "Dataset B"]]

for ind, setVariables in enumerate(datasetSelector):
    print("----Starting to process "+setVariables[4]+"----")
    wa = setVariables[0]
    wb = setVariables[1]
    datasetDir = os.path.join(homeDir + setVariables[2])
    

    pathlist = []
    filelist = []
    # Searching for Files in the directory
    for root, dirs, files in os.walk(datasetDir):
        for file in files:
            if file.endswith(".png"):
                filelist.append(file)
                pathlist.append(root)
    print('Read file names --> Done')
    
    # +++++ This imports all images that are in the selected folder
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
    print("Load all images --> Done")
    # Set coordinates of the Instrument center for the first image of the dataset
    if wa == True and wb == False : 
        # Data Set A
        first_center_x = 348  
        first_center_y = 191  
    elif wa == False and wb == True:
        # Data Set B
        first_center_x = 439  
        first_center_y = 272  

    # +++++++++++++++ Main +++++++++++++++++++
    # ++++ Image Preprocessing ++++
    img_corr = []
    # align image all images to the first one
    img_corr, shift_vector = alignImages(imagelist_gray)
    print("Image alignment --> Done")

    # ++++ Tool detection ++++
    tool_pos =[]
    det_lines = []
    points = []
    # Generates a vector with angels for hough transform
    angles1 = np.linspace(0,np.pi/4,45)
    angles2 = np.linspace(np.pi*1.5,np.pi*2,45)
    angles = np.concatenate((angles1,angles2), axis=0)
    # Set patch coordinates to first images tool center
    patch_center_x = first_center_x
    patch_center_y = first_center_y
    
    patch_half_size = 50
    patch_size = 2*patch_half_size+1
    best_corner = [patch_half_size, patch_half_size]
    middle_end = best_corner
    print("Starting the image Processing")
    for ind, img in enumerate(img_corr):
        if ind == 0: 
            tool_pos.append((first_center_x,first_center_y))
        else:
            # --- Patch Preprocessing ----
            img = exposure.equalize_adapthist(img)
            img = np.uint8(255*(img/img.max()))
            patch = cutpatch(img,patch_center_x,patch_center_y, patch_size,patch_size)
            # Apply Sobel filter to detect edges
            edges = filters.sobel(patch)
            edges = filters.gaussian(edges, sigma=1)
            edges = np.uint8(255*(edges/edges.max()))
            # Binarize image by means of Otsu thresholding
            otsu = threshold_otsu(edges)
            skeleton = np.zeros(patch.shape)
            skeleton[edges>= otsu] = 1
            # Skeletonize image in order to get sharp edges
            skeleton = morphology.skeletonize(skeleton)
            edges[skeleton == True] = 255
            edges[skeleton == False] = 0
            # Apply Harris Corner detector to detect corners on the patch
            harris = feature.corner_harris(edges, method = 'k', sigma =2, k =0.1)
            # Select the 6 best edges in the image
            peak_data = feature.corner_peaks(harris, min_distance=10, num_peaks=6, exclude_border=True)
            points.append(peak_data)
            # Apply hough transform to detect the instrument shaft
            lines = transform.probabilistic_hough_line(edges,threshold=10, line_gap= 3, line_length=20, theta=angles)
            det_lines.append(lines)
        
            # uncomment this section to see the detailed performance (creates a lot of plots)
#            for bla, line in enumerate(lines):
#                rr,cc = draw.line(line[0][0],line[0][1],line[1][0],line[1][1])
#                patch[cc,rr] = 255
#        
#            for bla, peak in enumerate(peak_data):
#                rr, cc = draw.circle(peak[1], peak[0], 3, patch.shape)
#                patch[cc,rr] = 255 # draw circles around peaks in patch
#    
#            plt.subplot(1,3,2)
#            plt.imshow(harris)
#            plt.title("harris") 
#    
#            plt.figure()
#            plt.subplot(1,3,1)
#            plt.imshow(edges, cmap = 'gray')
#            plt.title("edges")
#
#            plt.subplot(1,3,3)
#            plt.imshow(patch, cmap = 'gray')
#            plt.title("patch + corners and lines")

            # +++++++ Optimal corner's choice +++++++

            patch_center = [patch_half_size, patch_half_size]
            lines = np.asarray(lines)
            if (not (len(lines)==0)): # if we don't find any lines we use the longest line from the previous iteration...
                lines_lengths = [np.sqrt(np.square(l[0,0]-l[1,0])+np.square(l[0,1]-l[1,1])) for l in lines]
                longest_line = lines[np.argmax(lines_lengths)]
                start = longest_line[0]
                end = longest_line[1]
                angle = np.arctan((end[1]-start[1])/(end[0]-start[0]))+np.pi/2
                angles = np.linspace(angle-np.pi/30,angle+np.pi/30,20)
                middle_end = findClosest(longest_line,(best_corner[1],best_corner[0]))
            '''
            plt.plot([start[0], end[0]],[start[1],end[1]], color="yellow",linewidth=2.0)
            '''

            best_corner = (best_corner+findClosest(peak_data,(middle_end[1],middle_end[0])))//2
            best_corner_img = (patch_center_x-patch_half_size+best_corner[1],patch_center_y-patch_half_size+best_corner[0])
            tool_pos.append((best_corner_img[0]+shift_vector[ind][1],best_corner_img[1]+shift_vector[ind][0]))

            # Uncomment this section to plot the results:
            rr, cc = draw.circle(tool_pos[ind][0], tool_pos[ind][1],5,imagelist[ind].shape)
            imagelist[ind][cc,rr] = 255, 255, 0 # draw circles around peaks in patch
            plt.figure()
            plt.imshow(imagelist[ind])
            plt.title('Image Nr. '+str(ind))
    
    print("Finished image Processing")

    # ++++ Generating the text file output:
    # Output: Text file Tool_Coordinates.txt located in the same directory as
    # this script
    print("saving data to file:" + setVariables[3])
    file = open(homeDir+setVariables[3],"w")
    for ind, coord in enumerate(tool_pos):
        file.writelines(filelist[ind]+"\t"+str(coord[0])+"\t"+str(coord[1])+"\n")
    file.close()




