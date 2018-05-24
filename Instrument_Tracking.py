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
first_center_x = 0  # x-Coordinate of the Instrument center in the first image
first_center_y = 0  # y-Coordinate of the Instrument center in the first image






#Some chunck
#circles_vector = []
##    cimg, circle = findPupilCenter(img)
#    cimg,x_center, y_center = findPupilCenter2(img)
##    if circle.shape == (1,1,3):
##        circles_vector.append(circle)
##    else:
##        circles_vector.append([0,0,100])
#
#    # ++++ Save annotated images +++++
#    save_img_path = os.path.normpath(os.path.join(savepath,filelist[i]))
#    print(save_img_path)
#    cv2.imwrite(save_img_path, cimg)
