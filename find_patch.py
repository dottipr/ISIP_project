'''
Author: Prisca Dotti
Updated: 22.05.2018

script containing code for finding a patch according to the previous one (unfinished)
'''

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
from skimage import filters
import tools_hw2 as tools


def extractPatch(img, x, y, patch_half_size):
    # input: greyscale image img
    #        coordinates x and y of the center of the patch
    #        patch half patch size
    # output: patch of img centered in x,y with size 2*patch_half_size+1 x 2*patch_half_size+1

    # !!!! arrays coordinates are switched with respect to usual coordinates

    return img[y-patch_half_size : y+patch_half_size+1, x-patch_half_size : x+patch_half_size+1]
