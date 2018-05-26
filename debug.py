import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from PIL import Image
from utils import *

# if you agree I think it could be a good idea to have all functions in separate
# files (on which we can work separately) and then put the rest in the main
# I let you modify it when your code is working :-)

filenames = glob.glob(os.path.join('project_data', 'b', '*.png'))
images = [np.asarray(Image.open(f).convert('L')) for f in filenames]
images = [gconv(i,3,7) for i in images]
#images = [cv2.bilateralFilter(i,5,150,150) for i in images]


#position in first image
x,y = 439-3,272-3 # -3 as gaussian filter reduces the size of the images

currentImg = images[0]

plt.figure()
plt.subplot(4,5,1)
plt.imshow(currentImg)
plt.axis("off")
plt.title(1)
plt.scatter(x,y,color='red')

for i in range(1,len(images)-1):
    if ((i+1)%5==0):
        plt.subplot(4,5,(i+1)/5+1)
        plt.imshow(currentImg)
        plt.axis("off")
        plt.title(i+1)
        plt.scatter(x,y,color='red')

    previousImg = images[i-1]
    currentImg = images[i]
    position = findTool(currentImg, x, y)
    x = position[0]
    y = position[1]

plt.show()
