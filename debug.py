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

filenames = glob.glob(os.path.join('project_data', 'a', '*.png'))
images = [np.asarray(Image.open(f).convert('L')) for f in filenames]

#position in first image
x,y = 348,191

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

'''
patch = cutpatch(images[1], x, y, 41,41)

response = computeResponseMatrix(patch)

binary = binaryResponse(response)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(patch)
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(response)
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(binary)
plt.axis('off')
plt.show()

a = findTool(images[1],348,191)

plt.figure()
plt.imshow(images[1])
plt.scatter(a[0],a[1])
plt.show()
'''
