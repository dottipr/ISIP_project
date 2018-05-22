'''
Author: Prisca Dotti
Updated: 22.05.2018

script using harris corner detection's function from skimage
'''

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import feature

########################### main ###########################

# load images
filenames = glob.glob(os.path.join('project_data', 'a_train_copy', '*.png'))
images = [np.asarray(Image.open(f).convert('L')) for f in filenames]
nb_images = np.shape(images)[0]

responseMatrices = [feature.corner_harris(images[i], method = 'eps', sigma = 3) for i in range(0, nb_images)]

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
