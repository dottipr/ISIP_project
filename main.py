import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from PIL import Image
import tools_hw2 as tools
#import Instrument_Tracking as tracking
import corner_detection as corner
import find_patch

# if you agree I think it could be a good idea to have all functions in separate
# files (on which we can work separately) and then put the rest in the main
# I let you modify it when your code is working :-)

filenames = [os.path.join('project_data', 'a', '000224.png'), os.path.join('project_data', 'a', '000225.png')]
images = [np.asarray(Image.open(f).convert('L')) for f in filenames]

#position in previous image
x,y = 348,191

patch = find_patch.extractPatch(images[1], 348, 191, 20)

response = corner.computeResponseMatrix(patch)

binary = corner.binaryResponse(response)

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
