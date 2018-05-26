# ISIP_project

24.5.18

Updates : 

put all functions in utils.py

in utils. py:
-in function cutpatch: converted x_top_left and y_top_left to integers (it didn't work on my code before)
-added function harrisCorners that returns a binary image containing the highlighted corners of the input image (while function responseMatrix returns the response matrix of harris corners algorithm)
-added function findTool which returns the position of the tool in the new image according to the previous one. The idea is the following: we cut out a patch in the new image centered at the position of the tool in the previous image and apply harris corner algorithm to it to mask out the regions that are candidates for being the new position of the tool. We then apply a Gaussian filter in order to give priority to pixel that are close to the previous position (if you have better ideas they are welcomed :-)) We then choose the pixel which has a minimal value [corners are negative] in that region to be the new position.

in debug.py:
-plot of some images and the selected position of the tool 
