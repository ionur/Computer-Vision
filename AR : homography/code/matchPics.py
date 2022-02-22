import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

"""
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
"""
def matchPics(I1, I2, opts):

        
        ratio       = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma       = opts.sigma  #'threshold for corner detection using FAST feature detector'
        

        # Convert Images to GrayScale
        I1          = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY) 
        I2          = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        
        # Detect Features in Both Images
        locs1       = corner_detection(I1, sigma)
        locs2       = corner_detection(I2, sigma)
        
        # Obtain descriptors for the computed feature locations
        desc1,locs1 = computeBrief(I1, locs1)
        desc2,locs2 = computeBrief(I2, locs2)

        # Match features using the descriptors
        # matches is of size [Q,2] where Q is the total #matches, col 1 corresponds to indices of first descriptor and  col 2 corresponds to indices of first descriptor
        matches = briefMatch(desc1,desc2,ratio)
        
        return matches, locs1, locs2
