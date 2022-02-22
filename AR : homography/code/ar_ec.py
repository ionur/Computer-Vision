import numpy as np
import cv2
import sys
import sys

sys.path.append('../python/')

import time
from helper import loadVid, crop_center,save_video
from matplotlib import pyplot as plt
from planarH import *
from matchPics import matchPics
from opts import get_opts

"""
    Creates an Augmented Reality application faster
    
    source_path: source image to match
    dest_path  : video onto warping will happen
    temp_path  : video to be warped
"""
class AR():
    def __init__(self):
        pass

    def create_ar_ec(self, source_path, dest_path, temp_path, opts):
        dest_mov   = loadVid(dest_path)
        temp_mov   = loadVid(temp_path)

        _,          H1, W1, C1 = dest_mov.shape
        num_frames, H2, W2, C2 = temp_mov.shape

        out_mov    = []
        border     = 100

        #Reads cv_cover.jpg, cv_desk.png, and hp_cover.jpg.
        source     = cv2.imread(source_path)

        dx         = int(source.shape[1]/2)
        dy         = int((H2 - border )/2)

        # Initiate ORB detector
        orb        = cv2.ORB_create()
        bf         = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        start_time = time.time()
        for i in range(num_frames):
            #get the relevant frame
            dest   = dest_mov[i,:,:,:]
            im     = temp_mov[i,:,:,:]
            #crop the im to be the same size as source
            im     = crop_center(im, dy, dx)
            im     = cv2.resize(im, (source.shape[1],source.shape[0]), interpolation = cv2.INTER_AREA)

            # find the keypoints and descriptors with SIFT
            locs1, desc1 = orb.detectAndCompute(dest,None)
            locs2, desc2 = orb.detectAndCompute(source,None)

            matches = bf.knnMatch(desc1, desc2, k = 2)

            self.tmp1 = []
            self.tmp2 = []
            
            def get_match(row, args):
                locs1,locs2 = args
                match1, match2 = row
                if match1.distance < opts.ratio * match2.distance:
                    self.tmp1.append(locs1[match1.queryIdx].pt)
                    self.tmp2.append(locs2[match1.trainIdx].pt)

            args    = (locs1,locs2)
            np.apply_along_axis(get_match, 1, matches,args)
            locs1   = np.array(self.tmp1)
            locs2   = np.array(self.tmp2)

            bestH2to1, inliers    = computeH_ransac(locs1, locs2, opts)

            #reshape im to be like source
            im_ = cv2.resize(im, (source.shape[1],source.shape[0]))
            out = compositeH(bestH2to1, im_, dest)
            out_mov.append(out)
        print('Processed {} frames in {}s. FPS is {}'.format(num_frames, time.time() - start_time,(time.time() - start_time)/num_frames))
        return out_mov

        
if __name__ == "__main__":
    source_path = '../data/cv_cover.jpg'
    dest_path   = '../data/book.mov'
    temp_path   = '../data/ar_source.mov'
    outvid      = "../result/output.mov"
    
    opts        = get_opts()
    out_mov     = AR().create_ar(source_path, dest_path, temp_path, opts)
    save_video(out_mov, 10, outvid)

