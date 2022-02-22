import numpy as np
import cv2

from helper import loadVid, crop_center,save_video
from matplotlib import pyplot as plt
from planarH import *
from matchPics import matchPics
from opts import get_opts

"""
    Creates an Augmented Reality application
    
    source_path: source image to match
    dest_path  : video onto warping will happen
    temp_path  : video to be warped
"""
def create_ar(source_path, dest_path, temp_path):
    dest_mov   = loadVid(dest_path)
    temp_mov   = loadVid(vid_path)

    _,          H1, W1, C1 = dest_mov.shape
    num_frames, H2, W2, C2 = temp_mov.shape

    out_mov    = []
    border     = 100
    
    #Reads cv_cover.jpg, cv_desk.png, and hp_cover.jpg.
    source     = cv2.imread(source_path)

    dx         = int(source.shape[1]/2)
    dy         = int((H2 - border )/2)

    for i in range(num_frames):
        #get the relevant frame
        dest   = dest_mov[i,:,:,:]
        im     = temp_mov[i,:,:,:]
        #crop the im to be the same size as source
        im     = crop_center(im, dy, dx)
        im     = cv2.resize(im, (source.shape[1],source.shape[0]), interpolation = cv2.INTER_AREA)

        matches, locs1, locs2 = matchPics(dest, source, opts)
        #first get the matching locations
        locs1 = locs1[matches[:,0], :]
        locs2 = locs2[matches[:,1], :]
        #locations are of the form (y,x) make them (x,y)
        locs1 = locs1[:,(1,0)]
        locs2 = locs2[:,(1,0)]

        bestH2to1, inliers    = computeH_ransac(locs1, locs2, opts)

        #reshape im to be like source
        im_ = cv2.resize(im, (source.shape[1],source.shape[0]))
        out = compositeH(bestH2to1, im_, dest)
        out_mov.append(out)
    return out_mov

        
if __name__ == "__main__":
    source_path = '../data/cv_cover.jpg'
    dest_path   = '../data/book.mov'
    temp_path   = '../data/ar_source.mov'
    outvid      = "../result/output.mov"
    
    opts        = get_opts()
    out_mov     = create_ar(source_path, dest_path, temp_path)
    save_video(out_mov, 25, outvid)
