import numpy as np
import cv2
from matplotlib import pyplot as plt
"""
    Computes the homography between two sets of points
        x1 = Hx2
    
    inputs x1 and x2 are N × 2 matrices containing the coordinates (x, y)
    mapping is from x2 to x1 so x1 = H(x2)
"""
def computeH(x1, x2):
    num_points = x1.shape[0]
    #calculate matrix A
    A = np.zeros((num_points * 2, 9))
    
    for i in range(num_points):
        x, y       = x2[i]
        x_,y_      = x1[i]
        start      = i*2
        stop       = i*2 + 1
        A[start,:] = [-x, -y, -1,  0,  0,  0,  x_*x, x_*y, x_]
        A[stop, :] = [ 0,  0,  0, -x, -y, -1,  y_*x, y_*y, y_]
    
    #get the SVD decomposition. The columns of V are the eigenvectors of ATA. 
    u,s,v          = np.linalg.svd(A)
    
    #we choose the smallest eigenvalue of ATA, which is the least-squares solution to Ah = 0 
    #eigenvalues are sorted in descending order, so get the last col of v
    H2to1          = np.reshape(v[-1], (3, 3))
    return H2to1

"""
    Computes normalized homography
    inputs x1 and x2 are N × 2 matrices containing the coordinates (x, y)
"""
def computeH_norm(x1, x2):
    N     = x1.shape[0]
    #Compute the centroid of the points
    mean1 = np.mean(x1, axis = 0)
    mean2 = np.mean(x2, axis = 0)

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    scale1 = np.sqrt(2) / np.mean(np.sqrt(np.sum(np.power(x1,2),1)))
    scale2 = np.sqrt(2) / np.mean(np.sqrt(np.sum(np.power(x2,2),1)))
    
    T1    = np.array([[scale1, 0, -mean1[0]*scale1], [0, scale1, -mean1[1]*scale1], [0, 0, 1]])
    T2    = np.array([[scale2, 0, -mean2[0]*scale2], [0, scale2, -mean2[1]*scale2], [0, 0, 1]])
    
    #Shift the origin of the points to the centroid and scale
    x1_   = (x1 - mean1) * scale1
    x2_   = (x2 - mean2) * scale2
    
    #Compute homography
    H_norm = computeH(x1_, x2_)
    
    #Denormalization
    H2to1 =  np.dot(np.linalg.inv(T1), np.dot(H_norm, T2))

    return H2to1



"""
    Compute the best fitting homography given a list of matching points
        x1 = Hx2
    locs1 and locs2 are N × 2 matrices containing the matched points.
    each row of loc is of the form (x,y) !
"""
def computeH_ransac(locs1, locs2, opts):
    num_points      = locs1.shape[0]
    max_iters       = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol      = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    min_points      = 4               # min number of points needed to calculate H
    best_inliers    = None
    bestH2to1       = None
    best_num        = 0
    for epoch in range(max_iters):
        #pick random points
        start = np.random.choice(num_points, min_points, replace = False)
        x1    = locs1[start, :]
        x2    = locs2[start, :]
        H     = computeH_norm(x1, x2)
        #find all points that satifsy locs1 = lamda * H(locs2)
        #transform locs2 to be in the form [3,N] where cols correspond to (x,y,1)
        locs2_= np.concatenate((np.transpose(locs2), np.ones((1, num_points))), axis = 0)
        out   = np.dot(H, locs2_)
        alphas= out[0,:]
        betas = out[1,:]
        lamda = out[2, :]
        out   = np.transpose(np.array([np.array(alphas/lamda), np.array(betas/lamda)]))
                         
        #check the distance between x1 and out and get the ones smaller than threshold
        dist  = np.linalg.norm(locs1-out, axis = 1)
        tmp   = dist < inlier_tol
        if sum(tmp) > best_num:
            best_inliers = tmp * 1
            bestH2to1    = H
            best_num     = sum(tmp)
    return bestH2to1, best_inliers


'''
    Takes the template image, warps it and combines with the destination image
'''
def compositeH(H2to1, template, img):
    #create ones to be mapped to the image location
    mask          = np.ones(template.shape)
    #warp the mask
    warped_mask   = cv2.warpPerspective(mask, H2to1, (img.shape[1], img.shape[0])) 
    #warp the template
    temp_warp     = cv2.warpPerspective(template, H2to1, (img.shape[1], img.shape[0]))
   
    #invert the mask
    inv_warp_mask = np.where(warped_mask>=1,0,1)  
    #clear the location of the template in the image
    clean_img   = np.multiply(inv_warp_mask, img)
    #compine the warped template with the cropped destination image
    composite_img  = clean_img + temp_warp
    return composite_img


