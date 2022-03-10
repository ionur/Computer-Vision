import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
from scipy.ndimage import interpolation

"""
    Dominant Motion Estimation
    
     a majority of the pixels correspond to the stationary objects in the scene whose depth variation is small relative to their distance from the camera

    It        : template image
    It1       : Current image
    threshold : if the length of dp is smaller than the threshold, terminate the optimization
    num_iters : number of iterations of the optimization
    
    return
    M         : the Affine warp matrix [2x3 numpy array] put your implementation here
 """
def LucasKanadeAffine(It, It1, threshold, num_iters):
    H,W       = It.shape
    
    spline_t  = RectBivariateSpline(np.arange(0,H), np.arange(0,W), It)
    spline_t1 = RectBivariateSpline(np.arange(0,H), np.arange(0,W), It1)
    
    dp        = None
    M         = np.eye(3)
    
    for i in range(num_iters):
        #if the length of dp is smaller than the threshold, terminate the optimization
        if dp is not None and np.linalg.norm(dp)**2 < threshold:
            break
          
        #manually update parameters instead of affine_transform because affine transform 
        #transforms the entire image
        col,row   = np.meshgrid(np.arange(0, W + 0.01), np.arange(0, H + 0.01))
        x, y      = col.flatten(), row.flatten()
        #warp the coordinates
        x_w       = M[0,0] * x + M[0,1] * y + M[0,2]
        y_w       = M[1,0] * x + M[1,1] * y + M[1,2]   
        
        #only get the points lying within It
        valid_idx = (0 <= x_w) & (x_w < W) & (0 <= y_w) & (y_w < H)
        x, y      = x[valid_idx], y[valid_idx]
        x_w, y_w  = x_w[valid_idx], y_w[valid_idx]
                          
        #get the template values    
        interp_t  = spline_t.ev(x,y)
        #get the warped img values
        interp_t1 = spline_t1.ev(x_w,y_w)
        
        # calculate gradient and wrap
        #first sample, then get image gradients
        dIw       = np.zeros((len(x_w),2))
        dIw[:, 0] = spline_t1.ev(x_w,y_w, dx = 1)
        dIw[:, 1] = spline_t1.ev(x_w,y_w, dy = 1)        
        
        # b = It - It1(W(x,p)) 
        b        = interp_t - interp_t1
        
        #steepest descent images is A = dIwdW/dp 
        #dW/dp is of the form [ [x y 1 0 0 0] [ 0 0 0 x y 1]]       
        A        = np.array([ [ dIw[i, 0] * x[i] ,
                                dIw[i, 0] * y[i] ,
                                dIw[i, 0] * 1 , 
                                dIw[i, 1] * x[i] ,
                                dIw[i, 1] * y[i] , 
                                dIw[i, 1] * 1 ] for i in range(len(x_w))])
        #x = (AtA)-1 Ab
        dp       = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,b)).reshape(2,3)
        M[:2,:] += dp
    return M