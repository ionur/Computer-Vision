import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

"""
    It         : template image
    It1        : Current image
    threshold  : if the length of dp is smaller than the threshold, terminate the optimization
    num_iters  : number of iterations of the optimization
    
    returns
    M          : the Affine warp matrix [2x3 numpy array]
"""
def InverseCompositionAffine(It, It1, threshold, num_iters):
    H, W      = It.shape
    
    spline_t  = RectBivariateSpline(np.arange(0,H), np.arange(0,W), It)
    spline_t1 = RectBivariateSpline(np.arange(0,H), np.arange(0,W), It1)
    
    #first compute the entire image gradients Ix and Iy of the template
    dy,dx     = np.gradient(It)
    spline_dx = RectBivariateSpline(np.arange(0,H), np.arange(0,W), dx)
    spline_dy = RectBivariateSpline(np.arange(0,H), np.arange(0,W), dy)
    
    col,row   = np.meshgrid(np.arange(0, W + 0.01), np.arange(0, H + 0.01))
    x, y      = col.flatten(), row.flatten()
    
    dIt       = np.zeros((len(x),2))
    dIt[:, 0] = spline_dx.ev(x, y)
    dIt[:, 1] = spline_dy.ev(x, y)        

        
    #steepest descent images is A = dIwdW/dp 
    #dW/dp is of the form [ [x y 1 0 0 0] [ 0 0 0 x y 1]] 
    A         = np.array([ [ dIt[i, 0] * x[i] ,
                                dIt[i, 0] * y[i] ,
                                dIt[i, 0] * 1 , 
                                dIt[i, 1] * x[i] ,
                                dIt[i, 1] * y[i] , 
                                dIt[i, 1] * 1 ] for i in range(len(x))])
    inv_hess  = np.linalg.inv(np.dot(A.T,A))
    
    dp        = None
    M         = np.eye(3)
    
    for i in range(num_iters):
        #if the length of dp is smaller than the threshold, terminate the optimization
        if dp is not None and np.linalg.norm(dp)**2 < threshold:
            break
            
        #warp the coordinates
        x_w          = M[0,0] * x + M[0,1] * y + M[0,2]
        y_w          = M[1,0] * x + M[1,1] * y + M[1,2]   
        
        #get the template values    
        interp_t     = spline_t.ev(x,y)
        #get the warped img values
        interp_t1    = spline_t1.ev(x_w,y_w)
        
        # b = It1(W(x,p)) - It
        b            = interp_t1 - interp_t
        
        #find invalid coordinates and set the to 0 in the error img
        inval_idx    = (0 > x_w) | (x_w >= W) | (0 > y_w) | (y_w >= H)
        b[inval_idx] = 0
              
        dp           = np.dot(inv_hess,np.dot(A.T,b)).reshape(2,3)
        dM           = np.eye(3)
        dM[:2,:]    += dp
        M            = np.dot(M, np.linalg.inv(dM))
        
        return M