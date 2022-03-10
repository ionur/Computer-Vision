import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage     import shift, convolve,gaussian_filter
import cv2             

"""

  Implements Lucas Kanade Alg.
    
    It        : template image
    It1       : Current image
    rect      : Current position of the car (top left, bot right coordinates)
    threshold : stopping threshold for dp 
    num_iters : number of iterations of the optimization
    p0        : Initial movement vector [dp_x0, dp_y0]
    
  return:
    p         : movement vector [dp_x, dp_y]
"""

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    H,W       = It.shape
    
    x         = np.arange(rect[0], rect[2] + 0.01)
    y         = np.arange(rect[1], rect[3] + 0.01)
    col, row  = np.meshgrid(x, y)

    p         = p0
    dp        = None
    
    spline_t  = RectBivariateSpline(np.arange(0,H), np.arange(0,W), It)
    spline_t1 = RectBivariateSpline(np.arange(0,H), np.arange(0,W), It1)
    
    #first compute the entire image gradients Ix and Iy and sample them at warp later
    dx        = cv2.Sobel(It1, cv2.CV_64F, 1, 0, ksize=3)
    dy        = cv2.Sobel(It1, cv2.CV_64F, 0, 1, ksize=3)
    spline_dx = RectBivariateSpline(np.arange(0,H), np.arange(0,W), dx)
    spline_dy = RectBivariateSpline(np.arange(0,H), np.arange(0,W), dy)
    
    for i in range(num_iters):
        #if the length of dp is smaller than the threshold, terminate the optimization
        if dp is not None and np.linalg.norm(dp)**2 < threshold:
            break
        #coordinates to wrap the image -- add shift to the coordinates
        x_w         = np.arange(rect[0], rect[2] + 0.01) + p[0]
        y_w         = np.arange(rect[1], rect[3] + 0.01) + p[1]
        col_w, row_w= np.meshgrid(x_w, y_w)
        
        #get the template values    
        interp_t  = spline_t.ev(row,col)
        #get the warped img values
        interp_t1 = spline_t1.ev(row_w,col_w)
        
        # calculate gradient and wrap
        dIw       = np.zeros((x.shape[0] * y.shape[0],2))
        dIw[:, 0] = spline_dx.ev(row_w,col_w).flatten()
        dIw[:, 1] = spline_dy.ev(row_w,col_w).flatten()
        
        # b = It - It1(W(x,p)) 
        b        = interp_t.flatten() - interp_t1.flatten()
        #steepest descent images is A = dIw since dW/dp is identity
        A        = dIw
        #x = (AtA)-1 Ab
        dp       = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,b))
        p       += dp 
    return p
