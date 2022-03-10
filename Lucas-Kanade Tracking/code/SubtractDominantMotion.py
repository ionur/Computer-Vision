import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage     import affine_transform,binary_erosion,binary_dilation
from InverseCompositionAffine import InverseCompositionAffine

"""
    image1    : Images at time t
    image2    : Images at time t+1
    threshold : used for LucasKanadeAffine
    num_iters : used for LucasKanadeAffine
    tolerance : binary threshold of intensity difference when computing the mask
    
    return    :
       mask   : [nxm] pixels corresponding to moving objects
"""
def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance, lk_type = 'affine'):
    if lk_type == 'affine':
        M                      = LucasKanadeAffine(image1, image2, threshold, num_iters)  
    else:
        print("here")
        M                      = InverseCompositionAffine(image1, image2, threshold, num_iters)
        
    #take the inverse as the mapping is inverse now 
    warped_im              = affine_transform(image1,np.linalg.inv(M))
    
    #erode and dilate
    eroded_im              = binary_erosion(warped_im)
    dilated_im             = binary_dilation(eroded_im)
    
    diff                   = abs(image2- dilated_im) #take their difference
    
    mask                   = np.zeros(image1.shape, dtype=bool)
    mask[diff > tolerance] = 1
    #only consider valid points. Affine puts 0 to out of bounds
    mask[warped_im == 0]   = 0

    return mask
