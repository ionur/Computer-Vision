import os
from multiprocessing import Pool
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from opts import get_options
from scipy.spatial import distance

def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """
    mode          = opts.mode
    num_filters   = opts.num_filters
    num_channels  = opts.num_channels
    filter_scales = opts.filter_scales
    h, w, c       = img.shape
    filter_responses = np.empty([h, w, num_filters * len(filter_scales) * num_channels])
    #check if the image is grayscale, if it is grayscale, create 3 channels using that single channe;
    if c > 3:
        img = img[:,:,:3]
    elif c == 1:
        img = np.stack((img,)*num_channels, axis=-1)

    #convert image into the Lab color space, which is designed to more effectively quantify color differences with respect to human perception
    img = skimage.color.rgb2lab(img)

    for scale_inx, scale in enumerate(filter_scales):
        start_inx = scale_inx * num_filters * num_channels
        for channel in range(0,num_channels): 
            #Gaussian
            filter_responses[:, :, start_inx + channel] = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma = scale, mode = mode)   
            #Laplacian of Gaussian
            filter_responses[:, :, start_inx + (3 + channel)] = scipy.ndimage.gaussian_laplace(img[:,:,channel], sigma = scale, mode = mode)
            #derivative of Gaussian in the x direction
            filter_responses[:, :, start_inx + (6 + channel)] = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma = scale, order=[0,1], mode = mode)
            #derivative of Gaussian in the y direction.
            filter_responses[:, :, start_inx + (9 + channel)] = scipy.ndimage.gaussian_filter(img[:,:,channel], sigma = scale, order=[1, 0], mode = mode)


    return filter_responses


def compute_dictionary_one_image(args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    """
    opts = get_options()
    alpha = opts.alpha
    num_filters   = opts.num_filters
    num_channels  = opts.num_channels
    
    idx, img_path = args
    img = plt.imread(img_path) / 255.
    h, w, c = img.shape
    
    filter_responses = extract_filter_responses(opts, img)
    #uses responses at alpha random pixels
    random_indices = np.random.permutation(h*w)[:alpha] 
    cropped_response = np.empty([alpha, filter_responses.shape[2]])
    for i, random_idx in enumerate(random_indices):
        row = int(random_idx / w)
        col = random_idx - (w * row)
        cropped_response[i, :] = filter_responses[row, col, : ]
    np.save(os.path.join(opts.res_dir, "img_dict", str(idx)), cropped_response)
    return filter_responses

def compute_dictionary(opts, dict_name, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    res_dir = opts.res_dir
    alpha   = opts.alpha
    K       = opts.K

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_files = train_files
    M = len(train_files)
    img_path = [join(opts.data_dir, img_name) for img_name in train_files]
    
    #pool to parallelize
    p = Pool(n_worker)
    
    args = zip(range(M), img_path)
    output = p.map(compute_dictionary_one_image, args)
    
    p.close()
    # Wait for all thread.
    p.join()
    
    sampled_responses = []
    
    for _,_, files in os.walk(join(res_dir,"img_dict")):
        for i, file in enumerate(files):
            resp = np.load(join(res_dir,"img_dict",file))
            sampled_responses.append(resp)        
    sampled_responses = np.array(sampled_responses)
    sampled_responses = sampled_responses.reshape(-1, sampled_responses.shape[-1])
    
    kmeans = KMeans(n_clusters=K).fit(sampled_responses) 
    dictionary = kmeans.cluster_centers_
    np.save(join(res_dir, dict_name), dictionary)


def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    filter_responses = extract_filter_responses(opts, img)
    h, w, x = filter_responses.shape
    #reshape 
    filter_responses = filter_responses.reshape([h * w, x])
    d = distance.cdist(filter_responses, dictionary, 'euclidean')

    wordmap = np.argmin(d, axis = 1)
    wordmap = wordmap.reshape([h, w])
    return wordmap
