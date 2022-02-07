import os
import math
import multiprocessing
from os.path import join
from copy import copy
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from sklearn import svm

from visual_words import *


def get_feature_from_wordmap(opts, wordmap, dictionary, normalize = True):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    hist,edges = np.histogram(wordmap, bins = len(dictionary), range = (0,len(dictionary)))
    if normalize:
        hist = hist / np.linalg.norm(hist, ord=1)
    return hist


def get_feature_from_wordmap_SPM(opts, wordmap, dictionary):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    """

    K = opts.K
    L = opts.L  
    
    h, w = wordmap.shape
    hist_all = np.array([])
    
    #make sure that both the width and height of your image is divisible by 2^L
    #if not pad by reflecting
    row_padding = 0
    col_padding = 0
    if h % (2**L) != 0:
        row_padding = (int(h / (2**L)) + 1) * (2**L)
    if w % (2**L) != 0:
        col_padding = (int(w / (2**L)) + 1) * (2**L)
    wordmap = np.pad(wordmap, ((row_padding, 0), (col_padding, 0)), 'reflect')
    
    #create initial base from the wordmap for the finest features
    finest = (2**L) * (2**L)
    
    #initially calculate the histogram for the finest layer
    cell_size    = (2**L)
    num_patches  =  cell_size * cell_size
    layer_weight = 2**(L - L - 1)
    x_width = int(w / cell_size)
    y_width = int(h / cell_size)

    aggregation_hist = np.zeros([cell_size , cell_size, len(dictionary)])
    for i in range(cell_size):
        row_start = i*y_width
        row_end   = (i*y_width) + y_width
        for j in range(cell_size):
            col_start = j*x_width
            col_end   = (j*x_width) + x_width
            cropped_wordmap = wordmap[row_start:row_end, col_start:col_end]
            hist = get_feature_from_wordmap(opts, cropped_wordmap, dictionary, normalize = False)
            #add to aggreation
            aggregation_hist[i,j,:] = hist
            #normalize and weigh  
            hist = layer_weight * (hist / np.linalg.norm(hist, ord=1))
            hist_all = np.append(hist_all, hist)
            
    #from here onwards it's just aggregation
    for l in reversed(range(L)):
        layer_weight = 2**(l - L - 1)
        if l == 0 or l == 1:
            layer_weight = 2**(-L)
        
        h,w,c = aggregation_hist.shape
        cell_size    = (2**l)
        x_width = int(h / cell_size)
        y_width = int(w / cell_size)
        num_patches  =  cell_size * cell_size
        aggregation_hist_tmp = np.zeros([cell_size , cell_size, len(dictionary)])
        for i in range(cell_size):
            row_start = i*y_width
            row_end   = (i*y_width) + y_width
            for j in range(cell_size):
                col_start = j*x_width
                col_end   = (j*x_width) + x_width
                aggregation_hist_tmp[i,j] = sum(sum(aggregation_hist[row_start:row_end, col_start:col_end]))
                #normalize and weigh
                hist = layer_weight * (aggregation_hist_tmp[i,j] / np.linalg.norm(hist, ord=1))
                hist_all = np.append(hist_all, hist)
        aggregation_hist = aggregation_hist_tmp 
    return hist_all / np.linalg.norm(hist_all, ord=1)


def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """

    img = plt.imread(img_path) / 255.
    
    #extract word map from the image
    wordmap = get_visual_words(opts, img, dictionary)
    
    #computes the SPM
    feature = get_feature_from_wordmap_SPM(opts, wordmap, dictionary)

    return feature


def build_recognition_system(opts, dict_file_name, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    res_dir  = opts.res_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K = opts.K

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(res_dir, dict_file_name))

    # ----- TODO -----
    N = len(train_files)
    num_features = int(K * (4**(SPM_layer_num+1) - 1) / 3)
    features = np.zeros([N, num_features])
    
    for i, img_name in enumerate(train_files):
        img_path = join(data_dir, img_name)
        feature  = get_image_feature(opts, img_path, dictionary)
        features[i,:] = feature

    # example code snippet to save the learned system
    np.savez_compressed(join(res_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    """
    Compute distance between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * distance: numpy.ndarray of shape (N)
    """

    dists = np.minimum(histograms, word_hist)
    dists = np.sum(dists, axis = 1)
    
    return 1 - dists


def evaluate_recognition_system(opts, distance_func, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    res_dir  = opts.res_dir

    trained_system = np.load(join(res_dir, "trained_system.npz"))
    
    dictionary = trained_system["dictionary"]
    trained_features = trained_system["features"]
    train_labels = trained_system["labels"]
    
    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    N = len(test_files)
    num_features = test_opts.K * (4**(test_opts.L+1) - 1) / 3 
    
    conf = np.zeros((8,8))
    for i, img_name in enumerate(test_files):
        img_path = join(data_dir, img_name)
        feature  = get_image_feature(opts, img_path, dictionary)
        dist = distance_func(feature, trained_features)
        predicted = train_labels[np.argmin(dist)]
        conf[test_labels[i]][predicted] += 1

    accuracy = sum(np.diagonal(conf)) / N
    return conf, accuracy

def svm_classifier(opts, n_worker=1):
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    res_dir  = opts.res_dir

    trained_system = np.load(join(res_dir, "trained_system.npz"))
    
    dictionary = trained_system["dictionary"]
    trained_features = trained_system["features"]
    train_labels = trained_system["labels"]
    
    clf = svm.SVC()
    clf.fit(trained_features, train_labels)
    
    return clf

def evaluate_rec_svm(opts, clf):
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    res_dir  = opts.res_dir
    trained_system = np.load(join(res_dir, "trained_system.npz"))
    
    dictionary = trained_system["dictionary"]
    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)
    
    N = len(test_files)
    num_features = int(opts.K * (4**(opts.L+1) - 1) / 3)
    
    conf = np.zeros((8,8))

    test_features = np.zeros((len(test_files), num_features))
    
    for i, img_name in enumerate(test_files):
        img_path = join(data_dir, img_name)
        feature  = get_image_feature(opts, img_path, dictionary)
        test_features[i,:] = feature
        
    y_pred = clf.predict(test_features)
    
    for i, img_name in enumerate(test_files):
        conf[test_labels[i],y_pred[i]] += 1
        
    accuracy = sum(np.diagonal(conf)) / N
    return conf, accuracy
