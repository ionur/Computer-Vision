'''
Hyperparameters wrapped in argparse

'''

import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='Homography')

    ## Feature detection (requires tuning)
    parser.add_argument('--sigma', type=float, default=0.15,
                        help='threshold for corner detection using FAST feature detector')
    parser.add_argument('--ratio', type=float, default=0.7,
                        help='ratio for BRIEF feature descriptor')

    ## Ransac (requires tuning)
    parser.add_argument('--max_iters', type=int, default=500,
                        help='the number of iterations to run RANSAC for')
    parser.add_argument('--inlier_tol', type=float, default=2.0,
                        help='the tolerance value for considering a point to be an inlier')


    ##
    opts = parser.parse_args()

    return opts
