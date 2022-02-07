'''
Hyperparameters wrapped in argparse
'''

import argparse


def get_opts():
    parser = argparse.ArgumentParser(
        description='16-720 HW1: Scene Recognition')

    # Paths
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='data folder')
    parser.add_argument('--feat_dir', type=str, default='../feat',
                        help='feature folder')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='output folder')

    # Visual words (requires tuning)
    parser.add_argument('--filter_scales', nargs='+', type=float,
                        default=[1, 2],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=10,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=25,
                        help='Using only a subset of alpha pixels in each image')

    # Recognition system (requires tuning)
    parser.add_argument('--L', type=int, default=1,
                        help='# of layers in spatial pyramid matching (SPM)')

    # Additional options (add your own hyperparameters here)
    parser.add_argument('--res_dir', type=str, default='./result',
                        help='result folder')
    parser.add_argument('--num_filters', type=int, default=4,
                        help='# of filters')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='# of color channels')
    parser.add_argument('--mode', type=str, default='reflect',
                        help='convolution mode')

    ##
    opts = parser.parse_args()
    return opts

def get_options():
    class Opts(object):
        
        def __init__(
            self,
            data_dir="../data",
            feat_dir="../feat",
            out_dir=".",
            filter_scales=[1,2],
            K=120,
            alpha=60,
            L=4,
            res_dir = "../result",
            num_filters = 4,
            num_channels  = 3,
            mode = "reflect"
        ):
            '''
            Manage tunable hyperparameters.

            [input]
            * data_dir: Data directory.
            * feat_dir: Feature directory.
            * out_dir: Output directory.
            * filter_scales: A list of scales for all the filters.
            * K: Number of words.
            * alpha: Subset of alpha pixels in each image.
            * L: Number of layers in spatial pyramid matching (SPM).

            '''
            self.data_dir = data_dir
            self.feat_dir = feat_dir
            self.out_dir = out_dir
            self.filter_scales = list(filter_scales)
            self.K = K
            self.alpha = alpha
            self.L = L
            self.num_filters = num_filters
            self.num_channels  = num_channels
            self.mode = mode
            self.res_dir = res_dir

    return Opts()
